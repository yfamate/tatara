//! YaneuraOu SFNNWithoutPsqt evaluation-file serialization.
//!
//! Two writers share the byte layout of the affine blocks:
//!
//! - [`save_yaneuraou`]: the YaneuraOu KingRank9 file and the fixed Cyfamate GP5 file
//!   (`1024 / 7+1 / 64 / 5 stacks`, boundaries `0.4,0.6,0.8,1`).
//! - [`save_yaneuraou_cyfamate`]: the Cyfamate GP-bucket research file. The header
//!   carries the boundary list (`Bucket=GP(...)`), the FT width, the fc_0 width (7 or 15
//!   + shortcut) and, when the net has threat features, the threat tokens
//!   (`Threat=<dims>,ThreatProfile=<id>,ThreatMirror=off`). A threat block (raw i8 rows
//!   at the FT scale, preceded by the `u32` profile id, as in the tatara `.bin`) follows
//!   the FT block. [`read_yaneuraou_cyfamate`] reads both forms back.

use std::io::{self, BufRead, Read, Write};

use shogi_features::{FeatureSet, FeatureSetSpec, ThreatMirror, ThreatProfile};

use crate::LayerStackWeights;
use crate::layerstack_weights::{QA, QB, pad32, read_leb128_tensor_i16, write_leb128_tensor_i16};

const YO_VERSION: u32 = 0x7af3_2f16;
const YO_TOP_HASH: u32 = 0x3c20_3b32;
const YO_FT_HASH: u32 = 0x5f13_4ab8;
const YO_NETWORK_HASH: u32 = 0x6333_718a;
/// Cyfamate GP5 identity tag ("5GP\0").
pub const GP5_HASH_TAG: u32 = 0x3547_5000;
/// Cyfamate GP5 H15 prototype tag ("1H15"): fc_0 hidden 7 -> 15 (l1_out 16). Old builds reject it.
pub const GP5_H15_HASH_TAG: u32 = 0x3531_4831;
pub const GP5_ARCH: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA2(Friend)[131949->1024x2],Network=SFNN_HALFKA2_1024_7_64_5GP{LayerStack=5;Bucket=GP(0.4,0.6,0.8,1)}";
pub const GP5_H15_ARCH: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA2(Friend)[131949->1024x2],Network=SFNN_HALFKA2_1024_15_64_5GP{LayerStack=5;Bucket=GP(0.4,0.6,0.8,1)}";

/// Top-level hash of a 5-bucket GP5 file for the given L1 output count (8 or 16).
pub fn gp5_top_hash(l1_out: usize) -> u32 {
    YO_TOP_HASH ^ GP5_HASH_TAG ^ if l1_out == 16 { GP5_H15_HASH_TAG } else { 0 }
}

/// Cyfamate GP-bucket research tag ("GPK\0"): the boundary list is read from the header
/// instead of being a compile-time constant of the engine.
pub const GPK_HASH_TAG: u32 = 0x004B_5047;
/// Cyfamate threat-block tag ("THRT"): the file carries a threat FT block after the base FT.
pub const THREAT_HASH_TAG: u32 = 0x5452_4854;
/// Cyfamate half-mirror tag ("HM\0\0"): base FT rows are `halfka-hm-merged` (45 king buckets).
pub const HM_HASH_TAG: u32 = 0x0000_4D48;
/// Boundaries of the fixed GP5 routing (`Bucket=GP(0.4,0.6,0.8,1)`).
pub const GP5_FIXED_BOUNDS: [f64; 4] = [0.4, 0.6, 0.8, 1.0];
/// Largest stack count the Cyfamate GP-bucket format accepts (tatara `GpBounds::MAX_LEN` + 1).
pub const CYFAMATE_MAX_STACKS: usize = 9;
/// Alternative per-stack hash some GP5 files carry (accepted on read, never written).
const YO_NETWORK_HASH_GP5_ALT: u32 = 0x6333_76a4;
/// Alternative FT hash some GP5 files carry (accepted on read, never written).
const YO_FT_HASH_GP5_ALT: u32 = 0x5f13_48b8;
const YO_LEB128_PREFIX: u8 = b'_';

/// Architecture of a Cyfamate GP-bucket evaluation file.
#[derive(Clone, Debug, PartialEq)]
pub struct CyfamateArch {
    /// FT output width per perspective (multiple of 128).
    pub ft_out: usize,
    /// fc_0 outputs including the shortcut: 8 (7+1) or 16 (15+1).
    pub l1_out: usize,
    /// fc_1 width (64).
    pub l2_out: usize,
    /// GP boundaries; the file has `gp_bounds.len() + 1` stacks.
    pub gp_bounds: Vec<f64>,
    /// Threat profile and square normalisation of the threat block, if any.
    pub threat: Option<(ThreatProfile, ThreatMirror)>,
    /// Half-mirrored base features (`halfka-hm-merged`, 73,305 rows; header token
    /// `Features=HalfKA2_hm(Friend)`). The threat block, if any, keeps raw squares.
    pub hm: bool,
}

impl CyfamateArch {
    /// Number of LayerStacks.
    pub fn buckets(&self) -> usize {
        self.gp_bounds.len() + 1
    }

    /// The fixed GP5 form that [`save_yaneuraou`] writes and old engine builds read.
    pub fn is_fixed_gp5(&self) -> bool {
        self.threat.is_none()
            && !self.hm
            && self.ft_out == 1024
            && (self.l1_out == 8 || self.l1_out == 16)
            && self.l2_out == 64
            && self.gp_bounds == GP5_FIXED_BOUNDS
    }

    /// Top-level hash the engine checks before the architecture string.
    pub fn top_hash(&self) -> u32 {
        if self.is_fixed_gp5() {
            return gp5_top_hash(self.l1_out);
        }
        YO_TOP_HASH
            ^ GP5_HASH_TAG
            ^ GPK_HASH_TAG
            ^ if self.l1_out == 16 { GP5_H15_HASH_TAG } else { 0 }
            ^ if self.threat.is_some() { THREAT_HASH_TAG } else { 0 }
            ^ if self.hm { HM_HASH_TAG } else { 0 }
    }

    /// Base feature set of the FT rows.
    pub fn base_feature_set(&self) -> FeatureSet {
        if self.hm { FeatureSet::HalfKaHmMerged } else { FeatureSet::HalfKaMerged }
    }

    /// Architecture string. The fixed GP5 form reproduces [`GP5_ARCH`] / [`GP5_H15_ARCH`].
    pub fn arch_string(&self) -> String {
        let spec = self.base_feature_set().spec();
        let feature_name = if self.hm { "HalfKA2_hm" } else { "HalfKA2" };
        let threat = match self.threat {
            None => String::new(),
            Some((profile, mirror)) => format!(
                ",Threat={},ThreatProfile={}{}",
                shogi_features::threat_dimensions_of(profile),
                profile.profile_id(),
                if mirror == ThreatMirror::Off { ",ThreatMirror=off" } else { "" }
            ),
        };
        let bounds = self
            .gp_bounds
            .iter()
            .map(|&b| format_bound(b))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "ModelType=SFNNWithoutPsqt;Features={feature_name}(Friend)[{}->{}x2]{threat},Network=SFNN_HALFKA2_{}_{}_{}_{}GP{{LayerStack={};Bucket=GP({bounds})}}",
            spec.ft_in(),
            self.ft_out,
            self.ft_out,
            self.l1_out - 1,
            self.l2_out,
            self.buckets(),
            self.buckets(),
        )
    }

    /// Feature-set spec of the weights this file holds.
    pub fn spec(&self) -> FeatureSetSpec {
        let base = self.base_feature_set().spec();
        match self.threat {
            None => base,
            Some((profile, mirror)) => base.with_threat_profile(profile).with_threat_mirror(mirror),
        }
    }

    /// Validates the dimensions and boundaries the engine research build accepts.
    pub fn validate(&self) -> io::Result<()> {
        if self.ft_out == 0 || self.ft_out % 128 != 0 || self.ft_out > MAX_FT_OUT {
            return invalid_input(format!(
                "Cyfamate GP format needs an FT width that is a positive multiple of 128 (got {})",
                self.ft_out
            ));
        }
        if self.l1_out != 8 && self.l1_out != 16 {
            return invalid_input(format!(
                "Cyfamate GP format needs fc_0 7+1 or 15+1 outputs (l1_out 8 or 16, got {})",
                self.l1_out
            ));
        }
        if self.l2_out != 64 {
            return invalid_input(format!(
                "Cyfamate GP format needs fc_1 width 64 (got {})",
                self.l2_out
            ));
        }
        let n = self.buckets();
        if n > CYFAMATE_MAX_STACKS {
            return invalid_input(format!(
                "Cyfamate GP format supports at most {CYFAMATE_MAX_STACKS} stacks (got {n})"
            ));
        }
        for (i, &b) in self.gp_bounds.iter().enumerate() {
            if !b.is_finite() || b <= 0.0 || b > 1.5 {
                return invalid_input(format!("GP boundary {b} is outside (0, 1.5]"));
            }
            if i > 0 && b <= self.gp_bounds[i - 1] {
                return invalid_input(format!(
                    "GP boundaries must be strictly increasing ({} then {b})",
                    self.gp_bounds[i - 1]
                ));
            }
        }
        if let Some((_, mirror)) = self.threat
            && mirror != ThreatMirror::Off
        {
            return invalid_input(
                "the Cyfamate threat indexer uses raw squares; export a net trained with \
                 --threat-mirror off",
            );
        }
        Ok(())
    }
}

/// Shortest round-trip decimal of a boundary with a trailing `.0` removed (`1.0` → `1`),
/// so the fixed GP5 string stays `GP(0.4,0.6,0.8,1)` and other values survive `strtod`.
fn format_bound(b: f64) -> String {
    let s = format!("{b:?}");
    match s.strip_suffix(".0") {
        Some(t) if !t.contains('e') => t.to_string(),
        _ => s,
    }
}

/// Parses the `Bucket=GP(...)` list.
fn parse_bounds(list: &str) -> io::Result<Vec<f64>> {
    if list.trim().is_empty() {
        return Ok(Vec::new());
    }
    list.split(',')
        .map(|item| {
            item.trim()
                .parse::<f64>()
                .map_err(|_| invalid_data_err(format!("GP boundary `{item}` is not a number")))
        })
        .collect()
}

fn profile_from_id(id: u32) -> Option<ThreatProfile> {
    [
        ThreatProfile::Full,
        ThreatProfile::SameClass,
        ThreatProfile::SameClassMajorPawn,
        ThreatProfile::StepAttacker,
        ThreatProfile::FullSymDedup,
        ThreatProfile::CrossSide,
    ]
    .into_iter()
    .find(|p| p.profile_id() == id)
}

/// Parses a Cyfamate GP-bucket architecture string (either form).
pub fn parse_cyfamate_arch(arch: &str) -> io::Result<CyfamateArch> {
    let prefix = format!(
        "ModelType=SFNNWithoutPsqt;Features=HalfKA2(Friend)[{}->",
        FeatureSet::HalfKaMerged.spec().ft_in()
    );
    let prefix_hm = format!(
        "ModelType=SFNNWithoutPsqt;Features=HalfKA2_hm(Friend)[{}->",
        FeatureSet::HalfKaHmMerged.spec().ft_in()
    );
    let (hm, rest) = match (arch.strip_prefix(&prefix), arch.strip_prefix(&prefix_hm)) {
        (Some(rest), _) => (false, rest),
        (None, Some(rest)) => (true, rest),
        (None, None) => {
            return invalid_data(format!(
                "not a Cyfamate HalfKA2 architecture (expected prefix `{prefix}` or `{prefix_hm}`): `{arch}`"
            ));
        }
    };
    let (ft_out_str, rest) = rest
        .split_once("x2]")
        .ok_or_else(|| invalid_data_err(format!("Features token has no `<ft>x2]`: `{arch}`")))?;
    let ft_out = parse_usize_data(ft_out_str)?;
    let (tokens, network) = rest
        .split_once(",Network=")
        .ok_or_else(|| invalid_data_err(format!("architecture has no `,Network=`: `{arch}`")))?;
    let mut threat_dims: Option<usize> = None;
    let mut threat_id: Option<u32> = None;
    let mut mirror_off = false;
    for tok in tokens.split(',').filter(|t| !t.is_empty()) {
        if let Some(v) = tok.strip_prefix("Threat=") {
            threat_dims = Some(parse_usize_data(v)?);
        } else if let Some(v) = tok.strip_prefix("ThreatProfile=") {
            threat_id = Some(parse_usize_data(v)? as u32);
        } else if let Some(v) = tok.strip_prefix("ThreatMirror=") {
            match v {
                "off" => mirror_off = true,
                "hm" => mirror_off = false,
                other => return invalid_data(format!("unknown ThreatMirror value `{other}`")),
            }
        } else {
            return invalid_data(format!("unknown architecture token `{tok}`: `{arch}`"));
        }
    }
    let threat = match (threat_dims, threat_id) {
        (None, None) => {
            if mirror_off {
                return invalid_data("ThreatMirror=off without a Threat token");
            }
            None
        }
        (Some(dims), id) => {
            let id = id.unwrap_or(0);
            let profile = profile_from_id(id)
                .ok_or_else(|| invalid_data_err(format!("unknown ThreatProfile id {id}")))?;
            let expected = shogi_features::threat_dimensions_of(profile);
            if dims != expected {
                return invalid_data(format!(
                    "Threat dims {dims} disagree with profile {profile} ({expected})"
                ));
            }
            Some((
                profile,
                if mirror_off {
                    ThreatMirror::Off
                } else {
                    ThreatMirror::Hm
                },
            ))
        }
        (None, Some(_)) => return invalid_data("ThreatProfile token without Threat token"),
    };
    let (name, brace) = network
        .split_once('{')
        .ok_or_else(|| invalid_data_err(format!("Network token has no `{{`: `{network}`")))?;
    let brace = brace
        .strip_suffix('}')
        .ok_or_else(|| invalid_data_err(format!("Network token has no closing `}}`: `{network}`")))?;
    let body = name.strip_prefix("SFNN_HALFKA2_").ok_or_else(|| {
        invalid_data_err(format!("Network `{name}` is not SFNN_HALFKA2_<ft>_<h1>_<h2>_<n>GP"))
    })?;
    let parts: Vec<&str> = body.split('_').collect();
    if parts.len() != 4 {
        return invalid_data(format!("Network `{name}` is not SFNN_HALFKA2_<ft>_<h1>_<h2>_<n>GP"));
    }
    if parse_usize_data(parts[0])? != ft_out {
        return invalid_data(format!("Network `{name}` FT width disagrees with Features ({ft_out})"));
    }
    let h1 = parse_usize_data(parts[1])?;
    let l2_out = parse_usize_data(parts[2])?;
    let n_name = parse_usize_data(parts[3].strip_suffix("GP").ok_or_else(|| {
        invalid_data_err(format!("Network `{name}` must end with `<n>GP`"))
    })?)?;
    let mut layer_stack: Option<usize> = None;
    let mut bounds: Option<Vec<f64>> = None;
    for item in brace.split(';') {
        if let Some(v) = item.strip_prefix("LayerStack=") {
            layer_stack = Some(parse_usize_data(v)?);
        } else if let Some(v) = item.strip_prefix("Bucket=GP(") {
            let v = v
                .strip_suffix(')')
                .ok_or_else(|| invalid_data_err(format!("Bucket token has no `)`: `{item}`")))?;
            bounds = Some(parse_bounds(v)?);
        } else {
            return invalid_data(format!("unknown Network attribute `{item}`: `{arch}`"));
        }
    }
    let gp_bounds = bounds.ok_or_else(|| invalid_data_err(format!("no Bucket=GP(...) in `{arch}`")))?;
    let layer_stack = layer_stack.ok_or_else(|| invalid_data_err(format!("no LayerStack= in `{arch}`")))?;
    if layer_stack != gp_bounds.len() + 1 || n_name != layer_stack {
        return invalid_data(format!(
            "LayerStack={layer_stack}, name {n_name}GP and {} boundaries disagree: `{arch}`",
            gp_bounds.len()
        ));
    }
    if h1 + 1 > MAX_HIDDEN_DIM {
        return invalid_data(format!("fc_0 width {h1} out of range"));
    }
    let parsed = CyfamateArch {
        ft_out,
        l1_out: h1 + 1,
        l2_out,
        gp_bounds,
        threat,
        hm,
    };
    parsed.validate()?;
    Ok(parsed)
}

fn parse_usize_data(value: &str) -> io::Result<usize> {
    value
        .parse::<usize>()
        .map_err(|error| invalid_data_err(format!("expected integer, got `{value}`: {error}")))
}

/// Writes a Cyfamate GP-bucket evaluation file. `gp_bounds` has `num_buckets - 1`
/// entries. The fixed GP5 shape (`1024 / 7+1 or 15+1 / 64 / 5`, boundaries
/// `0.4,0.6,0.8,1`, no threat) is written exactly as [`save_yaneuraou`] does.
pub fn save_yaneuraou_cyfamate<W: Write>(
    writer: &mut W,
    weights: &LayerStackWeights,
    gp_bounds: &[f64],
) -> io::Result<()> {
    let spec = weights.feature_set;
    let hm = match spec.feature_set() {
        FeatureSet::HalfKaMerged => false,
        FeatureSet::HalfKaHmMerged => true,
        _ => {
            return invalid_input(
                "Cyfamate GP format needs the halfka-merged (HalfKA2) or halfka-hm-merged feature set",
            );
        }
    };
    if spec.effect_bucket_config().is_some() {
        return invalid_input("effect bucket models are not representable in Cyfamate GP format");
    }
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT models are not representable in Cyfamate GP format");
    }
    if weights.num_buckets != gp_bounds.len() + 1 {
        return invalid_input(format!(
            "{} boundaries define {} stacks, but the weights have {} buckets",
            gp_bounds.len(),
            gp_bounds.len() + 1,
            weights.num_buckets
        ));
    }
    let arch = CyfamateArch {
        ft_out: weights.ft_b.len(),
        l1_out: weights.l1_shared_bias.len(),
        l2_out: weights.l2_b.len().checked_div(weights.num_buckets).unwrap_or(0),
        gp_bounds: gp_bounds.to_vec(),
        threat: spec.threat_profile().map(|p| (p, spec.threat_mirror())),
        hm,
    };
    arch.validate()?;
    if arch.is_fixed_gp5() {
        return save_yaneuraou(writer, weights);
    }
    let ft_out = arch.ft_out;
    let l1_out = arch.l1_out;
    let l2_out = arch.l2_out;
    let l2_in = (l1_out - 1) * 2;
    let buckets = arch.buckets();
    let base_ft_w_n = spec.base_ft_in() * ft_out;
    let lengths = [
        ("ft_b", weights.ft_b.len(), ft_out),
        ("ft_w", weights.ft_w.len(), spec.ft_in() * ft_out),
        ("l1_b", weights.l1_b.len(), buckets * l1_out),
        ("l1_w", weights.l1_w.len(), buckets * l1_out * ft_out),
        ("l1_shared_bias", weights.l1_shared_bias.len(), l1_out),
        ("l1_shared_weight", weights.l1_shared_weight.len(), ft_out * l1_out),
        ("l2_b", weights.l2_b.len(), buckets * l2_out),
        ("l2_w", weights.l2_w.len(), buckets * l2_out * l2_in),
        ("l3_b", weights.l3_b.len(), buckets),
        ("l3_w", weights.l3_w.len(), buckets * l2_out),
    ];
    for (name, actual, expected) in lengths {
        if actual != expected {
            return invalid_input(format!(
                "{name} length mismatch: expected {expected}, got {actual}"
            ));
        }
    }

    write_u32(writer, YO_VERSION)?;
    write_u32(writer, arch.top_hash())?;
    let arch_string = arch.arch_string();
    write_u32(writer, u32::try_from(arch_string.len()).expect("architecture string fits"))?;
    writer.write_all(arch_string.as_bytes())?;

    write_u32(writer, YO_FT_HASH)?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_b, QA as f64))?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_w[..base_ft_w_n], QA as f64))?;
    if let Some((profile, _)) = arch.threat {
        // Same block as the tatara `.bin`: u32 profile id, then raw i8 rows at the FT scale.
        write_u32(writer, profile.profile_id())?;
        let threat_w = &weights.ft_w[base_ft_w_n..];
        let mut bytes = Vec::with_capacity(threat_w.len());
        for &v in threat_w {
            let q = (v as f64 * QA as f64).round().clamp(-127.0, 127.0) as i8;
            bytes.push(q as u8);
        }
        writer.write_all(&bytes)?;
    }

    for bucket in 0..buckets {
        write_u32(writer, YO_NETWORK_HASH)?;
        let l1_biases = (0..l1_out)
            .map(|output| weights.l1_b[bucket * l1_out + output] + weights.l1_shared_bias[output]);
        let l1_weights = (0..l1_out).flat_map(|output| {
            (0..ft_out).map(move |input| {
                weights.l1_w[bucket * l1_out * ft_out + output * ft_out + input]
                    + weights.l1_shared_weight[input * l1_out + output]
            })
        });
        write_affine(writer, l1_biases, l1_weights, ft_out, l1_out)?;
        let l2_biases = (0..l2_out).map(|output| weights.l2_b[bucket * l2_out + output]);
        let l2_weights = (0..l2_out).flat_map(|output| {
            (0..l2_in)
                .map(move |input| weights.l2_w[bucket * l2_out * l2_in + output * l2_in + input])
        });
        write_affine(writer, l2_biases, l2_weights, l2_in, l2_out)?;
        write_affine(
            writer,
            std::iter::once(weights.l3_b[bucket]),
            (0..l2_out).map(|input| weights.l3_w[bucket * l2_out + input]),
            l2_out,
            1,
        )?;
    }
    Ok(())
}

/// Reads a Cyfamate GP-bucket evaluation file (either form) back into trainer weights.
/// The merged fc_0 goes to `l1_w` with the shared term zero, as `load_quantised` does.
pub fn read_yaneuraou_cyfamate<R: BufRead>(
    reader: &mut R,
) -> io::Result<(LayerStackWeights, CyfamateArch)> {
    let version = read_u32(reader)?;
    if version != YO_VERSION {
        return invalid_data(format!("version mismatch: expected {YO_VERSION:#010x}, got {version:#010x}"));
    }
    let top_hash = read_u32(reader)?;
    let arch_len = read_u32(reader)? as usize;
    if arch_len == 0 || arch_len > 16_384 {
        return invalid_data(format!("invalid architecture string length: {arch_len}"));
    }
    let mut arch_bytes = vec![0_u8; arch_len];
    reader.read_exact(&mut arch_bytes)?;
    let arch_str = std::str::from_utf8(&arch_bytes)
        .map_err(|e| invalid_data_err(format!("architecture string is not UTF-8: {e}")))?;
    let arch = parse_cyfamate_arch(arch_str)?;
    if top_hash != arch.top_hash() {
        return invalid_data(format!(
            "top-level hash {top_hash:#010x} disagrees with architecture `{arch_str}` (expected {:#010x})",
            arch.top_hash()
        ));
    }
    let ft_hash = read_u32(reader)?;
    if ft_hash != YO_FT_HASH && !(arch.is_fixed_gp5() && ft_hash == YO_FT_HASH_GP5_ALT) {
        return invalid_data("invalid feature-transformer hash");
    }
    if reader.fill_buf()?.first() == Some(&YO_LEB128_PREFIX) {
        reader.consume(1);
    }
    let spec = arch.spec();
    let ft_out = arch.ft_out;
    let l1_out = arch.l1_out;
    let l2_out = arch.l2_out;
    let l2_in = (l1_out - 1) * 2;
    let buckets = arch.buckets();
    let qa = QA as f32;
    let mut weights = LayerStackWeights::zeroed(spec, ft_out, l1_out, l2_out, buckets);
    let ft_b = read_leb128_tensor_i16(reader, Some(ft_out))?;
    let ft_w = read_leb128_tensor_i16(reader, Some(spec.base_ft_in() * ft_out))?;
    weights.ft_b = ft_b.into_iter().map(|v| v as f32 / qa).collect();
    let mut ft_w: Vec<f32> = ft_w.into_iter().map(|v| v as f32 / qa).collect();
    if let Some((profile, _)) = arch.threat {
        let id = read_u32(reader)?;
        if id != profile.profile_id() {
            return invalid_data(format!(
                "threat block profile id {id} disagrees with header profile {}",
                profile.profile_id()
            ));
        }
        let n = spec.threat_dims() * ft_out;
        let mut bytes = vec![0_u8; n];
        reader.read_exact(&mut bytes)?;
        ft_w.extend(bytes.into_iter().map(|b| b as i8 as f32 / qa));
    }
    weights.ft_w = ft_w;
    for bucket in 0..buckets {
        let hash = read_u32(reader)?;
        if hash != YO_NETWORK_HASH && !(arch.is_fixed_gp5() && hash == YO_NETWORK_HASH_GP5_ALT) {
            return invalid_data(format!("invalid LayerStack hash at stack {bucket}"));
        }
        let (biases, dense) = read_affine(reader, ft_out, l1_out)?;
        weights.l1_b[bucket * l1_out..(bucket + 1) * l1_out].copy_from_slice(&biases);
        let start = bucket * l1_out * ft_out;
        weights.l1_w[start..start + dense.len()].copy_from_slice(&dense);
        let (biases, dense) = read_affine(reader, l2_in, l2_out)?;
        weights.l2_b[bucket * l2_out..(bucket + 1) * l2_out].copy_from_slice(&biases);
        let start = bucket * l2_out * l2_in;
        weights.l2_w[start..start + dense.len()].copy_from_slice(&dense);
        let (biases, dense) = read_affine(reader, l2_out, 1)?;
        weights.l3_b[bucket] = biases[0];
        weights.l3_w[bucket * l2_out..(bucket + 1) * l2_out].copy_from_slice(&dense);
    }
    Ok((weights, arch))
}

fn read_affine<R: Read>(
    reader: &mut R,
    input_dimensions: usize,
    output_dimensions: usize,
) -> io::Result<(Vec<f32>, Vec<f32>)> {
    let bias_scale = (QA * QB) as f32;
    let mut biases = Vec::with_capacity(output_dimensions);
    for _ in 0..output_dimensions {
        biases.push(read_i32(reader)? as f32 / bias_scale);
    }
    let weight_scale = QB as f32;
    let padded = pad32(input_dimensions);
    let mut weights = Vec::with_capacity(output_dimensions * input_dimensions);
    let mut row = vec![0_u8; padded];
    for _ in 0..output_dimensions {
        reader.read_exact(&mut row)?;
        weights.extend(
            row[..input_dimensions]
                .iter()
                .map(|&v| v as i8 as f32 / weight_scale),
        );
    }
    Ok((biases, weights))
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut b = [0_u8; 4];
    reader.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32<R: Read>(reader: &mut R) -> io::Result<i32> {
    let mut b = [0_u8; 4];
    reader.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn invalid_data<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_data_err(message.into()))
}

fn invalid_data_err(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// YaneuraOu SFNN が要求する KingRank9 LayerStack 数。
pub const YANEURAOU_LAYER_STACKS: usize = 9;

const MAX_FT_OUT: usize = 8192;
const MAX_HIDDEN_DIM: usize = 4096;

struct YoFeature {
    feature_set: FeatureSet,
    yo_name: &'static str,
    gen_key: &'static str,
}

const YO_FEATURES: [YoFeature; 5] = [
    YoFeature {
        feature_set: FeatureSet::HalfKp,
        yo_name: "HalfKP",
        gen_key: "halfkp",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaSplit,
        yo_name: "HalfKA1",
        gen_key: "halfka1",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaMerged,
        yo_name: "HalfKA2",
        gen_key: "halfka2",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaHmSplit,
        yo_name: "HalfKA_hm1",
        gen_key: "halfkahm1",
    },
    YoFeature {
        feature_set: FeatureSet::HalfKaHmMerged,
        yo_name: "HalfKA_hm2",
        gen_key: "halfkahm2",
    },
];

/// LayerStack weights を YaneuraOu SFNNWithoutPsqt 形式で書き出す。
///
/// feature set と各層次元は weights の shape から決定する。YaneuraOu SFNN が
/// 表現できない拡張 feature、PSQT、KingRank9 / Cyfamate GP5 以外は reject する。
/// bucket routing mode 自体は weights に含まれないため、caller は学習 config 等から
/// 9 組は KingRank9、5 組は GP5 であることを確認してから呼ぶ必要がある。
pub fn save_yaneuraou<W: Write>(writer: &mut W, weights: &LayerStackWeights) -> io::Result<()> {
    let arch = architecture(weights)?;
    validate_weights(&arch, weights)?;

    let ft_out = arch.ft_out;
    let l1_out = arch.l1_out;
    let l2_out = arch.l2_out;
    let l2_in = (l1_out - 1) * 2;

    write_u32(writer, YO_VERSION)?;
    write_u32(
        writer,
        if arch.buckets == 5 {
            gp5_top_hash(arch.l1_out)
        } else {
            YO_TOP_HASH
        },
    )?;
    let arch_string = arch_string(&arch);
    write_u32(
        writer,
        u32::try_from(arch_string.len()).expect("architecture string length fits in u32"),
    )?;
    writer.write_all(arch_string.as_bytes())?;

    write_u32(writer, YO_FT_HASH)?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_b, QA as f64))?;
    write_leb128_tensor_i16(writer, &quantize_i16(&weights.ft_w, QA as f64))?;

    for bucket in 0..arch.buckets {
        write_u32(writer, YO_NETWORK_HASH)?;

        // L1 shared term は通常 export 前に L1 へ fold 済み。未 fold の weights を
        // caller が渡した場合にも同じ推論 weight になるよう加算する。
        let l1_biases = (0..l1_out)
            .map(|output| weights.l1_b[bucket * l1_out + output] + weights.l1_shared_bias[output]);
        let l1_weights = (0..l1_out).flat_map(|output| {
            (0..ft_out).map(move |input| {
                weights.l1_w[bucket * l1_out * ft_out + output * ft_out + input]
                    + weights.l1_shared_weight[input * l1_out + output]
            })
        });
        write_affine(writer, l1_biases, l1_weights, ft_out, l1_out)?;

        let l2_biases = (0..l2_out).map(|output| weights.l2_b[bucket * l2_out + output]);
        let l2_weights = (0..l2_out).flat_map(|output| {
            (0..l2_in)
                .map(move |input| weights.l2_w[bucket * l2_out * l2_in + output * l2_in + input])
        });
        write_affine(writer, l2_biases, l2_weights, l2_in, l2_out)?;

        write_affine(
            writer,
            std::iter::once(weights.l3_b[bucket]),
            (0..l2_out).map(|input| weights.l3_w[bucket * l2_out + input]),
            l2_out,
            1,
        )?;
    }
    Ok(())
}

#[derive(Debug)]
struct Architecture {
    buckets: usize,
    feature_set: FeatureSet,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
}

fn architecture(weights: &LayerStackWeights) -> io::Result<Architecture> {
    let feature_set = FeatureSet::ALL
        .into_iter()
        .find(|feature_set| feature_set.spec() == weights.feature_set)
        .ok_or_else(|| invalid_input_err("feature set is not representable in YaneuraOu SFNN"))?;
    let ft_out = weights.ft_b.len();
    let l1_out = weights.l1_shared_bias.len();
    let num_buckets = weights.num_buckets;
    if num_buckets != YANEURAOU_LAYER_STACKS && num_buckets != 5 {
        return invalid_input(format!(
            "YaneuraOu SFNN requires {} LayerStacks (KingRank9), but weights have {num_buckets} buckets",
            YANEURAOU_LAYER_STACKS
        ));
    }
    let l2_out = weights.l2_b.len().checked_div(num_buckets).unwrap_or(0);
    if num_buckets == 5
        && (feature_set != FeatureSet::HalfKaMerged
            || ft_out != 1024
            || (l1_out != 8 && l1_out != 16)
            || l2_out != 64)
    {
        return invalid_input("Cyfamate GP5 requires HalfKA2 / 1024 / 8 (or 16, H15) / 64 / 5 weights");
    }
    Ok(Architecture {
        buckets: num_buckets,
        feature_set,
        ft_out,
        l1_out,
        l2_out,
    })
}

fn arch_string(arch: &Architecture) -> String {
    if arch.buckets == 5 {
        return if arch.l1_out == 16 { GP5_H15_ARCH } else { GP5_ARCH }.into();
    }
    let feature = YO_FEATURES
        .iter()
        .find(|feature| feature.feature_set == arch.feature_set)
        .expect("every FeatureSet has a YaneuraOu mapping");
    let input_size = arch.feature_set.spec().ft_in();
    let h1 = arch.l1_out - 1;
    let network = if arch.feature_set == FeatureSet::HalfKaHmMerged
        && arch.ft_out == 1536
        && arch.l1_out == 16
        && arch.l2_out == 32
    {
        "SFNN-1536".to_string()
    } else {
        format!(
            "SFNN_{}_{}_{}_{}_k3k3",
            feature.gen_key, arch.ft_out, h1, arch.l2_out
        )
        .to_ascii_uppercase()
    };
    format!(
        "ModelType=SFNNWithoutPsqt;Features={}(Friend)[{input_size}->{}x2],Network={network}{{LayerStack={YANEURAOU_LAYER_STACKS}}}",
        feature.yo_name, arch.ft_out
    )
}

fn validate_weights(arch: &Architecture, weights: &LayerStackWeights) -> io::Result<()> {
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT models are not representable in YaneuraOu SFNN");
    }
    if arch.ft_out == 0 || arch.ft_out > MAX_FT_OUT || !arch.ft_out.is_multiple_of(32) {
        return invalid_input(format!(
            "unsupported FT output dimension {} (expected a positive multiple of 32 up to {MAX_FT_OUT})",
            arch.ft_out
        ));
    }
    if arch.l1_out < 2 || arch.l1_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L1 output dimension {} (expected 2..={MAX_HIDDEN_DIM})",
            arch.l1_out
        ));
    }
    if arch.l2_out == 0 || arch.l2_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L2 output dimension {} (expected 1..={MAX_HIDDEN_DIM})",
            arch.l2_out
        ));
    }
    let l2_in = (arch.l1_out - 1) * 2;
    let spec = arch.feature_set.spec();
    let lengths = [
        ("ft_b", weights.ft_b.len(), arch.ft_out),
        ("ft_w", weights.ft_w.len(), spec.ft_in() * arch.ft_out),
        ("l1_b", weights.l1_b.len(), arch.buckets * arch.l1_out),
        (
            "l1_w",
            weights.l1_w.len(),
            arch.buckets * arch.l1_out * arch.ft_out,
        ),
        ("l1_shared_bias", weights.l1_shared_bias.len(), arch.l1_out),
        (
            "l1_shared_weight",
            weights.l1_shared_weight.len(),
            arch.ft_out * arch.l1_out,
        ),
        ("l2_b", weights.l2_b.len(), arch.buckets * arch.l2_out),
        ("l3_b", weights.l3_b.len(), arch.buckets),
        (
            "l2_w",
            weights.l2_w.len(),
            arch.buckets * arch.l2_out * l2_in,
        ),
        ("l3_w", weights.l3_w.len(), arch.buckets * arch.l2_out),
    ];
    for (name, actual, expected) in lengths {
        if actual != expected {
            return invalid_input(format!(
                "{name} length mismatch: expected {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

fn write_affine<W, B, V>(
    writer: &mut W,
    biases: B,
    weights: V,
    input_dimensions: usize,
    output_dimensions: usize,
) -> io::Result<()>
where
    W: Write,
    B: IntoIterator<Item = f32>,
    V: IntoIterator<Item = f32>,
{
    for bias in biases {
        writer.write_all(&quantize_i32(bias, (QA * QB) as f64).to_le_bytes())?;
    }
    let padded_input = input_dimensions.div_ceil(32) * 32;
    let mut weights = weights.into_iter();
    for _ in 0..output_dimensions {
        for input in 0..padded_input {
            let value = if input < input_dimensions {
                weights
                    .next()
                    .ok_or_else(|| invalid_input_err("affine weight iterator is short"))?
            } else {
                0.0
            };
            writer.write_all(&[quantize_i8(value, QB as f64) as u8])?;
        }
    }
    if weights.next().is_some() {
        return invalid_input("affine weight iterator has extra values");
    }
    Ok(())
}

fn quantize_i16(values: &[f32], scale: f64) -> Vec<i16> {
    values
        .iter()
        .map(|&value| {
            (value as f64 * scale)
                .round()
                .clamp(i16::MIN as f64, i16::MAX as f64) as i16
        })
        .collect()
}

fn quantize_i32(value: f32, scale: f64) -> i32 {
    (value as f64 * scale)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn quantize_i8(value: f32, scale: f64) -> i8 {
    (value as f64 * scale)
        .round()
        .clamp(i8::MIN as f64, i8::MAX as f64) as i8
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_input_err(message))
}

fn invalid_input_err(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gp5_export_has_engine_header_and_five_stacks() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaMerged.spec(), 1024, 8, 64, 5);
        let mut bytes = Vec::new();
        save_yaneuraou(&mut bytes, &weights).unwrap();
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            YO_TOP_HASH ^ 0x3547_5000
        );
        let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert!(
            std::str::from_utf8(&bytes[12..12 + len])
                .unwrap()
                .ends_with("{LayerStack=5;Bucket=GP(0.4,0.6,0.8,1)}")
        );
        let stack_bytes = 4 + 8 * 4 + 8 * 1024 + 64 * 4 + 64 * 32 + 4 + 64;
        for i in 0..5 {
            let offset = bytes.len() - (5 - i) * stack_bytes;
            assert_eq!(
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()),
                YO_NETWORK_HASH
            );
        }
        if let Some(path) = std::env::var_os("CYF_GP5_EXPORT_FIXTURE") {
            std::fs::write(path, &bytes).unwrap();
        }
    }

    #[test]
    fn gp5_rejects_other_engine_dimensions() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaMerged.spec(), 128, 8, 64, 5);
        assert!(
            save_yaneuraou(&mut Vec::new(), &weights)
                .unwrap_err()
                .to_string()
                .contains("GP5")
        );
    }

    #[test]
    fn baseline_architecture_string_matches_yaneuraou() {
        let weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            1536,
            16,
            32,
            YANEURAOU_LAYER_STACKS,
        );
        assert_eq!(
            arch_string(&architecture(&weights).unwrap()),
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}"
        );
    }

    #[test]
    fn generated_architecture_names_match_yaneuraou_loader_contract() {
        let cases = [
            (
                FeatureSet::HalfKaHmMerged,
                1536,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaHmMerged,
                512,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->512x2],Network=SFNN_HALFKAHM2_512_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKp,
                1536,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKP(Friend)[125388->1536x2],Network=SFNN_HALFKP_1536_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaSplit,
                768,
                8,
                16,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA1(Friend)[138510->768x2],Network=SFNN_HALFKA1_768_7_16_K3K3{LayerStack=9}",
            ),
        ];

        for (feature_set, ft_out, l1_out, l2_out, expected) in cases {
            let weights = LayerStackWeights::zeroed(
                feature_set.spec(),
                ft_out,
                l1_out,
                l2_out,
                YANEURAOU_LAYER_STACKS,
            );
            assert_eq!(arch_string(&architecture(&weights).unwrap()), expected);
        }
    }

    #[test]
    fn rejects_non_kingrank9_shape() {
        let weights = LayerStackWeights::zeroed(FeatureSet::HalfKaHmMerged.spec(), 128, 16, 32, 8);
        let error = save_yaneuraou(&mut Vec::new(), &weights).unwrap_err();
        assert!(error.to_string().contains("KingRank9"), "{error}");
    }

    #[test]
    fn direct_export_matches_tatara_reload_export_byte_for_byte() {
        let mut weights = LayerStackWeights::zeroed(
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            4,
            3,
            YANEURAOU_LAYER_STACKS,
        );
        weights.ft_b[0] = 0.25;
        weights.ft_w[17] = -0.5;
        weights.l1_b[2] = 0.75;
        weights.l1_w[31] = -0.25;
        weights.l1_shared_bias[1] = 0.25;
        weights.l1_shared_weight[7] = -0.5;
        weights.l2_b[4] = 0.5;
        weights.l2_w[9] = -0.75;
        weights.l3_b[3] = 0.125;
        weights.l3_w[5] = -0.125;

        let mut direct = Vec::new();
        save_yaneuraou(&mut direct, &weights).unwrap();

        let mut tatara = Vec::new();
        weights.save_quantised(&mut tatara, Some(28)).unwrap();
        let reloaded = LayerStackWeights::load_quantised(
            &mut tatara.as_slice(),
            FeatureSet::HalfKaHmMerged.spec(),
            128,
            4,
            3,
            YANEURAOU_LAYER_STACKS,
        )
        .unwrap();
        let mut post_hoc = Vec::new();
        save_yaneuraou(&mut post_hoc, &reloaded).unwrap();

        assert_eq!(direct, post_hoc);
    }

    // ---- Cyfamate GP-bucket research format ----

    /// Deterministic grid-valued weights so quantisation round-trips exactly.
    fn grid_weights(spec: FeatureSetSpec, ft_out: usize, l1_out: usize, l2_out: usize, buckets: usize, seed: u64) -> LayerStackWeights {
        let mut w = LayerStackWeights::zeroed(spec, ft_out, l1_out, l2_out, buckets);
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let base_n = spec.base_ft_in() * ft_out;
        for (i, v) in w.ft_w.iter_mut().enumerate() {
            let r = next();
            *v = if i < base_n {
                ((r % 601) as i32 - 300) as f32 / QA as f32
            } else {
                ((r % 255) as i32 - 127) as f32 / QA as f32
            };
        }
        for v in w.ft_b.iter_mut() {
            *v = ((next() % 2001) as i32 - 1000) as f32 / QA as f32;
        }
        for v in w.l1_w.iter_mut() {
            *v = ((next() % 255) as i32 - 127) as f32 / QB as f32;
        }
        for v in w.l2_w.iter_mut() {
            *v = ((next() % 255) as i32 - 127) as f32 / QB as f32;
        }
        for v in w.l3_w.iter_mut() {
            *v = ((next() % 255) as i32 - 127) as f32 / QB as f32;
        }
        for v in w.l1_b.iter_mut() {
            *v = ((next() % 20001) as i32 - 10000) as f32 / (QA * QB) as f32;
        }
        for v in w.l2_b.iter_mut() {
            *v = ((next() % 20001) as i32 - 10000) as f32 / (QA * QB) as f32;
        }
        for v in w.l3_b.iter_mut() {
            *v = ((next() % 20001) as i32 - 10000) as f32 / (QA * QB) as f32;
        }
        w
    }

    const GP8_BOUNDS: [f64; 7] = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05];

    #[test]
    fn cyfamate_fixed_gp5_is_byte_identical_to_save_yaneuraou() {
        for l1_out in [8usize, 16] {
            let w = grid_weights(FeatureSet::HalfKaMerged.spec(), 1024, l1_out, 64, 5, 7);
            let mut a = Vec::new();
            save_yaneuraou(&mut a, &w).unwrap();
            let mut b = Vec::new();
            save_yaneuraou_cyfamate(&mut b, &w, &GP5_FIXED_BOUNDS).unwrap();
            assert_eq!(a, b, "l1_out {l1_out}: fixed GP5 form must not change");
            let (back, arch) = read_yaneuraou_cyfamate(&mut std::io::Cursor::new(&b)).unwrap();
            assert!(arch.is_fixed_gp5());
            assert_eq!(arch.l1_out, l1_out);
            let mut q1 = Vec::new();
            w.save_quantised(&mut q1, Some(36)).unwrap();
            let mut q2 = Vec::new();
            back.save_quantised(&mut q2, Some(36)).unwrap();
            assert_eq!(q1, q2, "l1_out {l1_out}: tatara round trip");
        }
    }

    #[test]
    fn cyfamate_arch_string_and_hash_round_trip() {
        let cases = [
            (768usize, 16usize, None),
            (512, 8, None),
            (1536, 16, Some((ThreatProfile::StepAttacker, ThreatMirror::Off))),
            (768, 16, Some((ThreatProfile::StepAttacker, ThreatMirror::Off))),
        ];
        for (ft_out, l1_out, threat) in cases {
            let arch = CyfamateArch {
                ft_out,
                l1_out,
                l2_out: 64,
                gp_bounds: GP8_BOUNDS.to_vec(),
                threat,
                hm: false,
            };
            let s = arch.arch_string();
            let parsed = parse_cyfamate_arch(&s).unwrap();
            assert_eq!(parsed, arch, "{s}");
            assert!(s.contains("{LayerStack=8;Bucket=GP(0.15,0.3,0.45,0.6,0.75,0.9,1.05)}"), "{s}");
            assert!(s.contains(&format!("SFNN_HALFKA2_{ft_out}_{}_64_8GP", l1_out - 1)), "{s}");
            if threat.is_some() {
                assert!(s.contains("],Threat=33408,ThreatProfile=3,ThreatMirror=off,Network="), "{s}");
                assert_ne!(arch.top_hash() & THREAT_HASH_TAG, 0);
            } else {
                assert!(!s.contains("Threat"), "{s}");
            }
            assert_ne!(arch.top_hash(), gp5_top_hash(l1_out));
        }
        // 固定 GP5 文字列は既存定数と一致する。
        let gp5 = CyfamateArch {
            ft_out: 1024,
            l1_out: 8,
            l2_out: 64,
            gp_bounds: GP5_FIXED_BOUNDS.to_vec(),
            threat: None,
            hm: false,
        };
        assert_eq!(gp5.arch_string(), GP5_ARCH);
        assert_eq!(parse_cyfamate_arch(GP5_ARCH).unwrap(), gp5);
        assert_eq!(parse_cyfamate_arch(GP5_H15_ARCH).unwrap().l1_out, 16);
        // Half-mirrored base rows: own feature token, 73,305 rows, "HM" hash tag; the
        // engine (cyf_gp8_threat.h kTagHm) expects the same top hash.
        let hm_base = CyfamateArch {
            ft_out: 1536,
            l1_out: 16,
            l2_out: 64,
            gp_bounds: GP8_BOUNDS.to_vec(),
            threat: Some((ThreatProfile::StepAttacker, ThreatMirror::Off)),
            hm: true,
        };
        let s = hm_base.arch_string();
        assert!(s.starts_with("ModelType=SFNNWithoutPsqt;Features=HalfKA2_hm(Friend)[73305->1536x2],Threat=33408,"), "{s}");
        assert_eq!(parse_cyfamate_arch(&s).unwrap(), hm_base);
        assert_eq!(hm_base.top_hash() ^ HM_HASH_TAG, CyfamateArch { hm: false, ..hm_base.clone() }.top_hash());
        assert_eq!(hm_base.spec().base_ft_in(), 73_305);
        // threat + hm mirror は拒否。
        let hm = CyfamateArch {
            threat: Some((ThreatProfile::StepAttacker, ThreatMirror::Hm)),
            ..gp5.clone()
        };
        assert!(hm.validate().is_err());
    }

    #[test]
    fn cyfamate_round_trip_over_widths_fc0_and_threat() {
        // T6: 幅 {512,768,1024,1536} × fc_0 {7+1,15+1} × fc_1 64 を GP8 (7 境界) で往復し、
        // tatara 形式の再保存 byte が一致する。threat (step-attacker, mirror off) は 2 幅で。
        let base = FeatureSet::HalfKaMerged.spec();
        let threat = base
            .with_threat_profile(ThreatProfile::StepAttacker)
            .with_threat_mirror(ThreatMirror::Off);
        let mut cases: Vec<(FeatureSetSpec, usize, usize)> = Vec::new();
        for ft_out in [512usize, 768, 1024, 1536] {
            for l1_out in [8usize, 16] {
                cases.push((base, ft_out, l1_out));
            }
        }
        cases.push((threat, 512, 16));
        cases.push((threat, 768, 16));
        // Half-mirrored base rows (73,305), with and without a raw-square threat block.
        let hm = FeatureSet::HalfKaHmMerged.spec();
        cases.push((hm, 512, 16));
        cases.push((
            hm.with_threat_profile(ThreatProfile::StepAttacker).with_threat_mirror(ThreatMirror::Off),
            512,
            16,
        ));
        for (i, (spec, ft_out, l1_out)) in cases.into_iter().enumerate() {
            let w = grid_weights(spec, ft_out, l1_out, 64, 8, 11 + i as u64);
            let mut bytes = Vec::new();
            save_yaneuraou_cyfamate(&mut bytes, &w, &GP8_BOUNDS).unwrap();
            let mut cursor = std::io::Cursor::new(&bytes);
            let (back, arch) = read_yaneuraou_cyfamate(&mut cursor).unwrap();
            assert_eq!(cursor.position() as usize, bytes.len(), "trailing bytes");
            assert_eq!(arch.ft_out, ft_out);
            assert_eq!(arch.l1_out, l1_out);
            assert_eq!(arch.gp_bounds, GP8_BOUNDS.to_vec());
            assert_eq!(arch.threat.is_some(), spec.threat_profile().is_some());
            assert_eq!(arch.hm, spec.feature_set() == FeatureSet::HalfKaHmMerged);
            assert_eq!(back.feature_set, spec);
            assert_eq!(back.ft_w, w.ft_w, "ft_w {ft_out}/{l1_out}");
            assert_eq!(back.ft_b, w.ft_b, "ft_b {ft_out}/{l1_out}");
            assert_eq!(back.l2_w, w.l2_w);
            assert_eq!(back.l3_w, w.l3_w);
            let mut q1 = Vec::new();
            w.save_quantised(&mut q1, Some(36)).unwrap();
            let mut q2 = Vec::new();
            back.save_quantised(&mut q2, Some(36)).unwrap();
            assert_eq!(q1, q2, "tatara .bin byte identity {ft_out}/{l1_out} threat={}", spec.threat_profile().is_some());
            // 期待 byte 数: header + FT + (threat) + 8 stacks。
            let arch_len = arch.arch_string().len();
            let stack = 4 + 4 * l1_out + l1_out * ft_out + 4 * 64 + 64 * 32 + 4 + 64;
            let threat_bytes = if spec.threat_profile().is_some() { 4 + 33_408 * ft_out } else { 0 };
            let prefix = 12 + arch_len + 4;
            let leb = bytes.len() - prefix - threat_bytes - 8 * stack;
            assert!(leb > 2 * 21, "two LEB128 blocks expected");
        }
    }

    #[test]
    fn affine_weights_are_row_major_and_padded() {
        let mut output = Vec::new();
        write_affine(
            &mut output,
            [1.0, -1.0],
            [
                1.0 / 64.0,
                2.0 / 64.0,
                3.0 / 64.0,
                -1.0 / 64.0,
                -2.0 / 64.0,
                -3.0 / 64.0,
            ],
            3,
            2,
        )
        .unwrap();
        assert_eq!(
            i32::from_le_bytes(output[0..4].try_into().unwrap()),
            QA * QB
        );
        assert_eq!(
            i32::from_le_bytes(output[4..8].try_into().unwrap()),
            -(QA * QB)
        );
        assert_eq!(&output[8..11], &[1, 2, 3]);
        assert!(output[11..40].iter().all(|&byte| byte == 0));
        assert_eq!(&output[40..43], &[255, 254, 253]);
        assert!(output[43..72].iter().all(|&byte| byte == 0));
    }
}
