//! Build a LayerStack net with fewer output buckets by keeping a chosen subset of an existing net's buckets.
//!
//! The FT (incl. threat rows) is copied unchanged; L1/L2/L3 of each kept bucket are copied in the given order.
//! Use it to start an N-bucket training run (`nnue_train --init-from`) from a trained M-bucket net, e.g. merging
//! adjacent GP buckets: `--select 1,3,5,7` turns 8 buckets into 4 (train with the matching `gpk:` boundaries).
//!
//!   net_bucket_select --input in.bin --output out.bin --feature-set halfka-merged --select 1,3,5,7
use std::io::Cursor;
use std::path::PathBuf;

use clap::Parser;
use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::{
    DEFAULT_L1_OUT, DEFAULT_L2_OUT, DEFAULT_NUM_BUCKETS, LEGACY_NNUE_VERSION_BUCKETS9, NNUE_VERSION,
};
use shogi_features::{EffectBucketConfig, FeatureSet, ThreatMirror, ThreatProfile};

#[derive(Parser)]
struct Args {
    /// Quantised LayerStack net (tatara format).
    #[arg(long)]
    input: PathBuf,
    /// Output path (must differ from the input).
    #[arg(long)]
    output: PathBuf,
    /// Input feature set the net was trained with (quantised files do not record it).
    #[arg(long, default_value = "halfka-merged")]
    feature_set: String,
    /// Comma-separated source bucket indices to keep, in output order.
    #[arg(long)]
    select: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input == args.output {
        return Err("input and output must differ".into());
    }
    let feature_set = FeatureSet::from_canonical_name(&args.feature_set)
        .ok_or_else(|| format!("unknown --feature-set {}", args.feature_set))?;
    let bytes = std::fs::read(&args.input)?;
    let meta = read_arch_meta(&bytes)?;
    let len = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
    let arch = std::str::from_utf8(&bytes[12..12 + len])?;
    let fv_scale = parse_token_usize(arch, "fv_scale=")?.map(|v| v as i32);
    let mut spec = feature_set.spec();
    if let Some(cfg) = meta.effect_bucket_config {
        spec = spec.with_effect_bucket_config(cfg);
    }
    if let Some(profile) = meta.threat_profile {
        spec = spec.with_threat_profile(profile);
        if meta.threat_mirror_off {
            spec = spec.with_threat_mirror(ThreatMirror::Off);
        }
    }
    let src = LayerStackWeights::load_quantised(
        &mut Cursor::new(&bytes),
        spec,
        meta.ft_out,
        meta.l1_out,
        meta.l2_out,
        meta.num_buckets,
    )?;
    if src.psqt_w.is_some() {
        return Err("PSQT nets are not supported".into());
    }
    let sel: Vec<usize> = args
        .select
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<_, _>>()?;
    let n = src.num_buckets;
    if sel.is_empty() || sel.iter().any(|&b| b >= n) {
        return Err(format!("--select must list buckets in 0..{n}").into());
    }
    let pick = |v: &[f32]| -> Vec<f32> {
        let per = v.len() / n;
        sel.iter()
            .flat_map(|&b| v[b * per..(b + 1) * per].iter().copied())
            .collect()
    };
    let mut dst = src.clone();
    dst.num_buckets = sel.len();
    dst.l1_w = pick(&src.l1_w);
    dst.l1_b = pick(&src.l1_b);
    dst.l2_w = pick(&src.l2_w);
    dst.l2_b = pick(&src.l2_b);
    dst.l3_w = pick(&src.l3_w);
    dst.l3_b = pick(&src.l3_b);
    let mut out = Vec::new();
    dst.save_quantised(&mut out, fv_scale)?;
    // The written file must load back as a sel.len()-bucket net with the kept layers unchanged.
    let back = LayerStackWeights::load_quantised(
        &mut Cursor::new(&out),
        spec,
        meta.ft_out,
        meta.l1_out,
        meta.l2_out,
        sel.len(),
    )?;
    if back.l1_w != dst.l1_w || back.l2_w != dst.l2_w || back.l3_b != dst.l3_b || back.ft_w != src.ft_w {
        return Err("round-trip mismatch".into());
    }
    std::fs::write(&args.output, &out)?;
    eprintln!(
        "{n} buckets -> {} (kept {sel:?}), fv_scale {fv_scale:?}, {} bytes",
        sel.len(),
        out.len()
    );
    Ok(())
}

struct ArchMeta {
    effect_bucket_config: Option<EffectBucketConfig>,
    threat_profile: Option<ThreatProfile>,
    threat_mirror_off: bool,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
}

fn read_arch_meta(bytes: &[u8]) -> Result<ArchMeta, Box<dyn std::error::Error>> {
    if bytes.len() < 16 {
        return Err("NNUE file is too short".into());
    }
    let version = u32::from_le_bytes(bytes[0..4].try_into().expect("slice has 4 bytes"));
    if version != NNUE_VERSION && version != LEGACY_NNUE_VERSION_BUCKETS9 {
        return Err(format!("unsupported NNUE version: {version:#x}").into());
    }
    let arch_len = u32::from_le_bytes(bytes[8..12].try_into().expect("slice has 4 bytes")) as usize;
    let arch_start = 12;
    let arch_end = arch_start + arch_len;
    if arch_end > bytes.len() {
        return Err("NNUE arch string extends past end of file".into());
    }
    let arch = std::str::from_utf8(&bytes[arch_start..arch_end])?;
    let num_buckets = if version == LEGACY_NNUE_VERSION_BUCKETS9 {
        DEFAULT_NUM_BUCKETS
    } else {
        if arch_end + 4 > bytes.len() {
            return Err("NNUE num_buckets field extends past end of file".into());
        }
        u32::from_le_bytes(
            bytes[arch_end..arch_end + 4]
                .try_into()
                .expect("slice has 4 bytes"),
        ) as usize
    };
    let affine_outs = parse_affine_outs(arch);
    Ok(ArchMeta {
        effect_bucket_config: parse_effect_bucket_config(arch)?,
        threat_profile: parse_threat_profile(arch)?,
        threat_mirror_off: match parse_token_str(arch, "ThreatMirror=") {
            None => false,
            Some("off") => true,
            Some(other) => return Err(format!("unsupported ThreatMirror value `{other}`").into()),
        },
        ft_out: parse_between(arch, "->", "x2]")?,
        l1_out: affine_outs.last().copied().unwrap_or(DEFAULT_L1_OUT),
        l2_out: affine_outs.get(1).copied().unwrap_or(DEFAULT_L2_OUT),
        num_buckets: if num_buckets == 0 {
            DEFAULT_NUM_BUCKETS
        } else {
            num_buckets
        },
    })
}

fn parse_effect_bucket_config(
    arch: &str,
) -> Result<Option<EffectBucketConfig>, Box<dyn std::error::Error>> {
    let Some(value) = parse_token_str(arch, "EffectBucket=") else {
        return Ok(None);
    };
    match value {
        "2x2fixed" => Ok(Some(EffectBucketConfig::KINGFIXED_2X2)),
        "2x2bucketed" => Ok(Some(EffectBucketConfig::KINGBUCKETED_2X2)),
        "3x3fixed" => Ok(Some(EffectBucketConfig::KINGFIXED_3X3)),
        "3x3bucketed" => Ok(Some(EffectBucketConfig::KINGBUCKETED_3X3)),
        _ => Err(format!("unsupported EffectBucket value `{value}`").into()),
    }
}

fn parse_threat_profile(arch: &str) -> Result<Option<ThreatProfile>, Box<dyn std::error::Error>> {
    let Some(dims) = parse_token_usize(arch, "Threat=")? else {
        return Ok(None);
    };
    let profile = match parse_token_usize(arch, "ThreatProfile=")? {
        None | Some(0) => ThreatProfile::Full,
        Some(1) => ThreatProfile::SameClass,
        Some(2) => ThreatProfile::SameClassMajorPawn,
        Some(3) => ThreatProfile::StepAttacker,
        Some(4) => ThreatProfile::FullSymDedup,
        Some(10) => ThreatProfile::CrossSide,
        Some(id) => return Err(format!("unsupported ThreatProfile id {id}").into()),
    };
    let expected = shogi_features::threat_dimensions_of(profile);
    if dims != expected {
        return Err(format!(
            "Threat dims/profile mismatch in arch string: dims={dims}, profile={profile}, expected={expected}"
        )
        .into());
    }
    Ok(Some(profile))
}

fn parse_affine_outs(arch: &str) -> Vec<usize> {
    let needle = "AffineTransform[";
    let mut start = 0usize;
    let mut vals = Vec::new();
    while let Some(pos) = arch[start..].find(needle) {
        let absolute = start + pos + needle.len();
        if let Some(end) = arch[absolute..].find("<-")
            && let Ok(v) = arch[absolute..absolute + end].parse()
        {
            vals.push(v);
        }
        start = absolute;
    }
    vals
}

fn parse_between(
    s: &str,
    start_pat: &str,
    end_pat: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let start = s
        .find(start_pat)
        .ok_or_else(|| format!("missing `{start_pat}` in arch string"))?
        + start_pat.len();
    let end = s[start..]
        .find(end_pat)
        .ok_or_else(|| format!("missing `{end_pat}` in arch string"))?
        + start;
    Ok(s[start..end].parse()?)
}

fn parse_token_usize(s: &str, token: &str) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    Ok(parse_token_str(s, token).map(str::parse).transpose()?)
}

fn parse_token_str<'a>(s: &'a str, token: &str) -> Option<&'a str> {
    let start = s.find(token).map(|pos| pos + token.len())?;
    let end = s[start..]
        .find(',')
        .map(|pos| start + pos)
        .unwrap_or(s.len());
    Some(s[start..end].split(']').next().unwrap_or(&s[start..end]))
}
