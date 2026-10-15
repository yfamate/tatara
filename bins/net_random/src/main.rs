//! Writes a tatara LayerStack `.bin` with deterministic pseudo-random weights on the
//! quantisation grid. Used to build reference nets for engine reader parity tests (no
//! training data involved): the same seed always gives the same file.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::{QA, QB};
use shogi_features::{FeatureSet, ThreatMirror, ThreatProfile};

#[derive(Parser)]
#[command(about = "Write a deterministic random LayerStack net (tatara .bin)")]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "halfka-merged")]
    feature_set: String,
    #[arg(long, default_value_t = 1024)]
    ft_out: usize,
    #[arg(long, default_value_t = 8)]
    l1: usize,
    #[arg(long, default_value_t = 64)]
    l2: usize,
    #[arg(long, default_value_t = 5)]
    num_buckets: usize,
    /// off or a threat profile name (full, same-class, ..., step-attacker, cross-side).
    #[arg(long, default_value = "off")]
    threat_profile: String,
    /// hm or off.
    #[arg(long, default_value = "hm")]
    threat_mirror: String,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// `fv_scale` token written to the arch string.
    #[arg(long, default_value_t = 36)]
    fv_scale: i32,
    /// Largest |FT weight| in QA units (base rows); threat rows are always within ±127.
    #[arg(long, default_value_t = 300)]
    ft_range: i32,
    /// Write an all-zero net (FT, threat FT, every layer and bias = 0): the "null" evaluation, eval = 0 everywhere.
    #[arg(long)]
    zero: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let feature_set = FeatureSet::from_canonical_name(&args.feature_set)
        .ok_or_else(|| format!("unknown --feature-set {}", args.feature_set))?;
    let mut spec = feature_set.spec();
    if args.threat_profile != "off" {
        let profile = ThreatProfile::from_cli(&args.threat_profile)
            .ok_or_else(|| format!("unknown --threat-profile {}", args.threat_profile))?;
        let mirror = ThreatMirror::from_cli(&args.threat_mirror)
            .ok_or_else(|| format!("unknown --threat-mirror {}", args.threat_mirror))?;
        spec = spec.with_threat_profile(profile).with_threat_mirror(mirror);
    }
    let mut w = LayerStackWeights::zeroed(spec, args.ft_out, args.l1, args.l2, args.num_buckets);
    let mut s = args.seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let base_n = spec.base_ft_in() * args.ft_out;
    if args.zero {
        let mut out = BufWriter::new(File::create(&args.output)?);
        w.save_quantised(&mut out, Some(args.fv_scale))?;
        out.flush()?;
        println!("wrote {} (all-zero null net, feature_set={} ft_out={} l1={} l2={} buckets={})", args.output.display(),
                 spec.canonical_name(), args.ft_out, args.l1, args.l2, args.num_buckets);
        return Ok(());
    }
    let ft_span = (2 * args.ft_range + 1) as u64;
    for (i, v) in w.ft_w.iter_mut().enumerate() {
        let r = next();
        *v = if i < base_n {
            ((r % ft_span) as i32 - args.ft_range) as f32 / QA as f32
        } else {
            ((r % 255) as i32 - 127) as f32 / QA as f32
        };
    }
    for v in w.ft_b.iter_mut() {
        *v = ((next() % 2001) as i32 - 1000) as f32 / QA as f32;
    }
    for v in w
        .l1_w
        .iter_mut()
        .chain(w.l2_w.iter_mut())
        .chain(w.l3_w.iter_mut())
    {
        *v = ((next() % 255) as i32 - 127) as f32 / QB as f32;
    }
    for v in w
        .l1_b
        .iter_mut()
        .chain(w.l2_b.iter_mut())
        .chain(w.l3_b.iter_mut())
    {
        *v = ((next() % 20001) as i32 - 10000) as f32 / (QA * QB) as f32;
    }
    let mut out = BufWriter::new(File::create(&args.output)?);
    w.save_quantised(&mut out, Some(args.fv_scale))?;
    out.flush()?;
    println!(
        "wrote {} (feature_set={} ft_in={} ft_out={} l1={} l2={} buckets={} seed={})",
        args.output.display(),
        spec.canonical_name(),
        spec.ft_in(),
        args.ft_out,
        args.l1,
        args.l2,
        args.num_buckets,
        args.seed
    );
    Ok(())
}
