//! `gp-quantile`: GP8 boundary tool (design T1).
//!
//! Streams sampled PSV positions, computes the board progress with the same function the
//! trainer routes on (`shogi_features::gp5_progress_board`), and writes the k-quantile
//! boundaries of that value together with the evidence needed to fix them for every arm:
//! tie counts at each boundary, per-group counts under the new and the current GP5 bounds, and
//! the share of positions that carry the entering-king bonus (GP > 1).
//!
//! ```bash
//! gp-quantile --data a.psv,b.psv --samples 4000000 --stride 250 --groups 8 --out GP8_BOUNDS.json
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use shogi_features::{GP5_BOUNDS, gp_bucket_from_progress, gp5_progress_board};
use shogi_format::PackedSfenValue;

#[derive(Parser, Debug)]
#[command(name = "gp-quantile")]
#[command(about = "k-quantile boundaries of the board progress (GP) over PSV data")]
struct Args {
    /// PSV data files, comma-separated. Samples are split in proportion to each file's record count
    /// (so the quantiles follow the training distribution of the concatenated data); --even splits them evenly.
    #[arg(long)]
    data: String,
    /// Old behaviour: the same number of samples from every file regardless of its size.
    #[arg(long)]
    even: bool,

    /// Total positions to sample.
    #[arg(long, default_value_t = 4_000_000)]
    samples: usize,

    /// Read every N-th record.
    #[arg(long, default_value_t = 1)]
    stride: u64,

    /// Starting record offset in each file.
    #[arg(long, default_value_t = 0)]
    offset: u64,

    /// Number of groups (boundaries = groups - 1).
    #[arg(long, default_value_t = 8)]
    groups: usize,

    /// Output JSON path.
    #[arg(long)]
    out: PathBuf,
}

/// Reads up to `max` records starting at `offset`, one every `stride` records, in 4 MiB chunks so
/// a dense or small-stride scan stays sequential.
fn read_gp(path: &PathBuf, offset: u64, stride: u64, max: usize, out: &mut Vec<f64>) -> io::Result<usize> {
    let record = size_of::<PackedSfenValue>();
    let mut file = File::open(path)?;
    let total = file.metadata()?.len() / record as u64;
    if offset >= total {
        return Ok(0);
    }
    file.seek(SeekFrom::Start(offset * record as u64))?;
    let per_chunk = (4 << 20) / record;
    let mut buf = vec![0u8; per_chunk * record];
    let mut index: u64 = offset;
    let mut taken = 0usize;
    while taken < max && index < total {
        let n = per_chunk.min((total - index) as usize);
        file.read_exact(&mut buf[..n * record])?;
        for i in 0..n {
            if (index + i as u64 - offset) % stride != 0 || taken >= max {
                continue;
            }
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut().copy_from_slice(&buf[i * record..(i + 1) * record]);
            out.push(gp5_progress_board(&psv.decode()));
            taken += 1;
        }
        index += n as u64;
        if stride > per_chunk as u64 {
            // skip whole chunks that contain no sampled record
            let next = offset + (index - offset).div_ceil(stride) * stride;
            if next > index {
                file.seek(SeekFrom::Start(next * record as u64))?;
                index = next;
            }
        }
    }
    Ok(taken)
}

fn histogram(gps: &[f64], bounds: &[f64]) -> Vec<u64> {
    let mut h = vec![0u64; bounds.len() + 1];
    for &g in gps {
        h[gp_bucket_from_progress(g, bounds) as usize] += 1;
    }
    h
}

fn json_list<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.stride == 0 || args.groups < 2 || args.groups > 9 {
        return Err("--stride must be >= 1 and --groups in [2, 9]".into());
    }
    let paths: Vec<PathBuf> =
        args.data.split(',').map(str::trim).filter(|s| !s.is_empty()).map(PathBuf::from).collect();
    if paths.is_empty() {
        return Err("--data is required".into());
    }
    let mut gps = Vec::with_capacity(args.samples);
    let sizes: Vec<u64> = paths.iter().map(|p| std::fs::metadata(p).map(|m| m.len() / 40).unwrap_or(0)).collect();
    let total: u64 = sizes.iter().sum::<u64>().max(1);
    let mut per_file_counts = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        // [2026-10-16] share ∝ record count (gp-threat-review: per-file equal sampling did not follow the training mix)
        let share = if args.even { args.samples.div_ceil(paths.len()) }
                    else { ((args.samples as u128 * sizes[i] as u128).div_ceil(total as u128)) as usize };
        let want = share.min(args.samples - gps.len());
        let got = read_gp(p, args.offset, args.stride, want, &mut gps)?;
        eprintln!("{}: {got} positions", p.display());
        per_file_counts.push((p.display().to_string(), got));
    }
    if gps.is_empty() {
        return Err("no positions read".into());
    }
    let mut sorted = gps.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let k = args.groups;
    let bounds: Vec<f64> = (1..k).map(|i| sorted[(i * n) / k]).collect();
    let strictly_increasing = bounds.windows(2).all(|w| w[0] < w[1]);
    let ties: Vec<u64> = bounds.iter().map(|b| sorted.iter().filter(|&&g| g == *b).count() as u64).collect();
    let hist_new = histogram(&gps, &bounds);
    let hist_gp5 = histogram(&gps, &GP5_BOUNDS);
    let bonus = gps.iter().filter(|&&g| g > 1.0).count();
    let bounds_arg = bounds.iter().map(|b| format!("{b:.6}")).collect::<Vec<_>>().join(",");
    // the trainer and engine see the rounded list; recount with it so the record matches what is used
    let rounded: Vec<f64> = bounds_arg.split(',').map(|s| s.parse().unwrap()).collect();
    let hist_rounded = histogram(&gps, &rounded);

    let mut w = BufWriter::new(File::create(&args.out)?);
    writeln!(w, "{{")?;
    writeln!(w, "  \"tool\": \"tatara gp-quantile (shogi_features::gp5_progress_board)\",")?;
    writeln!(w, "  \"samples\": {n},")?;
    writeln!(w, "  \"stride\": {}, \"offset\": {},", args.stride, args.offset)?;
    writeln!(
        w,
        "  \"files\": [{}],",
        per_file_counts.iter().map(|(p, c)| format!("{{\"path\": {:?}, \"samples\": {c}}}", p)).collect::<Vec<_>>().join(", ")
    )?;
    writeln!(w, "  \"groups\": {k},")?;
    writeln!(w, "  \"bounds\": [{}],", json_list(&bounds))?;
    writeln!(w, "  \"bounds_arg\": \"{bounds_arg}\",")?;
    writeln!(w, "  \"strictly_increasing\": {strictly_increasing},")?;
    writeln!(w, "  \"ties_at_bound\": [{}],", json_list(&ties))?;
    writeln!(w, "  \"group_counts\": [{}],", json_list(&hist_new))?;
    writeln!(w, "  \"group_counts_rounded_bounds\": [{}],", json_list(&hist_rounded))?;
    writeln!(w, "  \"gp5_bounds\": [{}],", json_list(&GP5_BOUNDS))?;
    writeln!(w, "  \"gp5_group_counts\": [{}],", json_list(&hist_gp5))?;
    writeln!(w, "  \"gp_min_median_max\": [{}, {}, {}],", sorted[0], sorted[n / 2], sorted[n - 1])?;
    writeln!(w, "  \"entering_king_bonus_share\": {}", bonus as f64 / n as f64)?;
    writeln!(w, "}}")?;
    w.flush()?;
    println!("bounds {bounds_arg} strictly_increasing={strictly_increasing} counts {:?}", hist_rounded);
    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::histogram;

    #[test]
    fn histogram_uses_ge_routing() {
        // a value equal to a boundary goes to the upper group, as in gp_bucket_from_progress
        assert_eq!(histogram(&[0.1, 0.5, 0.5, 0.9], &[0.5]), vec![1, 3]);
    }
}
