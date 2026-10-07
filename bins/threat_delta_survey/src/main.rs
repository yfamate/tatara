//! Threat active-set change (Δedges) survey over game-ordered positions.
//!
//! Input: a text file of SFEN lines in game order. A line starting with `#game` starts a
//! new game (the rest of the line is kept as a label); consecutive SFEN lines inside one
//! game are consecutive positions. For every transition the tool computes, per threat
//! profile and per mirror mode, the size of the symmetric difference between the active
//! threat index sets of the two positions (removed + added edges) from one perspective.
//! This is the per-move update cost an engine with an incremental threat accumulator pays
//! (one perspective; the other perspective sees the same number of edges).
//!
//! Output: a JSON document with the distribution (count, mean, median, p90, p99, max,
//! histogram) of Δedges for all transitions, for transitions without an own-king move
//! (king moves refresh the accumulator anyway), split by board progress (gp5 buckets and
//! opening / middle / end), plus the active count per position.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use clap::Parser;
use shogi_features::{ThreatIndexer, ThreatMirror, ThreatProfile, gp5_bucket_board};
use shogi_format::ShogiBoard;
use shogi_format::types::{Color, Hand, Piece, PieceType, Square};

#[derive(Parser)]
#[command(about = "Δedges survey of threat features over game-ordered SFEN positions")]
struct Args {
    /// SFEN lines in game order; `#game ...` lines separate games.
    #[arg(long)]
    input: PathBuf,
    /// JSON output path.
    #[arg(long)]
    output: PathBuf,
    /// Threat profiles to survey (default: all six).
    #[arg(long, value_delimiter = ',')]
    profiles: Vec<String>,
    /// Seed for the 10,000-transition random subsample.
    #[arg(long, default_value_t = 20261007)]
    seed: u64,
}

const ALL_PROFILES: [ThreatProfile; 6] = [
    ThreatProfile::Full,
    ThreatProfile::SameClass,
    ThreatProfile::SameClassMajorPawn,
    ThreatProfile::StepAttacker,
    ThreatProfile::FullSymDedup,
    ThreatProfile::CrossSide,
];

const MIRRORS: [ThreatMirror; 2] = [ThreatMirror::Off, ThreatMirror::Hm];

/// One transition's observations, shared across profiles.
struct Transition {
    /// Own king of the Black perspective moved (Black king square changed).
    black_king_moved: bool,
    /// White king square changed.
    white_king_moved: bool,
    /// gp5 bucket of the position before the move.
    gp5_bucket: u8,
    /// Ply of the position before the move.
    ply: u16,
}

#[derive(Default)]
struct Series {
    values: Vec<u32>,
}

impl Series {
    fn push(&mut self, v: usize) {
        self.values.push(v as u32);
    }

    fn stats_json(&self) -> String {
        let mut v = self.values.clone();
        v.sort_unstable();
        let n = v.len();
        if n == 0 {
            return "{\"n\":0}".to_string();
        }
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let q = |p: f64| -> u32 {
            let idx = ((n as f64 - 1.0) * p).round() as usize;
            v[idx.min(n - 1)]
        };
        let median = if n % 2 == 1 {
            v[n / 2] as f64
        } else {
            (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.0
        };
        let mut hist = BTreeMap::new();
        for &x in &v {
            *hist.entry(x.min(64)).or_insert(0usize) += 1;
        }
        let mut h = String::from("{");
        for (i, (k, c)) in hist.iter().enumerate() {
            if i > 0 {
                h.push(',');
            }
            let key = if *k == 64 { "64+".to_string() } else { k.to_string() };
            let _ = write!(h, "\"{key}\":{c}");
        }
        h.push('}');
        format!(
            "{{\"n\":{n},\"mean\":{mean:.4},\"median\":{median},\"p90\":{},\"p99\":{},\"max\":{},\"le6\":{:.4},\"le12\":{:.4},\"hist\":{h}}}",
            q(0.90),
            q(0.99),
            v[n - 1],
            v.iter().filter(|&&x| x <= 6).count() as f64 / n as f64,
            v.iter().filter(|&&x| x <= 12).count() as f64 / n as f64,
        )
    }

    fn median_of(values: &[u32]) -> f64 {
        let mut v = values.to_vec();
        v.sort_unstable();
        let n = v.len();
        if n == 0 {
            return f64::NAN;
        }
        if n % 2 == 1 {
            v[n / 2] as f64
        } else {
            (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.0
        }
    }
}

/// Per (profile, mirror) accumulators.
#[derive(Default)]
struct Bucket {
    all_black: Series,
    all_white: Series,
    no_own_king_black: Series,
    no_own_king_white: Series,
    no_any_king_black: Series,
    by_gp5: [Series; 5],
    by_phase: [Series; 3], // opening (gp<0.4), middle ([0.4,0.8)), end (>=0.8)
    by_ply: [Series; 3],   // ply <40, 40..99, >=100
    active_black: Series,
    king_move_full_black: Series, // |active| at positions reached by an own-king move (refresh size)
}

fn symmetric_difference_len(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut d) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
            d += 1;
        } else {
            j += 1;
            d += 1;
        }
    }
    d + (a.len() - i) + (b.len() - j)
}

fn active_sorted(
    indexer: &ThreatIndexer,
    board: &ShogiBoard,
    perspective: Color,
    mirror: ThreatMirror,
) -> Vec<u32> {
    let mut v: Vec<u32> = indexer
        .active_threat_indices_with_mirror(board, perspective, mirror)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    v.sort_unstable();
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let profiles: Vec<ThreatProfile> = if args.profiles.is_empty() {
        ALL_PROFILES.to_vec()
    } else {
        args.profiles
            .iter()
            .map(|s| ThreatProfile::from_cli(s).ok_or_else(|| format!("unknown profile {s}")))
            .collect::<Result<_, _>>()?
    };
    let text = fs::read_to_string(&args.input)?;

    // Parse games.
    let mut games: Vec<(String, Vec<ShogiBoard>)> = Vec::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(label) = line.strip_prefix("#game") {
            games.push((label.trim().to_string(), Vec::new()));
            continue;
        }
        if games.is_empty() {
            games.push((String::from("(unlabeled)"), Vec::new()));
        }
        let board = parse_sfen(line)?;
        games.last_mut().expect("game exists").1.push(board);
    }
    let n_games = games.len();
    let n_positions: usize = games.iter().map(|g| g.1.len()).sum();
    let n_transitions: usize = games.iter().map(|g| g.1.len().saturating_sub(1)).sum();
    eprintln!("[survey] games={n_games} positions={n_positions} transitions={n_transitions}");

    let mut buckets: BTreeMap<(String, &'static str), Bucket> = BTreeMap::new();
    let mut transitions: Vec<Transition> = Vec::with_capacity(n_transitions);

    for (profile_i, &profile) in profiles.iter().enumerate() {
        let indexer = ThreatIndexer::new(profile);
        for &mirror in &MIRRORS {
            let key = (profile.to_string(), mirror.name());
            let bucket = buckets.entry(key).or_default();
            let record_transitions = profile_i == 0 && mirror == ThreatMirror::Off;
            for (_label, boards) in &games {
                let mut prev_b: Option<Vec<u32>> = None;
                let mut prev_w: Option<Vec<u32>> = None;
                let mut prev_board: Option<&ShogiBoard> = None;
                for board in boards {
                    let cur_b = active_sorted(&indexer, board, Color::Black, mirror);
                    let cur_w = active_sorted(&indexer, board, Color::White, mirror);
                    bucket.active_black.push(cur_b.len());
                    if let (Some(pb), Some(pw), Some(pboard)) = (&prev_b, &prev_w, prev_board) {
                        let db = symmetric_difference_len(pb, &cur_b);
                        let dw = symmetric_difference_len(pw, &cur_w);
                        let bk = pboard.black_king_sq != board.black_king_sq;
                        let wk = pboard.white_king_sq != board.white_king_sq;
                        let gp = gp5_bucket_board(pboard);
                        bucket.all_black.push(db);
                        bucket.all_white.push(dw);
                        if !bk {
                            bucket.no_own_king_black.push(db);
                        } else {
                            bucket.king_move_full_black.push(cur_b.len());
                        }
                        if !wk {
                            bucket.no_own_king_white.push(dw);
                        }
                        if !bk && !wk {
                            bucket.no_any_king_black.push(db);
                        }
                        bucket.by_gp5[(gp as usize).min(4)].push(db);
                        let phase = match gp {
                            0 => 0,
                            1 | 2 => 1,
                            _ => 2,
                        };
                        bucket.by_phase[phase].push(db);
                        let ply_bin = if pboard.ply < 40 {
                            0
                        } else if pboard.ply < 100 {
                            1
                        } else {
                            2
                        };
                        bucket.by_ply[ply_bin].push(db);
                        if record_transitions {
                            transitions.push(Transition {
                                black_king_moved: bk,
                                white_king_moved: wk,
                                gp5_bucket: gp,
                                ply: pboard.ply,
                            });
                        }
                    }
                    prev_b = Some(cur_b);
                    prev_w = Some(cur_w);
                    prev_board = Some(board);
                }
            }
            eprintln!(
                "[survey] {} mirror={}: median Δ(black, all)={} n={}",
                profile,
                mirror.name(),
                Series::median_of(&bucket.all_black.values),
                bucket.all_black.values.len()
            );
        }
    }

    // Random 10,000 subsample (fixed seed, xorshift) of transition indices.
    let sample_idx: Vec<usize> = {
        let mut s = args.seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let n = transitions.len();
        let mut pool: Vec<usize> = (0..n).collect();
        // partial Fisher–Yates
        let k = n.min(10_000);
        for i in 0..k {
            let j = i + (next() as usize) % (n - i);
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool
    };

    let king_moves = transitions.iter().filter(|t| t.black_king_moved).count();
    let any_king_moves = transitions
        .iter()
        .filter(|t| t.black_king_moved || t.white_king_moved)
        .count();
    let gp_counts: [usize; 5] = {
        let mut c = [0usize; 5];
        for t in &transitions {
            c[(t.gp5_bucket as usize).min(4)] += 1;
        }
        c
    };
    let ply_max = transitions.iter().map(|t| t.ply).max().unwrap_or(0);

    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(
        out,
        "  \"input\": {:?},\n  \"games\": {n_games},\n  \"positions\": {n_positions},\n  \"transitions\": {n_transitions},",
        args.input.display().to_string()
    );
    let _ = writeln!(
        out,
        "  \"transitions_black_king_moved\": {king_moves},\n  \"transitions_any_king_moved\": {any_king_moves},\n  \"transitions_by_gp5_bucket_of_pre_move_position\": {gp_counts:?},\n  \"ply_max\": {ply_max},\n  \"subsample_seed\": {},\n  \"subsample_size\": {},",
        args.seed,
        sample_idx.len()
    );
    out.push_str("  \"definition\": \"delta = |active(prev) xor active(next)| of threat indices from one perspective (black = Black perspective, white = White perspective); no_own_king excludes transitions where that perspective's own king moved (engine refreshes there); by_gp5 / by_phase use the gp5 progress of the pre-move position (phase: opening gp<0.4, middle 0.4<=gp<0.8, end gp>=0.8); by_ply uses the pre-move ply (<40, 40-99, >=100); active_black is the active count per position; king_move_full_black is the active count at positions reached by an own (Black) king move (refresh size).\",\n");
    out.push_str("  \"profiles\": {\n");
    let n_keys = buckets.len();
    for (ki, ((profile, mirror), b)) in buckets.iter().enumerate() {
        let first10k = Series::median_of(&b.all_black.values[..b.all_black.values.len().min(10_000)]);
        let sample: Vec<u32> = sample_idx
            .iter()
            .map(|&i| b.all_black.values[i])
            .collect();
        let sample_median = Series::median_of(&sample);
        let _ = writeln!(out, "    \"{profile}/mirror-{mirror}\": {{");
        let _ = writeln!(out, "      \"all_black\": {},", b.all_black.stats_json());
        let _ = writeln!(out, "      \"all_white\": {},", b.all_white.stats_json());
        let _ = writeln!(
            out,
            "      \"no_own_king_black\": {},",
            b.no_own_king_black.stats_json()
        );
        let _ = writeln!(
            out,
            "      \"no_own_king_white\": {},",
            b.no_own_king_white.stats_json()
        );
        let _ = writeln!(
            out,
            "      \"no_any_king_black\": {},",
            b.no_any_king_black.stats_json()
        );
        let _ = writeln!(
            out,
            "      \"by_gp5_black\": [{}],",
            b.by_gp5
                .iter()
                .map(|s| s.stats_json())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            out,
            "      \"by_phase_black\": {{\"opening\": {}, \"middle\": {}, \"end\": {}}},",
            b.by_phase[0].stats_json(),
            b.by_phase[1].stats_json(),
            b.by_phase[2].stats_json()
        );
        let _ = writeln!(
            out,
            "      \"by_ply_black\": {{\"lt40\": {}, \"40to99\": {}, \"ge100\": {}}},",
            b.by_ply[0].stats_json(),
            b.by_ply[1].stats_json(),
            b.by_ply[2].stats_json()
        );
        let _ = writeln!(out, "      \"active_black\": {},", b.active_black.stats_json());
        let _ = writeln!(
            out,
            "      \"king_move_full_black\": {},",
            b.king_move_full_black.stats_json()
        );
        let _ = writeln!(
            out,
            "      \"median_first_10000\": {first10k},\n      \"median_random_10000\": {sample_median}"
        );
        let _ = writeln!(out, "    }}{}", if ki + 1 < n_keys { "," } else { "" });
    }
    out.push_str("  }\n}\n");
    fs::write(&args.output, out)?;
    eprintln!("[survey] wrote {}", args.output.display());
    Ok(())
}

// ---- SFEN parsing (same rules as bins/eval_sfen) -------------------------------------

fn parse_sfen(sfen: &str) -> Result<ShogiBoard, String> {
    let mut parts = sfen.split_whitespace();
    let board_part = parts.next().ok_or("missing board")?;
    let side_part = parts.next().ok_or("missing side-to-move")?;
    let hand_part = parts.next().ok_or("missing hand")?;
    let ply_part = parts.next().unwrap_or("1");

    let mut board = ShogiBoard {
        side_to_move: match side_part {
            "b" => Color::Black,
            "w" => Color::White,
            _ => return Err(format!("invalid side-to-move: {side_part}")),
        },
        ply: ply_part
            .parse::<u16>()
            .map_err(|_| format!("invalid ply: {ply_part}"))?,
        ..Default::default()
    };
    parse_board(board_part, &mut board)?;
    parse_hand(hand_part, &mut board)?;
    Ok(board)
}

fn parse_board(board_part: &str, board: &mut ShogiBoard) -> Result<(), String> {
    let ranks: Vec<&str> = board_part.split('/').collect();
    if ranks.len() != 9 {
        return Err(format!("board must have 9 ranks: {board_part}"));
    }
    for (rank, row) in ranks.iter().enumerate() {
        let mut file_from_left = 0usize;
        let mut promoted = false;
        for ch in row.chars() {
            if ch == '+' {
                promoted = true;
                continue;
            }
            if let Some(skip) = ch.to_digit(10) {
                file_from_left += skip as usize;
                continue;
            }
            if file_from_left >= 9 {
                return Err(format!("too many files in rank: {row}"));
            }
            let color = if ch.is_ascii_uppercase() {
                Color::Black
            } else {
                Color::White
            };
            let mut pt = piece_type_from_sfen(ch)?;
            if promoted {
                pt = pt.promote();
                promoted = false;
            }
            let sq = Square::new((8 - file_from_left) as u8, rank as u8);
            board.board[sq.index()] = Piece::new(color, pt);
            if pt == PieceType::King {
                match color {
                    Color::Black => board.black_king_sq = sq,
                    Color::White => board.white_king_sq = sq,
                }
            }
            file_from_left += 1;
        }
        if file_from_left != 9 {
            return Err(format!("rank does not contain 9 files: {row}"));
        }
    }
    Ok(())
}

fn parse_hand(hand_part: &str, board: &mut ShogiBoard) -> Result<(), String> {
    if hand_part == "-" {
        return Ok(());
    }
    let mut count = 0u8;
    for ch in hand_part.chars() {
        if let Some(digit) = ch.to_digit(10) {
            count = count.saturating_mul(10).saturating_add(digit as u8);
            continue;
        }
        let n = if count == 0 { 1 } else { count };
        count = 0;
        let color = if ch.is_ascii_uppercase() {
            Color::Black
        } else {
            Color::White
        };
        let pt = piece_type_from_sfen(ch)?;
        hand_mut(board, color).add(pt, n);
    }
    Ok(())
}

fn piece_type_from_sfen(ch: char) -> Result<PieceType, String> {
    match ch.to_ascii_uppercase() {
        'P' => Ok(PieceType::Pawn),
        'L' => Ok(PieceType::Lance),
        'N' => Ok(PieceType::Knight),
        'S' => Ok(PieceType::Silver),
        'G' => Ok(PieceType::Gold),
        'B' => Ok(PieceType::Bishop),
        'R' => Ok(PieceType::Rook),
        'K' => Ok(PieceType::King),
        _ => Err(format!("invalid piece char: {ch}")),
    }
}

fn hand_mut(board: &mut ShogiBoard, color: Color) -> &mut Hand {
    match color {
        Color::Black => &mut board.black_hand,
        Color::White => &mut board.white_hand,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_difference_counts_removed_plus_added() {
        assert_eq!(symmetric_difference_len(&[1, 2, 3], &[2, 3, 4]), 2);
        assert_eq!(symmetric_difference_len(&[], &[1, 2]), 2);
        assert_eq!(symmetric_difference_len(&[5, 6], &[5, 6]), 0);
        assert_eq!(symmetric_difference_len(&[1, 9], &[2, 3, 9]), 3);
    }

    #[test]
    fn startpos_parses_and_has_step_threats() {
        let b = parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .unwrap();
        let idx = ThreatIndexer::new(ThreatProfile::StepAttacker);
        let a = active_sorted(&idx, &b, Color::Black, ThreatMirror::Off);
        assert!(!a.is_empty());
        assert!(a.windows(2).all(|w| w[0] < w[1]));
    }
}
