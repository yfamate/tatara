use shogi_features::{
    GP5_BOUNDS, gp_bucket_board, gp_bucket_from_progress, gp5_bucket_board, gp5_progress_board,
};
use shogi_format::{PackedSfenValue, ShogiBoard};

// Packed positions and full-scan values from Cyfamate's legal-move/undo corpus.
fn oracle_positions() -> Vec<(ShogiBoard, f64, u8, &'static str)> {
    include_str!("fixtures/gp5_oracle.dat")
        .lines()
        .map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            let mut record = PackedSfenValue::default();
            for i in 0..32 {
                record.as_bytes_mut()[i] =
                    u8::from_str_radix(&fields[0][2 * i..2 * i + 2], 16).unwrap();
            }
            (
                record.decode(),
                fields[1].parse().unwrap(),
                fields[2].parse().unwrap(),
                line,
            )
        })
        .collect()
}

#[test]
fn progress_matches_cpp_full_scan() {
    let mut buckets = [0; 5];
    for (board, expected, bucket, line) in oracle_positions() {
        assert!(
            (gp5_progress_board(&board) - expected).abs() < 1e-10,
            "{line}"
        );
        assert_eq!(gp5_bucket_board(&board), bucket, "{line}");
        buckets[bucket as usize] += 1;
    }
    assert!(buckets.iter().all(|&n| n > 0), "{buckets:?}");
}

#[test]
fn configurable_bounds_reproduce_gp5_buckets() {
    let positions = oracle_positions();
    assert!(!positions.is_empty());
    for (board, _, bucket, line) in &positions {
        assert_eq!(gp_bucket_board(board, &GP5_BOUNDS), *bucket, "{line}");
        assert_eq!(gp_bucket_board(board, &[0.4, 0.6, 0.8, 1.0]), *bucket, "{line}");
        assert_eq!(gp_bucket_board(board, &[]), 0, "{line}");
        // Merging the two upper boundaries merges buckets 3 and 4 and leaves the rest.
        assert_eq!(
            gp_bucket_board(board, &[0.4, 0.6, 0.8]),
            (*bucket).min(3),
            "{line}"
        );
    }
}

#[test]
fn boundaries_are_inclusive_lower_edges() {
    assert_eq!(gp_bucket_from_progress(0.0, &GP5_BOUNDS), 0);
    assert_eq!(gp_bucket_from_progress(0.399_999, &GP5_BOUNDS), 0);
    assert_eq!(gp_bucket_from_progress(0.4, &GP5_BOUNDS), 1);
    assert_eq!(gp_bucket_from_progress(0.6, &GP5_BOUNDS), 2);
    assert_eq!(gp_bucket_from_progress(0.8, &GP5_BOUNDS), 3);
    assert_eq!(gp_bucket_from_progress(1.0, &GP5_BOUNDS), 4);
    assert_eq!(gp_bucket_from_progress(1.5, &GP5_BOUNDS), 4);
    assert_eq!(gp_bucket_from_progress(1.5, &[]), 0);
}
