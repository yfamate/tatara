//! Board-derived five-bucket progress, including entering-king progress above one.
use shogi_format::types::Square;
use shogi_format::{Color, PieceType, ShogiBoard};

pub const GP5_NUM_BUCKETS: usize = 5;
pub const GP5_BOUNDS: [f64; 4] = [0.4, 0.6, 0.8, 1.0];

fn index(pt: PieceType) -> Option<usize> {
    use PieceType::*;
    match pt {
        Pawn => Some(0),
        Lance => Some(1),
        Knight => Some(2),
        Silver => Some(3),
        Gold => Some(4),
        Bishop => Some(5),
        Rook => Some(6),
        ProPawn => Some(7),
        ProLance => Some(8),
        ProKnight => Some(9),
        ProSilver => Some(10),
        Horse => Some(11),
        Dragon => Some(12),
        _ => Option::None,
    }
}

/// Progress is independent of side to move, move number and earlier root positions.
pub fn gp5_progress_board(board: &ShogiBoard) -> f64 {
    const HAND: [f64; 7] = [
        1.0, 2.775113, 2.066253, 3.120539, 3.911216, 4.574237, 7.863081,
    ];
    const CAMP: [f64; 13] = [
        1.396302, 1.177588, 1.248085, 1.297911, 1.444452, 1.066618, 1.40945, 0.563566, 1.186525,
        0.91197, 1.020019, 0.592588, 1.458039,
    ];
    let mut hand = 0.0;
    let mut camp = 0.0;
    let mut king = 0.0;
    for (color, held, square) in [
        (Color::Black, &board.black_hand, board.black_king_sq),
        (Color::White, &board.white_hand, board.white_king_sq),
    ] {
        for (n, value) in held.counts.iter().zip(HAND) {
            hand += f64::from(*n) * value;
        }
        for (sq, piece) in board.board.iter().enumerate() {
            if piece.color != color {
                continue;
            }
            let rank = Square(sq as u8).rank();
            if (color == Color::Black && rank <= 2) || (color == Color::White && rank >= 6) {
                if let Some(i) = index(piece.piece_type) {
                    camp += CAMP[i];
                }
            }
        }
        if !square.is_valid() {
            continue;
        }
        let kr = i32::from(square.rank()) + 1;
        let raw = if color == Color::Black {
            7 - kr
        } else {
            kr - 3
        };
        if raw < 0 {
            continue;
        }
        let support = board
            .board
            .iter()
            .enumerate()
            .filter(|(sq, piece)| {
                if piece.color != color || piece.is_none() || piece.piece_type == PieceType::King {
                    return false;
                }
                let s = Square(*sq as u8);
                let sr = i32::from(s.rank()) + 1;
                s.file().abs_diff(square.file()) <= 2
                    && s.rank().abs_diff(square.rank()) <= 2
                    && if color == Color::Black {
                        sr <= kr + 1
                    } else {
                        sr >= kr - 1
                    }
            })
            .count()
            .min(3);
        king += (f64::from(raw) + 0.5) * (0.25 + 0.25 * support as f64);
    }
    let sum = 0.334803 * hand / 2.035839 + 0.312461 * camp / 1.173523 + 0.540983 * king / 0.121706;
    let neg = -sum / 1.285300;
    let core = if neg < -10.0 { 1.0 } else { 1.0 - neg.exp() };
    let bonus = if king > 0.514070 {
        0.5 * ((king - 0.514070) / (4.757159 - 0.514070)).min(1.0)
    } else {
        0.0
    };
    (core + bonus).clamp(0.0, 1.5)
}

/// Bucket index of a progress value for a boundary list: the number of boundaries `b` with
/// `b <= gp`. With `n` boundaries the result is in `0..=n`. Callers keep `bounds` finite and
/// strictly increasing and at most 255 long.
#[inline]
pub fn gp_bucket_from_progress(gp: f64, bounds: &[f64]) -> u8 {
    bounds.iter().filter(|&&bound| gp >= bound).count() as u8
}

/// Board-progress bucket for an arbitrary boundary list (see [`gp_bucket_from_progress`]).
#[inline]
pub fn gp_bucket_board(board: &ShogiBoard, bounds: &[f64]) -> u8 {
    gp_bucket_from_progress(gp5_progress_board(board), bounds)
}

pub fn gp5_bucket_board(board: &ShogiBoard) -> u8 {
    gp_bucket_board(board, &GP5_BOUNDS)
}
