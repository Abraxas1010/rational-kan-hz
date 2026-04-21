use crate::normalization::{log1000_floor, HybridNumber};
use rug::Integer;

pub fn proved_active_levels_bound(n: &Integer) -> u32 {
    if *n <= 0 {
        0
    } else {
        log1000_floor(n) + 2
    }
}

pub fn gap_to_bound(hz: &HybridNumber, n: &Integer) -> i64 {
    let bound = proved_active_levels_bound(n) as i64;
    bound - hz.active_levels() as i64
}

pub fn density_trend_nonincreasing(means: &[(u32, f64)]) -> bool {
    if means.len() < 4 {
        return false;
    }
    let mut consecutive = 0usize;
    for win in means.windows(2) {
        if win[1].1 <= win[0].1 {
            consecutive += 1;
            if consecutive >= 3 {
                return true;
            }
        } else {
            consecutive = 0;
        }
    }
    false
}

pub fn simple_linear_fit(xs: &[f64], ys: &[f64]) -> Option<(f64, f64)> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let n = xs.len() as f64;
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|x| x * x).sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return None;
    }
    let alpha = (n * sum_xy - sum_x * sum_y) / denom;
    let beta = (sum_y - alpha * sum_x) / n;
    Some((alpha, beta))
}
