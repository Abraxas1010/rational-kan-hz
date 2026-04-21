use crate::base_phi::{BasePhiDigits, ShiftSupport};
use rand::Rng;
use rug::Integer;
use std::collections::BTreeSet;

pub fn support_from_density(span: i32, rho: f64) -> usize {
    let width = (2 * span + 1).max(1) as f64;
    ((rho * width).round() as usize).clamp(1, width as usize)
}

pub fn random_sparse_digits(
    span: i32,
    support: usize,
    coeff_bits: u32,
    rng: &mut impl Rng,
) -> BasePhiDigits {
    let mut exponents = BTreeSet::new();
    while exponents.len() < support {
        exponents.insert(rng.gen_range(-span..=span));
    }

    let mut out = BasePhiDigits::new();
    let bit_cap = coeff_bits.min(30);
    let max_mag = (1u64 << bit_cap).saturating_sub(1).max(1);
    for exponent in exponents {
        let magnitude = rng.gen_range(1..=max_mag);
        let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
        out.insert(exponent, Integer::from(sign * magnitude as i64));
    }
    out
}

pub fn random_shift_support(
    span: i32,
    support: usize,
    coeff_bits: u32,
    rng: &mut impl Rng,
) -> ShiftSupport {
    random_sparse_digits(span, support, coeff_bits, rng)
}
