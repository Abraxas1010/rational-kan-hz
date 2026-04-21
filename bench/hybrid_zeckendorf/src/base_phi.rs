use crate::fib_table::fib;
use crate::normalization::HybridNumber;
use crate::weight::weight;
use rug::ops::Pow;
use rug::{Float, Integer, Rational};
use std::collections::BTreeMap;

pub type ShiftSupport = BTreeMap<i32, Integer>;
pub type BasePhiDigits = BTreeMap<i32, Integer>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhiPair {
    pub constant: Integer,
    pub phi: Integer,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FibPartCarrier {
    pub parts: Vec<u32>,
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.partRatRaw
pub fn part_rat_raw(n: u32) -> Rational {
    Rational::from((1, fib(n + 1))) - Rational::from((1, fib(n + 2)))
}

impl FibPartCarrier {
    // Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.FibPartCarrier.eval
    pub fn eval(&self) -> Rational {
        self.parts
            .iter()
            .fold(Rational::from(0), |acc, &n| acc + part_rat_raw(n))
    }
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.telescopic_prefix_sum
pub fn telescopic_prefix_sum(start: u32, len: u32) -> Rational {
    let mut acc = Rational::from(0);
    for i in 0..len {
        acc += part_rat_raw(start + i);
    }
    acc
}

pub fn telescopic_prefix_closed_form(start: u32, len: u32) -> Rational {
    Rational::from((1, fib(start + 1))) - Rational::from((1, fib(start + len + 1)))
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.signedFib
pub fn signed_fib(i: i32) -> Integer {
    if i >= 0 {
        fib(i as u32)
    } else {
        let n = (-i) as u32;
        let value = fib(n);
        if n % 2 == 1 {
            value
        } else {
            -value
        }
    }
}

pub fn phi_pair_pow(index: i32) -> PhiPair {
    PhiPair {
        constant: signed_fib(index - 1),
        phi: signed_fib(index),
    }
}

pub fn phi_pair_add(lhs: &PhiPair, rhs: &PhiPair) -> PhiPair {
    PhiPair {
        constant: Integer::from(&lhs.constant + &rhs.constant),
        phi: Integer::from(&lhs.phi + &rhs.phi),
    }
}

pub fn phi_pair_mul(lhs: &PhiPair, rhs: &PhiPair) -> PhiPair {
    let constant = Integer::from(&lhs.constant * &rhs.constant) + Integer::from(&lhs.phi * &rhs.phi);
    let phi = Integer::from(&lhs.constant * &rhs.phi)
        + Integer::from(&lhs.phi * &rhs.constant)
        + Integer::from(&lhs.phi * &rhs.phi);
    PhiPair { constant, phi }
}

pub fn base_phi_pair_eval(d: &BasePhiDigits) -> PhiPair {
    d.iter().fold(PhiPair::default(), |acc, (index, coeff)| {
        let pow = phi_pair_pow(*index);
        phi_pair_add(
            &acc,
            &PhiPair {
                constant: Integer::from(coeff * &pow.constant),
                phi: Integer::from(coeff * &pow.phi),
            },
        )
    })
}

pub fn integer_like_value(d: &BasePhiDigits) -> Option<Integer> {
    let pair = base_phi_pair_eval(d);
    if pair.phi == 0 {
        Some(pair.constant)
    } else {
        None
    }
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.shiftEval
pub fn shift_eval(s: &ShiftSupport) -> Integer {
    s.iter().fold(Integer::from(0), |acc, (i, coeff)| {
        acc + coeff * signed_fib(*i)
    })
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.shiftSingle
pub fn shift_single(i: i32, coeff: Integer) -> ShiftSupport {
    let mut out = ShiftSupport::new();
    if coeff != 0 {
        out.insert(i, coeff);
    }
    out
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.shiftBy
pub fn shift_by(k: i32, s: &ShiftSupport) -> ShiftSupport {
    let mut out = ShiftSupport::new();
    for (i, coeff) in s {
        let dst = out.entry(i + k).or_insert_with(|| Integer::from(0));
        *dst += coeff;
    }
    out.retain(|_, coeff| *coeff != 0);
    out
}

pub fn shift_eval_shifted_rhs(k: i32, s: &ShiftSupport) -> Integer {
    s.iter().fold(Integer::from(0), |acc, (i, coeff)| {
        acc + coeff * signed_fib(i + k)
    })
}

pub fn golden_ratio(prec: u32) -> Float {
    let work_prec = prec + 32;
    let sqrt5 = Float::with_val(work_prec, 5).sqrt();
    Float::with_val(work_prec, (1 + sqrt5) / 2)
}

fn golden_ratio_pow(index: i32, prec: u32) -> Float {
    let work_prec = prec + 32;
    let phi = golden_ratio(work_prec);
    if index >= 0 {
        Float::with_val(work_prec, phi.pow(index as u32))
    } else {
        let denom = phi.pow((-index) as u32);
        Float::with_val(work_prec, 1) / denom
    }
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.basePhiEval
pub fn base_phi_eval(d: &BasePhiDigits, prec: u32) -> Float {
    let work_prec = prec + 32;
    let mut acc = Float::with_val(work_prec, 0);
    for (i, coeff) in d {
        let term = Float::with_val(work_prec, coeff) * golden_ratio_pow(*i, work_prec);
        acc += term;
    }
    Float::with_val(prec, acc)
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.basePhiCanonical
pub fn base_phi_canonical(d: &BasePhiDigits) -> bool {
    d.iter()
        .all(|(i, coeff)| *coeff == 0 || d.get(&(i + 1)).is_none_or(|next| *next == 0))
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.shiftToBasePhi
pub fn shift_to_base_phi(s: &ShiftSupport) -> BasePhiDigits {
    s.clone()
}

pub fn add_digits(a: &BasePhiDigits, b: &BasePhiDigits) -> BasePhiDigits {
    let mut out = a.clone();
    for (i, coeff) in b {
        let dst = out.entry(*i).or_insert_with(|| Integer::from(0));
        *dst += coeff;
    }
    out.retain(|_, coeff| *coeff != 0);
    out
}

fn add_monomial(out: &mut BasePhiDigits, exponent: i32, coeff: Integer) {
    if coeff == 0 {
        return;
    }
    let dst = out.entry(exponent).or_insert_with(|| Integer::from(0));
    *dst += coeff;
    if *dst == 0 {
        out.remove(&exponent);
    }
}

pub fn scale_shift(d: &BasePhiDigits, shift: i32, scale: &Integer) -> BasePhiDigits {
    if *scale == 0 {
        return BasePhiDigits::new();
    }
    let mut out = BasePhiDigits::new();
    for (i, coeff) in d {
        let value = Integer::from(coeff * scale);
        if value != 0 {
            out.insert(i + shift, value);
        }
    }
    out
}

// Mirrors the intended raw Laurent-product semantics behind Lean's basePhiEval_mul.
pub fn raw_mul(a: &BasePhiDigits, b: &BasePhiDigits) -> BasePhiDigits {
    let mut out = BasePhiDigits::new();
    for (ia, ca) in a {
        for (ib, cb) in b {
            let dst = out.entry(ia + ib).or_insert_with(|| Integer::from(0));
            *dst += ca * cb;
        }
    }
    out.retain(|_, coeff| *coeff != 0);
    out
}

pub fn repeated_add_mul(a: &BasePhiDigits, b: &BasePhiDigits) -> BasePhiDigits {
    let mut acc = BasePhiDigits::new();
    for (i, coeff) in b {
        let term = scale_shift(a, *i, coeff);
        acc = add_digits(&acc, &term);
    }
    acc
}

pub fn dense_mul(a: &BasePhiDigits, b: &BasePhiDigits) -> BasePhiDigits {
    let (min_a, max_a) = match exponent_span(a) {
        Some(span) => span,
        None => return BasePhiDigits::new(),
    };
    let (min_b, max_b) = match exponent_span(b) {
        Some(span) => span,
        None => return BasePhiDigits::new(),
    };

    let len_a = (max_a - min_a + 1) as usize;
    let len_b = (max_b - min_b + 1) as usize;
    let mut dense_a = vec![Integer::from(0); len_a];
    let mut dense_b = vec![Integer::from(0); len_b];

    for (i, coeff) in a {
        dense_a[(i - min_a) as usize] = coeff.clone();
    }
    for (i, coeff) in b {
        dense_b[(i - min_b) as usize] = coeff.clone();
    }

    let mut dense_out = vec![Integer::from(0); len_a + len_b - 1];
    for (ia, ca) in dense_a.iter().enumerate() {
        if *ca == 0 {
            continue;
        }
        for (ib, cb) in dense_b.iter().enumerate() {
            if *cb == 0 {
                continue;
            }
            dense_out[ia + ib] += ca * cb;
        }
    }

    let min_out = min_a + min_b;
    let mut out = BasePhiDigits::new();
    for (offset, coeff) in dense_out.into_iter().enumerate() {
        if coeff != 0 {
            out.insert(min_out + offset as i32, coeff);
        }
    }
    out
}

/// O(n log n) base-φ multiplication via four-prime NTT with Garner CRT.
///
/// Equivalent to `raw_mul` and `dense_mul` (Laurent polynomial convolution),
/// but uses FFT to achieve subquadratic complexity in the number of base-φ digits.
/// Falls back to `dense_mul` when coefficients exceed the CRT range (~115 bits).
pub fn fft_mul(a: &BasePhiDigits, b: &BasePhiDigits) -> BasePhiDigits {
    let (min_a, max_a) = match exponent_span(a) {
        Some(span) => span,
        None => return BasePhiDigits::new(),
    };
    let (min_b, max_b) = match exponent_span(b) {
        Some(span) => span,
        None => return BasePhiDigits::new(),
    };

    let len_a = (max_a - min_a + 1) as usize;
    let len_b = (max_b - min_b + 1) as usize;
    let mut dense_a = vec![Integer::from(0); len_a];
    let mut dense_b = vec![Integer::from(0); len_b];

    for (i, coeff) in a {
        dense_a[(i - min_a) as usize] = coeff.clone();
    }
    for (i, coeff) in b {
        dense_b[(i - min_b) as usize] = coeff.clone();
    }

    let max_bits_a = dense_a.iter().map(|v| v.significant_bits()).max().unwrap_or(0);
    let max_bits_b = dense_b.iter().map(|v| v.significant_bits()).max().unwrap_or(0);
    let n = len_a.min(len_b) as u32;
    let product_bits = max_bits_a + max_bits_b + n.max(1).ilog2() + 2;
    if product_bits > 112 {
        return dense_mul(a, b);
    }

    let dense_out = crate::ntt::fft_convolve(&dense_a, &dense_b);

    let min_out = min_a + min_b;
    let mut out = BasePhiDigits::new();
    for (offset, coeff) in dense_out.into_iter().enumerate() {
        if coeff != 0 {
            out.insert(min_out + offset as i32, coeff);
        }
    }
    out
}

/// Energy-optimized base-φ multiplication with automatic dispatch.
///
/// Three-tier strategy based on PMU energy measurements:
/// 1. Small inputs (digit_pairs < 16384): i64 schoolbook — zero Integer arithmetic
///    in the multiply kernel, highest IPC, minimal cache pressure
/// 2. Medium inputs: single-prime NTT on i64 — O(n log n) with Shoup modular multiply
/// 3. Large coefficients: four-prime NTT with i128 CRT — no rug::Integer allocation
/// Falls back to `fft_mul` only when coefficients exceed i64 range.
pub fn fft_mul_fast(a: &BasePhiDigits, b: &BasePhiDigits) -> BasePhiDigits {
    let (min_a, max_a) = match exponent_span(a) {
        Some(span) => span,
        None => return BasePhiDigits::new(),
    };
    let (min_b, max_b) = match exponent_span(b) {
        Some(span) => span,
        None => return BasePhiDigits::new(),
    };

    let len_a = (max_a - min_a + 1) as usize;
    let len_b = (max_b - min_b + 1) as usize;

    let mut dense_a = vec![0i64; len_a];
    let mut dense_b = vec![0i64; len_b];
    let mut max_abs_a: u64 = 0;
    let mut max_abs_b: u64 = 0;

    for (&exp, coeff) in a {
        let Some(v) = coeff.to_i64() else {
            return fft_mul(a, b);
        };
        dense_a[(exp - min_a) as usize] = v;
        max_abs_a = max_abs_a.max(v.unsigned_abs());
    }
    for (&exp, coeff) in b {
        let Some(v) = coeff.to_i64() else {
            return fft_mul(a, b);
        };
        dense_b[(exp - min_b) as usize] = v;
        max_abs_b = max_abs_b.max(v.unsigned_abs());
    }

    let min_out = min_a + min_b;

    if (len_a as u64) * (len_b as u64) < crate::ntt_fast::DENSE_ENERGY_CROSSOVER {
        let out_len = len_a + len_b - 1;
        let mut dense_out = vec![0i128; out_len];
        for (ia, &ca) in dense_a.iter().enumerate() {
            if ca == 0 { continue; }
            for (ib, &cb) in dense_b.iter().enumerate() {
                if cb == 0 { continue; }
                dense_out[ia + ib] += ca as i128 * cb as i128;
            }
        }
        let mut out = BasePhiDigits::new();
        for (offset, coeff) in dense_out.into_iter().enumerate() {
            if coeff != 0 {
                out.insert(min_out + offset as i32, Integer::from(coeff));
            }
        }
        return out;
    }

    let n_min = len_a.min(len_b) as u64;
    let max_output = (max_abs_a as u128) * (max_abs_b as u128) * (n_min as u128);

    if max_output < crate::ntt_fast::safe_coefficient_bound() as u128 {
        let dense_out = crate::ntt_fast::convolve_i64(&dense_a, &dense_b);
        let mut out = BasePhiDigits::new();
        for (offset, coeff) in dense_out.into_iter().enumerate() {
            if coeff != 0 {
                out.insert(min_out + offset as i32, Integer::from(coeff));
            }
        }
        return out;
    }

    let dense_out = crate::ntt_fast::convolve_i64_multi(&dense_a, &dense_b);
    let mut out = BasePhiDigits::new();
    for (offset, coeff) in dense_out.into_iter().enumerate() {
        if coeff != 0 {
            out.insert(min_out + offset as i32, Integer::from(coeff));
        }
    }
    out
}

pub fn exponent_span(d: &BasePhiDigits) -> Option<(i32, i32)> {
    Some((*d.first_key_value()?.0, *d.last_key_value()?.0))
}

pub fn support_density(d: &BasePhiDigits) -> f64 {
    let Some((lo, hi)) = exponent_span(d) else {
        return 0.0;
    };
    d.len() as f64 / (hi - lo + 1) as f64
}

pub fn fib_to_base_phi(index: u32) -> BasePhiDigits {
    let mut out = BasePhiDigits::new();
    if index == 0 {
        return out;
    }

    if index % 2 == 0 {
        let m = ((index - 2) / 2) as i32;
        for j in 0..=m {
            add_monomial(&mut out, 4 * j - 2 * m, Integer::from(1));
        }
    } else {
        let m = ((index - 1) / 2) as i32;
        add_monomial(&mut out, -2 * m, Integer::from(1));
        for j in 0..m {
            add_monomial(&mut out, 4 * j - (2 * m - 3), Integer::from(1));
        }
    }

    out
}

pub fn hybrid_to_base_phi(value: &HybridNumber) -> BasePhiDigits {
    let mut out = BasePhiDigits::new();
    for (&level, payload) in &value.levels {
        let scale = weight(level);
        for &fib_index in payload {
            let lifted = fib_to_base_phi(fib_index);
            for (exponent, coeff) in lifted {
                add_monomial(&mut out, exponent, Integer::from(coeff * &scale));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HybridNumber;

    fn approx_eq(a: &Float, b: &Float, tol: f64) -> bool {
        let diff = Float::with_val(a.prec().max(b.prec()), a - b).abs();
        diff.to_f64() <= tol
    }

    #[test]
    fn telescopic_prefix_matches_closed_form() {
        for start in 0..8 {
            for len in 0..8 {
                assert_eq!(
                    telescopic_prefix_sum(start, len),
                    telescopic_prefix_closed_form(start, len),
                    "start={start} len={len}"
                );
            }
        }
    }

    #[test]
    fn signed_fib_matches_negafibonacci() {
        assert_eq!(signed_fib(-1), 1);
        assert_eq!(signed_fib(-2), -1);
        assert_eq!(signed_fib(-3), 2);
        assert_eq!(signed_fib(0), 0);
        assert_eq!(signed_fib(1), 1);
        assert_eq!(signed_fib(7), 13);
    }

    #[test]
    fn phi_pair_pow_matches_small_powers() {
        assert_eq!(
            phi_pair_pow(0),
            PhiPair {
                constant: Integer::from(1),
                phi: Integer::from(0),
            }
        );
        assert_eq!(
            phi_pair_pow(1),
            PhiPair {
                constant: Integer::from(0),
                phi: Integer::from(1),
            }
        );
        assert_eq!(
            phi_pair_pow(-1),
            PhiPair {
                constant: Integer::from(-1),
                phi: Integer::from(1),
            }
        );
    }

    #[test]
    fn phi_pair_mul_matches_phi_relation() {
        let phi = PhiPair {
            constant: Integer::from(0),
            phi: Integer::from(1),
        };
        assert_eq!(
            phi_pair_mul(&phi, &phi),
            PhiPair {
                constant: Integer::from(1),
                phi: Integer::from(1),
            }
        );
    }

    #[test]
    fn fib_to_base_phi_is_integer_like() {
        let expected = [0u32, 1, 1, 2, 3, 5, 8, 13, 21];
        for (index, &fib_value) in expected.iter().enumerate() {
            let digits = fib_to_base_phi(index as u32);
            assert_eq!(
                integer_like_value(&digits),
                Some(Integer::from(fib_value)),
                "fib index={index}"
            );
        }
    }

    #[test]
    fn hybrid_to_base_phi_preserves_integer_value() {
        let value = Integer::from(12_345_678u64);
        let hz = HybridNumber::from_integer(&value);
        let digits = hybrid_to_base_phi(&hz);
        assert_eq!(integer_like_value(&digits), Some(value));
    }

    #[test]
    fn shift_by_matches_shifted_rhs_semantics() {
        let mut s = ShiftSupport::new();
        s.insert(-2, Integer::from(3));
        s.insert(1, Integer::from(-5));
        s.insert(4, Integer::from(2));
        let shifted = shift_by(3, &s);
        assert_eq!(shift_eval(&shifted), shift_eval_shifted_rhs(3, &s));
    }

    #[test]
    fn shift_to_base_phi_preserves_eval_formula() {
        let mut s = ShiftSupport::new();
        s.insert(-1, Integer::from(1));
        s.insert(2, Integer::from(3));
        let lhs = base_phi_eval(&shift_to_base_phi(&s), 256);
        let rhs = base_phi_eval(&s, 256);
        assert!(approx_eq(&lhs, &rhs, 1e-25));
    }

    #[test]
    fn raw_mul_multiplies_base_phi_eval() {
        let mut a = BasePhiDigits::new();
        a.insert(0, Integer::from(1));
        a.insert(2, Integer::from(1));
        let mut b = BasePhiDigits::new();
        b.insert(-1, Integer::from(2));
        b.insert(1, Integer::from(1));
        let lhs = base_phi_eval(&raw_mul(&a, &b), 256);
        let rhs = Float::with_val(256, base_phi_eval(&a, 256) * base_phi_eval(&b, 256));
        assert!(approx_eq(&lhs, &rhs, 1e-22));
    }

    #[test]
    fn fft_mul_matches_raw_mul() {
        let mut a = BasePhiDigits::new();
        a.insert(-3, Integer::from(2));
        a.insert(1, Integer::from(-5));
        a.insert(4, Integer::from(3));
        let mut b = BasePhiDigits::new();
        b.insert(-2, Integer::from(7));
        b.insert(0, Integer::from(1));
        b.insert(5, Integer::from(-2));
        assert_eq!(fft_mul(&a, &b), raw_mul(&a, &b));
    }

    #[test]
    fn fft_mul_on_hybrid_numbers() {
        for &(x, y) in &[(100u64, 200), (12345, 67890), (999999, 1000001)] {
            let hx = HybridNumber::from_u64(x);
            let hy = HybridNumber::from_u64(y);
            let px = hybrid_to_base_phi(&hx);
            let py = hybrid_to_base_phi(&hy);
            let product = fft_mul(&px, &py);
            assert_eq!(
                integer_like_value(&product),
                Some(Integer::from(x as u128 * y as u128)),
                "fft_mul incorrect for {x}×{y}"
            );
        }
    }

    #[test]
    fn fft_mul_large_values() {
        let val_a = Integer::from(10u32).pow(20);
        let val_b = Integer::from(10u32).pow(20) + Integer::from(42);
        let ha = HybridNumber::from_integer(&val_a);
        let hb = HybridNumber::from_integer(&val_b);
        let pa = hybrid_to_base_phi(&ha);
        let pb = hybrid_to_base_phi(&hb);
        let product = fft_mul(&pa, &pb);
        assert_eq!(
            integer_like_value(&product),
            Some(Integer::from(&val_a * &val_b)),
        );
    }

    #[test]
    fn repeated_add_and_dense_match_raw_mul() {
        let mut a = BasePhiDigits::new();
        a.insert(-3, Integer::from(2));
        a.insert(1, Integer::from(-5));
        a.insert(4, Integer::from(3));
        let mut b = BasePhiDigits::new();
        b.insert(-2, Integer::from(7));
        b.insert(0, Integer::from(1));
        b.insert(5, Integer::from(-2));
        let raw = raw_mul(&a, &b);
        assert_eq!(repeated_add_mul(&a, &b), raw);
        assert_eq!(dense_mul(&a, &b), raw);
    }

    #[test]
    fn fft_mul_fast_matches_raw_mul() {
        let mut a = BasePhiDigits::new();
        a.insert(-3, Integer::from(2));
        a.insert(1, Integer::from(-5));
        a.insert(4, Integer::from(3));
        let mut b = BasePhiDigits::new();
        b.insert(-2, Integer::from(7));
        b.insert(0, Integer::from(1));
        b.insert(5, Integer::from(-2));
        assert_eq!(fft_mul_fast(&a, &b), raw_mul(&a, &b));
    }

    #[test]
    fn fft_mul_fast_on_hybrid_numbers() {
        for &(x, y) in &[(100u64, 200), (12345, 67890), (999999, 1000001)] {
            let hx = HybridNumber::from_u64(x);
            let hy = HybridNumber::from_u64(y);
            let px = hybrid_to_base_phi(&hx);
            let py = hybrid_to_base_phi(&hy);
            let product = fft_mul_fast(&px, &py);
            assert_eq!(
                integer_like_value(&product),
                Some(Integer::from(x as u128 * y as u128)),
                "fft_mul_fast incorrect for {x}×{y}"
            );
        }
    }

    #[test]
    fn fft_mul_fast_matches_fft_mul_on_large() {
        let val_a = Integer::from(10u32).pow(20);
        let val_b = Integer::from(10u32).pow(20) + Integer::from(42);
        let ha = HybridNumber::from_integer(&val_a);
        let hb = HybridNumber::from_integer(&val_b);
        let pa = hybrid_to_base_phi(&ha);
        let pb = hybrid_to_base_phi(&hb);
        assert_eq!(fft_mul_fast(&pa, &pb), fft_mul(&pa, &pb));
    }

    #[test]
    fn fft_mul_fast_falls_back_for_huge_coefficients() {
        let mut a = BasePhiDigits::new();
        a.insert(0, Integer::from(10u32).pow(30));
        a.insert(1, Integer::from(1));
        let mut b = BasePhiDigits::new();
        b.insert(0, Integer::from(10u32).pow(30));
        b.insert(1, Integer::from(-1));
        // Coefficients don't fit i64 — should fall back to fft_mul and still be correct
        assert_eq!(fft_mul_fast(&a, &b), raw_mul(&a, &b));
    }
}
