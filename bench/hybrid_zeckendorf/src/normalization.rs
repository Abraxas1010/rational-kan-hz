use crate::flat_hybrid::FlatHybridNumber;
use crate::weight::weight;
use crate::zeckendorf::{lazy_eval_fib, zeckendorf};
use rug::Integer;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct HybridNumber {
    pub levels: BTreeMap<u32, Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedHybridNumber {
    pub levels: BTreeMap<u32, Vec<i8>>,
}

pub fn naf_digits_u64(mut value: u64) -> Vec<i8> {
    let mut digits = Vec::new();
    while value > 0 {
        if value & 1 == 1 {
            let digit = if value & 3 == 1 { 1 } else { -1 };
            digits.push(digit);
            if digit > 0 {
                value = (value - 1) >> 1;
            } else {
                value = (value >> 1) + 1;
            }
        } else {
            digits.push(0);
            value >>= 1;
        }
    }
    while digits.last() == Some(&0) {
        digits.pop();
    }
    digits
}

pub fn naf_is_canonical(digits: &[i8]) -> bool {
    let mut previous_nonzero = false;
    for &digit in digits {
        if !matches!(digit, -1..=1) {
            return false;
        }
        let is_nonzero = digit != 0;
        if previous_nonzero && is_nonzero {
            return false;
        }
        previous_nonzero = is_nonzero;
    }
    true
}

pub fn naf_support_card(digits: &[i8]) -> u32 {
    digits.iter().filter(|&&digit| digit != 0).count() as u32
}

pub fn naf_eval(digits: &[i8]) -> Integer {
    let mut value = Integer::from(0);
    for (index, &digit) in digits.iter().enumerate() {
        if digit == 0 {
            continue;
        }
        let term = Integer::from(1) << index;
        if digit > 0 {
            value += term;
        } else {
            value -= term;
        }
    }
    value
}

pub fn decompose_weight_coefficients_u64(value: u64) -> BTreeMap<u32, u64> {
    let mut coeffs = BTreeMap::new();
    let mut level = 0u32;
    loop {
        let place = weight(level)
            .to_u64()
            .expect("u64 coefficient decomposition only supports u64-resident weights");
        if place > value && level > 0 {
            break;
        }

        let radix = weight(level + 1).to_u64().map(|next| next / place);
        let coeff = match radix {
            Some(base) => value / place % base,
            None => value / place,
        };
        if coeff != 0 {
            coeffs.insert(level, coeff);
        }

        if radix.is_none() {
            break;
        }
        level += 1;
    }
    coeffs
}

impl SignedHybridNumber {
    pub fn empty() -> Self {
        Self {
            levels: BTreeMap::new(),
        }
    }

    pub fn from_u64(value: u64) -> Self {
        if value == 0 {
            return Self::empty();
        }
        let mut levels = BTreeMap::new();
        for (level, coeff) in decompose_weight_coefficients_u64(value) {
            let digits = naf_digits_u64(coeff);
            if !digits.is_empty() {
                levels.insert(level, digits);
            }
        }
        let mut out = Self { levels };
        out.canonicalize();
        out
    }

    pub fn canonicalize(&mut self) {
        self.levels.retain(|_, digits| {
            while digits.last() == Some(&0) {
                digits.pop();
            }
            !digits.is_empty()
        });
    }

    pub fn eval(&self) -> Integer {
        self.levels
            .iter()
            .map(|(&level, digits)| naf_eval(digits) * weight(level))
            .sum()
    }

    pub fn support_card(&self) -> u32 {
        self.levels.values().map(|digits| naf_support_card(digits)).sum()
    }

    pub fn active_levels(&self) -> u32 {
        self.levels.values().filter(|digits| !digits.is_empty()).count() as u32
    }
}

impl HybridNumber {
    // Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.eval
    pub fn eval(&self) -> Integer {
        self.levels
            .iter()
            .map(|(&level, payload)| lazy_eval_fib(payload) * weight(level))
            .sum()
    }

    pub fn active_levels(&self) -> u32 {
        self.levels.values().filter(|v| !v.is_empty()).count() as u32
    }

    // Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.supportCard
    pub fn support_card(&self) -> u32 {
        self.levels.values().map(|v| v.len() as u32).sum()
    }

    // Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.density
    pub fn density(&self) -> f64 {
        let val = self.eval();
        if val <= 1 {
            return 0.0;
        }
        let bits = val.significant_bits() as f64;
        let ln_n = bits * std::f64::consts::LN_2;
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let log_phi_val = ln_n / golden_ratio.ln();
        if log_phi_val <= 0.0 {
            0.0
        } else {
            self.support_card() as f64 / log_phi_val
        }
    }

    pub fn canonicalize(&mut self) {
        self.levels.retain(|_, payload| !payload.is_empty());
        for payload in self.levels.values_mut() {
            payload.sort_unstable();
        }
    }

    fn level_eval_at(&self, level: u32) -> Integer {
        self.levels
            .get(&level)
            .map(|payload| lazy_eval_fib(payload))
            .unwrap_or_else(|| Integer::from(0))
    }

    fn set_level_from_value(&mut self, level: u32, value: &Integer) {
        if *value <= 0 {
            self.levels.remove(&level);
            return;
        }
        self.levels.insert(level, zeckendorf(value));
    }

    // Stage 1: canonical Zeckendorf encoding inside each active level.
    fn normalize_intra_legacy(&mut self) {
        let keys: Vec<u32> = self.levels.keys().copied().collect();
        for level in keys {
            let val = self.level_eval_at(level);
            self.set_level_from_value(level, &val);
        }
    }

    fn carry_threshold(level: u32) -> Integer {
        if level == 0 {
            weight(1)
        } else {
            weight(level)
        }
    }

    fn normalize_inter_carry_legacy(&mut self) {
        loop {
            let mut changed = false;
            let mut level = 0u32;
            let mut max_level = self.levels.keys().copied().max().unwrap_or(0);

            while level <= max_level + 1 {
                let current = self.level_eval_at(level);
                let threshold = Self::carry_threshold(level);
                if threshold == 0 {
                    level += 1;
                    continue;
                }

                let q = current.clone() / threshold.clone();
                let r = current % threshold;

                if q > 0 {
                    self.set_level_from_value(level, &r);
                    changed = true;
                    let next = self.level_eval_at(level + 1);
                    let next_total = next + q;
                    self.set_level_from_value(level + 1, &next_total);
                    if level + 1 > max_level {
                        max_level = level + 1;
                    }
                }
                level += 1;
            }

            if !changed {
                break;
            }
        }
    }

    pub fn normalize_legacy(&mut self) {
        self.normalize_intra_legacy();
        self.normalize_inter_carry_legacy();
        self.canonicalize();
    }

    pub fn normalize_native(&mut self) {
        let mut flat = FlatHybridNumber::from_legacy(self);
        flat.normalize_native();
        *self = flat.to_legacy();
        self.canonicalize();
    }

    pub fn normalize(&mut self) {
        self.normalize_native();
    }

    pub fn is_zero(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn empty() -> Self {
        Self {
            levels: BTreeMap::new(),
        }
    }
}

pub fn log1000_floor(n: &Integer) -> u32 {
    assert!(*n > 0);
    let mut q = n.clone();
    let mut k = 0u32;
    while q >= 1000 {
        q /= 1000;
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::{
        decompose_weight_coefficients_u64, naf_digits_u64, naf_eval, naf_is_canonical,
        naf_support_card, SignedHybridNumber,
    };
    use crate::HybridNumber;
    use rug::Integer;

    #[test]
    fn naf_examples_match_known_patterns() {
        let seven = naf_digits_u64(7);
        assert_eq!(seven, vec![-1, 0, 0, 1]);
        assert!(naf_is_canonical(&seven));
        assert_eq!(naf_support_card(&seven), 2);
        assert_eq!(naf_eval(&seven), Integer::from(7));

        let fifteen = naf_digits_u64(15);
        assert_eq!(fifteen, vec![-1, 0, 0, 0, 1]);
        assert!(naf_is_canonical(&fifteen));
        assert_eq!(naf_support_card(&fifteen), 2);
        assert_eq!(naf_eval(&fifteen), Integer::from(15));

        let thirty_one = naf_digits_u64(31);
        assert_eq!(thirty_one, vec![-1, 0, 0, 0, 0, 1]);
        assert!(naf_is_canonical(&thirty_one));
        assert_eq!(naf_support_card(&thirty_one), 2);
        assert_eq!(naf_eval(&thirty_one), Integer::from(31));
    }

    #[test]
    fn weight_coefficients_roundtrip_exactly() {
        for value in [1u64, 42, 999, 1_000, 1_641, 12_345, 1_500_001, 999_999_999] {
            let coeffs = decompose_weight_coefficients_u64(value);
            let rebuilt: Integer = coeffs
                .into_iter()
                .map(|(level, coeff)| Integer::from(coeff) * crate::weight::weight(level))
                .sum();
            assert_eq!(rebuilt, Integer::from(value));
        }
    }

    #[test]
    fn signed_hybrid_roundtrips_exactly() {
        for value in 0u64..=2_048 {
            let signed = SignedHybridNumber::from_u64(value);
            assert_eq!(signed.eval(), Integer::from(value));
            assert!(signed.levels.values().all(|digits| naf_is_canonical(digits)));
        }
    }

    #[test]
    fn signed_support_improves_mersenne_like_coefficients() {
        let unsigned = HybridNumber::from_u64(31);
        let signed = SignedHybridNumber::from_u64(31);
        assert!(signed.support_card() < unsigned.support_card());
        assert_eq!(signed.support_card(), 2);
    }

    #[test]
    fn naf_handles_u64_max_without_overflow() {
        let digits = naf_digits_u64(u64::MAX);
        assert!(naf_is_canonical(&digits));
        assert_eq!(naf_eval(&digits), Integer::from(u64::MAX));
    }
}
