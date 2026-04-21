use crate::flat_hybrid::FlatHybridNumber;
use crate::normalization::HybridNumber;
use crate::weight::weight;
use crate::zeckendorf::{lazy_eval_fib, zeckendorf};
use rug::Integer;
use std::collections::BTreeMap;

impl HybridNumber {
    pub fn add_legacy(&self, other: &HybridNumber) -> HybridNumber {
        let mut out = self.add_lazy_legacy(other);
        out.normalize_legacy();
        out
    }

    pub fn add_lazy_legacy(&self, other: &HybridNumber) -> HybridNumber {
        let (base, extra) = if self.levels.len() >= other.levels.len() {
            (&self.levels, &other.levels)
        } else {
            (&other.levels, &self.levels)
        };

        let mut levels = base.clone();
        for (&level, payload) in extra {
            let dst = levels.entry(level).or_default();
            dst.reserve(payload.len());
            dst.extend(payload.iter().copied());
        }
        HybridNumber { levels }
    }

    pub fn add_lazy_flat(&self, other: &HybridNumber) -> FlatHybridNumber {
        let flat_a = FlatHybridNumber::from_legacy(self);
        let flat_b = FlatHybridNumber::from_legacy(other);
        flat_a.add_lazy(&flat_b)
    }

    pub fn add_flat(&self, other: &HybridNumber) -> FlatHybridNumber {
        let mut out = self.add_lazy_flat(other);
        out.normalize_native();
        out
    }

    // Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.add
    pub fn add(&self, other: &HybridNumber) -> HybridNumber {
        self.add_legacy(other)
    }

    // Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.addLazy
    pub fn add_lazy(&self, other: &HybridNumber) -> HybridNumber {
        self.add_lazy_legacy(other)
    }

    // Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.multiplyBinary
    pub fn multiply(&self, other: &HybridNumber) -> HybridNumber {
        self.multiply_legacy(other)
    }

    pub fn multiply_legacy(&self, other: &HybridNumber) -> HybridNumber {
        if self.is_zero() || other.is_zero() {
            return HybridNumber::empty();
        }
        let mut acc = HybridNumber::empty();
        for (&level, payload) in &other.levels {
            if payload.is_empty() {
                continue;
            }
            let coeff = lazy_eval_fib(payload);
            if coeff == 0 {
                continue;
            }
            let factor = coeff * weight(level);
            let term = self.scale_by_integer_legacy(&factor);
            acc = acc.add_legacy(&term);
        }
        acc
    }

    pub fn modulo(&self, modulus: &Integer) -> HybridNumber {
        let v = self.eval() % modulus;
        HybridNumber::from_integer(&v)
    }

    pub fn from_integer(n: &Integer) -> HybridNumber {
        Self::from_integer_legacy(n)
    }

    pub fn from_integer_legacy(n: &Integer) -> HybridNumber {
        if *n <= 0 {
            return HybridNumber::empty();
        }
        let mut levels = BTreeMap::new();
        levels.insert(0, zeckendorf(n));
        let mut out = HybridNumber { levels };
        out.normalize_legacy();
        out
    }

    pub fn to_integer(&self) -> Integer {
        self.eval()
    }

    pub fn from_u64(n: u64) -> HybridNumber {
        HybridNumber::from_integer(&Integer::from(n))
    }

    pub fn modpow(base: &Integer, exp: &Integer, modulus: &Integer) -> Integer {
        if *exp < 0 {
            panic!("negative exponent is not supported");
        }
        let hz_exp = HybridNumber::from_integer(exp);
        Self::modpow_from_hybrid(base, &hz_exp, modulus)
    }

    pub fn modpow_from_hybrid(
        base: &Integer,
        exp_hybrid: &HybridNumber,
        modulus: &Integer,
    ) -> Integer {
        if *modulus == 0 {
            return Integer::from(0);
        }
        if *modulus == 1 {
            return Integer::from(0);
        }
        let mut result = Integer::from(1) % modulus;
        if exp_hybrid.is_zero() {
            return result;
        }

        let max_level = exp_hybrid.levels.keys().copied().max().unwrap_or(0);
        let mut base_at_level = Vec::<Integer>::with_capacity(max_level as usize + 1);
        let base_mod = base.clone() % modulus;
        base_at_level.push(base_mod.clone());
        if max_level >= 1 {
            let level1 = base_mod
                .clone()
                .pow_mod(&Integer::from(1000), modulus)
                .expect("pow_mod failed for weight(1)");
            base_at_level.push(level1);
            for level in 2..=max_level {
                let prev = base_at_level[(level - 1) as usize].clone();
                let exp_weight = weight(level - 1);
                let next = prev
                    .pow_mod(&exp_weight, modulus)
                    .expect("pow_mod failed while precomputing base^weight(level)");
                base_at_level.push(next);
            }
        }

        for (&level, payload) in &exp_hybrid.levels {
            if payload.is_empty() {
                continue;
            }
            let level_coeff = lazy_eval_fib(payload);
            if level_coeff == 0 {
                continue;
            }
            let factor = base_at_level[level as usize]
                .clone()
                .pow_mod(&level_coeff, modulus)
                .expect("pow_mod failed for hybrid coefficient factor");
            result = (result * factor) % modulus;
        }
        result
    }

    #[allow(dead_code)]
    fn scale_by_integer(&self, scalar: &Integer) -> HybridNumber {
        if *scalar <= 0 || self.is_zero() {
            return HybridNumber::empty();
        }
        let mut n = scalar.clone();
        let mut addend = self.clone();
        let mut acc = HybridNumber::empty();

        while n > 0 {
            if n.is_odd() {
                acc = acc.add(&addend);
            }
            n >>= 1;
            if n > 0 {
                addend = addend.add(&addend);
            }
        }
        acc
    }

    fn scale_by_integer_legacy(&self, scalar: &Integer) -> HybridNumber {
        if *scalar <= 0 || self.is_zero() {
            return HybridNumber::empty();
        }
        let mut n = scalar.clone();
        let mut addend = self.clone();
        let mut acc = HybridNumber::empty();

        while n > 0 {
            if n.is_odd() {
                acc = acc.add_legacy(&addend);
            }
            n >>= 1;
            if n > 0 {
                addend = addend.add_legacy(&addend);
            }
        }
        acc
    }
}
