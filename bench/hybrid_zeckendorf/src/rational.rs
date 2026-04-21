use crate::{FlatHybridNumber, HybridNumber};
use rug::{Integer, Rational};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct HZRational {
    pub sign: i8,
    pub num: FlatHybridNumber,
    pub den: FlatHybridNumber,
}

impl HZRational {
    pub fn zero() -> Self {
        Self {
            sign: 0,
            num: FlatHybridNumber::empty(),
            den: FlatHybridNumber::from_legacy(&HybridNumber::from_u64(1)),
        }
    }

    pub fn one() -> Self {
        Self {
            sign: 1,
            num: FlatHybridNumber::from_legacy(&HybridNumber::from_u64(1)),
            den: FlatHybridNumber::from_legacy(&HybridNumber::from_u64(1)),
        }
    }

    pub fn from_integer_pair(num: Integer, den: Integer) -> Self {
        assert!(den > 0, "HZRational denominator must be positive");
        if num == 0 {
            return Self::zero();
        }
        let sign = if num < 0 { -1 } else { 1 };
        let mut n = num.abs();
        let mut d = den;
        let gcd = n.clone().gcd(&d);
        n /= gcd.clone();
        d /= gcd;
        Self {
            sign,
            num: FlatHybridNumber::from_legacy(&HybridNumber::from_integer(&n)),
            den: FlatHybridNumber::from_legacy(&HybridNumber::from_integer(&d)),
        }
    }

    pub fn from_rug(value: &Rational) -> Self {
        let (num, den) = value.clone().into_numer_denom();
        Self::from_integer_pair(num, den)
    }

    pub fn to_rug(&self) -> Rational {
        if self.sign == 0 {
            return Rational::from((Integer::from(0), Integer::from(1)));
        }
        let mut num = self.num.eval();
        if self.sign < 0 {
            num = -num;
        }
        Rational::from((num, self.den.eval()))
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::from_rug(&(self.to_rug() + other.to_rug()))
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::from_rug(&(self.to_rug() - other.to_rug()))
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self::from_rug(&(self.to_rug() * other.to_rug()))
    }

    pub fn div(&self, other: &Self) -> Self {
        assert!(other.sign != 0, "division by zero HZRational");
        Self::from_rug(&(self.to_rug() / other.to_rug()))
    }

    pub fn neg(&self) -> Self {
        let mut out = self.clone();
        out.sign = -out.sign;
        out
    }

    pub fn canonical_hash(&self) -> u64 {
        let value = self.to_rug();
        let mut hasher = DefaultHasher::new();
        value.to_string().hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::HZRational;
    use rug::{Integer, Rational};

    #[test]
    fn canonical_representations_match() {
        let a = HZRational::from_integer_pair(Integer::from(2), Integer::from(4));
        let b = HZRational::from_integer_pair(Integer::from(1), Integer::from(2));
        assert_eq!(a.to_rug(), b.to_rug());
        assert_eq!(a.canonical_hash(), b.canonical_hash());
    }

    #[test]
    fn arithmetic_matches_rug() {
        let a = HZRational::from_rug(&Rational::from((Integer::from(1), Integer::from(3))));
        let b = HZRational::from_rug(&Rational::from((Integer::from(5), Integer::from(7))));
        assert_eq!(a.add(&b).to_rug(), Rational::from((Integer::from(22), Integer::from(21))));
        assert_eq!(a.mul(&b).to_rug(), Rational::from((Integer::from(5), Integer::from(21))));
    }
}
