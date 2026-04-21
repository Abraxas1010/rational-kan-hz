use crate::base_phi::{
    add_digits, hybrid_to_base_phi, integer_like_value, raw_mul, support_density, BasePhiDigits,
};
use crate::reference::gmp_wrapper::modpow_gmp;
use crate::{FlatHybridNumber, HybridNumber};
use rug::Integer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionRoute {
    NativeAdd,
    BasePhiMultiplyBridge,
    GmpMultiplyFallback,
    GmpModPowFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionDecision {
    pub route: ProductionRoute,
    pub reason: &'static str,
}

#[derive(Debug, Clone)]
pub enum ProductionNumber {
    Hybrid {
        value: HybridNumber,
        integer: Integer,
    },
    BasePhi {
        digits: BasePhiDigits,
        integer: Integer,
    },
    Integer(Integer),
}

pub fn add_decision() -> ProductionDecision {
    ProductionDecision {
        route: ProductionRoute::NativeAdd,
        reason:
            "exp4/exp5 show the native flat-normalization add path dominates the legacy add path",
    }
}

pub fn multiply_decision() -> ProductionDecision {
    ProductionDecision {
        route: ProductionRoute::GmpMultiplyFallback,
        reason:
            "the current theorem-mirrored multiply path is repeated-add / normalization-heavy and fails even tiny bounded production benchmarks",
    }
}

pub fn multiply_decision_for(lhs: &HybridNumber, rhs: &HybridNumber) -> ProductionDecision {
    let lhs_digits = hybrid_to_base_phi(lhs);
    let rhs_digits = hybrid_to_base_phi(rhs);
    let avg_density = (support_density(&lhs_digits) + support_density(&rhs_digits)) * 0.5;
    let avg_support = (lhs_digits.len() + rhs_digits.len()) / 2;
    let avg_levels = (lhs.active_levels() + rhs.active_levels()) / 2;

    if avg_density <= 0.35 && avg_support <= 512 && avg_levels <= 4 {
        ProductionDecision {
            route: ProductionRoute::BasePhiMultiplyBridge,
            reason:
                "sparse lifted base-phi digits stay below the measured raw-convolution crossover and can defer HZ re-embedding",
        }
    } else {
        ProductionDecision {
            route: ProductionRoute::GmpMultiplyFallback,
            reason:
                "lifted base-phi digits are too dense or broad for the current exact bridge to beat the integer fallback",
        }
    }
}

pub fn modpow_decision() -> ProductionDecision {
    ProductionDecision {
        route: ProductionRoute::GmpModPowFallback,
        reason:
            "exp1/exp6 show no operational win for the current HZ modpow path; production should route directly to GMP",
    }
}

impl HybridNumber {
    pub fn add_production(&self, other: &HybridNumber) -> HybridNumber {
        let mut out =
            FlatHybridNumber::from_legacy(self).add_lazy(&FlatHybridNumber::from_legacy(other));
        out.normalize_native();
        out.to_legacy()
    }

    pub fn multiply_production(&self, other: &HybridNumber) -> HybridNumber {
        self.multiply_production_deferred(other).normalize_to_hybrid()
    }

    pub fn multiply_production_deferred(&self, other: &HybridNumber) -> ProductionNumber {
        match multiply_decision_for(self, other).route {
            ProductionRoute::BasePhiMultiplyBridge => {
                let lhs_digits = hybrid_to_base_phi(self);
                let rhs_digits = hybrid_to_base_phi(other);
                let digits = raw_mul(&lhs_digits, &rhs_digits);
                let integer =
                    integer_like_value(&digits).expect("hybrid->base-phi bridge left integer lane");
                ProductionNumber::BasePhi { digits, integer }
            }
            ProductionRoute::NativeAdd
            | ProductionRoute::GmpMultiplyFallback
            | ProductionRoute::GmpModPowFallback => {
                ProductionNumber::Integer(self.eval() * other.eval())
            }
        }
    }

    pub fn modpow_production(base: &Integer, exp: &Integer, modulus: &Integer) -> Integer {
        modpow_gmp(base, exp, modulus)
    }
}

impl ProductionNumber {
    pub fn from_hybrid(value: HybridNumber) -> Self {
        let integer = value.eval();
        Self::Hybrid { value, integer }
    }

    pub fn from_integer(value: Integer) -> Self {
        Self::Integer(value)
    }

    pub fn to_integer(&self) -> Integer {
        match self {
            Self::Hybrid { integer, .. } => integer.clone(),
            Self::BasePhi { integer, .. } => integer.clone(),
            Self::Integer(value) => value.clone(),
        }
    }

    pub fn normalize_to_hybrid(&self) -> HybridNumber {
        match self {
            Self::Hybrid { value, .. } => value.clone(),
            Self::BasePhi { integer, .. } => HybridNumber::from_integer(integer),
            Self::Integer(value) => HybridNumber::from_integer(value),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (
                Self::Hybrid {
                    value: a,
                    integer: ai,
                },
                Self::Hybrid {
                    value: b,
                    integer: bi,
                },
            ) => Self::Hybrid {
                value: a.add_production(b),
                integer: (ai + bi).into(),
            },
            (
                Self::BasePhi {
                    digits: a,
                    integer: ai,
                },
                Self::BasePhi {
                    digits: b,
                    integer: bi,
                },
            ) => Self::BasePhi {
                digits: add_digits(a, b),
                integer: (ai + bi).into(),
            },
            _ => Self::Integer(self.to_integer() + other.to_integer()),
        }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        match (self, other) {
            (
                Self::Hybrid {
                    value: a, ..
                },
                Self::Hybrid {
                    value: b, ..
                },
            ) => a.multiply_production_deferred(b),
            (
                Self::BasePhi { digits: a, .. },
                Self::BasePhi { digits: b, .. },
            ) => {
                let digits = raw_mul(a, b);
                let integer =
                    integer_like_value(&digits).expect("base-phi multiply left integer lane");
                Self::BasePhi { digits, integer }
            }
            (
                Self::BasePhi { digits: a, .. },
                Self::Hybrid { value: b, .. },
            )
            | (
                Self::Hybrid { value: b, .. },
                Self::BasePhi { digits: a, .. },
            ) => {
                let digits = raw_mul(a, &hybrid_to_base_phi(b));
                let integer =
                    integer_like_value(&digits).expect("mixed base-phi multiply left integer lane");
                Self::BasePhi { digits, integer }
            }
            _ => Self::Integer(self.to_integer() * other.to_integer()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_add_preserves_semantics() {
        let a = HybridNumber::from_u64(12_345);
        let b = HybridNumber::from_u64(67_890);
        assert_eq!(
            a.add_production(&b).eval(),
            Integer::from(12_345u64 + 67_890u64)
        );
        assert_eq!(add_decision().route, ProductionRoute::NativeAdd);
    }

    #[test]
    fn production_multiply_preserves_semantics() {
        let a = HybridNumber::from_u64(12_345);
        let b = HybridNumber::from_u64(67_890);
        let deferred = a.multiply_production_deferred(&b);
        assert_eq!(deferred.to_integer(), Integer::from(12_345u64 * 67_890u64));
        assert_eq!(
            a.multiply_production(&b).eval(),
            Integer::from(12_345u64 * 67_890u64)
        );
    }

    #[test]
    fn production_modpow_matches_gmp() {
        let base = Integer::from(123_456_789u64);
        let exp = Integer::from(65_537u64);
        let modulus = Integer::from(1_000_003u64);
        let got = HybridNumber::modpow_production(&base, &exp, &modulus);
        let want = modpow_gmp(&base, &exp, &modulus);
        assert_eq!(got, want);
        assert_eq!(modpow_decision().route, ProductionRoute::GmpModPowFallback);
    }

    #[test]
    fn production_number_multiply_stays_in_integer_carrier() {
        let a = ProductionNumber::from_hybrid(HybridNumber::from_u64(12_345));
        let b = ProductionNumber::from_hybrid(HybridNumber::from_u64(67_890));
        let prod = a.multiply(&b);
        match prod {
            ProductionNumber::BasePhi { integer: value, .. }
            | ProductionNumber::Integer(value) => {
                assert_eq!(value, Integer::from(12_345u64 * 67_890u64))
            }
            ProductionNumber::Hybrid { .. } => panic!("expected integer result"),
        }
    }

    #[test]
    fn production_base_phi_bridge_route_is_available_for_sparse_inputs() {
        let a = HybridNumber::from_u64(144);
        let b = HybridNumber::from_u64(233);
        let decision = multiply_decision_for(&a, &b);
        assert_eq!(decision.route, ProductionRoute::BasePhiMultiplyBridge);
        match a.multiply_production_deferred(&b) {
            ProductionNumber::BasePhi { integer, .. } => {
                assert_eq!(integer, Integer::from(144u64 * 233u64))
            }
            other => panic!("expected base-phi deferred carrier, got {other:?}"),
        }
    }

    #[test]
    fn production_number_add_uses_hybrid_when_both_inputs_are_hybrid() {
        let a = ProductionNumber::from_hybrid(HybridNumber::from_u64(12));
        let b = ProductionNumber::from_hybrid(HybridNumber::from_u64(34));
        let sum = a.add(&b);
        match sum {
            ProductionNumber::Hybrid { value, integer } => {
                assert_eq!(value.eval(), Integer::from(46));
                assert_eq!(integer, Integer::from(46));
            }
            ProductionNumber::BasePhi { .. } => panic!("expected hybrid result"),
            ProductionNumber::Integer(_) => panic!("expected hybrid result"),
        }
    }
}
