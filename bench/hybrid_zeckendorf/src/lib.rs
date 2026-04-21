pub mod active_set;
pub mod arithmetic;
pub mod base_phi;
pub mod base_phi_bench;
pub mod bench_config;
pub mod density;
pub mod fib_table;
pub mod flat_hybrid;
pub mod normalization;
pub mod ntt;
pub mod ntt_fast;
pub mod production;
pub mod rational;
pub mod reference;
pub mod sparse;
pub mod weight;
pub mod zeckendorf;
pub mod zeckendorf_native;

pub use flat_hybrid::FlatHybridNumber;
pub use normalization::HybridNumber;
pub use rational::HZRational;

#[cfg(test)]
mod tests {
    use crate::sparse::{construct_sparse_hz, validate_sparse_hz};
    use crate::weight::weight;
    use crate::zeckendorf::lazy_eval_fib;
    use crate::HybridNumber;
    use rand::thread_rng;
    use rug::ops::Pow;
    use rug::Integer;
    use std::collections::BTreeMap;

    #[test]
    fn weight_system_sanity() {
        assert_eq!(weight(0), Integer::from(1));
        assert_eq!(weight(1), Integer::from(1000));
        assert_eq!(weight(2), Integer::from(1_000_000));
    }

    #[test]
    fn roundtrip_from_integer_eval() {
        let cases = [0u64, 1, 42, 999, 1000, 123_456_789];
        for n in cases {
            let hz = HybridNumber::from_u64(n);
            assert_eq!(hz.eval(), Integer::from(n));
        }
    }

    #[test]
    fn from_integer_activates_higher_levels() {
        let n = Integer::from(1_500_000u64);
        let hz = HybridNumber::from_integer(&n);
        assert!(hz.active_levels() > 1);
        assert_eq!(hz.eval(), n);
    }

    #[test]
    fn add_and_multiply_match_integer_semantics() {
        let a = HybridNumber::from_u64(12345);
        let b = HybridNumber::from_u64(67890);
        let sum = a.add(&b);
        let prod = a.multiply(&b);
        assert_eq!(sum.eval(), Integer::from(12345u64 + 67890u64));
        assert_eq!(prod.eval(), Integer::from(12345u64 * 67890u64));
    }

    #[test]
    fn small_domain_semantics_regression() {
        for x in 0u64..50 {
            for y in 0u64..50 {
                let hx = HybridNumber::from_u64(x);
                let hy = HybridNumber::from_u64(y);
                assert_eq!(hx.add(&hy).eval(), Integer::from(x + y));
                assert_eq!(hx.multiply(&hy).eval(), Integer::from(x * y));
            }
        }
    }

    #[test]
    fn modpow_matches_gmp_reference_on_small_cases() {
        for base in 2u64..15 {
            for exp in 0u64..20 {
                let modulus = Integer::from(97u64);
                let b = Integer::from(base);
                let e = Integer::from(exp);
                let hz = HybridNumber::modpow(&b, &e, &modulus);
                let gmp = b.clone().pow_mod(&e, &modulus).expect("pow_mod");
                assert_eq!(hz, gmp, "base={base}, exp={exp}");
            }
        }
    }

    #[test]
    fn normalize_inter_carry_cascades_through_multiple_levels() {
        let n = weight(0) + weight(1) + weight(2) + weight(3) + weight(4);
        let hz = HybridNumber::from_integer(&n);
        assert_eq!(hz.eval(), n);
        assert!(hz.active_levels() >= 4);
        let max_level = hz.levels.keys().copied().max().unwrap_or(0);
        assert!(max_level >= 3);
    }

    #[test]
    fn multiply_multi_level_inputs_matches_integer_semantics() {
        let a_val = Integer::from(10).pow(18u32) + Integer::from(123_456_789u64);
        let b_val = Integer::from(10).pow(15u32) + Integer::from(987_654_321u64);
        let a = HybridNumber::from_integer(&a_val);
        let b = HybridNumber::from_integer(&b_val);
        assert!(a.active_levels() > 1);
        assert!(b.active_levels() > 1);
        let prod = a.multiply(&b);
        assert_eq!(prod.eval(), a_val * b_val);
    }

    #[test]
    fn modpow_large_exponent_exercises_multi_level_path() {
        let base = Integer::from(123_456_789u64);
        let exp = Integer::from(1_234_567u64);
        let modulus = Integer::from(1_000_003u64);
        let hz_exp = HybridNumber::from_integer(&exp);
        assert!(hz_exp.active_levels() > 1);
        let hz = HybridNumber::modpow(&base, &exp, &modulus);
        let gmp = base
            .clone()
            .pow_mod(&exp, &modulus)
            .expect("pow_mod with large exponent");
        assert_eq!(hz, gmp);
    }

    #[test]
    fn normalize_from_prepopulated_levels_preserves_eval() {
        let mut levels = BTreeMap::new();
        levels.insert(0u32, vec![2, 2, 3, 5, 8, 8, 13]);
        levels.insert(2u32, vec![4, 7, 7]);
        let before = HybridNumber {
            levels: levels.clone(),
        };
        let mut after = HybridNumber { levels };
        let eval_before = before.eval();
        after.normalize();
        assert_eq!(after.eval(), eval_before);
    }

    #[test]
    fn construct_sparse_produces_controlled_density() {
        let mut rng = thread_rng();
        for target_rho in [1e-5, 1e-4, 1e-3, 1e-2] {
            let hz = construct_sparse_hz(1_000_000, target_rho, &mut rng);
            let measured = hz.density();
            assert!(
                measured < 2.0 * target_rho,
                "target_rho={target_rho}, measured={measured}"
            );
            assert!(measured > 0.0);
            assert!(hz.eval() > 0);
            validate_sparse_hz(1_000_000, target_rho, &hz).expect("sparse validation");
        }
    }

    #[test]
    fn add_lazy_preserves_eval() {
        let a = HybridNumber::from_u64(12_345);
        let b = HybridNumber::from_u64(67_890);
        let lazy = a.add_lazy(&b);
        assert_eq!(lazy.eval(), Integer::from(12_345u64 + 67_890u64));
        let mut normalized = lazy.clone();
        normalized.normalize();
        assert_eq!(normalized.eval(), Integer::from(12_345u64 + 67_890u64));
    }

    #[test]
    fn sparse_hz_levels_satisfy_carry_invariant() {
        let mut rng = thread_rng();
        let hz = construct_sparse_hz(100_000, 1e-3, &mut rng);
        for (&level, payload) in &hz.levels {
            let coeff = lazy_eval_fib(payload);
            let divisor = if level == 0 { weight(1) } else { weight(level) };
            assert!(
                coeff < divisor,
                "level {level}: coeff={coeff} >= divisor={divisor}"
            );
        }
    }
}
