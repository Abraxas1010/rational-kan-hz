// Energy-optimized NTT for base-φ polynomial convolution.
//
// Three key optimizations over the naive implementation:
// 1. u64 arithmetic with const modulus — all primes < 2^30, products fit u64,
//    LLVM optimizes `% CONST` to multiply-shift (no runtime division)
// 2. Shoup precomputed quotients (Harvey 2014) — twiddle multiply becomes
//    UMULH + MUL + conditional subtract, eliminating all modular division
// 3. Precomputed twiddle tables — one allocation per NTT, not per butterfly

const FAST_MOD: u64 = 998244353; // 119 × 2^23 + 1; max NTT length = 2^23

pub const fn safe_coefficient_bound() -> u64 {
    FAST_MOD / 2
}

pub const DENSE_ENERGY_CROSSOVER: u64 = 16384;

// Shoup quotient: floor(w * 2^64 / p)
#[inline(always)]
fn shoup_precompute(w: u64, p: u64) -> u64 {
    (((w as u128) << 64) / p as u128) as u64
}

// Modular multiply using Shoup's precomputed quotient.
// Computes (t * w) mod p without division.
// On AArch64: ((wq as u128 * t as u128) >> 64) compiles to single UMULH.
#[inline(always)]
fn mul_shoup(t: u64, w: u64, wq: u64, p: u64) -> u64 {
    let q = ((wq as u128 * t as u128) >> 64) as u64;
    let r = t.wrapping_mul(w).wrapping_sub(q.wrapping_mul(p));
    if r >= p { r.wrapping_sub(p) } else { r }
}

// Specialized NTT engine per prime via macro.
// Each module uses a compile-time constant prime, so `(a * b) % P` is optimized
// by LLVM to a multiply-shift sequence with zero runtime divisions.
macro_rules! ntt_engine {
    ($name:ident, $prime:expr, $gen:expr) => {
        mod $name {
            pub const P: u64 = $prime;
            const G: u64 = $gen;

            #[inline(always)]
            pub fn mul_mod(a: u64, b: u64) -> u64 {
                (a * b) % P
            }

            pub fn pow_mod(mut base: u64, mut exp: u64) -> u64 {
                let mut result = 1u64;
                base %= P;
                while exp > 0 {
                    if exp & 1 == 1 {
                        result = mul_mod(result, base);
                    }
                    exp >>= 1;
                    base = mul_mod(base, base);
                }
                result
            }

            pub fn ntt(a: &mut [u64], invert: bool) {
                let n = a.len();
                debug_assert!(n.is_power_of_two());
                if n == 1 {
                    return;
                }

                let mut j = 0usize;
                for i in 1..n {
                    let mut bit = n >> 1;
                    while j & bit != 0 {
                        j ^= bit;
                        bit >>= 1;
                    }
                    j ^= bit;
                    if i < j {
                        a.swap(i, j);
                    }
                }

                let max_half = n / 2;
                let mut tw = vec![0u64; max_half];
                let mut twq = vec![0u64; max_half];

                let mut len = 2;
                while len <= n {
                    let w = if invert {
                        pow_mod(G, P - 1 - (P - 1) / len as u64)
                    } else {
                        pow_mod(G, (P - 1) / len as u64)
                    };
                    let half = len / 2;

                    tw[0] = 1;
                    twq[0] = super::shoup_precompute(1, P);
                    for k in 1..half {
                        tw[k] = mul_mod(tw[k - 1], w);
                        twq[k] = super::shoup_precompute(tw[k], P);
                    }

                    for i in (0..n).step_by(len) {
                        for k in 0..half {
                            let u = a[i + k];
                            let v = super::mul_shoup(a[i + k + half], tw[k], twq[k], P);
                            a[i + k] = if u + v >= P { u + v - P } else { u + v };
                            a[i + k + half] = if u >= v { u - v } else { u + P - v };
                        }
                    }
                    len <<= 1;
                }

                if invert {
                    let n_inv = pow_mod(n as u64, P - 2);
                    let n_inv_q = super::shoup_precompute(n_inv, P);
                    for x in a.iter_mut() {
                        *x = super::mul_shoup(*x, n_inv, n_inv_q, P);
                    }
                }
            }

            pub fn convolve(a: &[u64], b: &[u64]) -> Vec<u64> {
                if a.is_empty() || b.is_empty() {
                    return vec![];
                }
                let result_len = a.len() + b.len() - 1;
                let n = result_len.next_power_of_two();

                let mut fa = vec![0u64; n];
                let mut fb = vec![0u64; n];
                fa[..a.len()].copy_from_slice(a);
                fb[..b.len()].copy_from_slice(b);

                ntt(&mut fa, false);
                ntt(&mut fb, false);

                for i in 0..n {
                    fa[i] = mul_mod(fa[i], fb[i]);
                }

                ntt(&mut fa, true);
                fa.truncate(result_len);
                fa
            }
        }
    };
}

ntt_engine!(p0, 998244353, 3);
ntt_engine!(p1, 469762049, 3);
ntt_engine!(p2, 167772161, 3);
ntt_engine!(p3, 985661441, 3);

pub fn convolve_u64(a: &[u64], b: &[u64]) -> Vec<u64> {
    p0::convolve(a, b)
}

pub fn convolve_i64(a: &[i64], b: &[i64]) -> Vec<i64> {
    let ua: Vec<u64> = a
        .iter()
        .map(|&x| {
            if x >= 0 {
                (x as u64) % FAST_MOD
            } else {
                FAST_MOD - ((-x) as u64 % FAST_MOD)
            }
        })
        .collect();
    let ub: Vec<u64> = b
        .iter()
        .map(|&x| {
            if x >= 0 {
                (x as u64) % FAST_MOD
            } else {
                FAST_MOD - ((-x) as u64 % FAST_MOD)
            }
        })
        .collect();

    let raw = convolve_u64(&ua, &ub);
    let half = FAST_MOD / 2;
    raw.into_iter()
        .map(|v| {
            if v > half {
                v as i64 - FAST_MOD as i64
            } else {
                v as i64
            }
        })
        .collect()
}

// --- Multi-prime with CRT in i128, no rug::Integer ---

fn to_residue(x: i64, p: u64) -> u64 {
    x.rem_euclid(p as i64) as u64
}

fn sub_mod_u64(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { p - (b - a) }
}

struct CrtI64 {
    inv: [u64; 3],
    acc: [u128; 3],
    total: u128,
    half: u128,
}

impl CrtI64 {
    fn compute() -> Self {
        let acc0 = p0::P as u128;
        let acc1 = acc0 * p1::P as u128;
        let acc2 = acc1 * p2::P as u128;
        let total = acc2 * p3::P as u128;

        let inv0 = p1::pow_mod(p0::P % p1::P, p1::P - 2);
        let acc1_mod_p2 = (acc1 % p2::P as u128) as u64;
        let inv1 = p2::pow_mod(acc1_mod_p2, p2::P - 2);
        let acc2_mod_p3 = (acc2 % p3::P as u128) as u64;
        let inv2 = p3::pow_mod(acc2_mod_p3, p3::P - 2);

        Self {
            inv: [inv0, inv1, inv2],
            acc: [acc0, acc1, acc2],
            total,
            half: total / 2,
        }
    }

    fn reconstruct_i128(&self, r: [u64; 4]) -> i128 {
        let x1 = r[0] as u128;

        let x1_mod_p1 = (x1 % p1::P as u128) as u64;
        let d2 = sub_mod_u64(r[1], x1_mod_p1, p1::P);
        let k2 = (d2 as u128 * self.inv[0] as u128 % p1::P as u128) as u64;
        let x2 = x1 + k2 as u128 * self.acc[0];

        let x2_mod_p2 = (x2 % p2::P as u128) as u64;
        let d3 = sub_mod_u64(r[2], x2_mod_p2, p2::P);
        let k3 = (d3 as u128 * self.inv[1] as u128 % p2::P as u128) as u64;
        let x3 = x2 + k3 as u128 * self.acc[1];

        let x3_mod_p3 = (x3 % p3::P as u128) as u64;
        let d4 = sub_mod_u64(r[3], x3_mod_p3, p3::P);
        let k4 = (d4 as u128 * self.inv[2] as u128 % p3::P as u128) as u64;
        let x4 = x3 + k4 as u128 * self.acc[2];

        if x4 > self.half {
            -(self.total.wrapping_sub(x4) as i128)
        } else {
            x4 as i128
        }
    }
}

/// Four-prime NTT convolution on i64 slices with i128 output.
///
/// Uses 4 const-specialized NTT engines (no runtime modular division).
/// Correct for |c_k| < M/2 ≈ 3.87 × 10^34.
pub fn convolve_i64_multi(a: &[i64], b: &[i64]) -> Vec<i128> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    let ra0: Vec<u64> = a.iter().map(|&x| to_residue(x, p0::P)).collect();
    let rb0: Vec<u64> = b.iter().map(|&x| to_residue(x, p0::P)).collect();
    let ra1: Vec<u64> = a.iter().map(|&x| to_residue(x, p1::P)).collect();
    let rb1: Vec<u64> = b.iter().map(|&x| to_residue(x, p1::P)).collect();
    let ra2: Vec<u64> = a.iter().map(|&x| to_residue(x, p2::P)).collect();
    let rb2: Vec<u64> = b.iter().map(|&x| to_residue(x, p2::P)).collect();
    let ra3: Vec<u64> = a.iter().map(|&x| to_residue(x, p3::P)).collect();
    let rb3: Vec<u64> = b.iter().map(|&x| to_residue(x, p3::P)).collect();

    let c0 = p0::convolve(&ra0, &rb0);
    let c1 = p1::convolve(&ra1, &rb1);
    let c2 = p2::convolve(&ra2, &rb2);
    let c3 = p3::convolve(&ra3, &rb3);

    let crt = CrtI64::compute();
    let result_len = c0.len();
    (0..result_len)
        .map(|i| crt.reconstruct_i128([c0[i], c1[i], c2[i], c3[i]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntt_roundtrip() {
        let mut a = vec![1u64, 2, 3, 4, 0, 0, 0, 0];
        let original = a.clone();
        p0::ntt(&mut a, false);
        assert_ne!(a, original);
        p0::ntt(&mut a, true);
        assert_eq!(a, original);
    }

    #[test]
    fn ntt_roundtrip_large() {
        let n = 1024;
        let mut a: Vec<u64> = (0..n).map(|i| i as u64 % FAST_MOD).collect();
        let original = a.clone();
        p0::ntt(&mut a, false);
        p0::ntt(&mut a, true);
        assert_eq!(a, original);
    }

    #[test]
    fn convolve_u64_small() {
        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5];
        let result = convolve_u64(&a, &b);
        assert_eq!(result, vec![4, 13, 22, 15]);
    }

    #[test]
    fn convolve_u64_single() {
        assert_eq!(convolve_u64(&[7], &[3]), vec![21]);
    }

    #[test]
    fn convolve_u64_empty() {
        assert!(convolve_u64(&[], &[1]).is_empty());
        assert!(convolve_u64(&[1], &[]).is_empty());
    }

    #[test]
    fn convolve_i64_signed() {
        let a = vec![3i64, -2, 1];
        let b = vec![1i64, 4];
        let result = convolve_i64(&a, &b);
        assert_eq!(result, vec![3, 10, -7, 4]);
    }

    #[test]
    fn convolve_i64_negative_only() {
        let a = vec![-5i64, -3];
        let b = vec![-2i64, 4];
        let result = convolve_i64(&a, &b);
        assert_eq!(result, vec![10, -14, -12]);
    }

    #[test]
    fn convolve_i64_matches_naive() {
        let a: Vec<i64> = (1..=8).map(|i| i * 3 - 7).collect();
        let b: Vec<i64> = (1..=4).map(|i| -(i * 2 - 5)).collect();

        let result = convolve_i64(&a, &b);

        let mut naive = vec![0i64; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                naive[i + j] += ai * bj;
            }
        }
        assert_eq!(result, naive);
    }

    #[test]
    fn convolve_i64_stress() {
        let n = 200;
        let a: Vec<i64> = (0..n).map(|i| (i as i64) * 37 - 3700).collect();
        let b: Vec<i64> = (0..n).map(|i| -(i as i64) * 13 + 1300).collect();

        let result = convolve_i64(&a, &b);

        let mut naive = vec![0i128; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                naive[i + j] += ai as i128 * bj as i128;
            }
        }

        for (k, (&r, &n)) in result.iter().zip(naive.iter()).enumerate() {
            assert_eq!(
                r as i128, n,
                "mismatch at k={k}: got {r}, expected {n}"
            );
        }
    }

    #[test]
    fn convolve_binary_sequences() {
        let a = vec![1i64, 0, 1, 0, 0, 1, 0, 1, 0, 1];
        let b = vec![0i64, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1];

        let result = convolve_i64(&a, &b);

        let mut naive = vec![0i64; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                naive[i + j] += ai * bj;
            }
        }
        assert_eq!(result, naive);
        assert!(result.iter().all(|&c| c.unsigned_abs() <= 10));
    }

    #[test]
    fn convolve_large_binary_sequences() {
        let n = 1000;
        let a: Vec<i64> = (0..n).map(|i| if i % 3 != 0 { 1 } else { 0 }).collect();
        let b: Vec<i64> = (0..n).map(|i| if i % 5 != 0 { 1 } else { 0 }).collect();

        let result = convolve_i64(&a, &b);

        assert_eq!(result[0], a[0] * b[0]);
        assert_eq!(result.len(), 2 * n - 1);
        assert!(result.iter().all(|&c| c.unsigned_abs() < safe_coefficient_bound()));
    }

    #[test]
    fn convolve_i64_multi_matches_naive() {
        let a: Vec<i64> = (1..=8).map(|i| i * 3 - 7).collect();
        let b: Vec<i64> = (1..=4).map(|i| -(i * 2 - 5)).collect();
        let result = convolve_i64_multi(&a, &b);
        let mut naive = vec![0i128; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                naive[i + j] += ai as i128 * bj as i128;
            }
        }
        assert_eq!(result, naive);
    }

    #[test]
    fn convolve_i64_multi_large_coefficients() {
        let big: i64 = 1_000_000_000_000;
        let a = vec![big, 1, 0, big];
        let b = vec![big, -1, 0, -big];
        let result = convolve_i64_multi(&a, &b);
        let mut naive = vec![0i128; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                naive[i + j] += ai as i128 * bj as i128;
            }
        }
        assert_eq!(result, naive);
        assert_eq!(result[0], 1_000_000_000_000_000_000_000_000i128);
    }

    #[test]
    fn convolve_i64_multi_stress() {
        let n = 200;
        let a: Vec<i64> = (0..n).map(|i| (i as i64) * 1_000_000 - 100_000_000).collect();
        let b: Vec<i64> = (0..n).map(|i| -(i as i64) * 500_000 + 50_000_000).collect();
        let result = convolve_i64_multi(&a, &b);
        let mut naive = vec![0i128; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                naive[i + j] += ai as i128 * bj as i128;
            }
        }
        assert_eq!(result, naive);
    }

    #[test]
    fn pow_mod_basic() {
        assert_eq!(p0::pow_mod(3, 0), 1);
        assert_eq!(p0::pow_mod(3, 1), 3);
        assert_eq!(p0::pow_mod(2, 10), 1024);
        assert_eq!(p0::pow_mod(2, 23), 1 << 23);
    }

    #[test]
    fn primitive_root_order() {
        let half_order = p0::pow_mod(3, (FAST_MOD - 1) / 2);
        assert_ne!(half_order, 1, "g=3 is not a primitive root of {FAST_MOD}");
        assert_eq!(p0::pow_mod(3, FAST_MOD - 1), 1);
    }

    #[test]
    fn max_ntt_length_is_power_of_two_23() {
        let p_minus_1 = FAST_MOD - 1;
        assert_eq!(p_minus_1 % (1 << 23), 0);
        assert_ne!(p_minus_1 % (1 << 24), 0);
    }

    #[test]
    fn shoup_mul_matches_naive() {
        for &(a, b) in &[(1u64, 1), (100, 200), (FAST_MOD - 1, FAST_MOD - 1), (0, 12345)] {
            let expected = ((a as u128 * b as u128) % FAST_MOD as u128) as u64;
            let bq = shoup_precompute(b, FAST_MOD);
            let result = mul_shoup(a, b, bq, FAST_MOD);
            assert_eq!(result, expected, "shoup({a}, {b}) = {result}, expected {expected}");
        }
    }

    #[test]
    fn all_four_primes_ntt_roundtrip() {
        let n = 64;
        let data: Vec<u64> = (0..n).map(|i| i as u64 * 7 + 3).collect();

        let mut a0 = data.iter().map(|&x| x % p0::P).collect::<Vec<_>>();
        let orig0 = a0.clone();
        p0::ntt(&mut a0, false);
        p0::ntt(&mut a0, true);
        assert_eq!(a0, orig0);

        let mut a1 = data.iter().map(|&x| x % p1::P).collect::<Vec<_>>();
        let orig1 = a1.clone();
        p1::ntt(&mut a1, false);
        p1::ntt(&mut a1, true);
        assert_eq!(a1, orig1);

        let mut a2 = data.iter().map(|&x| x % p2::P).collect::<Vec<_>>();
        let orig2 = a2.clone();
        p2::ntt(&mut a2, false);
        p2::ntt(&mut a2, true);
        assert_eq!(a2, orig2);

        let mut a3 = data.iter().map(|&x| x % p3::P).collect::<Vec<_>>();
        let orig3 = a3.clone();
        p3::ntt(&mut a3, false);
        p3::ntt(&mut a3, true);
        assert_eq!(a3, orig3);
    }
}
