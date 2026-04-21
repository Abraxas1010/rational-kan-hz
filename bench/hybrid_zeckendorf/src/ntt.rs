use rug::Integer;

const MOD: [u64; 4] = [998244353, 469762049, 167772161, 985661441];
const G: u64 = 3;

pub fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result as u128 * base as u128 % modulus as u128) as u64;
        }
        exp >>= 1;
        base = (base as u128 * base as u128 % modulus as u128) as u64;
    }
    result
}

fn ntt_transform(a: &mut [u64], invert: bool, modulus: u64) {
    let n = a.len();
    assert!(n.is_power_of_two());
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

    let mut len = 2;
    while len <= n {
        let w = if invert {
            pow_mod(G, modulus - 1 - (modulus - 1) / len as u64, modulus)
        } else {
            pow_mod(G, (modulus - 1) / len as u64, modulus)
        };
        let half = len / 2;
        for i in (0..n).step_by(len) {
            let mut wn = 1u64;
            for k in 0..half {
                let u = a[i + k];
                let v = (a[i + k + half] as u128 * wn as u128 % modulus as u128) as u64;
                a[i + k] = if u + v >= modulus {
                    u + v - modulus
                } else {
                    u + v
                };
                a[i + k + half] = if u >= v { u - v } else { u + modulus - v };
                wn = (wn as u128 * w as u128 % modulus as u128) as u64;
            }
        }
        len <<= 1;
    }

    if invert {
        let n_inv = pow_mod(n as u64, modulus - 2, modulus);
        for x in a.iter_mut() {
            *x = (*x as u128 * n_inv as u128 % modulus as u128) as u64;
        }
    }
}

fn convolve_mod(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();

    let mut fa = vec![0u64; n];
    let mut fb = vec![0u64; n];
    fa[..a.len()].copy_from_slice(a);
    fb[..b.len()].copy_from_slice(b);

    ntt_transform(&mut fa, false, modulus);
    ntt_transform(&mut fb, false, modulus);

    for i in 0..n {
        fa[i] = (fa[i] as u128 * fb[i] as u128 % modulus as u128) as u64;
    }

    ntt_transform(&mut fa, true, modulus);
    fa.truncate(result_len);
    fa
}

fn to_mod(val: &Integer, p: u32) -> u64 {
    val.mod_u(p) as u64
}

fn sub_mod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        p - (b - a)
    }
}

struct CrtConstants {
    inv: [u64; 3],
    acc: [u128; 3],
    total: u128,
    half: u128,
}

impl CrtConstants {
    fn compute() -> Self {
        let acc0 = MOD[0] as u128;
        let acc1 = acc0 * MOD[1] as u128;
        let acc2 = acc1 * MOD[2] as u128;
        let total = acc2 * MOD[3] as u128;

        let inv0 = pow_mod(MOD[0] % MOD[1], MOD[1] - 2, MOD[1]);
        let acc1_mod_p2 = (acc1 % MOD[2] as u128) as u64;
        let inv1 = pow_mod(acc1_mod_p2, MOD[2] - 2, MOD[2]);
        let acc2_mod_p3 = (acc2 % MOD[3] as u128) as u64;
        let inv2 = pow_mod(acc2_mod_p3, MOD[3] - 2, MOD[3]);

        Self {
            inv: [inv0, inv1, inv2],
            acc: [acc0, acc1, acc2],
            total,
            half: total / 2,
        }
    }

    fn reconstruct(&self, r: [u64; 4]) -> Integer {
        // Garner step 1
        let x1 = r[0] as u128;

        // Garner step 2
        let x1_mod_p1 = (x1 % MOD[1] as u128) as u64;
        let d2 = sub_mod(r[1], x1_mod_p1, MOD[1]);
        let k2 = (d2 as u128 * self.inv[0] as u128 % MOD[1] as u128) as u64;
        let x2 = x1 + k2 as u128 * self.acc[0];

        // Garner step 3
        let x2_mod_p2 = (x2 % MOD[2] as u128) as u64;
        let d3 = sub_mod(r[2], x2_mod_p2, MOD[2]);
        let k3 = (d3 as u128 * self.inv[1] as u128 % MOD[2] as u128) as u64;
        let x3 = x2 + k3 as u128 * self.acc[1];

        // Garner step 4
        let x3_mod_p3 = (x3 % MOD[3] as u128) as u64;
        let d4 = sub_mod(r[3], x3_mod_p3, MOD[3]);
        let k4 = (d4 as u128 * self.inv[2] as u128 % MOD[3] as u128) as u64;
        let x4 = x3 + k4 as u128 * self.acc[2];

        if x4 > self.half {
            -Integer::from(self.total - x4)
        } else {
            Integer::from(x4)
        }
    }
}

/// FFT-based convolution of arbitrary-precision integer sequences.
///
/// Four-prime NTT (998244353, 469762049, 167772161, 985661441) with Garner CRT.
/// Correct for |c_k| < M/2 ≈ 3.87 × 10^34 (~115 bits).
pub fn fft_convolve(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    let primes: [u32; 4] = [MOD[0] as u32, MOD[1] as u32, MOD[2] as u32, MOD[3] as u32];

    let residues_a: Vec<Vec<u64>> = primes
        .iter()
        .map(|&p| a.iter().map(|x| to_mod(x, p)).collect())
        .collect();
    let residues_b: Vec<Vec<u64>> = primes
        .iter()
        .map(|&p| b.iter().map(|x| to_mod(x, p)).collect())
        .collect();

    let convolved: Vec<Vec<u64>> = (0..4)
        .map(|i| convolve_mod(&residues_a[i], &residues_b[i], MOD[i]))
        .collect();

    let crt = CrtConstants::compute();
    let result_len = convolved[0].len();
    (0..result_len)
        .map(|i| crt.reconstruct([convolved[0][i], convolved[1][i], convolved[2][i], convolved[3][i]]))
        .collect()
}

/// Maximum safe absolute value for convolution output coefficients.
pub fn crt_half_bound() -> Integer {
    let m = Integer::from(MOD[0])
        * Integer::from(MOD[1])
        * Integer::from(MOD[2])
        * Integer::from(MOD[3]);
    m / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::ops::Pow;

    #[test]
    fn ntt_roundtrip() {
        for &modulus in &MOD {
            let mut a = vec![1, 2, 3, 4, 0, 0, 0, 0];
            let original = a.clone();
            ntt_transform(&mut a, false, modulus);
            assert_ne!(a, original);
            ntt_transform(&mut a, true, modulus);
            assert_eq!(a, original, "roundtrip failed for modulus {modulus}");
        }
    }

    #[test]
    fn ntt_roundtrip_large() {
        let n = 1024;
        let mut a: Vec<u64> = (0..n).map(|i| i as u64 % MOD[0]).collect();
        let original = a.clone();
        ntt_transform(&mut a, false, MOD[0]);
        ntt_transform(&mut a, true, MOD[0]);
        assert_eq!(a, original);
    }

    #[test]
    fn convolve_small() {
        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5];
        let result = convolve_mod(&a, &b, MOD[0]);
        assert_eq!(result, vec![4, 13, 22, 15]);
    }

    #[test]
    fn convolve_single_element() {
        let a = vec![7u64];
        let b = vec![3u64];
        assert_eq!(convolve_mod(&a, &b, MOD[0]), vec![21]);
    }

    #[test]
    fn crt_positive() {
        let crt = CrtConstants::compute();
        for x in [0u64, 1, 42, 999_999, 1_000_000_000] {
            let r: [u64; 4] = std::array::from_fn(|i| x % MOD[i]);
            assert_eq!(crt.reconstruct(r), Integer::from(x), "CRT failed for {x}");
        }
    }

    #[test]
    fn crt_negative() {
        let crt = CrtConstants::compute();
        for x in [-1i64, -17, -999_999, -1_000_000_000] {
            let val = Integer::from(x);
            let r: [u64; 4] = std::array::from_fn(|i| to_mod(&val, MOD[i] as u32));
            assert_eq!(crt.reconstruct(r), val, "CRT failed for {x}");
        }
    }

    #[test]
    fn crt_large() {
        let crt = CrtConstants::compute();
        for x in [
            1_000_000_000_000i64,
            -1_000_000_000_000,
            999_999_999_999_999,
            -999_999_999_999_999,
        ] {
            let val = Integer::from(x);
            let r: [u64; 4] = std::array::from_fn(|i| to_mod(&val, MOD[i] as u32));
            assert_eq!(crt.reconstruct(r), val, "CRT failed for {x}");
        }
    }

    #[test]
    fn crt_very_large() {
        let crt = CrtConstants::compute();
        let x = Integer::from(10u32).pow(25);
        let r: [u64; 4] = std::array::from_fn(|i| to_mod(&x, MOD[i] as u32));
        assert_eq!(crt.reconstruct(r), x);

        let neg = -Integer::from(10u32).pow(25);
        let r2: [u64; 4] = std::array::from_fn(|i| to_mod(&neg, MOD[i] as u32));
        assert_eq!(crt.reconstruct(r2), neg);
    }

    #[test]
    fn fft_convolve_matches_naive_signed() {
        let a = vec![Integer::from(3), Integer::from(-2), Integer::from(1)];
        let b = vec![Integer::from(1), Integer::from(4)];
        let result = fft_convolve(&a, &b);
        assert_eq!(
            result,
            vec![
                Integer::from(3),
                Integer::from(10),
                Integer::from(-7),
                Integer::from(4),
            ]
        );
    }

    #[test]
    fn fft_convolve_matches_naive_larger() {
        let a: Vec<Integer> = (1..=8).map(Integer::from).collect();
        let b: Vec<Integer> = (1..=4).map(Integer::from).collect();
        let result = fft_convolve(&a, &b);

        let mut naive = vec![Integer::from(0); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                naive[i + j] += Integer::from(ai * bj);
            }
        }
        assert_eq!(result, naive);
    }

    #[test]
    fn fft_convolve_stress() {
        let n = 200;
        let a: Vec<Integer> = (1..=n).map(|i| Integer::from(i * 3 - 7)).collect();
        let b: Vec<Integer> = (1..=n).map(|i| Integer::from(-(i as i64) * 2 + 5)).collect();
        let result = fft_convolve(&a, &b);

        let mut naive = vec![Integer::from(0); a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                naive[i + j] += Integer::from(ai * bj);
            }
        }
        assert_eq!(result, naive);
    }

    #[test]
    fn fft_convolve_large_coefficients() {
        let big = Integer::from(10u32).pow(12);
        let a = vec![big.clone(), Integer::from(1)];
        let b = vec![big.clone(), Integer::from(-1)];
        let result = fft_convolve(&a, &b);
        // (10^12 + x)(10^12 - x) = 10^24 - x^2 evaluated at x^0, x^1, x^2
        // = [10^24, 10^12 - 10^12, -1] = [10^24, 0, -1]
        assert_eq!(
            result,
            vec![
                Integer::from(10u32).pow(24),
                Integer::from(0),
                Integer::from(-1),
            ]
        );
    }

    #[test]
    fn fft_convolve_empty() {
        assert!(fft_convolve(&[], &[Integer::from(1)]).is_empty());
        assert!(fft_convolve(&[Integer::from(1)], &[]).is_empty());
    }
}
