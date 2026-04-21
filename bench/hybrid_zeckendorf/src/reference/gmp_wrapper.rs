use rug::Integer;

pub fn modpow_gmp(base: &Integer, exp: &Integer, modulus: &Integer) -> Integer {
    base.clone()
        .pow_mod(exp, modulus)
        .expect("pow_mod failed for GMP reference")
}

pub fn polymul_schoolbook_gmp(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Integer::from(0); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            out[i + j] += ai.clone() * bj.clone();
        }
    }
    out
}
