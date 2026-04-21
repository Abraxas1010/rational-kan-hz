//! Precomputed Fibonacci number table.
//!
//! Computes Fibonacci numbers once and serves O(1) lookups for the benchmark's
//! native HZ path.

use rug::Integer;
use std::sync::OnceLock;

pub const MAX_FIB_TABLE: usize = 20_000;

static FIB_TABLE: OnceLock<Vec<Integer>> = OnceLock::new();

fn init_table() -> Vec<Integer> {
    let mut table = Vec::with_capacity(MAX_FIB_TABLE + 1);
    table.push(Integer::from(0));
    table.push(Integer::from(1));
    for i in 2..=MAX_FIB_TABLE {
        let next = table[i - 1].clone() + &table[i - 2];
        table.push(next);
    }
    table
}

fn table() -> &'static Vec<Integer> {
    FIB_TABLE.get_or_init(init_table)
}

pub fn fib(n: u32) -> Integer {
    let idx = n as usize;
    let table = table();
    if idx < table.len() {
        table[idx].clone()
    } else {
        crate::zeckendorf::fib(n)
    }
}

pub fn fib_ref(n: u32) -> &'static Integer {
    let idx = n as usize;
    assert!(
        idx <= MAX_FIB_TABLE,
        "Fibonacci index {n} exceeds MAX_FIB_TABLE={MAX_FIB_TABLE}"
    );
    &table()[idx]
}

pub fn zeckendorf_from_table(n: &Integer) -> Vec<u32> {
    if *n <= 0 {
        return vec![];
    }

    let table = table();
    if n > table.last().expect("fibonacci table is non-empty") {
        return crate::zeckendorf::zeckendorf(n);
    }

    let mut k = table.partition_point(|x| x <= n);
    if k == 0 {
        return vec![];
    }
    k -= 1;

    let mut remaining = n.clone();
    let mut out = Vec::new();
    while remaining > 0 && k >= 2 {
        if table[k] <= remaining {
            out.push(k as u32);
            remaining -= &table[k];
            if k >= 2 {
                k = k.saturating_sub(2);
            } else {
                break;
            }
        } else {
            k -= 1;
        }
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fib_small_values() {
        assert_eq!(fib(0), 0);
        assert_eq!(fib(1), 1);
        assert_eq!(fib(2), 1);
        assert_eq!(fib(10), 55);
        assert_eq!(fib(20), 6765);
    }

    #[test]
    fn fib_matches_legacy_fast_doubling() {
        use crate::zeckendorf::fib as fib_old;

        for i in [
            0, 1, 2, 10, 50, 100, 500, 1000, 5000, 10000, 15_000, 100_000, 500_000,
        ] {
            assert_eq!(fib(i), fib_old(i), "mismatch at F({i})");
        }
    }

    #[test]
    fn zeckendorf_matches_legacy() {
        use crate::zeckendorf::zeckendorf as zeck_old;

        for n in [1u64, 5, 11, 42, 100, 999, 12_345, 1_000_000] {
            let value = Integer::from(n);
            assert_eq!(
                zeckendorf_from_table(&value),
                zeck_old(&value),
                "mismatch at n={n}"
            );
        }
    }

    #[test]
    fn zeckendorf_large_value_falls_back_correctly() {
        use crate::weight::weight;
        use crate::zeckendorf::zeckendorf as zeck_old;

        let value = weight(15);
        assert_eq!(zeckendorf_from_table(&value), zeck_old(&value));
    }
}
