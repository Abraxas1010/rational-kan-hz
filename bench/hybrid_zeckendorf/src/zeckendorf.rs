use rug::Integer;

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.lazyEvalFib
pub fn lazy_eval_fib(indices: &[u32]) -> Integer {
    indices.iter().map(|&i| fib(i)).sum()
}

pub fn fib(n: u32) -> Integer {
    fast_doubling(n).0
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.intraNormalize
pub fn zeckendorf(n: &Integer) -> Vec<u32> {
    if *n <= 0 {
        return vec![];
    }

    let mut fibs: Vec<Integer> = vec![Integer::from(0), Integer::from(1)];
    while fibs.last().expect("nonempty") <= n {
        let next = fibs[fibs.len() - 1].clone() + fibs[fibs.len() - 2].clone();
        fibs.push(next);
    }

    let mut remaining = n.clone();
    let mut result = Vec::<u32>::new();
    let mut k = fibs.len().saturating_sub(1);

    while remaining > 0 && k >= 2 {
        if fibs[k] <= remaining {
            result.push(k as u32);
            remaining -= fibs[k].clone();
            if k >= 2 {
                k -= 2; // enforce non-consecutive indices
            } else {
                break;
            }
        } else {
            k -= 1;
        }
    }

    result.sort_unstable();
    result
}

fn fast_doubling(n: u32) -> (Integer, Integer) {
    if n == 0 {
        return (Integer::from(0), Integer::from(1));
    }
    let (a, b) = fast_doubling(n / 2);
    let two_b_minus_a = b.clone() * 2 - a.clone();
    let c = a.clone() * two_b_minus_a;
    let d = a.clone() * a + b.clone() * b;
    if n % 2 == 0 {
        (c, d)
    } else {
        (d.clone(), c + d)
    }
}
