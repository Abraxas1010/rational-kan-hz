use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{
    get_machine_info, median, now_utc, time_fn, write_result, BenchConfig, BenchResult, DataPoint,
};
use hybrid_zeckendorf_bench::normalization::log1000_floor;
use hybrid_zeckendorf_bench::reference::gmp_wrapper::modpow_gmp;
use hybrid_zeckendorf_bench::HybridNumber;
use rand::RngCore;
use rug::integer::Order;
use rug::Integer;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![512u32, 1024, 2048, 4096])]
    bit_sizes: Vec<u32>,
    #[arg(long, default_value_t = 20)]
    repeats: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
}

fn main() {
    let args = Args::parse();
    let bit_sizes = if args.full {
        vec![512, 1024, 2048, 4096]
    } else {
        args.bit_sizes.clone()
    };

    let config = BenchConfig {
        experiment_id: "exp1_modexp".to_string(),
        description: "Cryptographic modular exponentiation: HZ vs GMP".to_string(),
        parameters: serde_json::json!({
            "bit_sizes": bit_sizes,
            "repeats_per_size": args.repeats,
            "warmup": args.warmup,
            "algorithm": "square_and_multiply",
        }),
        repeats: args.repeats,
        warmup_repeats: args.warmup,
    };

    let mut data_points = Vec::new();
    let mut rng = rand::thread_rng();

    for &bits in &bit_sizes {
        let (n, a, e) = generate_modexp_params(bits, &mut rng);

        for _ in 0..args.warmup {
            std::hint::black_box(modpow_gmp(&a, &e, &n));
            std::hint::black_box(HybridNumber::modpow(&a, &e, &n));
        }

        let ref_times = time_fn(&mut || modpow_gmp(&a, &e, &n), args.repeats);
        let hz_times = time_fn(&mut || HybridNumber::modpow(&a, &e, &n), args.repeats);

        let gmp_result = modpow_gmp(&a, &e, &n);
        let hz_result = HybridNumber::modpow(&a, &e, &n);
        assert_eq!(gmp_result, hz_result, "Results diverge at {bits} bits");

        let hz_exp = HybridNumber::from_integer(&e);

        let hz_median = median(&hz_times);
        let ref_median = median(&ref_times);

        data_points.push(DataPoint {
            input_description: format!("{bits}-bit modular exponentiation"),
            input_size_bits: bits as u64,
            hz_time_ns: hz_times,
            native_time_ns: None,
            ref_time_ns: ref_times,
            hz_median_ns: hz_median,
            native_median_ns: None,
            ref_median_ns: ref_median,
            speedup_ratio: ref_median as f64 / hz_median as f64,
            native_speedup_ratio: None,
            hz_density_rho: Some(hz_exp.density()),
            hz_active_levels: Some(hz_exp.active_levels()),
            extra: serde_json::json!({
                "n_digits": n.to_string().len(),
                "e_digits": e.to_string().len(),
                "hz_support_card": hz_exp.support_card(),
                "proved_bound_active_levels_exp": log1000_floor(&e) + 2,
                "proved_bound_active_levels_modulus": log1000_floor(&n) + 2,
            }),
        });
    }

    let n = data_points.len();
    let faster_count = data_points
        .iter()
        .filter(|d| d.hz_median_ns < d.ref_median_ns)
        .count();
    let slower_2x_count = data_points
        .iter()
        .filter(|d| d.hz_median_ns > d.ref_median_ns.saturating_mul(2))
        .count();
    let all_competitive = data_points
        .iter()
        .all(|d| d.hz_median_ns <= d.ref_median_ns.saturating_mul(2));

    let decision = if faster_count * 4 >= 3 * n {
        "advantage"
    } else if all_competitive {
        "competitive"
    } else if slower_2x_count * 4 >= 3 * n {
        "disadvantage"
    } else {
        "mixed"
    };

    let result = BenchResult {
        config,
        data_points,
        timestamp: now_utc(),
        machine_info: get_machine_info(),
        decision: Some(decision.to_string()),
        summary: serde_json::json!({
            "decision": decision,
            "faster_count": faster_count,
            "slower_2x_count": slower_2x_count,
            "total_sizes": n,
            "rule": "advantage if >=3/4 faster; competitive if all <=2x slower; disadvantage if >=3/4 slower by >2x"
        }),
    };

    write_result(&result, "results/exp1_modexp.json");
    println!(
        "Wrote results/exp1_modexp.json with {} data points",
        result.data_points.len()
    );
}

fn generate_modexp_params(bits: u32, rng: &mut impl RngCore) -> (Integer, Integer, Integer) {
    let n = loop {
        let mut candidate = random_integer_bits(bits, rng);
        candidate.set_bit(0, true);
        if candidate > 3 {
            break candidate;
        }
    };

    let a = random_integer_below(&n, rng);
    let mut e = random_integer_bits(bits, rng);
    if e <= 1 {
        e = Integer::from(65537);
    }
    (n, a, e)
}

fn random_integer_bits(bits: u32, rng: &mut impl RngCore) -> Integer {
    let byte_len = ((bits + 7) / 8) as usize;
    let mut bytes = vec![0u8; byte_len];
    rng.fill_bytes(&mut bytes);
    let mut n = Integer::from_digits(&bytes, Order::Lsf);
    if bits > 0 {
        n.set_bit(bits - 1, true);
    }
    n
}

fn random_integer_below(n: &Integer, rng: &mut impl RngCore) -> Integer {
    let bits = n.significant_bits() as u32;
    loop {
        let x = random_integer_bits(bits.max(2), rng);
        if &x < n {
            return x;
        }
    }
}
