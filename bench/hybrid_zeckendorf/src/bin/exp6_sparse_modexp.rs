use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{
    get_machine_info, median, now_utc, time_fn, write_result, BenchConfig, BenchResult, DataPoint,
};
use hybrid_zeckendorf_bench::sparse::{construct_sparse_hz, validate_sparse_hz};
use hybrid_zeckendorf_bench::HybridNumber;
use rand::RngCore;
use rug::integer::Order;
use rug::Integer;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![1_000_000u64])]
    bit_sizes: Vec<u64>,
    #[arg(long, value_delimiter = ',', default_values_t = vec![1e-4f64, 1e-3, 1e-2])]
    rhos: Vec<f64>,
    #[arg(long, default_value_t = 10)]
    samples: usize,
    #[arg(long, default_value_t = 10)]
    repeats: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
}

fn main() {
    let args = Args::parse();
    let bit_sizes = if args.full {
        vec![1_000_000]
    } else {
        args.bit_sizes.clone()
    };
    let rhos = if args.full {
        vec![1e-4, 1e-3, 1e-2]
    } else {
        args.rhos.clone()
    };
    let samples = if args.full { 20 } else { args.samples };
    let repeats = if args.full { 20 } else { args.repeats };
    let warmup = if args.full { 3 } else { args.warmup };

    let config = BenchConfig {
        experiment_id: "exp6_sparse_modexp".to_string(),
        description: "Sparse-exponent modpow: HZ modpow_from_hybrid vs GMP pow_mod".to_string(),
        parameters: serde_json::json!({
            "bit_sizes": bit_sizes,
            "target_rhos": rhos,
            "samples_per_config": samples,
            "repeats_per_sample": repeats,
            "warmup": warmup
        }),
        repeats,
        warmup_repeats: warmup,
    };

    let mut rng = rand::thread_rng();
    let mut data_points = Vec::<DataPoint>::new();

    for &bits in &bit_sizes {
        for &target_rho in &rhos {
            for sample_idx in 0..samples {
                let hz_exp = construct_sparse_hz(bits, target_rho, &mut rng);
                validate_sparse_hz(bits, target_rho, &hz_exp).expect("sparse exponent validation");
                let actual_rho = hz_exp.density();
                assert!(
                    actual_rho < 2.0 * target_rho,
                    "sparse exponent rho drift: target={target_rho:.6e}, actual={actual_rho:.6e}"
                );

                let exp = hz_exp.eval();
                let (modulus, base) = generate_modexp_params(bits as u32, &mut rng);

                for _ in 0..warmup {
                    std::hint::black_box(HybridNumber::modpow_from_hybrid(
                        &base, &hz_exp, &modulus,
                    ));
                    std::hint::black_box(base.clone().pow_mod(&exp, &modulus).expect("pow_mod"));
                }

                let hz_times = time_fn(
                    &mut || HybridNumber::modpow_from_hybrid(&base, &hz_exp, &modulus),
                    repeats,
                );
                let ref_times = time_fn(
                    &mut || base.clone().pow_mod(&exp, &modulus).expect("pow_mod"),
                    repeats,
                );

                let hz_result = HybridNumber::modpow_from_hybrid(&base, &hz_exp, &modulus);
                let gmp_result = base.clone().pow_mod(&exp, &modulus).expect("pow_mod");
                assert_eq!(hz_result, gmp_result, "modpow mismatch");

                let hz_median = median(&hz_times);
                let ref_median = median(&ref_times);
                data_points.push(DataPoint {
                    input_description: format!(
                        "bits={bits}, target_rho={target_rho:.6e}, sample={sample_idx}"
                    ),
                    input_size_bits: bits,
                    hz_time_ns: hz_times,
                    native_time_ns: None,
                    ref_time_ns: ref_times,
                    hz_median_ns: hz_median,
                    native_median_ns: None,
                    ref_median_ns: ref_median,
                    speedup_ratio: ref_median as f64 / hz_median.max(1) as f64,
                    native_speedup_ratio: None,
                    hz_density_rho: Some(actual_rho),
                    hz_active_levels: Some(hz_exp.active_levels()),
                    extra: serde_json::json!({
                        "target_rho": target_rho,
                        "actual_rho": actual_rho,
                        "support_card": hz_exp.support_card(),
                        "active_levels": hz_exp.active_levels(),
                        "exp_bits": exp.significant_bits(),
                        "modulus_bits": modulus.significant_bits()
                    }),
                });
            }
        }
    }

    let n = data_points.len();
    let faster = data_points
        .iter()
        .filter(|dp| dp.hz_median_ns < dp.ref_median_ns)
        .count();
    let slower_2x = data_points
        .iter()
        .filter(|dp| dp.hz_median_ns > dp.ref_median_ns.saturating_mul(2))
        .count();
    let decision = if n > 0 && faster * 4 >= 3 * n {
        "advantage"
    } else if n > 0 && slower_2x * 4 >= 3 * n {
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
            "faster_count": faster,
            "slower_2x_count": slower_2x,
            "total_data_points": n,
            "rule": "advantage if >=3/4 faster; disadvantage if >=3/4 slower by >2x"
        }),
    };

    write_result(&result, "results/exp6_sparse_modexp.json");
    println!(
        "Wrote results/exp6_sparse_modexp.json with {} data points",
        result.data_points.len()
    );
}

fn generate_modexp_params(bits: u32, rng: &mut impl RngCore) -> (Integer, Integer) {
    let modulus = loop {
        let mut candidate = random_integer_bits(bits.max(2), rng);
        candidate.set_bit(0, true);
        if candidate > 3 {
            break candidate;
        }
    };
    let base = random_integer_below(&modulus, rng);
    (modulus, base)
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
