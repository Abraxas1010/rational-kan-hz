use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{
    BenchConfig, BenchResult, DataPoint, get_machine_info, median, median_f64, now_utc, time_fn,
    write_result,
};
use hybrid_zeckendorf_bench::sparse::{
    construct_sparse_hz, density_from_value, validate_sparse_hz_with_value,
};
use rug::Integer;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![10_000u64, 100_000])]
    bit_sizes: Vec<u64>,
    #[arg(long, value_delimiter = ',', default_values_t = vec![1e-4f64, 1e-3, 1e-2])]
    rhos: Vec<f64>,
    #[arg(long, default_value_t = 5)]
    samples: usize,
    #[arg(long, default_value_t = 5)]
    repeats: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
}

fn main() {
    let args = Args::parse();
    let bit_sizes = if args.full {
        vec![10_000, 100_000, 1_000_000]
    } else {
        args.bit_sizes.clone()
    };
    let rhos = if args.full {
        vec![1e-4, 1e-3, 1e-2]
    } else {
        args.rhos.clone()
    };
    let samples = if args.full { 8 } else { args.samples };
    let repeats = if args.full { 8 } else { args.repeats };
    let warmup = if args.full { 2 } else { args.warmup };

    let config = BenchConfig {
        experiment_id: "exp11_production_multiply".to_string(),
        description:
            "Production HybridNumber.multiply vs GMP integer multiplication on sparse inputs"
                .to_string(),
        parameters: serde_json::json!({
            "bit_sizes": bit_sizes,
            "target_rhos": rhos,
            "samples_per_config": samples,
            "repeats_per_sample": repeats,
            "warmup": warmup,
            "production_surface": "HybridNumber::multiply"
        }),
        repeats,
        warmup_repeats: warmup,
    };

    let mut rng = rand::thread_rng();
    let mut data_points = Vec::<DataPoint>::new();

    for &bits in &bit_sizes {
        for &target_rho in &rhos {
            for sample_idx in 0..samples {
                let hz_a = construct_sparse_hz(bits, target_rho, &mut rng);
                let hz_b = construct_sparse_hz(bits, target_rho, &mut rng);
                let gmp_a = hz_a.eval();
                let gmp_b = hz_b.eval();

                validate_sparse_hz_with_value(bits, target_rho, &hz_a, &gmp_a)
                    .expect("lhs sparse validation");
                validate_sparse_hz_with_value(bits, target_rho, &hz_b, &gmp_b)
                    .expect("rhs sparse validation");

                let rho_a = density_from_value(&hz_a, &gmp_a);
                let rho_b = density_from_value(&hz_b, &gmp_b);

                for _ in 0..warmup {
                    std::hint::black_box(hz_a.multiply(&hz_b));
                    std::hint::black_box(gmp_mul(&gmp_a, &gmp_b));
                }

                let hz_times = time_fn(&mut || hz_a.multiply(&hz_b), repeats);
                let ref_times = time_fn(&mut || gmp_mul(&gmp_a, &gmp_b), repeats);

                let hz_result = hz_a.multiply(&hz_b).eval();
                let gmp_result = gmp_mul(&gmp_a, &gmp_b);
                assert_eq!(hz_result, gmp_result, "production multiply mismatch");

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
                    hz_density_rho: Some((rho_a + rho_b) * 0.5),
                    hz_active_levels: Some((hz_a.active_levels() + hz_b.active_levels()) / 2),
                    extra: serde_json::json!({
                        "target_rho": target_rho,
                        "actual_rho_a": rho_a,
                        "actual_rho_b": rho_b,
                        "support_card_a": hz_a.support_card(),
                        "support_card_b": hz_b.support_card(),
                        "active_levels_a": hz_a.active_levels(),
                        "active_levels_b": hz_b.active_levels(),
                        "lhs_bits": gmp_a.significant_bits(),
                        "rhs_bits": gmp_b.significant_bits(),
                    }),
                });
            }
        }
    }

    let faster_count = data_points
        .iter()
        .filter(|dp| dp.hz_median_ns < dp.ref_median_ns)
        .count();
    let slower_2x_count = data_points
        .iter()
        .filter(|dp| dp.hz_median_ns > dp.ref_median_ns.saturating_mul(2))
        .count();
    let all_competitive = !data_points.is_empty()
        && data_points
            .iter()
            .all(|dp| dp.hz_median_ns <= dp.ref_median_ns.saturating_mul(2));

    let mut sweep_rows = Vec::new();
    for &bits in &bit_sizes {
        for &rho in &rhos {
            let rows: Vec<_> = data_points
                .iter()
                .filter(|dp| {
                    dp.input_size_bits == bits
                        && dp
                            .extra
                            .get("target_rho")
                            .and_then(|v| v.as_f64())
                            .map(|x| (x - rho).abs() <= rho.abs() * 1e-12 + 1e-15)
                            .unwrap_or(false)
                })
                .collect();
            if rows.is_empty() {
                continue;
            }
            let speedups: Vec<f64> = rows.iter().map(|dp| dp.speedup_ratio).collect();
            sweep_rows.push(serde_json::json!({
                "bits": bits,
                "target_rho": rho,
                "median_speedup": median_f64(&speedups),
                "samples": rows.len(),
            }));
        }
    }

    let decision = if data_points.is_empty() {
        "blocked"
    } else if faster_count * 4 >= data_points.len() * 3 {
        "advantage"
    } else if all_competitive {
        "competitive"
    } else if slower_2x_count * 4 >= data_points.len() * 3 {
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
            "total_data_points": bit_sizes.len() * rhos.len() * samples,
            "sweep_rows": sweep_rows,
            "claim": "Tests the actual production multiply surface rather than the exact base-phi carrier alone",
        }),
    };

    write_result(&result, "results/exp11_production_multiply.json");
    println!(
        "Wrote results/exp11_production_multiply.json with {} data points",
        result.data_points.len()
    );
}

fn gmp_mul(a: &Integer, b: &Integer) -> Integer {
    a.clone() * b.clone()
}
