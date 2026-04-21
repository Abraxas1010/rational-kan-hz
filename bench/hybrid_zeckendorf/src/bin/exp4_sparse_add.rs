use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{
    approx_eq, get_machine_info, median, median_f64, now_utc, time_fn, write_result, BenchConfig,
    BenchResult, DataPoint,
};
use hybrid_zeckendorf_bench::sparse::{
    construct_sparse_hz, density_from_value, validate_sparse_hz_with_value,
};
use hybrid_zeckendorf_bench::FlatHybridNumber;
use rug::Integer;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![10_000u64, 100_000, 1_000_000])]
    bit_sizes: Vec<u64>,
    #[arg(long, value_delimiter = ',', default_values_t = vec![1e-5f64, 1e-4, 1e-3, 1e-2, 1e-1, 0.27])]
    rhos: Vec<f64>,
    #[arg(long, default_value_t = 50)]
    samples: usize,
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
        vec![10_000, 100_000, 1_000_000]
    } else {
        args.bit_sizes.clone()
    };
    let rhos = if args.full {
        vec![1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.27]
    } else {
        args.rhos.clone()
    };
    let samples = if args.full { 50 } else { args.samples };
    let repeats = if args.full { 20 } else { args.repeats };
    let warmup = if args.full { 3 } else { args.warmup };

    let config = BenchConfig {
        experiment_id: "exp4_sparse_add".to_string(),
        description: "Sparse-addition benchmark with legacy and native HZ paths vs GMP".to_string(),
        parameters: serde_json::json!({
            "bit_sizes": bit_sizes,
            "target_rhos": rhos,
            "samples_per_config": samples,
            "repeats_per_sample": repeats,
            "warmup": warmup,
            "constraints": {
                "h10_sparse_constructor_only": true,
                "h11_same_mathematical_inputs": true,
                "h_bench_dual_path": true
            }
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
                let flat_a = FlatHybridNumber::from_legacy(&hz_a);
                let flat_b = FlatHybridNumber::from_legacy(&hz_b);
                let gmp_a = hz_a.eval();
                let gmp_b = hz_b.eval();

                validate_sparse_hz_with_value(bits, target_rho, &hz_a, &gmp_a)
                    .expect("sparse constructor validation failed for lhs");
                validate_sparse_hz_with_value(bits, target_rho, &hz_b, &gmp_b)
                    .expect("sparse constructor validation failed for rhs");

                let actual_rho_a = density_from_value(&hz_a, &gmp_a);
                let actual_rho_b = density_from_value(&hz_b, &gmp_b);
                assert!(
                    actual_rho_a < 2.0 * target_rho && actual_rho_b < 2.0 * target_rho,
                    "density drift: target={target_rho:.6e}, lhs={actual_rho_a:.6e}, rhs={actual_rho_b:.6e}"
                );

                for _ in 0..warmup {
                    std::hint::black_box(hz_a.add_legacy(&hz_b));
                    std::hint::black_box(hz_a.add_lazy_legacy(&hz_b));
                    std::hint::black_box(flat_a.add_native(&flat_b));
                    std::hint::black_box(flat_a.add_lazy(&flat_b));
                    std::hint::black_box(gmp_a.clone() + gmp_b.clone());
                }

                let hz_times_eager = time_fn(&mut || hz_a.add_legacy(&hz_b), repeats);
                let hz_times_lazy = time_fn(&mut || hz_a.add_lazy_legacy(&hz_b), repeats);
                let hz_native_times_eager = time_fn(&mut || flat_a.add_native(&flat_b), repeats);
                let hz_native_times_lazy = time_fn(&mut || flat_a.add_lazy(&flat_b), repeats);
                let ref_times = time_fn(
                    &mut || {
                        let out: Integer = gmp_a.clone() + gmp_b.clone();
                        out
                    },
                    repeats,
                );

                let gmp_result: Integer = gmp_a.clone() + gmp_b.clone();
                let hz_result_eager = hz_a.add_legacy(&hz_b).eval();
                assert_eq!(
                    hz_result_eager, gmp_result,
                    "legacy eager addition mismatch at bits={bits}, target_rho={target_rho}, sample={sample_idx}"
                );
                let hz_result_lazy = hz_a.add_lazy_legacy(&hz_b).eval();
                assert_eq!(
                    hz_result_lazy, gmp_result,
                    "legacy lazy addition mismatch at bits={bits}, target_rho={target_rho}, sample={sample_idx}"
                );
                let hz_native_result_eager = flat_a.add_native(&flat_b).eval();
                assert_eq!(
                    hz_native_result_eager, gmp_result,
                    "native eager addition mismatch at bits={bits}, target_rho={target_rho}, sample={sample_idx}"
                );
                let hz_native_result_lazy = flat_a.add_lazy(&flat_b).eval();
                assert_eq!(
                    hz_native_result_lazy, gmp_result,
                    "native lazy addition mismatch at bits={bits}, target_rho={target_rho}, sample={sample_idx}"
                );

                let hz_eager_median = median(&hz_times_eager);
                let hz_lazy_median = median(&hz_times_lazy);
                let hz_native_eager_median = median(&hz_native_times_eager);
                let hz_native_lazy_median = median(&hz_native_times_lazy);
                let ref_median = median(&ref_times);
                data_points.push(DataPoint {
                    input_description: format!(
                        "bits={bits}, target_rho={target_rho:.6e}, sample={sample_idx}"
                    ),
                    input_size_bits: bits,
                    hz_time_ns: hz_times_lazy.clone(),
                    native_time_ns: Some(hz_native_times_lazy.clone()),
                    ref_time_ns: ref_times.clone(),
                    hz_median_ns: hz_lazy_median,
                    native_median_ns: Some(hz_native_lazy_median),
                    ref_median_ns: ref_median,
                    speedup_ratio: ref_median as f64 / hz_lazy_median.max(1) as f64,
                    native_speedup_ratio: Some(ref_median as f64 / hz_native_lazy_median.max(1) as f64),
                    hz_density_rho: Some((actual_rho_a + actual_rho_b) * 0.5),
                    hz_active_levels: Some((hz_a.active_levels() + hz_b.active_levels()) / 2),
                    extra: serde_json::json!({
                        "target_rho": target_rho,
                        "actual_rho": (actual_rho_a + actual_rho_b) * 0.5,
                        "actual_rho_a": actual_rho_a,
                        "actual_rho_b": actual_rho_b,
                        "support_card_a": hz_a.support_card(),
                        "support_card_b": hz_b.support_card(),
                        "active_levels_a": hz_a.active_levels(),
                        "active_levels_b": hz_b.active_levels(),
                        "value_bits_a": gmp_a.significant_bits(),
                        "value_bits_b": gmp_b.significant_bits(),
                        "hz_lazy_time_ns": hz_times_lazy,
                        "hz_lazy_median_ns": hz_lazy_median,
                        "hz_lazy_speedup_ratio": ref_median as f64 / hz_lazy_median.max(1) as f64,
                        "hz_eager_time_ns": hz_times_eager,
                        "hz_eager_median_ns": hz_eager_median,
                        "hz_eager_speedup_ratio": ref_median as f64 / hz_eager_median.max(1) as f64,
                        "hz_native_lazy_time_ns": hz_native_times_lazy,
                        "hz_native_lazy_median_ns": hz_native_lazy_median,
                        "hz_native_lazy_speedup_ratio": ref_median as f64 / hz_native_lazy_median.max(1) as f64,
                        "hz_native_add_time_ns": hz_native_times_eager,
                        "hz_native_add_median_ns": hz_native_eager_median,
                        "hz_native_add_speedup_ratio": ref_median as f64 / hz_native_eager_median.max(1) as f64
                    }),
                });
            }
        }
    }

    let decision_bits = if bit_sizes.contains(&1_000_000) {
        1_000_000
    } else {
        *bit_sizes.iter().max().unwrap_or(&1_000_000)
    };
    let mut table_at_decision_bits = Vec::new();
    let mut monotonic = true;
    let mut prev_speedup: Option<f64> = None;
    for &rho in &rhos {
        let slice: Vec<&DataPoint> = data_points
            .iter()
            .filter(|dp| dp.input_size_bits == decision_bits)
            .filter(|dp| {
                dp.extra
                    .get("target_rho")
                    .and_then(|v| v.as_f64())
                    .map(|x| approx_eq(x, rho))
                    .unwrap_or(false)
            })
            .collect();
        if slice.is_empty() {
            continue;
        }
        let hz_lazy_medians: Vec<u64> = slice.iter().map(|dp| dp.hz_median_ns).collect();
        let hz_native_lazy_medians: Vec<u64> =
            slice.iter().filter_map(|dp| dp.native_median_ns).collect();
        let gmp_medians: Vec<u64> = slice.iter().map(|dp| dp.ref_median_ns).collect();
        let speedups_lazy: Vec<f64> = slice.iter().map(|dp| dp.speedup_ratio).collect();
        let speedups_native: Vec<f64> = slice
            .iter()
            .filter_map(|dp| dp.native_speedup_ratio)
            .collect();
        let hz_lazy_med = median(&hz_lazy_medians);
        let hz_native_lazy_med = median(&hz_native_lazy_medians);
        let gmp_med = median(&gmp_medians);
        let speed_lazy_med = median_f64(&speedups_lazy);
        let speed_native_med = median_f64(&speedups_native);
        if let Some(prev) = prev_speedup {
            if speed_lazy_med > prev {
                monotonic = false;
            }
        }
        prev_speedup = Some(speed_lazy_med);
        table_at_decision_bits.push(serde_json::json!({
            "target_rho": rho,
            "hz_lazy_median_ns": hz_lazy_med,
            "hz_native_lazy_median_ns": hz_native_lazy_med,
            "gmp_median_ns": gmp_med,
            "speedup_lazy_ratio": speed_lazy_med,
            "speedup_native_ratio": speed_native_med,
            "samples": slice.len(),
        }));
    }

    let rho_1e3_speedup_lazy = table_at_decision_bits
        .iter()
        .find_map(|row| {
            row.get("target_rho")
                .and_then(|v| v.as_f64())
                .filter(|x| approx_eq(*x, 1e-3))
                .and_then(|_| row.get("speedup_lazy_ratio").and_then(|v| v.as_f64()))
        })
        .unwrap_or(0.0);
    let rho_1e3_speedup_native = table_at_decision_bits
        .iter()
        .find_map(|row| {
            row.get("target_rho")
                .and_then(|v| v.as_f64())
                .filter(|x| approx_eq(*x, 1e-3))
                .and_then(|_| row.get("speedup_native_ratio").and_then(|v| v.as_f64()))
        })
        .unwrap_or(0.0);
    let rho_1e3_speedup_lazy_present = table_at_decision_bits.iter().any(|row| {
        row.get("target_rho")
            .and_then(|v| v.as_f64())
            .map(|x| approx_eq(x, 1e-3))
            .unwrap_or(false)
    });
    let decision = if !rho_1e3_speedup_lazy_present {
        "insufficient_rho_data"
    } else if rho_1e3_speedup_native >= 10.0 {
        "native_paper_confirmed"
    } else if rho_1e3_speedup_native > rho_1e3_speedup_lazy {
        "native_improved_over_legacy"
    } else {
        "native_not_improved"
    };

    let result = BenchResult {
        config,
        data_points,
        timestamp: now_utc(),
        machine_info: get_machine_info(),
        decision: Some(decision.to_string()),
        summary: serde_json::json!({
            "decision": decision,
            "decision_bits": decision_bits,
            "legacy_monotonic_vs_density": monotonic,
            "table_at_decision_bits": table_at_decision_bits,
            "rho_1e3_speedup_lazy": rho_1e3_speedup_lazy,
            "rho_1e3_speedup_native": rho_1e3_speedup_native,
        }),
    };

    write_result(&result, "results/exp4_sparse_add.json");
    println!(
        "Wrote results/exp4_sparse_add.json with {} data points",
        result.data_points.len()
    );
}
