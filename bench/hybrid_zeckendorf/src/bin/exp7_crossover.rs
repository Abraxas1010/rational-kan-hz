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
    #[arg(long, default_value_t = 1_000_000u64)]
    bits: u64,
    #[arg(long, default_value_t = 25)]
    rho_steps: usize,
    #[arg(long, default_value_t = -5.0)]
    rho_min_exp: f64,
    #[arg(long, default_value_t = 0.0)]
    rho_max_exp: f64,
    #[arg(long, default_value_t = 20)]
    trials: usize,
    #[arg(long, default_value_t = 10)]
    repeats: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
}

fn main() {
    let args = Args::parse();
    let bits = if args.full { 1_000_000 } else { args.bits };
    let rho_steps = if args.full { 25 } else { args.rho_steps };
    let trials = if args.full { 20 } else { args.trials };
    let repeats = if args.full { 10 } else { args.repeats };
    let warmup = if args.full { 2 } else { args.warmup };
    let rho_values = logspace(args.rho_min_exp, args.rho_max_exp, rho_steps.max(2));

    let config = BenchConfig {
        experiment_id: "exp7_crossover".to_string(),
        description: "Crossover sweep for sparse-add speedup vs density (legacy and native)"
            .to_string(),
        parameters: serde_json::json!({
            "bit_size": bits,
            "rho_values": rho_values,
            "trials_per_rho": trials,
            "repeats_per_trial": repeats,
            "warmup": warmup,
            "h13_rho_steps": rho_steps,
            "h_bench_dual_path": true
        }),
        repeats,
        warmup_repeats: warmup,
    };

    let mut rng = rand::thread_rng();
    let mut data_points = Vec::<DataPoint>::new();
    let mut skipped_invalid_density_trials = 0usize;

    for &rho in &rho_values {
        for trial in 0..trials {
            let hz_a = construct_sparse_hz(bits, rho, &mut rng);
            let hz_b = construct_sparse_hz(bits, rho, &mut rng);
            let flat_a = FlatHybridNumber::from_legacy(&hz_a);
            let flat_b = FlatHybridNumber::from_legacy(&hz_b);
            let gmp_a = hz_a.eval();
            let gmp_b = hz_b.eval();
            if let Err(err) = validate_sparse_hz_with_value(bits, rho, &hz_a, &gmp_a) {
                skipped_invalid_density_trials += 1;
                eprintln!(
                    "Skipping exp7 trial at rho={rho:.6e}, trial={trial}: lhs validation failed: {err}"
                );
                continue;
            }
            if let Err(err) = validate_sparse_hz_with_value(bits, rho, &hz_b, &gmp_b) {
                skipped_invalid_density_trials += 1;
                eprintln!(
                    "Skipping exp7 trial at rho={rho:.6e}, trial={trial}: rhs validation failed: {err}"
                );
                continue;
            }
            let rho_a = density_from_value(&hz_a, &gmp_a);
            let rho_b = density_from_value(&hz_b, &gmp_b);
            if !(rho_a < 2.0 * rho && rho_b < 2.0 * rho) {
                skipped_invalid_density_trials += 1;
                eprintln!(
                    "Skipping exp7 trial at rho={rho:.6e}, trial={trial}: rho drift lhs={rho_a:.6e}, rhs={rho_b:.6e}"
                );
                continue;
            }

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
                "legacy eager addition mismatch at rho={rho}"
            );
            let hz_result_lazy = hz_a.add_lazy_legacy(&hz_b).eval();
            assert_eq!(
                hz_result_lazy, gmp_result,
                "legacy lazy addition mismatch at rho={rho}"
            );
            let hz_native_result_eager = flat_a.add_native(&flat_b).eval();
            assert_eq!(
                hz_native_result_eager, gmp_result,
                "native eager addition mismatch at rho={rho}"
            );
            let hz_native_result_lazy = flat_a.add_lazy(&flat_b).eval();
            assert_eq!(
                hz_native_result_lazy, gmp_result,
                "native lazy addition mismatch at rho={rho}"
            );

            let hz_eager_median = median(&hz_times_eager);
            let hz_lazy_median = median(&hz_times_lazy);
            let hz_native_eager_median = median(&hz_native_times_eager);
            let hz_native_lazy_median = median(&hz_native_times_lazy);
            let ref_median = median(&ref_times);
            data_points.push(DataPoint {
                input_description: format!("bits={bits}, target_rho={rho:.6e}, trial={trial}"),
                input_size_bits: bits,
                hz_time_ns: hz_times_lazy.clone(),
                native_time_ns: Some(hz_native_times_lazy.clone()),
                ref_time_ns: ref_times.clone(),
                hz_median_ns: hz_lazy_median,
                native_median_ns: Some(hz_native_lazy_median),
                ref_median_ns: ref_median,
                speedup_ratio: ref_median as f64 / hz_lazy_median.max(1) as f64,
                native_speedup_ratio: Some(ref_median as f64 / hz_native_lazy_median.max(1) as f64),
                hz_density_rho: Some((rho_a + rho_b) * 0.5),
                hz_active_levels: Some((hz_a.active_levels() + hz_b.active_levels()) / 2),
                extra: serde_json::json!({
                    "target_rho": rho,
                    "actual_rho": (rho_a + rho_b) * 0.5,
                    "actual_rho_a": rho_a,
                    "actual_rho_b": rho_b,
                    "support_card_a": hz_a.support_card(),
                    "support_card_b": hz_b.support_card(),
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

    let mut sweep_rows = Vec::new();
    for &rho in &rho_values {
        let mut speedups_lazy = Vec::new();
        let mut speedups_native = Vec::new();
        let mut measured = Vec::new();
        for dp in &data_points {
            let is_rho = dp
                .extra
                .get("target_rho")
                .and_then(|v| v.as_f64())
                .map(|x| approx_eq(x, rho))
                .unwrap_or(false);
            if is_rho {
                speedups_lazy.push(dp.speedup_ratio);
                if let Some(speed_native) = dp.native_speedup_ratio {
                    speedups_native.push(speed_native);
                }
                measured.push(dp.hz_density_rho.unwrap_or(0.0));
            }
        }
        if !speedups_lazy.is_empty() {
            sweep_rows.push(serde_json::json!({
                "target_rho": rho,
                "median_lazy_speedup": median_f64(&speedups_lazy),
                "median_native_speedup": median_f64(&speedups_native),
                "mean_measured_rho": measured.iter().sum::<f64>() / measured.len() as f64,
                "samples": speedups_lazy.len(),
            }));
        }
    }

    let crossover_rho_lazy = find_crossover(&sweep_rows, "median_lazy_speedup");
    let crossover_rho_native = find_crossover(&sweep_rows, "median_native_speedup");
    let decision = if crossover_rho_native.is_some() && crossover_rho_lazy.is_some() {
        "legacy_and_native_crossover_found"
    } else if crossover_rho_native.is_some() {
        "native_crossover_found"
    } else if crossover_rho_lazy.is_some() {
        "legacy_crossover_found"
    } else {
        "no_crossover_in_range"
    };

    let result = BenchResult {
        config,
        data_points,
        timestamp: now_utc(),
        machine_info: get_machine_info(),
        decision: Some(decision.to_string()),
        summary: serde_json::json!({
            "decision": decision,
            "crossover_rho_lazy": crossover_rho_lazy,
            "crossover_rho_native": crossover_rho_native,
            "sweep_rows": sweep_rows,
            "h13_requirement_met": rho_steps >= 20,
            "skipped_invalid_density_trials": skipped_invalid_density_trials
        }),
    };

    write_result(&result, "results/exp7_crossover.json");
    println!(
        "Wrote results/exp7_crossover.json with {} data points",
        result.data_points.len()
    );
}

fn logspace(min_exp: f64, max_exp: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![10f64.powf(min_exp)];
    }
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let e = min_exp + t * (max_exp - min_exp);
            10f64.powf(e)
        })
        .collect()
}

fn find_crossover(rows: &[serde_json::Value], speed_key: &str) -> Option<f64> {
    let mut last = None::<(f64, f64)>;
    for row in rows {
        let rho = row.get("target_rho").and_then(|v| v.as_f64())?;
        let speed = row.get(speed_key).and_then(|v| v.as_f64())?;
        if let Some((prev_rho, prev_speed)) = last {
            if prev_speed > 1.0 && speed <= 1.0 {
                let delta = speed - prev_speed;
                if delta.abs() < 1e-15 {
                    return Some(rho);
                }
                let t = (1.0 - prev_speed) / delta;
                return Some(prev_rho + t * (rho - prev_rho));
            }
        }
        last = Some((rho, speed));
    }
    None
}
