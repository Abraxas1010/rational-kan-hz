use clap::Parser;
use hybrid_zeckendorf_bench::base_phi::{dense_mul, raw_mul};
use hybrid_zeckendorf_bench::base_phi_bench::{random_sparse_digits, support_from_density};
use hybrid_zeckendorf_bench::bench_config::{
    get_machine_info, median, median_f64, now_utc, time_fn, write_result, BenchConfig, BenchResult,
    DataPoint,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 2048)]
    span: i32,
    #[arg(long, default_value_t = 12)]
    rho_steps: usize,
    #[arg(long, default_value_t = -3.0)]
    rho_min_exp: f64,
    #[arg(long, default_value_t = 0.0)]
    rho_max_exp: f64,
    #[arg(long, default_value_t = 10)]
    repeats: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
}

fn main() {
    let args = Args::parse();
    let rho_values = logspace(
        args.rho_min_exp,
        args.rho_max_exp,
        if args.full { 16 } else { args.rho_steps.max(2) },
    );
    let coeff_bits = 16u32;

    let config = BenchConfig {
        experiment_id: "exp9_base_phi_crossover".to_string(),
        description: "Sparse-vs-dense crossover for base-phi raw multiplication".to_string(),
        parameters: serde_json::json!({
            "span": args.span,
            "rho_values": rho_values,
            "coeff_bits": coeff_bits,
            "repeats": args.repeats,
            "warmup": args.warmup,
        }),
        repeats: args.repeats,
        warmup_repeats: args.warmup,
    };

    let mut rng = rand::thread_rng();
    let mut data_points = Vec::new();

    for &rho in &rho_values {
        let support = support_from_density(args.span, rho);
        let a = random_sparse_digits(args.span, support, coeff_bits, &mut rng);
        let b = random_sparse_digits(args.span, support, coeff_bits, &mut rng);

        for _ in 0..args.warmup {
            std::hint::black_box(raw_mul(&a, &b));
            std::hint::black_box(dense_mul(&a, &b));
        }

        let hz_times = time_fn(&mut || raw_mul(&a, &b), args.repeats);
        let ref_times = time_fn(&mut || dense_mul(&a, &b), args.repeats);
        let raw = raw_mul(&a, &b);
        let dense = dense_mul(&a, &b);
        assert_eq!(raw, dense, "raw/dense mismatch at rho={rho:.6e}");

        let hz_median = median(&hz_times);
        let ref_median = median(&ref_times);
        data_points.push(DataPoint {
            input_description: format!("span={}, rho={rho:.6e}, support={support}", args.span),
            input_size_bits: args.span as u64,
            hz_time_ns: hz_times,
            native_time_ns: None,
            ref_time_ns: ref_times,
            hz_median_ns: hz_median,
            native_median_ns: None,
            ref_median_ns: ref_median,
            speedup_ratio: ref_median as f64 / hz_median.max(1) as f64,
            native_speedup_ratio: None,
            hz_density_rho: Some(rho),
            hz_active_levels: None,
            extra: serde_json::json!({
                "span": args.span,
                "target_rho": rho,
                "support": support,
                "coeff_bits": coeff_bits,
                "result_support": raw.len(),
            }),
        });
    }

    let sweep_rows: Vec<_> = rho_values
        .iter()
        .filter_map(|rho| {
            let mut speedups = Vec::new();
            for dp in &data_points {
                let target = dp.extra.get("target_rho").and_then(|v| v.as_f64())?;
                if (target - rho).abs() <= rho.abs() * 1e-12 + 1e-15 {
                    speedups.push(dp.speedup_ratio);
                }
            }
            if speedups.is_empty() {
                None
            } else {
                Some(serde_json::json!({
                    "target_rho": rho,
                    "median_speedup": median_f64(&speedups),
                    "samples": speedups.len(),
                }))
            }
        })
        .collect();

    let crossover_rho = find_crossover(&sweep_rows);
    let decision = if crossover_rho.is_some() {
        "crossover_found"
    } else if sweep_rows.iter().all(|row| {
        row.get("median_speedup")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
            > 1.0
    }) {
        "sparse_wins_in_range"
    } else {
        "dense_wins_in_range"
    };

    let result = BenchResult {
        config,
        data_points,
        timestamp: now_utc(),
        machine_info: get_machine_info(),
        decision: Some(decision.to_string()),
        summary: serde_json::json!({
            "decision": decision,
            "crossover_rho": crossover_rho,
            "sweep_rows": sweep_rows,
        }),
    };

    write_result(&result, "results/exp9_base_phi_crossover.json");
    println!(
        "Wrote results/exp9_base_phi_crossover.json with {} data points",
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
            10f64.powf(min_exp + t * (max_exp - min_exp))
        })
        .collect()
}

fn find_crossover(rows: &[serde_json::Value]) -> Option<f64> {
    let mut last: Option<(f64, f64)> = None;
    for row in rows {
        let rho = row.get("target_rho").and_then(|v| v.as_f64())?;
        let speed = row.get("median_speedup").and_then(|v| v.as_f64())?;
        if let Some((prev_rho, prev_speed)) = last {
            if (prev_speed >= 1.0 && speed <= 1.0) || (prev_speed <= 1.0 && speed >= 1.0) {
                return Some((prev_rho + rho) * 0.5);
            }
        }
        last = Some((rho, speed));
    }
    None
}
