use clap::Parser;
use hybrid_zeckendorf_bench::base_phi::{raw_mul, repeated_add_mul, support_density};
use hybrid_zeckendorf_bench::base_phi_bench::random_sparse_digits;
use hybrid_zeckendorf_bench::bench_config::{
    get_machine_info, median, now_utc, time_fn, write_result, BenchConfig, BenchResult, DataPoint,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 12)]
    repeats: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
}

fn main() {
    let args = Args::parse();
    let configs = if args.full {
        vec![(128, 8, 16), (512, 16, 16), (2048, 32, 16), (4096, 64, 16)]
    } else {
        vec![(128, 8, 16), (512, 16, 16), (2048, 32, 16)]
    };

    let config = BenchConfig {
        experiment_id: "exp8_base_phi_rawmul".to_string(),
        description: "Sparse base-phi raw convolution vs repeated-add multiplication".to_string(),
        parameters: serde_json::json!({
            "configs": configs,
            "repeats_per_config": args.repeats,
            "warmup": args.warmup,
        }),
        repeats: args.repeats,
        warmup_repeats: args.warmup,
    };

    let mut rng = rand::thread_rng();
    let mut data_points = Vec::new();

    for (span, support, coeff_bits) in configs {
        let a = random_sparse_digits(span, support, coeff_bits, &mut rng);
        let b = random_sparse_digits(span, support, coeff_bits, &mut rng);

        for _ in 0..args.warmup {
            std::hint::black_box(raw_mul(&a, &b));
            std::hint::black_box(repeated_add_mul(&a, &b));
        }

        let hz_times = time_fn(&mut || raw_mul(&a, &b), args.repeats);
        let ref_times = time_fn(&mut || repeated_add_mul(&a, &b), args.repeats);

        let raw = raw_mul(&a, &b);
        let repeated = repeated_add_mul(&a, &b);
        assert_eq!(
            raw, repeated,
            "raw/repeated mismatch at span={span}, support={support}"
        );

        let hz_median = median(&hz_times);
        let ref_median = median(&ref_times);
        data_points.push(DataPoint {
            input_description: format!("span={span}, support={support}, coeff_bits={coeff_bits}"),
            input_size_bits: span as u64,
            hz_time_ns: hz_times,
            native_time_ns: None,
            ref_time_ns: ref_times,
            hz_median_ns: hz_median,
            native_median_ns: None,
            ref_median_ns: ref_median,
            speedup_ratio: ref_median as f64 / hz_median.max(1) as f64,
            native_speedup_ratio: None,
            hz_density_rho: Some((support_density(&a) + support_density(&b)) * 0.5),
            hz_active_levels: None,
            extra: serde_json::json!({
                "span": span,
                "support": support,
                "coeff_bits": coeff_bits,
                "input_support_a": a.len(),
                "input_support_b": b.len(),
                "result_support": raw.len(),
            }),
        });
    }

    let faster_count = data_points
        .iter()
        .filter(|dp| dp.hz_median_ns < dp.ref_median_ns)
        .count();
    let decision = if data_points.is_empty() {
        "blocked"
    } else if faster_count * 4 >= data_points.len() * 3 {
        "advantage"
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
            "total_configs": configs_len(args.full),
            "claim": "raw sparse convolution outperforms repeated-add analogue when support remains sparse",
        }),
    };

    write_result(&result, "results/exp8_base_phi_rawmul.json");
    println!(
        "Wrote results/exp8_base_phi_rawmul.json with {} data points",
        result.data_points.len()
    );
}

fn configs_len(full: bool) -> usize {
    if full {
        4
    } else {
        3
    }
}
