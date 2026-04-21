use clap::Parser;
use hybrid_zeckendorf_bench::base_phi::{
    base_phi_eval, raw_mul, repeated_add_mul, shift_to_base_phi, support_density,
};
use hybrid_zeckendorf_bench::base_phi_bench::{random_shift_support, support_from_density};
use hybrid_zeckendorf_bench::bench_config::{
    get_machine_info, median, now_utc, time_fn, write_result, BenchConfig, BenchResult, DataPoint,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 512)]
    span: i32,
    #[arg(long, default_value_t = 0.05)]
    rho: f64,
    #[arg(long, default_value_t = 12)]
    repeats: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
}

fn main() {
    let args = Args::parse();
    let support = support_from_density(args.span, args.rho);
    let coeff_bits = if args.full { 20 } else { 16 };

    let config = BenchConfig {
        experiment_id: "exp10_shift_bridge".to_string(),
        description:
            "Shift-support bridge plus raw base-phi multiplication vs repeated-add baseline"
                .to_string(),
        parameters: serde_json::json!({
            "span": args.span,
            "rho": args.rho,
            "support": support,
            "coeff_bits": coeff_bits,
            "repeats": args.repeats,
            "warmup": args.warmup,
        }),
        repeats: args.repeats,
        warmup_repeats: args.warmup,
    };

    let mut rng = rand::thread_rng();
    let lhs_shift = random_shift_support(args.span, support, coeff_bits, &mut rng);
    let rhs_shift = random_shift_support(args.span, support, coeff_bits, &mut rng);

    for _ in 0..args.warmup {
        std::hint::black_box(raw_mul(
            &shift_to_base_phi(&lhs_shift),
            &shift_to_base_phi(&rhs_shift),
        ));
        std::hint::black_box(repeated_add_mul(
            &shift_to_base_phi(&lhs_shift),
            &shift_to_base_phi(&rhs_shift),
        ));
    }

    let hz_times = time_fn(
        &mut || {
            raw_mul(
                &shift_to_base_phi(&lhs_shift),
                &shift_to_base_phi(&rhs_shift),
            )
        },
        args.repeats,
    );
    let ref_times = time_fn(
        &mut || {
            repeated_add_mul(
                &shift_to_base_phi(&lhs_shift),
                &shift_to_base_phi(&rhs_shift),
            )
        },
        args.repeats,
    );

    let lhs_digits = shift_to_base_phi(&lhs_shift);
    let rhs_digits = shift_to_base_phi(&rhs_shift);
    let raw = raw_mul(&lhs_digits, &rhs_digits);
    let repeated = repeated_add_mul(&lhs_digits, &rhs_digits);
    assert_eq!(raw, repeated, "bridge multiply mismatch");

    let lhs_eval = base_phi_eval(&lhs_digits, 256);
    let rhs_eval = base_phi_eval(&rhs_digits, 256);
    let raw_eval = base_phi_eval(&raw, 256);
    let rhs_product = rug::Float::with_val(256, lhs_eval * rhs_eval);

    let hz_median = median(&hz_times);
    let ref_median = median(&ref_times);
    let result = BenchResult {
        config,
        data_points: vec![DataPoint {
            input_description: format!(
                "span={}, rho={:.6e}, support={support}",
                args.span, args.rho
            ),
            input_size_bits: args.span as u64,
            hz_time_ns: hz_times,
            native_time_ns: None,
            ref_time_ns: ref_times,
            hz_median_ns: hz_median,
            native_median_ns: None,
            ref_median_ns: ref_median,
            speedup_ratio: ref_median as f64 / hz_median.max(1) as f64,
            native_speedup_ratio: None,
            hz_density_rho: Some(
                (support_density(&lhs_digits) + support_density(&rhs_digits)) * 0.5,
            ),
            hz_active_levels: None,
            extra: serde_json::json!({
                "span": args.span,
                "rho": args.rho,
                "support": support,
                "coeff_bits": coeff_bits,
                "result_support": raw.len(),
                "raw_eval": raw_eval.to_string_radix(10, None),
                "input_eval_product": rhs_product.to_string_radix(10, None),
            }),
        }],
        timestamp: now_utc(),
        machine_info: get_machine_info(),
        decision: Some(
            if hz_median < ref_median {
                "advantage"
            } else {
                "mixed"
            }
            .to_string(),
        ),
        summary: serde_json::json!({
            "decision": if hz_median < ref_median { "advantage" } else { "mixed" },
            "claim": "bridge + raw base-phi multiplication can be measured end-to-end from shift-support inputs",
        }),
    };

    write_result(&result, "results/exp10_shift_bridge.json");
    println!("Wrote results/exp10_shift_bridge.json");
}
