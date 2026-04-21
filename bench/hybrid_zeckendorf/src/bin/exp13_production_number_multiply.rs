use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{
    BenchConfig, BenchResult, DataPoint, get_machine_info, median, now_utc, time_fn, write_result,
};
use hybrid_zeckendorf_bench::production::{multiply_decision_for, ProductionNumber};
use hybrid_zeckendorf_bench::sparse::{
    construct_sparse_hz, density_from_value, validate_sparse_hz_with_value,
};
use rug::Integer;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 10_000u64)]
    bits: u64,
    #[arg(long, value_delimiter = ',', default_values_t = vec![1e-4f64, 1e-3])]
    rhos: Vec<f64>,
    #[arg(long, default_value_t = 2)]
    samples: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
}

fn main() {
    let args = Args::parse();

    let config = BenchConfig {
        experiment_id: "exp13_production_number_multiply".to_string(),
        description:
            "Carrier-polymorphic production multiply vs raw GMP multiply on sparse-origin inputs"
                .to_string(),
        parameters: serde_json::json!({
            "bits": args.bits,
            "target_rhos": args.rhos,
            "samples": args.samples,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "carrier_policy": "multiply returns routed deferred carrier"
        }),
        repeats: args.repeats,
        warmup_repeats: args.warmup,
    };

    let mut rng = rand::thread_rng();
    let mut data_points = Vec::new();

    for &rho in &args.rhos {
        for sample_idx in 0..args.samples {
            let hz_a = construct_sparse_hz(args.bits, rho, &mut rng);
            let hz_b = construct_sparse_hz(args.bits, rho, &mut rng);
            let gmp_a = hz_a.eval();
            let gmp_b = hz_b.eval();
            validate_sparse_hz_with_value(args.bits, rho, &hz_a, &gmp_a)
                .expect("lhs sparse validation");
            validate_sparse_hz_with_value(args.bits, rho, &hz_b, &gmp_b)
                .expect("rhs sparse validation");
            let route = multiply_decision_for(&hz_a, &hz_b);

            let prod_a = ProductionNumber::from_hybrid(hz_a.clone());
            let prod_b = ProductionNumber::from_hybrid(hz_b.clone());

            for _ in 0..args.warmup {
                std::hint::black_box(prod_a.multiply(&prod_b));
                std::hint::black_box(gmp_mul(&gmp_a, &gmp_b));
            }

            let hz_times = time_fn(&mut || prod_a.multiply(&prod_b), args.repeats);
            let ref_times = time_fn(&mut || gmp_mul(&gmp_a, &gmp_b), args.repeats);

            let hz_product = prod_a.multiply(&prod_b);
            let hz_result = hz_product.to_integer();
            let gmp_result = gmp_mul(&gmp_a, &gmp_b);
            assert_eq!(hz_result, gmp_result, "production-number multiply mismatch");

            let hz_median = median(&hz_times);
            let ref_median = median(&ref_times);
            data_points.push(DataPoint {
                input_description: format!(
                    "bits={}, target_rho={rho:.6e}, sample={sample_idx}",
                    args.bits
                ),
                input_size_bits: args.bits,
                hz_time_ns: hz_times,
                native_time_ns: None,
                ref_time_ns: ref_times,
                hz_median_ns: hz_median,
                native_median_ns: None,
                ref_median_ns: ref_median,
                speedup_ratio: ref_median as f64 / hz_median.max(1) as f64,
                native_speedup_ratio: None,
                hz_density_rho: Some(
                    (density_from_value(&hz_a, &gmp_a) + density_from_value(&hz_b, &gmp_b)) * 0.5,
                ),
                hz_active_levels: Some((hz_a.active_levels() + hz_b.active_levels()) / 2),
                extra: serde_json::json!({
                    "carrier_policy": match hz_product {
                        ProductionNumber::BasePhi { .. } => "BasePhi",
                        ProductionNumber::Hybrid { .. } => "Hybrid",
                        ProductionNumber::Integer(_) => "Integer",
                    },
                    "multiply_route": format!("{:?}", route.route),
                    "multiply_reason": route.reason,
                }),
            });
        }
    }

    let competitive = data_points
        .iter()
        .all(|dp| dp.hz_median_ns <= dp.ref_median_ns.saturating_mul(2));
    let decision = if data_points
        .iter()
        .all(|dp| dp.hz_median_ns < dp.ref_median_ns)
    {
        "advantage"
    } else if competitive {
        "competitive"
    } else {
        "disadvantage"
    };

    let result = BenchResult {
        config,
        data_points,
        timestamp: now_utc(),
        machine_info: get_machine_info(),
        decision: Some(decision.to_string()),
        summary: serde_json::json!({
            "decision": decision,
            "claim": "Carrier-polymorphic production multiply avoids the catastrophic HZ re-embedding cost",
        }),
    };

    write_result(&result, "results/exp13_production_number_multiply.json");
    println!(
        "Wrote results/exp13_production_number_multiply.json with {} data points",
        result.data_points.len()
    );
}

fn gmp_mul(a: &Integer, b: &Integer) -> Integer {
    a.clone() * b.clone()
}
