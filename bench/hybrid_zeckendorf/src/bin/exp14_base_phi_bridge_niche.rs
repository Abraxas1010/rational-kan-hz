use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{
    get_machine_info, median, now_utc, time_fn, write_result, BenchConfig, BenchResult, DataPoint,
};
use hybrid_zeckendorf_bench::production::{multiply_decision_for, ProductionNumber, ProductionRoute};
use hybrid_zeckendorf_bench::HybridNumber;
use rug::Integer;
use std::collections::BTreeMap;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 5)]
    repeats: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
}

fn main() {
    let args = Args::parse();
    let configs = [(0u32, 12u32, 13u32), (1, 16, 17), (2, 20, 21), (3, 24, 25)];

    let config = BenchConfig {
        experiment_id: "exp14_base_phi_bridge_niche".to_string(),
        description:
            "Curated single-payload HybridNumber inputs that force the deferred base-phi bridge"
                .to_string(),
        parameters: serde_json::json!({
            "configs": configs,
            "repeats": args.repeats,
            "warmup": args.warmup,
        }),
        repeats: args.repeats,
        warmup_repeats: args.warmup,
    };

    let mut data_points = Vec::new();

    for (level, lhs_index, rhs_index) in configs {
        let lhs = single_payload(level, lhs_index);
        let rhs = single_payload(level, rhs_index);
        let decision = multiply_decision_for(&lhs, &rhs);
        assert_eq!(
            decision.route,
            ProductionRoute::BasePhiMultiplyBridge,
            "curated bridge config unexpectedly fell back"
        );

        let prod_lhs = ProductionNumber::from_hybrid(lhs.clone());
        let prod_rhs = ProductionNumber::from_hybrid(rhs.clone());
        let gmp_lhs = lhs.eval();
        let gmp_rhs = rhs.eval();

        for _ in 0..args.warmup {
            std::hint::black_box(prod_lhs.multiply(&prod_rhs));
            std::hint::black_box(gmp_mul(&gmp_lhs, &gmp_rhs));
        }

        let hz_times = time_fn(&mut || prod_lhs.multiply(&prod_rhs), args.repeats);
        let ref_times = time_fn(&mut || gmp_mul(&gmp_lhs, &gmp_rhs), args.repeats);

        let hz_product = prod_lhs.multiply(&prod_rhs);
        let hz_result = hz_product.to_integer();
        let gmp_result = gmp_mul(&gmp_lhs, &gmp_rhs);
        assert_eq!(hz_result, gmp_result, "bridge niche mismatch");

        let hz_median = median(&hz_times);
        let ref_median = median(&ref_times);
        data_points.push(DataPoint {
            input_description: format!(
                "level={level}, lhs_index={lhs_index}, rhs_index={rhs_index}"
            ),
            input_size_bits: gmp_result.significant_bits() as u64,
            hz_time_ns: hz_times,
            native_time_ns: None,
            ref_time_ns: ref_times,
            hz_median_ns: hz_median,
            native_median_ns: None,
            ref_median_ns: ref_median,
            speedup_ratio: ref_median as f64 / hz_median.max(1) as f64,
            native_speedup_ratio: None,
            hz_density_rho: Some((lhs.density() + rhs.density()) * 0.5),
            hz_active_levels: Some((lhs.active_levels() + rhs.active_levels()) / 2),
            extra: serde_json::json!({
                "route": format!("{:?}", decision.route),
                "reason": decision.reason,
                "carrier": match hz_product {
                    ProductionNumber::BasePhi { .. } => "BasePhi",
                    ProductionNumber::Hybrid { .. } => "Hybrid",
                    ProductionNumber::Integer(_) => "Integer",
                },
                "lhs_eval_bits": gmp_lhs.significant_bits(),
                "rhs_eval_bits": gmp_rhs.significant_bits(),
            }),
        });
    }

    let faster_count = data_points
        .iter()
        .filter(|dp| dp.hz_median_ns < dp.ref_median_ns)
        .count();
    let decision = if faster_count == data_points.len() {
        "advantage"
    } else if faster_count > 0 {
        "mixed"
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
            "faster_count": faster_count,
            "claim": "Measures the curated trigger domain of the deferred base-phi bridge rather than generic sparse-origin inputs",
        }),
    };

    write_result(&result, "results/exp14_base_phi_bridge_niche.json");
    println!("Wrote results/exp14_base_phi_bridge_niche.json");
}

fn single_payload(level: u32, fib_index: u32) -> HybridNumber {
    let mut levels = BTreeMap::new();
    levels.insert(level, vec![fib_index]);
    HybridNumber { levels }
}

fn gmp_mul(a: &Integer, b: &Integer) -> Integer {
    a.clone() * b.clone()
}
