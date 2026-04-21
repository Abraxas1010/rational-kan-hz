use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{
    get_machine_info, now_utc, write_result, BenchConfig, BenchResult, DataPoint,
};
use hybrid_zeckendorf_bench::density::{density_trend_nonincreasing, simple_linear_fit};
use hybrid_zeckendorf_bench::normalization::log1000_floor;
use hybrid_zeckendorf_bench::HybridNumber;
use rand::RngCore;
use rug::{Complete, Integer};
use std::collections::BTreeMap;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![3u32, 5, 10, 20, 50, 100, 200, 500, 1000])]
    digit_counts: Vec<u32>,
    #[arg(long, default_value_t = 50)]
    samples_per_size: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
}

fn main() {
    let args = Args::parse();
    let digit_counts = if args.full {
        vec![3u32, 5, 10, 20, 50, 100, 200, 500, 1000]
    } else {
        args.digit_counts.clone()
    };

    let config = BenchConfig {
        experiment_id: "exp3_density".to_string(),
        description: "Density scaling verification against proved bounds".to_string(),
        parameters: serde_json::json!({
            "digit_counts": digit_counts,
            "samples_per_size": args.samples_per_size,
        }),
        repeats: 1,
        warmup_repeats: 0,
    };

    let mut data_points = Vec::new();
    let mut rng = rand::thread_rng();
    let mut by_size: BTreeMap<u32, Vec<(u32, u32, f64)>> = BTreeMap::new();
    let mut fit_x = Vec::<f64>::new();
    let mut fit_y = Vec::<f64>::new();

    for &digits in &digit_counts {
        for sample_idx in 0..args.samples_per_size {
            let n = random_integer_with_digits(digits, &mut rng);
            let hz = HybridNumber::from_integer(&n);

            let active = hz.active_levels();
            let support = hz.support_card();
            let proved_bound = log1000_floor(&n) + 2;
            let density_rho = hz.density();
            let gap = proved_bound as i64 - active as i64;

            assert!(
                active <= proved_bound,
                "BOUND VIOLATION: active_levels={active} > proved_bound={proved_bound} for digits={digits}"
            );

            fit_x.push(log1000_floor(&n) as f64);
            fit_y.push(active as f64);
            by_size
                .entry(digits)
                .or_default()
                .push((active, proved_bound, density_rho));

            data_points.push(DataPoint {
                input_description: format!("digits={digits}, sample={sample_idx}"),
                input_size_bits: n.significant_bits() as u64,
                hz_time_ns: vec![0],
                native_time_ns: None,
                ref_time_ns: vec![0],
                hz_median_ns: 0,
                native_median_ns: None,
                ref_median_ns: 0,
                speedup_ratio: 1.0,
                native_speedup_ratio: None,
                hz_density_rho: Some(density_rho),
                hz_active_levels: Some(active),
                extra: serde_json::json!({
                    "digit_count": digits,
                    "sample_index": sample_idx,
                    "active_levels": active,
                    "support_card": support,
                    "proved_bound": proved_bound,
                    "gap": gap,
                    "bound_violation": active > proved_bound,
                }),
            });
        }
    }

    let mut class_summary = Vec::new();
    let mut density_means = Vec::<(u32, f64)>::new();
    let mut max_gap_large = i64::MIN;

    for (digits, rows) in &by_size {
        let n = rows.len() as f64;
        let mean_active = rows.iter().map(|r| r.0 as f64).sum::<f64>() / n;
        let mean_bound = rows.iter().map(|r| r.1 as f64).sum::<f64>() / n;
        let mean_density = rows.iter().map(|r| r.2).sum::<f64>() / n;
        let max_gap = rows
            .iter()
            .map(|r| r.1 as i64 - r.0 as i64)
            .max()
            .unwrap_or(0);
        if *digits > 100 {
            max_gap_large = max_gap_large.max(max_gap);
        }
        density_means.push((*digits, mean_density));
        class_summary.push(serde_json::json!({
            "digit_count": digits,
            "samples": rows.len(),
            "mean_active_levels": mean_active,
            "mean_proved_bound": mean_bound,
            "mean_density": mean_density,
            "max_gap": max_gap,
        }));
    }

    let bound_violations = data_points
        .iter()
        .filter(|p| {
            p.extra
                .get("bound_violation")
                .and_then(|x| x.as_bool())
                .unwrap_or(false)
        })
        .count();
    let bound_holds = bound_violations == 0;
    let bound_tight = max_gap_large != i64::MIN && max_gap_large < 5;
    let density_decreases = density_trend_nonincreasing(&density_means);
    let (alpha, beta) = simple_linear_fit(&fit_x, &fit_y).unwrap_or((f64::NAN, f64::NAN));

    let decision = if bound_holds && bound_tight && density_decreases {
        "bound_holds_tight_and_density_decreases"
    } else if bound_holds {
        "bound_holds"
    } else {
        "violation"
    };

    let result = BenchResult {
        config,
        data_points,
        timestamp: now_utc(),
        machine_info: get_machine_info(),
        decision: Some(decision.to_string()),
        summary: serde_json::json!({
            "decision": decision,
            "bound_violations": bound_violations,
            "bound_holds": bound_holds,
            "bound_tight": bound_tight,
            "max_gap_for_digits_gt_100": if max_gap_large == i64::MIN { serde_json::Value::Null } else { serde_json::json!(max_gap_large) },
            "density_decreases": density_decreases,
            "linear_fit_active_vs_log1000": {"alpha": alpha, "beta": beta},
            "size_class_summary": class_summary,
            "decision_rules": {
                "bound_holds": "active_levels <= proved_bound for 100% inputs",
                "bound_tight": "max gap < 5 for inputs with digits > 100",
                "density_decreases": "mean density non-increasing across >=3 consecutive size classes"
            }
        }),
    };

    write_result(&result, "results/exp3_density.json");
    println!(
        "Wrote results/exp3_density.json with {} data points",
        result.data_points.len()
    );
}

fn random_integer_with_digits(digits: u32, rng: &mut impl RngCore) -> Integer {
    if digits <= 1 {
        return Integer::from(1);
    }
    let mut s = String::with_capacity(digits as usize);
    s.push(char::from(b'1' + (rng.next_u32() % 9) as u8));
    for _ in 1..digits {
        s.push(char::from(b'0' + (rng.next_u32() % 10) as u8));
    }
    Integer::parse(s).expect("valid random decimal").complete()
}
