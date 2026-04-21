use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{
    get_machine_info, median, now_utc, time_fn, write_result, BenchConfig, BenchResult, DataPoint,
};
use hybrid_zeckendorf_bench::sparse::{
    construct_sparse_hz, density_from_value, validate_sparse_hz_with_value,
};
use hybrid_zeckendorf_bench::{FlatHybridNumber, HybridNumber};
use rug::Integer;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![10u32, 100, 1000])]
    counts: Vec<u32>,
    #[arg(long, value_delimiter = ',', default_values_t = vec![1e-4f64, 1e-3, 1e-2])]
    rhos: Vec<f64>,
    #[arg(long = "bits", alias = "bit-size", default_value_t = 1_000_000u64)]
    bits: u64,
    #[arg(long, default_value_t = 10)]
    repeats: usize,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
}

fn main() {
    let args = Args::parse();
    let counts = if args.full {
        vec![10, 100, 1000]
    } else {
        args.counts.clone()
    };
    let rhos = if args.full {
        vec![1e-4, 1e-3, 1e-2]
    } else {
        args.rhos.clone()
    };
    let bits = if args.full { 1_000_000 } else { args.bits };
    let repeats = if args.full { 10 } else { args.repeats };
    let warmup = if args.full { 1 } else { args.warmup };

    let config = BenchConfig {
        experiment_id: "exp5_lazy_accum".to_string(),
        description: "Lazy accumulation benchmark with legacy/native HZ totals and GMP".to_string(),
        parameters: serde_json::json!({
            "bit_size": bits,
            "counts": counts,
            "target_rhos": rhos,
            "repeats": repeats,
            "warmup": warmup,
            "constraint_h12_no_intermediate_normalize": true,
            "h_bench_dual_path": true
        }),
        repeats,
        warmup_repeats: warmup,
    };

    let mut rng = rand::thread_rng();
    let mut data_points = Vec::<DataPoint>::new();

    for &count in &counts {
        for &target_rho in &rhos {
            let summands: Vec<HybridNumber> = (0..count)
                .map(|_| construct_sparse_hz(bits, target_rho, &mut rng))
                .collect();
            let flat_summands: Vec<FlatHybridNumber> =
                summands.iter().map(FlatHybridNumber::from_legacy).collect();
            let gmp_summands: Vec<Integer> = summands.iter().map(HybridNumber::eval).collect();
            for (s, v) in summands.iter().zip(gmp_summands.iter()) {
                validate_sparse_hz_with_value(bits, target_rho, s, v)
                    .expect("summand sparse validation");
                let rho = density_from_value(s, v);
                assert!(rho < 2.0 * target_rho, "density control failed for summand");
            }

            let avg_rho = summands
                .iter()
                .zip(gmp_summands.iter())
                .map(|(s, v)| density_from_value(s, v))
                .sum::<f64>()
                / summands.len() as f64;
            let avg_levels = (summands
                .iter()
                .map(|s| s.active_levels() as u64)
                .sum::<u64>()
                / summands.len() as u64) as u32;

            for _ in 0..warmup {
                std::hint::black_box(accumulate_eager_legacy(&summands));
                std::hint::black_box(accumulate_lazy_no_normalize_legacy(&summands));
                std::hint::black_box(accumulate_lazy_legacy(&summands));
                std::hint::black_box(accumulate_eager_native(&flat_summands));
                std::hint::black_box(accumulate_lazy_no_normalize_native(&flat_summands));
                std::hint::black_box(accumulate_lazy_native(&flat_summands));
                std::hint::black_box(accumulate_gmp(&gmp_summands));
            }

            let eager_times = time_fn(&mut || accumulate_eager_legacy(&summands), repeats);
            let concat_times = time_fn(
                &mut || accumulate_lazy_no_normalize_legacy(&summands),
                repeats,
            );
            let concat_seed = accumulate_lazy_no_normalize_legacy(&summands);
            let normalize_times = time_fn(
                &mut || {
                    let mut acc = concat_seed.clone();
                    acc.normalize_legacy();
                    acc
                },
                repeats,
            );
            let lazy_times = time_fn(&mut || accumulate_lazy_legacy(&summands), repeats);

            let native_eager_times =
                time_fn(&mut || accumulate_eager_native(&flat_summands), repeats);
            let native_concat_times = time_fn(
                &mut || accumulate_lazy_no_normalize_native(&flat_summands),
                repeats,
            );
            let native_concat_seed = accumulate_lazy_no_normalize_native(&flat_summands);
            let native_normalize_times = time_fn(
                &mut || {
                    let mut acc = native_concat_seed.clone();
                    acc.normalize_native();
                    acc
                },
                repeats,
            );
            let native_lazy_times =
                time_fn(&mut || accumulate_lazy_native(&flat_summands), repeats);
            let gmp_times = time_fn(&mut || accumulate_gmp(&gmp_summands), repeats);

            let eager_eval = accumulate_eager_legacy(&summands).eval();
            let concat_eval = accumulate_lazy_no_normalize_legacy(&summands).eval();
            let lazy_eval = accumulate_lazy_legacy(&summands).eval();
            let native_eager_eval = accumulate_eager_native(&flat_summands).eval();
            let native_concat_eval = accumulate_lazy_no_normalize_native(&flat_summands).eval();
            let native_lazy_eval = accumulate_lazy_native(&flat_summands).eval();
            let gmp_eval = accumulate_gmp(&gmp_summands);
            assert_eq!(eager_eval, gmp_eval, "legacy eager/gmp mismatch");
            assert_eq!(concat_eval, gmp_eval, "legacy concat-only/gmp mismatch");
            assert_eq!(lazy_eval, gmp_eval, "legacy lazy/gmp mismatch");
            assert_eq!(native_eager_eval, gmp_eval, "native eager/gmp mismatch");
            assert_eq!(
                native_concat_eval, gmp_eval,
                "native concat-only/gmp mismatch"
            );
            assert_eq!(native_lazy_eval, gmp_eval, "native lazy/gmp mismatch");

            let eager_median = median(&eager_times);
            let concat_median = median(&concat_times);
            let normalize_median = median(&normalize_times);
            let lazy_median = median(&lazy_times);
            let native_eager_median = median(&native_eager_times);
            let native_concat_median = median(&native_concat_times);
            let native_normalize_median = median(&native_normalize_times);
            let native_lazy_median = median(&native_lazy_times);
            let gmp_median = median(&gmp_times);
            data_points.push(DataPoint {
                input_description: format!(
                    "bits={bits}, count={count}, target_rho={target_rho:.6e}"
                ),
                input_size_bits: bits,
                hz_time_ns: lazy_times.clone(),
                native_time_ns: Some(native_lazy_times.clone()),
                ref_time_ns: gmp_times.clone(),
                hz_median_ns: lazy_median,
                native_median_ns: Some(native_lazy_median),
                ref_median_ns: gmp_median,
                speedup_ratio: gmp_median as f64 / lazy_median.max(1) as f64,
                native_speedup_ratio: Some(gmp_median as f64 / native_lazy_median.max(1) as f64),
                hz_density_rho: Some(avg_rho),
                hz_active_levels: Some(avg_levels),
                extra: serde_json::json!({
                    "target_rho": target_rho,
                    "actual_rho": avg_rho,
                    "accum_count": count,
                    "gmp_time_ns": gmp_times,
                    "gmp_median_ns": gmp_median,
                    "eager_time_ns": eager_times,
                    "eager_median_ns": eager_median,
                    "concat_only_time_ns": concat_times,
                    "concat_only_median_ns": concat_median,
                    "normalize_only_time_ns": normalize_times,
                    "normalize_only_median_ns": normalize_median,
                    "concat_plus_normalize_time_ns": lazy_times,
                    "concat_plus_normalize_median_ns": lazy_median,
                    "hz_lazy_time_ns": lazy_times,
                    "hz_lazy_median_ns": lazy_median,
                    "hz_native_eager_time_ns": native_eager_times,
                    "hz_native_eager_median_ns": native_eager_median,
                    "hz_native_concat_only_time_ns": native_concat_times,
                    "hz_native_concat_only_median_ns": native_concat_median,
                    "hz_native_normalize_only_time_ns": native_normalize_times,
                    "hz_native_normalize_only_median_ns": native_normalize_median,
                    "hz_native_lazy_time_ns": native_lazy_times,
                    "hz_native_lazy_median_ns": native_lazy_median,
                    "lazy_vs_eager_ratio": eager_median as f64 / lazy_median.max(1) as f64,
                    "lazy_vs_gmp_ratio": gmp_median as f64 / lazy_median.max(1) as f64,
                    "native_lazy_vs_gmp_ratio": gmp_median as f64 / native_lazy_median.max(1) as f64,
                    "native_lazy_vs_legacy_lazy_ratio": lazy_median as f64 / native_lazy_median.max(1) as f64,
                    "eager_vs_gmp_ratio": gmp_median as f64 / eager_median.max(1) as f64,
                    "concat_only_vs_gmp_ratio": gmp_median as f64 / concat_median.max(1) as f64
                }),
            });
        }
    }

    let faster_lazy_vs_gmp = data_points
        .iter()
        .filter(|dp| dp.hz_median_ns < dp.ref_median_ns)
        .count();
    let faster_native_vs_gmp = data_points
        .iter()
        .filter(|dp| {
            dp.native_median_ns
                .map(|x| x < dp.ref_median_ns)
                .unwrap_or(false)
        })
        .count();
    let key_point = data_points
        .iter()
        .find(|dp| {
            dp.extra
                .get("accum_count")
                .and_then(|v| v.as_u64())
                .map(|x| x == 100)
                .unwrap_or(false)
                && dp
                    .extra
                    .get("target_rho")
                    .and_then(|v| v.as_f64())
                    .map(|x| (x - 1e-3).abs() < 1e-12)
                    .unwrap_or(false)
        })
        .cloned();
    let lazy_vs_gmp_at_key = key_point.as_ref().map(|dp| dp.speedup_ratio).unwrap_or(0.0);
    let native_vs_gmp_at_key = key_point
        .as_ref()
        .and_then(|dp| dp.native_speedup_ratio)
        .unwrap_or(0.0);
    let native_vs_legacy_at_key = key_point
        .as_ref()
        .and_then(|dp| {
            dp.extra
                .get("native_lazy_vs_legacy_lazy_ratio")
                .and_then(|v| v.as_f64())
        })
        .unwrap_or(0.0);

    let decision = if native_vs_gmp_at_key > 1.0 {
        "native_beats_gmp"
    } else if native_vs_legacy_at_key > 1.0 {
        "native_beats_legacy_not_gmp"
    } else if lazy_vs_gmp_at_key > 1.0 {
        "legacy_beats_gmp_only"
    } else {
        "no_hz_advantage"
    };

    let result = BenchResult {
        config,
        data_points,
        timestamp: now_utc(),
        machine_info: get_machine_info(),
        decision: Some(decision.to_string()),
        summary: serde_json::json!({
            "decision": decision,
            "key_ratio_count100_rho1e3_lazy_vs_gmp": lazy_vs_gmp_at_key,
            "key_ratio_count100_rho1e3_native_vs_gmp": native_vs_gmp_at_key,
            "key_ratio_count100_rho1e3_native_vs_legacy": native_vs_legacy_at_key,
            "legacy_faster_than_gmp_configs": faster_lazy_vs_gmp,
            "native_faster_than_gmp_configs": faster_native_vs_gmp,
            "total_configs": counts.len() * rhos.len(),
        }),
    };

    write_result(&result, "results/exp5_lazy_accum.json");
    println!(
        "Wrote results/exp5_lazy_accum.json with {} data points",
        result.data_points.len()
    );
}

fn accumulate_eager_legacy(summands: &[HybridNumber]) -> HybridNumber {
    let mut acc = HybridNumber::empty();
    for s in summands {
        acc = acc.add_legacy(s);
    }
    acc
}

fn accumulate_lazy_legacy(summands: &[HybridNumber]) -> HybridNumber {
    let mut acc = accumulate_lazy_no_normalize_legacy(summands);
    acc.normalize_legacy();
    acc
}

fn accumulate_lazy_no_normalize_legacy(summands: &[HybridNumber]) -> HybridNumber {
    let mut acc = HybridNumber::empty();
    for s in summands {
        acc = acc.add_lazy_legacy(s);
    }
    acc
}

fn accumulate_eager_native(summands: &[FlatHybridNumber]) -> FlatHybridNumber {
    let mut acc = FlatHybridNumber::empty();
    for s in summands {
        acc = acc.add_native(s);
    }
    acc
}

fn accumulate_lazy_native(summands: &[FlatHybridNumber]) -> FlatHybridNumber {
    let mut acc = accumulate_lazy_no_normalize_native(summands);
    acc.normalize_native();
    acc
}

fn accumulate_lazy_no_normalize_native(summands: &[FlatHybridNumber]) -> FlatHybridNumber {
    let mut acc = FlatHybridNumber::empty();
    for s in summands {
        acc = acc.add_lazy(s);
    }
    acc
}

fn accumulate_gmp(summands: &[Integer]) -> Integer {
    let mut acc = Integer::from(0);
    for s in summands {
        acc += s;
    }
    acc
}
