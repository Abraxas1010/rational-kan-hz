use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{
    get_machine_info, median, now_utc, time_fn, write_result, BenchConfig, BenchResult, DataPoint,
};
use hybrid_zeckendorf_bench::reference::gmp_wrapper::polymul_schoolbook_gmp;
use hybrid_zeckendorf_bench::HybridNumber;
use rand::RngCore;
use rug::{Complete, Integer};

#[derive(Debug, Clone)]
struct GiantPoly {
    coeffs: Vec<Integer>,
}

impl GiantPoly {
    fn random(
        degree: usize,
        digits: u64,
        rng: &mut impl RngCore,
        max_generate_digits: u64,
    ) -> Option<Self> {
        let mut coeffs = Vec::with_capacity(degree + 1);
        for _ in 0..=degree {
            coeffs.push(generate_integer_with_digits(
                digits,
                rng,
                max_generate_digits,
            )?);
        }
        Some(Self { coeffs })
    }

    fn multiply_gmp(&self, other: &GiantPoly) -> GiantPoly {
        GiantPoly {
            coeffs: polymul_schoolbook_gmp(&self.coeffs, &other.coeffs),
        }
    }

    fn multiply_hz(&self, other: &GiantPoly) -> GiantPoly {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return GiantPoly { coeffs: Vec::new() };
        }
        let mut out = vec![Integer::from(0); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, ai) in self.coeffs.iter().enumerate() {
            for (j, bj) in other.coeffs.iter().enumerate() {
                let hz_a = HybridNumber::from_integer(ai);
                let hz_b = HybridNumber::from_integer(bj);
                let prod = hz_a.multiply(&hz_b).eval();
                let curr = HybridNumber::from_integer(&out[i + j]);
                let updated = curr.add(&HybridNumber::from_integer(&prod));
                out[i + j] = updated.eval();
            }
        }
        GiantPoly { coeffs: out }
    }

    fn sample_density(&self) -> f64 {
        let take = self.coeffs.iter().take(3);
        let mut sum = 0.0;
        let mut count = 0.0;
        for c in take {
            let hz = HybridNumber::from_integer(c);
            sum += hz.density();
            count += 1.0;
        }
        if count == 0.0 {
            0.0
        } else {
            sum / count
        }
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    degree: Option<u32>,
    #[arg(long)]
    coeff_digits: Option<u64>,
    #[arg(long, default_value_t = 10)]
    repeats: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = false)]
    full: bool,
    #[arg(long, default_value_t = 1000000)]
    max_generate_digits: u64,
    #[arg(long, default_value_t = 200000000)]
    max_complexity_budget: u128,
}

fn main() {
    let args = Args::parse();
    let configs = choose_configs(&args);

    let config = BenchConfig {
        experiment_id: "exp2_polymul".to_string(),
        description: "Polynomial multiplication with giant coefficients: HZ vs GMP".to_string(),
        parameters: serde_json::json!({
            "configs": configs,
            "repeats_per_config": args.repeats,
            "warmup": args.warmup,
            "max_generate_digits": args.max_generate_digits,
            "max_complexity_budget": args.max_complexity_budget,
            "algorithm": "schoolbook_on_both_sides",
        }),
        repeats: args.repeats,
        warmup_repeats: args.warmup,
    };

    let mut data_points = Vec::new();
    let mut skipped = Vec::new();
    let mut rng = rand::thread_rng();

    for (degree, digits) in configs {
        let complexity = (degree as u128 + 1)
            * (degree as u128 + 1)
            * (digits.min(args.max_generate_digits) as u128);
        if complexity > args.max_complexity_budget {
            skipped.push(serde_json::json!({
                "degree": degree,
                "coeff_digits": digits,
                "reason": "complexity_budget_exceeded",
                "estimated_complexity": complexity,
            }));
            continue;
        }

        let p = match GiantPoly::random(degree as usize, digits, &mut rng, args.max_generate_digits)
        {
            Some(poly) => poly,
            None => {
                skipped.push(serde_json::json!({
                    "degree": degree,
                    "coeff_digits": digits,
                    "reason": "digit_count_not_supported_or_infeasible",
                    "max_generate_digits": args.max_generate_digits,
                }));
                continue;
            }
        };
        let q = match GiantPoly::random(degree as usize, digits, &mut rng, args.max_generate_digits)
        {
            Some(poly) => poly,
            None => {
                skipped.push(serde_json::json!({
                    "degree": degree,
                    "coeff_digits": digits,
                    "reason": "digit_count_not_supported_or_infeasible",
                    "max_generate_digits": args.max_generate_digits,
                }));
                continue;
            }
        };

        for _ in 0..args.warmup {
            std::hint::black_box(p.multiply_gmp(&q));
            std::hint::black_box(p.multiply_hz(&q));
        }

        let ref_times = time_fn(&mut || p.multiply_gmp(&q), args.repeats);
        let hz_times = time_fn(&mut || p.multiply_hz(&q), args.repeats);

        let gmp = p.multiply_gmp(&q);
        let hz = p.multiply_hz(&q);
        assert_eq!(
            gmp.coeffs, hz.coeffs,
            "Polynomial product mismatch at degree={degree}, digits={digits}"
        );

        let hz_median = median(&hz_times);
        let ref_median = median(&ref_times);
        data_points.push(DataPoint {
            input_description: format!("degree={degree}, coeff_digits={digits}"),
            input_size_bits: gmp
                .coeffs
                .first()
                .map(|c| c.significant_bits() as u64)
                .unwrap_or(0),
            hz_time_ns: hz_times,
            native_time_ns: None,
            ref_time_ns: ref_times,
            hz_median_ns: hz_median,
            native_median_ns: None,
            ref_median_ns: ref_median,
            speedup_ratio: ref_median as f64 / hz_median as f64,
            native_speedup_ratio: None,
            hz_density_rho: Some(p.sample_density()),
            hz_active_levels: None,
            extra: serde_json::json!({
                "degree": degree,
                "coeff_digits": digits,
                "result_coeff_count": gmp.coeffs.len(),
                "normalization_proxy": hz.coeffs.iter().take(3).map(|c| HybridNumber::from_integer(c).support_card()).collect::<Vec<_>>(),
            }),
        });
    }

    let total = data_points.len();
    let faster_count = data_points
        .iter()
        .filter(|d| d.hz_median_ns < d.ref_median_ns)
        .count();
    let slower_2x_count = data_points
        .iter()
        .filter(|d| d.hz_median_ns > d.ref_median_ns.saturating_mul(2))
        .count();
    let all_competitive = total > 0
        && data_points
            .iter()
            .all(|d| d.hz_median_ns <= d.ref_median_ns.saturating_mul(2));

    let decision = if total == 0 {
        "blocked"
    } else if faster_count * 4 >= 3 * total {
        "advantage"
    } else if all_competitive {
        "competitive"
    } else if slower_2x_count * 4 >= 3 * total {
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
            "executed_configs": total,
            "skipped_configs": skipped.len(),
            "faster_count": faster_count,
            "slower_2x_count": slower_2x_count,
            "skipped": skipped,
            "rule": "advantage if >=3/4 faster; competitive if all <=2x slower; disadvantage if >=3/4 slower by >2x"
        }),
    };

    write_result(&result, "results/exp2_polymul.json");
    println!(
        "Wrote results/exp2_polymul.json with {} executed configs",
        result.data_points.len()
    );
}

fn choose_configs(args: &Args) -> Vec<(u32, u64)> {
    if args.full {
        let degrees = [10u32, 50, 100];
        let digits = [1_000_000u64, 1_000_000_000, 1_000_000_000_000];
        return degrees
            .into_iter()
            .flat_map(|d| digits.into_iter().map(move |k| (d, k)))
            .collect();
    }

    if let (Some(degree), Some(coeff_digits)) = (args.degree, args.coeff_digits) {
        return vec![(degree, coeff_digits)];
    }

    vec![(10, 1000)]
}

fn generate_integer_with_digits(
    digits: u64,
    rng: &mut impl RngCore,
    max_generate_digits: u64,
) -> Option<Integer> {
    if digits == 0 || digits > max_generate_digits {
        return None;
    }
    if digits - 1 > u32::MAX as u64 {
        return None;
    }

    // Build an exact-length random decimal with non-zero leading digit.
    let mut s = String::with_capacity(digits as usize);
    s.push(char::from(b'1' + (rng.next_u32() % 9) as u8));
    for _ in 1..digits {
        s.push(char::from(b'0' + (rng.next_u32() % 10) as u8));
    }
    Integer::parse(s).ok().map(|n| n.complete())
}
