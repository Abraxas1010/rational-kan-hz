//! Targeted native-vs-GMP benchmark (skips legacy paths).

use clap::Parser;
use hybrid_zeckendorf_bench::bench_config::{median, time_fn};
use hybrid_zeckendorf_bench::flat_hybrid::MAX_LEVELS;
use hybrid_zeckendorf_bench::sparse::construct_sparse_hz;
use hybrid_zeckendorf_bench::zeckendorf_native::{profile_normalize_flat_native, NormalizeProfile};
use hybrid_zeckendorf_bench::{FlatHybridNumber, HybridNumber};
use rand::{rngs::StdRng, SeedableRng};
use rug::Integer;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_values_t = vec![100_000u64, 500_000, 1_000_000, 5_000_000])]
    bits: Vec<u64>,
    #[arg(long = "counts", alias = "count", value_delimiter = ',', default_values_t = vec![10u32])]
    counts: Vec<u32>,
    #[arg(long, value_delimiter = ',', default_values_t = vec![0.001f64])]
    rho: Vec<f64>,
    #[arg(long, default_value_t = 7)]
    repeats: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 42u64)]
    seed: u64,
    #[arg(long, default_value_t = false)]
    mixed: bool,
    #[arg(long, default_value_t = false)]
    profile_normalize: bool,
}

fn main() {
    let args = Args::parse();

    assert!(
        args.warmup >= 2,
        "--warmup must be >= 2 for reproducible results"
    );

    println!("# base_seed={}", args.seed);
    println!("# mixed={}", args.mixed);
    println!(
        "{:>12} {:>12} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>12}",
        "bits",
        "rho",
        "count",
        "gmp_ns",
        "native_ns",
        "native_mut",
        "concat_ns",
        "concat_mut",
        "clone_ns",
        "norm_ns",
        "norm_only",
        "speedup",
        "speedup_mut",
        "seed",
        "levels",
        "note"
    );

    for &bit_size in &args.bits {
        for &rho in &args.rho {
            for &count in &args.counts {
                let required_levels = estimated_required_levels(bit_size);
                if required_levels > MAX_LEVELS {
                    println!(
                        "{:>12} {:>12.6e} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>12}",
                        bit_size,
                        rho,
                        count,
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        args.seed,
                        required_levels,
                        "skip_levels"
                    );
                    continue;
                }

                let scenario_seed = scenario_seed(args.seed, bit_size, count, rho, args.mixed);
                let mut rng = StdRng::seed_from_u64(scenario_seed);
                let summands = construct_summands(bit_size, count, rho, args.mixed, &mut rng);
                let flat_summands: Vec<FlatHybridNumber> =
                    summands.iter().map(FlatHybridNumber::from_legacy).collect();
                let gmp_summands: Vec<Integer> = summands.iter().map(HybridNumber::eval).collect();

                let avg_levels = summands
                    .iter()
                    .map(|s| s.active_levels() as u64)
                    .sum::<u64>() as f64
                    / summands.len() as f64;

                for _ in 0..args.warmup {
                    std::hint::black_box(accum_native(&flat_summands));
                    std::hint::black_box(accum_native_mut(&flat_summands));
                    std::hint::black_box(accum_gmp(&gmp_summands));
                }

                let native_val = accum_native(&flat_summands).eval();
                let native_mut_val = accum_native_mut(&flat_summands).eval();
                let gmp_val = accum_gmp(&gmp_summands);
                assert_eq!(
                    native_val, gmp_val,
                    "native/gmp mismatch at bits={bit_size} rho={rho:.6e} count={count}"
                );
                assert_eq!(
                    native_mut_val, gmp_val,
                    "native_mut/gmp mismatch at bits={bit_size} rho={rho:.6e} count={count}"
                );

                let native_concat_times =
                    time_fn(&mut || accum_concat_only(&flat_summands), args.repeats);
                let native_concat_mut_times =
                    time_fn(&mut || accum_concat_only_mut(&flat_summands), args.repeats);
                let concat_seed_mut = accum_concat_only_mut(&flat_summands);
                let clone_times = time_fn(&mut || concat_seed_mut.clone(), args.repeats);
                let native_norm_times = time_fn(
                    &mut || {
                        let mut acc = concat_seed_mut.clone();
                        acc.normalize_native();
                        acc
                    },
                    args.repeats,
                );
                let native_norm_only_times = time_normalize_only(&concat_seed_mut, args.repeats);
                let native_times = time_fn(&mut || accum_native(&flat_summands), args.repeats);
                let native_mut_times =
                    time_fn(&mut || accum_native_mut(&flat_summands), args.repeats);
                let gmp_times = time_fn(&mut || accum_gmp(&gmp_summands), args.repeats);

                let gmp_med = median(&gmp_times);
                let native_med = median(&native_times);
                let native_mut_med = median(&native_mut_times);
                let concat_med = median(&native_concat_times);
                let concat_mut_med = median(&native_concat_mut_times);
                let clone_med = median(&clone_times);
                let norm_med = median(&native_norm_times);
                let norm_only_med = median(&native_norm_only_times);
                let speedup = gmp_med as f64 / native_med.max(1) as f64;
                let speedup_mut = gmp_med as f64 / native_mut_med.max(1) as f64;

                println!(
                    "{:>12} {:>12.6e} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12.4} {:>12.4} {:>12} {:>8.1} {:>12}",
                    bit_size,
                    rho,
                    count,
                    gmp_med,
                    native_med,
                    native_mut_med,
                    concat_med,
                    concat_mut_med,
                    clone_med,
                    norm_med,
                    norm_only_med,
                    speedup,
                    speedup_mut,
                    scenario_seed,
                    avg_levels,
                    if args.mixed { "mixed" } else { "-" }
                );

                if args.profile_normalize {
                    let mut prof_input = concat_seed_mut.clone();
                    let profile = profile_normalize_flat_native(&mut prof_input);
                    assert_eq!(
                        prof_input.eval(),
                        gmp_val,
                        "profile normalize/gmp mismatch at bits={bit_size} rho={rho:.6e} count={count}"
                    );
                    print_profile(bit_size, rho, count, scenario_seed, &profile);
                }
            }
        }
    }
}

fn accum_native(summands: &[FlatHybridNumber]) -> FlatHybridNumber {
    let mut acc = FlatHybridNumber::empty();
    for s in summands {
        acc = acc.add_lazy(s);
    }
    acc.normalize_native();
    acc
}

fn accum_native_mut(summands: &[FlatHybridNumber]) -> FlatHybridNumber {
    let mut acc = FlatHybridNumber::empty();
    for s in summands {
        acc.add_lazy_mut(s);
    }
    acc.normalize_native();
    acc
}

fn accum_concat_only(summands: &[FlatHybridNumber]) -> FlatHybridNumber {
    let mut acc = FlatHybridNumber::empty();
    for s in summands {
        acc = acc.add_lazy(s);
    }
    acc
}

fn accum_concat_only_mut(summands: &[FlatHybridNumber]) -> FlatHybridNumber {
    let mut acc = FlatHybridNumber::empty();
    for s in summands {
        acc.add_lazy_mut(s);
    }
    acc
}

fn accum_gmp(summands: &[Integer]) -> Integer {
    let mut acc = Integer::from(0);
    for s in summands {
        acc += s;
    }
    acc
}

fn construct_summands(
    bit_size: u64,
    count: u32,
    rho: f64,
    mixed: bool,
    rng: &mut StdRng,
) -> Vec<HybridNumber> {
    if !mixed {
        return (0..count)
            .map(|_| construct_sparse_hz(bit_size, rho, rng))
            .collect();
    }

    let dense_count = count / 10;
    let sparse_count = count.saturating_sub(dense_count);
    let mut out = Vec::with_capacity(count as usize);
    out.extend((0..sparse_count).map(|_| construct_sparse_hz(bit_size, rho, rng)));
    out.extend((0..dense_count).map(|_| construct_sparse_hz(bit_size, 1e-3, rng)));
    out
}

fn estimated_required_levels(target_bits: u64) -> usize {
    let mut level = 0u32;
    loop {
        let next_bits = level_bits_estimate(level + 1);
        if next_bits >= target_bits.max(1) {
            break;
        }
        level += 1;
        if level > 64 {
            break;
        }
    }
    (level + 1) as usize
}

fn level_bits_estimate(level: u32) -> u64 {
    match level {
        0 => 1,
        1 => 10,
        _ => {
            let exp = 2f64.powi((level - 1) as i32);
            (9.965_784_284_662_087 * exp).round().max(1.0) as u64
        }
    }
}

fn scenario_seed(base_seed: u64, bits: u64, count: u32, rho: f64, mixed: bool) -> u64 {
    let mut x = base_seed
        ^ bits.rotate_left(17)
        ^ u64::from(count).rotate_left(33)
        ^ rho.to_bits().rotate_left(7);
    if mixed {
        x ^= 0x9E37_79B9_7F4A_7C15;
    }
    splitmix64(x)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn time_normalize_only(seed: &FlatHybridNumber, repeats: usize) -> Vec<u64> {
    let mut inputs: Vec<Option<FlatHybridNumber>> =
        (0..repeats).map(|_| Some(seed.clone())).collect();
    let mut next = 0usize;
    time_fn(
        &mut || {
            let mut acc = inputs[next]
                .take()
                .expect("enough prebuilt normalize inputs");
            next += 1;
            acc.normalize_native();
            acc
        },
        repeats,
    )
}

fn print_profile(bits: u64, rho: f64, count: u32, seed: u64, profile: &NormalizeProfile) {
    println!("# normalize_profile bits={bits} rho={rho:.6e} count={count} seed={seed}");
    println!(
        "# {:>5} {:>9} {:>12} {:>8} {:>10} {:>11} {:>9} {:>12} {:>12} {:>8}",
        "level",
        "u64_calls",
        "sparse_calls",
        "skips",
        "carry_inv",
        "carry_units",
        "set_calls",
        "indices_in",
        "indices_out",
        "fuel"
    );
    for (level, stats) in profile.levels.iter().enumerate() {
        if *stats == Default::default() {
            continue;
        }
        println!(
            "# {:>5} {:>9} {:>12} {:>8} {:>10} {:>11} {:>9} {:>12} {:>12} {:>8}",
            level,
            stats.u64_calls,
            stats.sparse_calls,
            stats.canonical_skips,
            stats.carry_invocations,
            stats.carry_units,
            stats.set_calls,
            stats.indices_in,
            stats.indices_out,
            stats.fuel_used
        );
    }
}
