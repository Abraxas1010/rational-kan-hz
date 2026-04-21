use clap::Parser;
use hybrid_zeckendorf_bench::weight::weight;
use serde_json::json;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = 4)]
    max_level: u32,
}

fn main() {
    let args = Args::parse();
    let weights: Vec<String> = (0..=args.max_level).map(|level| weight(level).to_string()).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "max_level": args.max_level,
            "weights": weights,
        }))
        .expect("json")
    );
}
