use clap::Parser;
use hybrid_zeckendorf_bench::active_set::{replay_trace_with_options, ReplayOptions, TraceSample};
use serde_json::json;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    trace: PathBuf,
    #[arg(long, default_value_t = 1)]
    dense_levels: u32,
    #[arg(long, default_value_t = false)]
    signed: bool,
    #[arg(long, default_value_t = false)]
    adaptive: bool,
    #[arg(long, default_value_t = 0.01)]
    adaptive_threshold: f64,
    #[arg(long)]
    row_id: Option<String>,
}

fn main() {
    let args = Args::parse();
    let trace_text = match fs::read_to_string(&args.trace) {
        Ok(text) => text,
        Err(err) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "error",
                    "row_id": args.row_id,
                    "reason": format!("failed to read trace file: {err}"),
                }))
                .expect("json")
            );
            std::process::exit(1);
        }
    };
    let samples: Vec<TraceSample> = match serde_json::from_str(&trace_text) {
        Ok(samples) => samples,
        Err(err) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "error",
                    "row_id": args.row_id,
                    "reason": format!("failed to parse trace json: {err}"),
                }))
                .expect("json")
            );
            std::process::exit(1);
        }
    };
    let options = ReplayOptions {
        dense_level_limit: args.dense_levels,
        signed_representation: args.signed,
        adaptive_dense_levels: args.adaptive,
        adaptive_density_threshold: args.adaptive_threshold,
    };
    match replay_trace_with_options(&samples, &options) {
        Ok(summary) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "ok",
                    "row_id": args.row_id,
                    "summary": summary,
                }))
                .expect("json")
            );
        }
        Err(reason) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "error",
                    "row_id": args.row_id,
                    "reason": reason,
                }))
                .expect("json")
            );
            std::process::exit(1);
        }
    }
}
