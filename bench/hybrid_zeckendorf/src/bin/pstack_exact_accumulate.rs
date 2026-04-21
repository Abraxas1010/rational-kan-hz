use clap::Parser;
use hybrid_zeckendorf_bench::flat_hybrid::BatchQuery;
use hybrid_zeckendorf_bench::{FlatHybridNumber, HybridNumber};
use rug::Complete;
use rug::Integer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

#[derive(Parser, Debug)]
#[command(about = "Exact lazy Hybrid Zeckendorf accumulation over signed integer groups")]
struct Args {
    #[arg(long, default_value = "hz_lazy")]
    backend: String,
}

#[derive(Debug, Deserialize)]
struct GroupIn {
    values: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Request {
    groups: Vec<GroupIn>,
}

#[derive(Debug, Serialize)]
struct GroupOut {
    sum: String,
    support_card: u32,
    active_levels: u32,
    witness_bytes: usize,
    nonzero_terms: usize,
}

#[derive(Debug, Serialize)]
struct Response {
    backend: String,
    results: Vec<GroupOut>,
}

struct SignedLazyAccumulator {
    pos: FlatHybridNumber,
    neg: FlatHybridNumber,
    witness_bytes: usize,
    nonzero_terms: usize,
}

impl SignedLazyAccumulator {
    fn new() -> Self {
        Self {
            pos: FlatHybridNumber::empty(),
            neg: FlatHybridNumber::empty(),
            witness_bytes: 0,
            nonzero_terms: 0,
        }
    }

    fn add_signed_decimal(&mut self, raw: &str, cache: &mut HashMap<String, FlatHybridNumber>) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return;
        }
        let value = Integer::parse(trimmed).expect("valid integer").complete();
        if value == 0 {
            return;
        }
        let sign_negative = value < 0;
        let magnitude = value.abs();
        let magnitude_key = magnitude.to_string();
        let flat = cache
            .entry(magnitude_key)
            .or_insert_with(|| FlatHybridNumber::from_legacy(&HybridNumber::from_integer(&magnitude)))
            .clone();
        if sign_negative {
            self.neg.add_lazy_mut(&flat);
        } else {
            self.pos.add_lazy_mut(&flat);
        }
        self.witness_bytes += trimmed.len();
        self.nonzero_terms += 1;
    }

    fn finalize(mut self) -> GroupOut {
        let queries = [
            BatchQuery::FullEval,
            BatchQuery::SupportCardinality,
            BatchQuery::ActiveLevels,
        ];
        let pos_values = self.pos.batch_readout(&queries);
        let neg_values = self.neg.batch_readout(&queries);
        let sum = pos_values[0].clone() - neg_values[0].clone();
        let support_card = pos_values[1].to_u32_wrapping() + neg_values[1].to_u32_wrapping();
        let active_levels = pos_values[2].to_u32_wrapping() + neg_values[2].to_u32_wrapping();
        GroupOut {
            sum: sum.to_string(),
            support_card,
            active_levels,
            witness_bytes: self.witness_bytes,
            nonzero_terms: self.nonzero_terms,
        }
    }
}

fn main() {
    let args = Args::parse();
    assert_eq!(
        args.backend, "hz_lazy",
        "only hz_lazy backend is supported by this exact accumulator"
    );
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut cache: HashMap<String, FlatHybridNumber> = HashMap::new();
    for line in stdin.lock().lines() {
        let line = line.expect("stdin line");
        if line.trim().is_empty() {
            continue;
        }
        let request: Request = serde_json::from_str(&line).expect("valid request json");

        let mut results = Vec::with_capacity(request.groups.len());
        for group in request.groups {
            let mut acc = SignedLazyAccumulator::new();
            for value in group.values {
                acc.add_signed_decimal(&value, &mut cache);
            }
            results.push(acc.finalize());
        }

        let response = Response {
            backend: args.backend.clone(),
            results,
        };
        serde_json::to_writer(&mut stdout, &response).expect("serialize response");
        writeln!(&mut stdout).expect("newline");
        stdout.flush().expect("flush");
    }
}
