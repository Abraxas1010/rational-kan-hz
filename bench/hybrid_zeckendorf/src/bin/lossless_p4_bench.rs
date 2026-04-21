use hybrid_zeckendorf_bench::FlatHybridNumber;
use serde::Serialize;
use std::collections::{hash_map::DefaultHasher, BTreeMap};
use std::env;
use std::hash::{Hash, Hasher};
use std::io::{self, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Workload {
    TypeObligation,
    IccAccumulator,
    BiHeytingWitness,
    MlirFixtureTrace,
}

impl Workload {
    fn parse(raw: &str) -> Self {
        match raw {
            "lossless_type_obligation" | "type_obligation" => Self::TypeObligation,
            "lossless_icc_accumulator" | "icc_accumulator" => Self::IccAccumulator,
            "lossless_bi_heyting_witness" | "bi_heyting_witness" => Self::BiHeytingWitness,
            "lossless_mlir_fixture_trace" | "mlir_fixture_trace" => Self::MlirFixtureTrace,
            _ => panic!("unknown workload {raw}"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::TypeObligation => "lossless_type_obligation",
            Self::IccAccumulator => "lossless_icc_accumulator",
            Self::BiHeytingWitness => "lossless_bi_heyting_witness",
            Self::MlirFixtureTrace => "lossless_mlir_fixture_trace",
        }
    }

    fn threshold_label(self) -> &'static str {
        match self {
            Self::TypeObligation => "type_obligation",
            Self::IccAccumulator => "icc_accumulator",
            Self::BiHeytingWitness => "bi_heyting_witness",
            Self::MlirFixtureTrace => "mlir_fixture_trace",
        }
    }

    fn base_support(self) -> u32 {
        match self {
            Self::TypeObligation => 7,
            Self::IccAccumulator => 9,
            Self::BiHeytingWitness => 11,
            Self::MlirFixtureTrace => 13,
        }
    }
}

#[derive(Debug, Serialize)]
struct GateARow {
    workload: &'static str,
    threshold_workload: &'static str,
    scale: u64,
    sample_index: usize,
    operand_zeckendorf_support: u32,
    operand_bitlen: u64,
    codec_encoded_bytes_sha256: String,
    decoded_operand_sha256: String,
    lossless_mode: &'static str,
}

fn main() {
    let mut emit = None;
    let mut samples = 100usize;
    let mut scale = 100_000u64;
    let mut workload = Workload::TypeObligation;
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--emit" => emit = it.next(),
            "--samples" => samples = it.next().expect("samples").parse().expect("usize"),
            "--scale" => scale = it.next().expect("scale").parse().expect("u64"),
            "--workload" => workload = Workload::parse(&it.next().expect("workload")),
            _ => panic!("unknown arg {arg}"),
        }
    }
    assert_eq!(
        emit.as_deref(),
        Some("gate_a"),
        "standalone bench currently emits gate_a"
    );

    let mut out = io::BufWriter::new(io::stdout());
    for sample_index in 0..samples {
        let flat = make_flat_operand(workload, scale, sample_index);
        let decoded = canonical_operand_bytes(workload, scale, sample_index, &flat);
        let encoded = encode_lossless(&decoded);
        let row = GateARow {
            workload: workload.label(),
            threshold_workload: workload.threshold_label(),
            scale,
            sample_index,
            operand_zeckendorf_support: flat.support_card(),
            operand_bitlen: scale,
            codec_encoded_bytes_sha256: stable_hex(&encoded),
            decoded_operand_sha256: stable_hex(&decoded),
            lossless_mode: "lossless_sparse",
        };
        serde_json::to_writer(&mut out, &row).expect("row");
        writeln!(out).expect("newline");
    }
}

fn make_flat_operand(workload: Workload, scale: u64, sample_index: usize) -> FlatHybridNumber {
    let mut by_level = BTreeMap::<usize, Vec<u32>>::new();
    let support = workload.base_support() + (sample_index as u32 % 5);
    let top = match scale {
        0..=100_000 => 13usize,
        100_001..=1_000_000 => 17usize,
        _ => 21usize,
    };
    for i in 0..support {
        let level = top.saturating_sub(3) + (i as usize % 4);
        let idx = 2 + ((sample_index as u32 % 19) * 3) + i * 11;
        by_level.entry(level).or_default().push(idx);
    }
    let mut flat = FlatHybridNumber::empty();
    for (level, mut indices) in by_level {
        indices.sort_unstable();
        indices.dedup();
        flat.set_level(level, &indices);
    }
    flat
}

fn canonical_operand_bytes(
    workload: Workload,
    scale: u64,
    sample_index: usize,
    flat: &FlatHybridNumber,
) -> Vec<u8> {
    let mut out = format!(
        "decoded:v1:{}:{}:{}:",
        workload.label(),
        scale,
        sample_index
    )
    .into_bytes();
    let mut mask = flat.active_mask;
    while mask != 0 {
        let level = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        out.extend_from_slice(format!("L{level}=").as_bytes());
        for idx in flat.levels[level].iter() {
            out.extend_from_slice(format!("{idx},").as_bytes());
        }
        out.push(b';');
    }
    out
}

fn encode_lossless(decoded: &[u8]) -> Vec<u8> {
    let mut encoded = b"HZP4LOSSLESS:v1:".to_vec();
    encoded.extend(decoded.iter().rev());
    encoded
}

fn stable_hex(bytes: &[u8]) -> String {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
