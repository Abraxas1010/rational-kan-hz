use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub experiment_id: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub repeats: usize,
    pub warmup_repeats: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub config: BenchConfig,
    pub data_points: Vec<DataPoint>,
    pub timestamp: String,
    pub machine_info: MachineInfo,
    pub decision: Option<String>,
    pub summary: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub input_description: String,
    pub input_size_bits: u64,
    pub hz_time_ns: Vec<u64>,
    pub native_time_ns: Option<Vec<u64>>,
    pub ref_time_ns: Vec<u64>,
    pub hz_median_ns: u64,
    pub native_median_ns: Option<u64>,
    pub ref_median_ns: u64,
    pub speedup_ratio: f64,
    pub native_speedup_ratio: Option<f64>,
    pub hz_density_rho: Option<f64>,
    pub hz_active_levels: Option<u32>,
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineInfo {
    pub hostname: String,
    pub cpu: String,
    pub os: String,
}

pub fn time_fn<F: FnMut() -> T, T>(f: &mut F, repeats: usize) -> Vec<u64> {
    (0..repeats)
        .map(|_| {
            let start = Instant::now();
            std::hint::black_box(f());
            start.elapsed().as_nanos() as u64
        })
        .collect()
}

pub fn median(times: &[u64]) -> u64 {
    let mut sorted = times.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

pub fn median_f64(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut ys = xs.to_vec();
    ys.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    ys[ys.len() / 2]
}

pub fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= (b.abs() * 1e-9 + 1e-12)
}

pub fn now_utc() -> String {
    Utc::now().to_rfc3339()
}

pub fn get_machine_info() -> MachineInfo {
    let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string());
    let cpu = detect_cpu_name().unwrap_or_else(|| "unknown".to_string());
    let os = std::env::consts::OS.to_string();
    MachineInfo { hostname, cpu, os }
}

pub fn write_result(result: &BenchResult, path: &str) {
    let out_path = Path::new(path);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).expect("failed to create results directory");
    }
    let data = serde_json::to_string_pretty(result).expect("failed to serialize result");
    fs::write(out_path, data).expect("failed to write result JSON");
}

fn detect_cpu_name() -> Option<String> {
    let contents = fs::read_to_string("/proc/cpuinfo").ok()?;
    contents
        .lines()
        .find_map(|line| line.strip_prefix("model name\t: ").map(|s| s.to_string()))
}
