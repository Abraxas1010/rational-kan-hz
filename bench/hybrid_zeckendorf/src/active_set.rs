use crate::normalization::{HybridNumber, SignedHybridNumber};
use crate::weight::weight;
use rug::Integer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Deserialize)]
pub struct TraceSample {
    #[serde(default)]
    pub step: Option<u64>,
    #[serde(rename = "nodesUsed", default)]
    pub nodes_used: Option<u64>,
    #[serde(rename = "activeIds", default)]
    pub active_ids: Option<Vec<u64>>,
}

#[derive(Debug, Clone)]
pub struct ReplayOptions {
    pub dense_level_limit: u32,
    pub signed_representation: bool,
    pub adaptive_dense_levels: bool,
    pub adaptive_density_threshold: f64,
}

impl ReplayOptions {
    pub fn unsigned_static(dense_level_limit: u32) -> Self {
        Self {
            dense_level_limit,
            signed_representation: false,
            adaptive_dense_levels: false,
            adaptive_density_threshold: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum EncodedAddress {
    Dense {
        node_id: u64,
        level: u32,
    },
    Sparse {
        node_id: u64,
        level: u32,
        relative_offset: u64,
        payloads: Option<BTreeMap<u32, Vec<u32>>>,
        signed_payloads: Option<BTreeMap<u32, Vec<i8>>>,
        support_card: u32,
        active_levels: u32,
        unsigned_support_card: u32,
        signed_support_card: u32,
        support_mode: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayLevelSummary {
    pub representation: String,
    pub sample_count: u64,
    pub total_addresses: u64,
    pub mean_addresses_per_sample: f64,
    pub max_addresses_in_sample: u64,
    pub mean_support_card_per_address: Option<f64>,
    pub max_support_card_per_address: Option<u32>,
    pub support_mode: Option<String>,
    pub mean_unsigned_support_card_per_address: Option<f64>,
    pub mean_signed_support_card_per_address: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DensityBucket {
    pub lower: f64,
    pub upper: f64,
    pub count: u64,
    pub fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraceReplaySummary {
    pub sample_count: u64,
    pub active_sample_count: u64,
    pub dense_level_limit: u32,
    pub requested_dense_level_limit: u32,
    pub adaptive_dense_level_limit: Option<u32>,
    pub adaptive_dense_levels: bool,
    pub adaptive_density_threshold: Option<f64>,
    pub signed_representation: bool,
    pub support_card_mode: String,
    pub roundtrip_ok: bool,
    pub total_addresses: u64,
    pub dense_address_count: u64,
    pub sparse_address_count: u64,
    pub sparse_levels_seen: Vec<String>,
    pub max_dense_addresses_in_sample: u64,
    pub max_sparse_addresses_in_sample: u64,
    pub total_support_card: u64,
    pub unsigned_total_support_card: u64,
    pub signed_total_support_card: u64,
    pub mean_support_card_per_sparse_address: Option<f64>,
    pub level_stats: BTreeMap<String, ReplayLevelSummary>,
    pub density_histogram: Vec<DensityBucket>,
    pub per_level_density_histograms: BTreeMap<String, Vec<DensityBucket>>,
}

#[derive(Debug, Default, Clone)]
struct LevelAccumulator {
    sample_count: u64,
    total_addresses: u64,
    max_addresses_in_sample: u64,
    total_support_card: u64,
    max_support_card_per_address: u32,
    unsigned_total_support_card: u64,
    unsigned_max_support_card_per_address: u32,
    signed_total_support_card: u64,
    signed_max_support_card_per_address: u32,
}

pub fn level_start(level: u32) -> Integer {
    if level == 0 {
        Integer::from(0)
    } else {
        weight(level)
    }
}

pub fn level_for_node_id(node_id: u64) -> u32 {
    let target = Integer::from(node_id);
    let mut level = 0u32;
    loop {
        let next_start = weight(level + 1);
        if next_start > target {
            return level;
        }
        level += 1;
    }
}

fn level_end_u64(level: u32) -> u64 {
    weight(level + 1).to_u64().unwrap_or(u64::MAX)
}

fn used_span_in_level(nodes_used: u64, level: u32) -> u64 {
    let start = level_start(level).to_u64().unwrap_or(u64::MAX);
    if nodes_used <= start {
        return 0;
    }
    let end = level_end_u64(level);
    nodes_used.min(end).saturating_sub(start)
}

const DENSITY_BUCKETS: &[(f64, f64)] = &[
    (0.0, 1e-5),
    (1e-5, 1e-4),
    (1e-4, 1e-3),
    (1e-3, 1e-2),
    (1e-2, 1e-1),
    (1e-1, 1.0),
];

fn build_density_histogram(samples: &[f64]) -> Vec<DensityBucket> {
    let total = samples.len() as u64;
    DENSITY_BUCKETS
        .iter()
        .enumerate()
        .map(|(index, (lower, upper))| {
            let count = samples
                .iter()
                .filter(|sample| {
                    if index + 1 == DENSITY_BUCKETS.len() {
                        **sample >= *lower && **sample <= *upper
                    } else {
                        **sample >= *lower && **sample < *upper
                    }
                })
                .count() as u64;
            DensityBucket {
                lower: *lower,
                upper: *upper,
                count,
                fraction: if total > 0 {
                    count as f64 / total as f64
                } else {
                    0.0
                },
            }
        })
        .collect()
}

fn select_adaptive_dense_level_limit(
    samples: &[TraceSample],
    threshold: f64,
) -> u32 {
    let mut totals = BTreeMap::<u32, (u64, u64)>::new();
    let mut max_level = 0u32;

    for sample in samples {
        let mut ids = sample.active_ids.clone().unwrap_or_default();
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            continue;
        }
        let inferred_nodes_used = ids.iter().copied().max().unwrap_or(0).saturating_add(1);
        let nodes_used = sample.nodes_used.unwrap_or(inferred_nodes_used).max(inferred_nodes_used);
        let mut sample_counts = BTreeMap::<u32, u64>::new();
        for node_id in ids {
            let level = level_for_node_id(node_id);
            *sample_counts.entry(level).or_insert(0) += 1;
            max_level = max_level.max(level);
        }
        for level in 0..=max_level {
            let span = used_span_in_level(nodes_used, level);
            if span == 0 {
                continue;
            }
            let entry = totals.entry(level).or_insert((0, 0));
            entry.0 += sample_counts.get(&level).copied().unwrap_or(0);
            entry.1 += span;
        }
    }

    let mut dense_level_limit = 0u32;
    for level in 0..=max_level {
        let Some((active, capacity)) = totals.get(&level).copied() else {
            break;
        };
        if capacity == 0 {
            break;
        }
        let density = active as f64 / capacity as f64;
        if density >= threshold {
            dense_level_limit = level + 1;
        } else {
            break;
        }
    }
    dense_level_limit
}

pub fn encode_address(node_id: u64, dense_level_limit: u32) -> Result<EncodedAddress, String> {
    encode_address_with_options(node_id, dense_level_limit, false)
}

pub fn encode_address_with_options(
    node_id: u64,
    dense_level_limit: u32,
    signed_representation: bool,
) -> Result<EncodedAddress, String> {
    let level = level_for_node_id(node_id);
    if level < dense_level_limit {
        return Ok(EncodedAddress::Dense { node_id, level });
    }

    let start = level_start(level);
    let target = Integer::from(node_id);
    let relative = target - start;
    let relative_offset = relative
        .to_u64()
        .ok_or_else(|| {
            format!(
                "relative offset exceeded u64 during encode for node_id={node_id}, level={level}"
            )
        })?;
    let relative_plus_one = relative_offset.checked_add(1).ok_or_else(|| {
        format!(
            "relative offset overflowed +1 sentinel during encode for node_id={node_id}, level={level}"
        )
    })?;
    let hz = HybridNumber::from_u64(relative_plus_one);
    let signed_hz = SignedHybridNumber::from_u64(relative_plus_one);
    let (support_card, active_levels, support_mode, payloads, signed_payloads) =
        if signed_representation {
            (
                signed_hz.support_card(),
                signed_hz.active_levels(),
                "signed_naf".to_string(),
                None,
                Some(signed_hz.levels.clone()),
            )
        } else {
            (
                hz.support_card(),
                hz.active_levels(),
                "unsigned_zeckendorf".to_string(),
                Some(hz.levels.clone()),
                None,
            )
        };
    Ok(EncodedAddress::Sparse {
        node_id,
        level,
        relative_offset,
        payloads,
        signed_payloads,
        support_card,
        active_levels,
        unsigned_support_card: hz.support_card(),
        signed_support_card: signed_hz.support_card(),
        support_mode,
    })
}

pub fn decode_address(encoded: &EncodedAddress) -> Result<u64, String> {
    match encoded {
        EncodedAddress::Dense { node_id, .. } => Ok(*node_id),
        EncodedAddress::Sparse {
            level,
            relative_offset: _,
            payloads,
            signed_payloads,
            ..
        } => {
            let relative_plus_one = if let Some(signed_payloads) = signed_payloads {
                SignedHybridNumber {
                    levels: signed_payloads.clone(),
                }
                .eval()
            } else {
                let hz = HybridNumber {
                    levels: payloads.clone().unwrap_or_default(),
                };
                hz.eval()
            };
            let rel_plus_one_u64 = relative_plus_one
                .to_u64()
                .ok_or_else(|| "encoded sparse address overflowed u64 on decode".to_string())?;
            if rel_plus_one_u64 == 0 {
                return Err("encoded sparse address decoded to zero sentinel".to_string());
            }
            let start = level_start(*level);
            let start_u64 = start
                .to_u64()
                .ok_or_else(|| format!("weight(level={level}) exceeded u64 during decode"))?;
            Ok(start_u64 + (rel_plus_one_u64 - 1))
        }
    }
}

pub fn replay_trace(
    samples: &[TraceSample],
    dense_level_limit: u32,
) -> Result<TraceReplaySummary, String> {
    replay_trace_with_options(samples, &ReplayOptions::unsigned_static(dense_level_limit))
}

pub fn replay_trace_with_options(
    samples: &[TraceSample],
    options: &ReplayOptions,
) -> Result<TraceReplaySummary, String> {
    let mut level_acc = BTreeMap::<u32, LevelAccumulator>::new();
    let mut active_sample_count = 0u64;
    let mut total_addresses = 0u64;
    let mut dense_address_count = 0u64;
    let mut sparse_address_count = 0u64;
    let mut total_support_card = 0u64;
    let mut unsigned_total_support_card = 0u64;
    let mut signed_total_support_card = 0u64;
    let mut max_dense_addresses_in_sample = 0u64;
    let mut max_sparse_addresses_in_sample = 0u64;
    let mut sparse_levels_seen = BTreeSet::<String>::new();
    let mut all_density_samples = Vec::<f64>::new();
    let mut per_level_density_samples = BTreeMap::<u32, Vec<f64>>::new();
    let effective_dense_level_limit = if options.adaptive_dense_levels {
        select_adaptive_dense_level_limit(samples, options.adaptive_density_threshold)
    } else {
        options.dense_level_limit
    };
    let support_card_mode = if options.signed_representation {
        "signed_naf".to_string()
    } else {
        "unsigned_zeckendorf".to_string()
    };

    for sample in samples {
        let mut ids: Vec<u64> = sample.active_ids.clone().unwrap_or_default();
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            continue;
        }
        let inferred_nodes_used = ids.iter().copied().max().unwrap_or(0).saturating_add(1);
        let nodes_used = sample
            .nodes_used
            .unwrap_or(inferred_nodes_used)
            .max(inferred_nodes_used);
        active_sample_count += 1;
        total_addresses += ids.len() as u64;
        let mut dense_in_sample = 0u64;
        let mut sparse_in_sample = 0u64;

        let mut sample_level_counts = BTreeMap::<u32, u64>::new();
        for node_id in ids.iter().copied() {
            let encoded = encode_address_with_options(
                node_id,
                effective_dense_level_limit,
                options.signed_representation,
            )?;
            let decoded = decode_address(&encoded)?;
            if decoded != node_id {
                let step = sample.step.unwrap_or(0);
                return Err(format!(
                    "roundtrip mismatch at step={step}: node_id={node_id}, decoded={decoded}"
                ));
            }
            match encoded {
                EncodedAddress::Dense { level, .. } => {
                    dense_address_count += 1;
                    dense_in_sample += 1;
                    let acc = level_acc.entry(level).or_default();
                    acc.total_addresses += 1;
                    *sample_level_counts.entry(level).or_insert(0) += 1;
                }
                EncodedAddress::Sparse {
                    level,
                    support_card,
                    active_levels,
                    unsigned_support_card,
                    signed_support_card,
                    ..
                } => {
                    sparse_address_count += 1;
                    sparse_in_sample += 1;
                    total_support_card += support_card as u64;
                    unsigned_total_support_card += unsigned_support_card as u64;
                    signed_total_support_card += signed_support_card as u64;
                    sparse_levels_seen.insert(format!("level_{level}"));
                    let acc = level_acc.entry(level).or_default();
                    acc.total_addresses += 1;
                    acc.total_support_card += support_card as u64;
                    acc.unsigned_total_support_card += unsigned_support_card as u64;
                    acc.signed_total_support_card += signed_support_card as u64;
                    acc.max_support_card_per_address =
                        acc.max_support_card_per_address.max(support_card.max(active_levels));
                    acc.unsigned_max_support_card_per_address = acc
                        .unsigned_max_support_card_per_address
                        .max(unsigned_support_card);
                    acc.signed_max_support_card_per_address =
                        acc.signed_max_support_card_per_address.max(signed_support_card);
                    *sample_level_counts.entry(level).or_insert(0) += 1;
                }
            }
        }

        max_dense_addresses_in_sample = max_dense_addresses_in_sample.max(dense_in_sample);
        max_sparse_addresses_in_sample = max_sparse_addresses_in_sample.max(sparse_in_sample);

        for (level, sample_level_count) in sample_level_counts {
            let acc = level_acc.get_mut(&level).expect("level accumulator present");
            acc.sample_count += 1;
            acc.max_addresses_in_sample = acc.max_addresses_in_sample.max(sample_level_count);
            let span = used_span_in_level(nodes_used, level);
            if span > 0 {
                let sample_density = sample_level_count as f64 / span as f64;
                all_density_samples.push(sample_density);
                per_level_density_samples
                    .entry(level)
                    .or_default()
                    .push(sample_density);
            }
        }
    }

    let mut level_stats = BTreeMap::new();
    for (level, acc) in level_acc {
        let representation = if level < effective_dense_level_limit {
            "bitmap".to_string()
        } else {
            "zeckendorf".to_string()
        };
        let mean_support_card_per_address =
            if level < effective_dense_level_limit || acc.total_addresses == 0 {
            None
        } else {
            Some(acc.total_support_card as f64 / acc.total_addresses as f64)
        };
        let max_support_card_per_address =
            if level < effective_dense_level_limit || acc.total_addresses == 0 {
            None
        } else {
            Some(acc.max_support_card_per_address)
        };
        level_stats.insert(
            format!("level_{level}"),
            ReplayLevelSummary {
                representation,
                sample_count: acc.sample_count,
                total_addresses: acc.total_addresses,
                mean_addresses_per_sample: if acc.sample_count > 0 {
                    acc.total_addresses as f64 / acc.sample_count as f64
                } else {
                    0.0
                },
                max_addresses_in_sample: acc.max_addresses_in_sample,
                mean_support_card_per_address,
                max_support_card_per_address,
                support_mode: if level < effective_dense_level_limit {
                    None
                } else {
                    Some(support_card_mode.clone())
                },
                mean_unsigned_support_card_per_address: if level < effective_dense_level_limit
                    || acc.total_addresses == 0
                {
                    None
                } else {
                    Some(acc.unsigned_total_support_card as f64 / acc.total_addresses as f64)
                },
                mean_signed_support_card_per_address: if level < effective_dense_level_limit
                    || acc.total_addresses == 0
                {
                    None
                } else {
                    Some(acc.signed_total_support_card as f64 / acc.total_addresses as f64)
                },
            },
        );
    }

    let density_histogram = build_density_histogram(&all_density_samples);
    let per_level_density_histograms = per_level_density_samples
        .into_iter()
        .map(|(level, samples)| {
            (
                format!("level_{level}"),
                build_density_histogram(&samples),
            )
        })
        .collect();

    Ok(TraceReplaySummary {
        sample_count: samples.len() as u64,
        active_sample_count,
        dense_level_limit: effective_dense_level_limit,
        requested_dense_level_limit: options.dense_level_limit,
        adaptive_dense_level_limit: if options.adaptive_dense_levels {
            Some(effective_dense_level_limit)
        } else {
            None
        },
        adaptive_dense_levels: options.adaptive_dense_levels,
        adaptive_density_threshold: if options.adaptive_dense_levels {
            Some(options.adaptive_density_threshold)
        } else {
            None
        },
        signed_representation: options.signed_representation,
        support_card_mode,
        roundtrip_ok: true,
        total_addresses,
        dense_address_count,
        sparse_address_count,
        sparse_levels_seen: sparse_levels_seen.into_iter().collect(),
        max_dense_addresses_in_sample,
        max_sparse_addresses_in_sample,
        total_support_card,
        unsigned_total_support_card,
        signed_total_support_card,
        mean_support_card_per_sparse_address: if sparse_address_count > 0 {
            Some(total_support_card as f64 / sparse_address_count as f64)
        } else {
            None
        },
        level_stats,
        density_histogram,
        per_level_density_histograms,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        decode_address, encode_address, encode_address_with_options, replay_trace,
        replay_trace_with_options, EncodedAddress, ReplayOptions, TraceSample,
    };

    #[test]
    fn dense_level_zero_roundtrips_directly() {
        let encoded = encode_address(17, 1).expect("encode");
        match encoded {
            EncodedAddress::Dense { level, node_id } => {
                assert_eq!(level, 0);
                assert_eq!(node_id, 17);
            }
            EncodedAddress::Sparse { .. } => panic!("expected dense encoding"),
        }
        assert_eq!(decode_address(&encoded).expect("decode"), 17);
    }

    #[test]
    fn sparse_level_one_roundtrips_through_hz() {
        let encoded = encode_address(2641, 1).expect("encode");
        match &encoded {
            EncodedAddress::Sparse {
                level,
                relative_offset,
                support_card,
                ..
            } => {
                assert_eq!(*level, 1);
                assert_eq!(*relative_offset, 1641);
                assert!(*support_card > 0);
            }
            EncodedAddress::Dense { .. } => panic!("expected sparse encoding"),
        }
        assert_eq!(decode_address(&encoded).expect("decode"), 2641);
    }

    #[test]
    fn replay_trace_reports_sparse_levels() {
        let samples = vec![
            TraceSample {
                step: Some(0),
                nodes_used: Some(3000),
                active_ids: Some(vec![7, 1000, 1500]),
            },
            TraceSample {
                step: Some(1),
                nodes_used: Some(3000),
                active_ids: Some(vec![5, 1700]),
            },
        ];
        let summary = replay_trace(&samples, 1).expect("replay");
        assert!(summary.roundtrip_ok);
        assert_eq!(summary.dense_address_count, 2);
        assert_eq!(summary.sparse_address_count, 3);
        assert!(summary.sparse_levels_seen.contains(&"level_1".to_string()));
        let level0 = summary.level_stats.get("level_0").expect("level_0");
        assert_eq!(level0.representation, "bitmap");
        let level1 = summary.level_stats.get("level_1").expect("level_1");
        assert_eq!(level1.representation, "zeckendorf");
        assert!(level1.mean_support_card_per_address.expect("support") > 0.0);
    }

    #[test]
    fn replay_trace_dedups_per_level_sample_maxima() {
        let samples = vec![TraceSample {
            step: Some(0),
            nodes_used: Some(4000),
            active_ids: Some(vec![7, 7, 1000, 1000, 1500]),
        }];
        let summary = replay_trace(&samples, 1).expect("replay");
        let level0 = summary.level_stats.get("level_0").expect("level_0");
        let level1 = summary.level_stats.get("level_1").expect("level_1");
        assert_eq!(summary.total_addresses, 3);
        assert_eq!(level0.sample_count, 1);
        assert_eq!(level0.max_addresses_in_sample, 1);
        assert_eq!(level1.sample_count, 1);
        assert_eq!(level1.max_addresses_in_sample, 2);
    }

    #[test]
    fn signed_sparse_support_mode_roundtrips() {
        let encoded = encode_address_with_options(1030, 1, true).expect("encode");
        match &encoded {
            EncodedAddress::Sparse {
                payloads,
                signed_payloads,
                support_mode,
                signed_support_card,
                unsigned_support_card,
                ..
            } => {
                assert!(payloads.is_none());
                assert!(signed_payloads.is_some());
                assert_eq!(support_mode, "signed_naf");
                assert!(*signed_support_card <= *unsigned_support_card);
            }
            EncodedAddress::Dense { .. } => panic!("expected sparse encoding"),
        }
        assert_eq!(decode_address(&encoded).expect("decode"), 1030);
    }

    #[test]
    fn adaptive_dense_level_limit_returns_zero_when_level_zero_is_sparse() {
        let samples = vec![
            TraceSample {
                step: Some(0),
                nodes_used: Some(8),
                active_ids: Some(vec![0, 1, 2, 3, 4, 5]),
            },
            TraceSample {
                step: Some(1),
                nodes_used: Some(1505),
                active_ids: Some(vec![0, 1, 2, 1000, 1500]),
            },
        ];
        let options = ReplayOptions {
            dense_level_limit: 1,
            signed_representation: true,
            adaptive_dense_levels: true,
            adaptive_density_threshold: 0.01,
        };
        let summary = replay_trace_with_options(&samples, &options).expect("replay");
        assert!(summary.adaptive_dense_levels);
        assert_eq!(summary.dense_level_limit, 0);
        assert_eq!(summary.requested_dense_level_limit, 1);
        assert_eq!(summary.adaptive_dense_level_limit, Some(0));
        assert_eq!(summary.support_card_mode, "signed_naf");
        assert!(summary.sparse_levels_seen.contains(&"level_0".to_string()));
    }

    #[test]
    fn adaptive_dense_level_limit_truncates_at_first_sparse_level() {
        let samples = vec![TraceSample {
            step: Some(0),
            nodes_used: Some(1_000_010),
            active_ids: Some(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1_000, 1_000_000]),
        }];
        let options = ReplayOptions {
            dense_level_limit: 3,
            signed_representation: true,
            adaptive_dense_levels: true,
            adaptive_density_threshold: 0.01,
        };
        let summary = replay_trace_with_options(&samples, &options).expect("replay");
        assert!(summary.adaptive_dense_levels);
        assert_eq!(summary.requested_dense_level_limit, 3);
        assert_eq!(summary.dense_level_limit, 1);
        assert_eq!(summary.adaptive_dense_level_limit, Some(1));
        assert!(summary.sparse_levels_seen.contains(&"level_1".to_string()));
        assert!(summary.sparse_levels_seen.contains(&"level_2".to_string()));
    }

    #[test]
    fn replay_trace_emits_density_histograms() {
        let samples = vec![
            TraceSample {
                step: Some(0),
                nodes_used: Some(2_000),
                active_ids: Some(vec![0, 1000, 1001, 1500]),
            },
            TraceSample {
                step: Some(1),
                nodes_used: Some(1_000_100),
                active_ids: Some(vec![0, 1, 2, 3, 4, 1_000, 1_000_001]),
            },
        ];
        let summary = replay_trace(&samples, 1).expect("replay");
        assert_eq!(summary.density_histogram.len(), 6);
        assert!(summary.density_histogram.iter().any(|bucket| bucket.count > 0));
        let level0 = summary
            .per_level_density_histograms
            .get("level_0")
            .expect("level_0 density histogram");
        assert_eq!(level0.len(), 6);
        assert!(level0.iter().any(|bucket| bucket.count > 0));
        let total_fraction: f64 = summary.density_histogram.iter().map(|bucket| bucket.fraction).sum();
        assert!((total_fraction - 1.0).abs() < 1e-9);
    }
}
