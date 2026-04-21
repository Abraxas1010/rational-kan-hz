use crate::flat_hybrid::FlatHybridNumber;
use crate::weight::weight;
use crate::zeckendorf::lazy_eval_fib;
use crate::HybridNumber;
use rand::RngCore;
use rug::Integer;
use std::collections::{BTreeMap, BTreeSet};

const LOG2_PHI: f64 = 0.694_241_913_630_617_4;

#[derive(Debug, Clone)]
struct LevelPlan {
    level: u32,
    max_index: u32,
    capacity: u32,
}

pub fn expected_support_card(target_bits: u64, target_rho: f64) -> u32 {
    if target_bits == 0 || target_rho <= 0.0 {
        return 1;
    }
    ((target_rho * target_bits as f64 / LOG2_PHI).round() as u32).max(1)
}

pub fn construct_sparse_hz(
    target_bits: u64,
    target_rho: f64,
    rng: &mut impl RngCore,
) -> HybridNumber {
    let total_k = expected_support_card(target_bits.max(1), target_rho.max(1e-12));

    let mut max_level = max_level_for_bits(target_bits.max(1));
    while total_capacity(max_level) < total_k {
        max_level += 1;
        if max_level > 64 {
            break;
        }
    }

    let plan: Vec<LevelPlan> = (0..=max_level)
        .map(|level| {
            let max_index = max_fib_index_for_level(level);
            let capacity = slot_capacity(max_index);
            LevelPlan {
                level,
                max_index,
                capacity,
            }
        })
        .collect();

    let total_capacity = plan.iter().map(|x| x.capacity).sum::<u32>().max(1);
    let mut quotas = vec![0u32; plan.len()];
    let mut remaining_k = total_k.min(total_capacity);
    let mut remaining_capacity = total_capacity;

    // Reserve one payload entry on the top level to keep scale near target_bits.
    if !plan.is_empty() {
        let top_idx = plan.len() - 1;
        quotas[top_idx] = 1.min(plan[top_idx].capacity);
        remaining_k = remaining_k.saturating_sub(quotas[top_idx]);
        remaining_capacity = remaining_capacity.saturating_sub(plan[top_idx].capacity);
    }

    for i in 0..plan.len() {
        if remaining_k == 0 || remaining_capacity == 0 {
            break;
        }
        if quotas[i] >= plan[i].capacity {
            continue;
        }
        let cap_free = plan[i].capacity - quotas[i];
        let proportional =
            ((remaining_k as u128 * cap_free as u128) / remaining_capacity as u128) as u32;
        let take = proportional.min(cap_free).min(remaining_k);
        quotas[i] += take;
        remaining_k -= take;
        remaining_capacity -= cap_free;
    }

    // Fill any rounding residue.
    let mut cursor = plan.len();
    while remaining_k > 0 && !plan.is_empty() {
        cursor = cursor.wrapping_sub(1);
        let i = cursor % plan.len();
        if quotas[i] < plan[i].capacity {
            quotas[i] += 1;
            remaining_k -= 1;
        }
    }

    let mut levels = BTreeMap::<u32, Vec<u32>>::new();
    for (i, lp) in plan.iter().enumerate() {
        let count = quotas[i];
        if count == 0 {
            continue;
        }
        let indices = pick_non_consecutive_indices(count, lp.max_index, rng);
        if !indices.is_empty() {
            levels.insert(lp.level, indices);
        }
    }

    let mut hz = HybridNumber { levels };
    hz.canonicalize();
    enforce_carry_invariant(&mut hz);

    if hz.is_zero() {
        let mut levels = BTreeMap::new();
        levels.insert(0, vec![2]);
        hz = HybridNumber { levels };
    }

    hz
}

pub fn construct_sparse_hz_flat(
    target_bits: u64,
    target_rho: f64,
    rng: &mut impl RngCore,
) -> FlatHybridNumber {
    FlatHybridNumber::from_legacy(&construct_sparse_hz(target_bits, target_rho, rng))
}

pub fn validate_sparse_hz(
    target_bits: u64,
    target_rho: f64,
    hz: &HybridNumber,
) -> Result<(), String> {
    let val = hz.eval();
    validate_sparse_hz_with_value(target_bits, target_rho, hz, &val)
}

pub fn density_from_value(hz: &HybridNumber, val: &Integer) -> f64 {
    if *val <= 1 {
        return 0.0;
    }

    let bits = val.significant_bits() as f64;
    let ln_n = bits * std::f64::consts::LN_2;
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let log_phi_val = ln_n / golden_ratio.ln();
    if log_phi_val <= 0.0 {
        0.0
    } else {
        hz.support_card() as f64 / log_phi_val
    }
}

pub fn validate_sparse_hz_with_value(
    target_bits: u64,
    target_rho: f64,
    hz: &HybridNumber,
    val: &Integer,
) -> Result<(), String> {
    if *val <= 0 {
        return Err("eval <= 0".to_string());
    }

    let measured_rho = density_from_value(hz, val);
    if target_rho > 0.0 && measured_rho > 2.0 * target_rho {
        return Err(format!(
            "rho drift: target={target_rho:.6e}, measured={measured_rho:.6e}"
        ));
    }

    let min_bits = ((target_bits / 5).max(1)) as u32;
    if val.significant_bits() < min_bits {
        return Err(format!(
            "bit-length too small: target_bits={target_bits}, actual_bits={}",
            val.significant_bits()
        ));
    }

    for (&level, payload) in &hz.levels {
        let coeff = lazy_eval_fib(payload);
        let divisor = carry_divisor(level);
        if coeff >= divisor {
            return Err(format!(
                "carry invariant violated at level={level}: coeff >= divisor"
            ));
        }
        if has_consecutive_or_unsorted(payload) {
            return Err(format!(
                "payload invariant violated at level={level}: consecutive/unsorted indices"
            ));
        }
    }

    Ok(())
}

fn enforce_carry_invariant(hz: &mut HybridNumber) {
    // The sparse constructor bounds per-level payloads using a 0.90 safety
    // factor, so no level should reach the carry threshold.
    #[cfg(debug_assertions)]
    {
        for (&level, payload) in &hz.levels {
            let coeff = lazy_eval_fib(payload);
            let divisor = carry_divisor(level);
            assert!(
                coeff < divisor,
                "carry invariant violated at level {level}: coeff={coeff} divisor={divisor}"
            );
        }
    }
    hz.canonicalize();
}

fn carry_divisor(level: u32) -> Integer {
    if level == 0 {
        weight(1)
    } else {
        weight(level)
    }
}

fn has_consecutive_or_unsorted(indices: &[u32]) -> bool {
    indices.windows(2).any(|w| w[1] <= w[0] || w[1] == w[0] + 1)
}

fn total_capacity(max_level: u32) -> u32 {
    (0..=max_level)
        .map(max_fib_index_for_level)
        .map(slot_capacity)
        .sum()
}

fn max_level_for_bits(target_bits: u64) -> u32 {
    let mut level = 0u32;
    loop {
        let next_bits = level_bits_estimate(level + 1);
        if next_bits >= target_bits {
            break;
        }
        level += 1;
        if level > 64 {
            break;
        }
    }
    level
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

fn max_fib_index_for_level(level: u32) -> u32 {
    let divisor_bits = if level == 0 {
        10
    } else {
        level_bits_estimate(level).max(10)
    };
    let safe = (0.90 * divisor_bits as f64 / LOG2_PHI).floor() as u32;
    safe.max(4)
}

fn slot_capacity(max_index: u32) -> u32 {
    if max_index < 2 {
        return 0;
    }
    let even_slots = ((max_index - 2) / 2) + 1;
    let odd_slots = if max_index >= 3 {
        ((max_index - 3) / 2) + 1
    } else {
        0
    };
    even_slots.max(odd_slots)
}

fn pick_non_consecutive_indices(count: u32, max_index: u32, rng: &mut impl RngCore) -> Vec<u32> {
    if count == 0 || max_index < 2 {
        return vec![];
    }

    let parity = (rng.next_u32() & 1) as u32;
    let start = if parity == 0 || max_index < 3 { 2 } else { 3 };
    let slots = slot_count_from_start(start, max_index) as usize;
    let choose = (count as usize).min(slots);
    if choose == 0 {
        return vec![];
    }

    let selected_slots = if choose * 4 < slots {
        let mut picked = BTreeSet::new();
        while picked.len() < choose {
            let s = (rng.next_u64() as usize) % slots;
            picked.insert(s);
        }
        picked.into_iter().collect::<Vec<_>>()
    } else {
        // Dense draw: partial Fisher-Yates.
        let mut pool: Vec<usize> = (0..slots).collect();
        for i in 0..choose {
            let j = i + ((rng.next_u64() as usize) % (slots - i));
            pool.swap(i, j);
        }
        let mut out = pool.into_iter().take(choose).collect::<Vec<_>>();
        out.sort_unstable();
        out
    };

    selected_slots
        .into_iter()
        .map(|slot| start + (slot as u32) * 2)
        .collect()
}

fn slot_count_from_start(start: u32, max_index: u32) -> u32 {
    if start > max_index {
        0
    } else {
        ((max_index - start) / 2) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::enforce_carry_invariant;
    use crate::HybridNumber;
    use std::collections::BTreeMap;

    #[test]
    fn enforce_carry_invariant_preserves_eval() {
        let mut levels = BTreeMap::new();
        levels.insert(0u32, vec![2, 4, 6]);
        levels.insert(1u32, vec![3, 5, 7]);
        let hz = HybridNumber { levels };
        let before = hz.eval();
        let mut carried = hz.clone();
        enforce_carry_invariant(&mut carried);
        assert_eq!(carried.eval(), before);
    }
}
