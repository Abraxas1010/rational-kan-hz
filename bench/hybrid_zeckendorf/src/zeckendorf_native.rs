//! Fibonacci-native Zeckendorf normalization.
//!
//! The hot path stays in the index/digit domain. Tests may compare against the
//! legacy GMP-based oracle for verification only.

use crate::flat_hybrid::{FlatHybridNumber, MAX_LEVELS};
use crate::weight::weight;
use crate::zeckendorf::zeckendorf as zeckendorf_legacy;
use std::cmp::Ordering;
use std::sync::{LazyLock, OnceLock};

static CARRY_THRESHOLDS: LazyLock<[OnceLock<Vec<u32>>; MAX_LEVELS]> =
    LazyLock::new(|| std::array::from_fn(|_| OnceLock::new()));

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LevelNormalizeProfile {
    pub u64_calls: u64,
    pub sparse_calls: u64,
    pub canonical_skips: u64,
    pub carry_invocations: u64,
    pub carry_units: u64,
    pub set_calls: u64,
    pub indices_in: u64,
    pub indices_out: u64,
    pub fuel_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizeProfile {
    pub levels: [LevelNormalizeProfile; MAX_LEVELS],
}

impl Default for NormalizeProfile {
    fn default() -> Self {
        Self {
            levels: std::array::from_fn(|_| LevelNormalizeProfile::default()),
        }
    }
}

impl NormalizeProfile {
    fn note_u64_call(&mut self, level: usize, indices_in: usize) {
        self.levels[level].u64_calls += 1;
        self.levels[level].indices_in += indices_in as u64;
    }

    fn note_sparse_call(&mut self, level: usize, indices_in: usize) {
        self.levels[level].sparse_calls += 1;
        self.levels[level].indices_in += indices_in as u64;
    }

    fn note_canonical_skip(&mut self, level: usize, indices_in: usize) {
        self.levels[level].canonical_skips += 1;
        self.levels[level].indices_in += indices_in as u64;
    }

    fn note_set_call(&mut self, level: usize, indices_out: usize) {
        self.levels[level].set_calls += 1;
        self.levels[level].indices_out += indices_out as u64;
    }

    fn note_carry(&mut self, level: usize, carry_units: u64) {
        self.levels[level].carry_invocations += 1;
        self.levels[level].carry_units += carry_units;
    }
}

pub fn carry_threshold_indices(level: u32) -> &'static [u32] {
    let slot = &CARRY_THRESHOLDS[level as usize];
    slot.get_or_init(|| {
        let threshold = if level == 0 { weight(1) } else { weight(level) };
        zeckendorf_legacy(&threshold)
    })
}

pub fn indices_to_digits(indices: &[u32]) -> Vec<u8> {
    let max_idx = indices.iter().copied().max().unwrap_or(2) as usize;
    let mut digits = vec![0u8; max_idx + 8];
    for &idx in indices {
        let i = idx as usize;
        if i >= digits.len() {
            digits.resize(i + 8, 0);
        }
        digits[i] = digits[i].saturating_add(1);
    }
    digits
}

pub fn digits_to_indices(digits: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    for (idx, &digit) in digits.iter().enumerate().skip(2) {
        for _ in 0..digit {
            out.push(idx as u32);
        }
    }
    out
}

fn counts_from_indices(indices: &[u32]) -> Vec<(u32, u32)> {
    if indices.is_empty() {
        return vec![];
    }
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    let mut counts = Vec::new();
    let mut i = 0usize;
    while i < sorted.len() {
        let idx = sorted[i];
        let mut count = 1u32;
        i += 1;
        while i < sorted.len() && sorted[i] == idx {
            count += 1;
            i += 1;
        }
        counts.push((idx, count));
    }
    counts
}

fn indices_from_counts(counts: &[(u32, u32)]) -> Vec<u32> {
    let mut out = Vec::with_capacity(counts.iter().map(|&(_, c)| c as usize).sum());
    for &(idx, count) in counts {
        for _ in 0..count {
            out.push(idx);
        }
    }
    out
}

fn upsert_signed(counts: &mut Vec<(u32, i32)>, idx: u32, amount: i32) {
    if amount == 0 {
        return;
    }
    match counts.binary_search_by_key(&idx, |&(k, _)| k) {
        Ok(pos) => {
            counts[pos].1 += amount;
            if counts[pos].1 == 0 {
                counts.remove(pos);
            }
        }
        Err(pos) => counts.insert(pos, (idx, amount)),
    }
}

fn canonical_sparse(counts: &[(u32, u32)]) -> bool {
    if counts.iter().any(|&(_, c)| c > 1) {
        return false;
    }
    let mut prev: Option<u32> = None;
    for &(idx, _) in counts {
        if let Some(last) = prev {
            if idx == last + 1 {
                return false;
            }
        }
        prev = Some(idx);
    }
    true
}

fn trim_i32(digits: &mut Vec<i32>) {
    while digits.len() > 3 && digits.last() == Some(&0) {
        digits.pop();
    }
}

fn trim_u8(digits: &mut Vec<u8>) {
    while digits.len() > 3 && digits.last() == Some(&0) {
        digits.pop();
    }
}

fn is_canonical_u8(digits: &[u8]) -> bool {
    if digits.iter().skip(2).any(|&d| d > 1) {
        return false;
    }
    !digits.windows(2).skip(2).any(|w| w[0] == 1 && w[1] == 1)
}

fn ensure_room_i32(digits: &mut Vec<i32>, idx: usize) {
    if idx >= digits.len() {
        digits.resize(idx + 4, 0);
    }
}

fn normalize_positive_i32(digits: &mut Vec<i32>) {
    ensure_room_i32(digits, 4);
    let mut fuel = 0usize;
    loop {
        fuel += 1;
        assert!(fuel < 1_000_000, "positive normalization did not converge");
        let mut changed = false;

        if digits.get(1).copied().unwrap_or(0) > 0 {
            ensure_room_i32(digits, 2);
            let carry = digits[1];
            digits[2] += carry;
            digits[1] = 0;
            changed = true;
        }
        if !digits.is_empty() && digits[0] != 0 {
            digits[0] = 0;
            changed = true;
        }

        let upper = digits.len().saturating_sub(1);
        for k in (2..upper).rev() {
            while digits[k] >= 2 {
                ensure_room_i32(digits, k + 1);
                digits[k] -= 2;
                digits[k + 1] += 1;
                if k >= 4 {
                    digits[k - 2] += 1;
                } else if k == 3 {
                    digits[2] += 1;
                }
                changed = true;
            }
        }

        if digits.get(1).copied().unwrap_or(0) > 0 {
            let carry = digits[1];
            digits[2] += carry;
            digits[1] = 0;
            changed = true;
        }

        let upper = digits.len().saturating_sub(1);
        for k in (3..upper).rev() {
            while digits[k] > 0 && digits[k - 1] > 0 {
                ensure_room_i32(digits, k + 1);
                digits[k] -= 1;
                digits[k - 1] -= 1;
                digits[k + 1] += 1;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
    trim_i32(digits);
}

fn split_one_down(digits: &mut Vec<i32>, idx: usize) -> bool {
    if idx < 3 || digits[idx] <= 0 {
        return false;
    }
    digits[idx] -= 1;
    if idx == 3 {
        digits[2] += 2;
    } else {
        digits[idx - 1] += 1;
        digits[idx - 2] += 1;
    }
    true
}

fn ensure_available_at(work: &mut Vec<i32>, idx: usize) -> bool {
    ensure_room_i32(work, idx + 2);
    if work[idx] > 0 {
        return true;
    }

    loop {
        if work[idx] > 0 {
            return true;
        }
        let Some(mut source) = (idx + 1..work.len()).find(|&j| work[j] > 0) else {
            return false;
        };
        while source > idx && work[idx] == 0 {
            if !split_one_down(work, source) {
                return false;
            }
            source -= 1;
        }
    }
}

fn cmp_trimmed_i32(a: &[i32], b: &[i32]) -> Ordering {
    let mut i = a.len();
    while i > 0 && a[i - 1] == 0 {
        i -= 1;
    }
    let mut j = b.len();
    while j > 0 && b[j - 1] == 0 {
        j -= 1;
    }
    let max_len = i.max(j);
    for idx in (0..max_len).rev() {
        let av = a.get(idx).copied().unwrap_or(0);
        let bv = b.get(idx).copied().unwrap_or(0);
        match av.cmp(&bv) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    Ordering::Equal
}

fn cmp_trimmed_u8(a: &[u8], b: &[u8]) -> Ordering {
    let mut i = a.len();
    while i > 0 && a[i - 1] == 0 {
        i -= 1;
    }
    let mut j = b.len();
    while j > 0 && b[j - 1] == 0 {
        j -= 1;
    }
    let max_len = i.max(j);
    for idx in (0..max_len).rev() {
        let av = a.get(idx).copied().unwrap_or(0);
        let bv = b.get(idx).copied().unwrap_or(0);
        match av.cmp(&bv) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    Ordering::Equal
}

fn reduce_large_multiplicities(digits: &mut Vec<u8>) {
    let mut work: Vec<i32> = digits.iter().map(|&d| i32::from(d)).collect();
    ensure_room_i32(&mut work, 4);
    let mut fuel = 0usize;
    loop {
        fuel += 1;
        assert!(
            fuel < 1_000_000,
            "large-multiplicity reduction did not converge"
        );
        let mut changed = false;
        let upper = work.len().max(5);
        if work.len() < upper {
            work.resize(upper, 0);
        }
        for idx in 2..work.len().saturating_sub(1) {
            if work[idx] <= 3 {
                continue;
            }
            let q = work[idx] / 2;
            work[idx] %= 2;
            if idx == 2 {
                work[3] += q;
            } else if idx == 3 {
                work[4] += q;
                work[2] += q;
            } else {
                ensure_room_i32(&mut work, idx + 1);
                work[idx + 1] += q;
                work[idx - 2] += q;
            }
            changed = true;
        }
        if !changed {
            break;
        }
    }
    digits.clear();
    digits.extend(
        work.into_iter()
            .map(|d| u8::try_from(d).expect("multiplicity reduction stayed nonnegative")),
    );
    trim_u8(digits);
}

fn ascending_to_msb_window(digits: &[u8]) -> Vec<u8> {
    let mut hi = digits.len();
    while hi > 2 && digits[hi - 1] == 0 {
        hi -= 1;
    }
    if hi <= 2 {
        return vec![0, 0];
    }
    let mut out = Vec::with_capacity(hi);
    out.push(0);
    for idx in (2..hi).rev() {
        out.push(digits[idx]);
    }
    out
}

fn msb_window_to_ascending(window: &[u8]) -> Vec<u8> {
    let mut digits = vec![0u8; 3];
    for (pos, &digit) in window.iter().enumerate() {
        if digit == 0 {
            continue;
        }
        let fib_idx = window.len() + 1 - pos;
        if fib_idx >= digits.len() {
            digits.resize(fib_idx + 1, 0);
        }
        digits[fib_idx] = digit;
    }
    trim_u8(&mut digits);
    digits
}

fn cleanup_pass1_suffix(window: &mut [u8]) {
    loop {
        let n = window.len();
        let changed = if n >= 4 && window[n - 4..] == [0, 1, 2, 0] {
            window[n - 4..].copy_from_slice(&[1, 0, 1, 0]);
            true
        } else if n >= 3 && window[n - 3..] == [0, 3, 0] {
            window[n - 3..].copy_from_slice(&[1, 1, 1]);
            true
        } else if n >= 3 && window[n - 3..] == [0, 2, 0] {
            window[n - 3..].copy_from_slice(&[1, 0, 1]);
            true
        } else if n >= 3 && window[n - 3..] == [0, 1, 2] {
            window[n - 3..].copy_from_slice(&[1, 0, 1]);
            true
        } else if n >= 2 && window[n - 2..] == [0, 3] {
            window[n - 2..].copy_from_slice(&[1, 1]);
            true
        } else if n >= 2 && window[n - 2..] == [0, 2] {
            window[n - 2..].copy_from_slice(&[1, 0]);
            true
        } else {
            false
        };
        if !changed {
            break;
        }
    }
}

pub fn normalize_digits(digits: &mut Vec<u8>) {
    let mut candidate = digits.clone();
    reduce_large_multiplicities(&mut candidate);
    let mut window = ascending_to_msb_window(&candidate);
    pass1_fission(&mut window);
    cleanup_pass1_suffix(&mut window);
    pass2_fusion_rtl(&mut window);
    pass3_fusion_ltr(&mut window);
    let exact = msb_window_to_ascending(&window);
    if is_canonical_u8(&exact) {
        *digits = exact;
        return;
    }

    // The strict three-pass pipeline is the intended hot path. For arbitrary
    // high-multiplicity multiset inputs outside the paper's two-operand model,
    // retain the older convergence normalizer as a correctness fallback.
    let mut work: Vec<i32> = digits.iter().map(|&d| i32::from(d)).collect();
    normalize_positive_i32(&mut work);
    digits.clear();
    digits.extend(
        work.into_iter()
            .map(|d| u8::try_from(d).expect("nonnegative digit")),
    );
    trim_u8(digits);
}

fn pass1_fission(digits: &mut Vec<u8>) {
    if digits.len() < 4 {
        return;
    }
    for i in 0..=digits.len() - 4 {
        match (digits[i], digits[i + 1], digits[i + 2]) {
            (0, 2, 0) => {
                digits[i] = 1;
                digits[i + 1] = 0;
                digits[i + 2] = 0;
                digits[i + 3] = digits[i + 3].saturating_add(1);
            }
            (0, 3, 0) => {
                digits[i] = 1;
                digits[i + 1] = 1;
                digits[i + 2] = 0;
                digits[i + 3] = digits[i + 3].saturating_add(1);
            }
            (0, 2, 1) => {
                digits[i] = 1;
                digits[i + 1] = 1;
                digits[i + 2] = 0;
            }
            (0, 1, 2) => {
                digits[i] = 1;
                digits[i + 1] = 0;
                digits[i + 2] = 1;
            }
            _ => {}
        }
    }
}

fn pass2_fusion_rtl(digits: &mut Vec<u8>) {
    if digits.len() < 3 {
        return;
    }
    for i in (0..=digits.len() - 3).rev() {
        if digits[i] == 0 && digits[i + 1] == 1 && digits[i + 2] == 1 {
            digits[i] = 1;
            digits[i + 1] = 0;
            digits[i + 2] = 0;
        }
    }
}

fn pass3_fusion_ltr(digits: &mut Vec<u8>) {
    if digits.len() < 3 {
        return;
    }
    for i in 0..=digits.len() - 3 {
        if digits[i] == 0 && digits[i + 1] == 1 && digits[i + 2] == 1 {
            digits[i] = 1;
            digits[i + 1] = 0;
            digits[i + 2] = 0;
        }
    }
}

pub fn add_digits(a: &[u8], b: &[u8]) -> Vec<u8> {
    let len = a.len().max(b.len()) + 8;
    let mut result = vec![0u8; len];
    for (idx, &digit) in a.iter().enumerate() {
        result[idx] = result[idx].saturating_add(digit);
    }
    for (idx, &digit) in b.iter().enumerate() {
        result[idx] = result[idx].saturating_add(digit);
    }
    normalize_digits(&mut result);
    result
}

pub fn subtract_digits(a: &[u8], b: &[u8]) -> Option<Vec<u8>> {
    let mut work = a.iter().map(|&d| i32::from(d)).collect::<Vec<_>>();
    let rhs = b.iter().map(|&d| i32::from(d)).collect::<Vec<_>>();
    if cmp_trimmed_i32(&work, &rhs) == Ordering::Less {
        return None;
    }

    for idx in (2..rhs.len()).rev() {
        let count = rhs[idx];
        for _ in 0..count {
            if !ensure_available_at(&mut work, idx) {
                return None;
            }
            work[idx] -= 1;
        }
    }

    normalize_positive_i32(&mut work);
    Some(
        work.into_iter()
            .map(|d| u8::try_from(d).ok())
            .collect::<Option<Vec<_>>>()?,
    )
}

pub fn normalize_indices(indices: &[u32]) -> Vec<u32> {
    if indices.is_empty() {
        return vec![];
    }
    normalize_indices_sparse(indices)
}

pub fn exceeds_threshold(canonical_digits: &[u8], threshold_digits: &[u8]) -> bool {
    cmp_trimmed_u8(canonical_digits, threshold_digits) != Ordering::Less
}

fn merge_new_entries_into(counts: &mut Vec<(u32, u32)>, new_entries: &mut Vec<(u32, u32)>) {
    if new_entries.is_empty() {
        return;
    }

    new_entries.sort_unstable_by_key(|&(idx, _)| idx);
    let mut coalesced: Vec<(u32, u32)> = Vec::with_capacity(new_entries.len());
    for &(idx, count) in new_entries.iter() {
        if let Some(last) = coalesced.last_mut() {
            if last.0 == idx {
                last.1 += count;
                continue;
            }
        }
        coalesced.push((idx, count));
    }

    let old = std::mem::take(counts);
    counts.reserve(old.len() + coalesced.len());
    let mut oi = 0usize;
    let mut ni = 0usize;
    while oi < old.len() && ni < coalesced.len() {
        match old[oi].0.cmp(&coalesced[ni].0) {
            Ordering::Less => {
                counts.push(old[oi]);
                oi += 1;
            }
            Ordering::Greater => {
                counts.push(coalesced[ni]);
                ni += 1;
            }
            Ordering::Equal => {
                counts.push((old[oi].0, old[oi].1 + coalesced[ni].1));
                oi += 1;
                ni += 1;
            }
        }
    }
    counts.extend_from_slice(&old[oi..]);
    counts.extend_from_slice(&coalesced[ni..]);
}

fn add_existing_or_buffer(
    counts: &mut [(u32, u32)],
    new_entries: &mut Vec<(u32, u32)>,
    idx: u32,
    amount: u32,
) {
    if amount == 0 {
        return;
    }
    match counts.binary_search_by_key(&idx, |&(key, _)| key) {
        Ok(pos) => counts[pos].1 += amount,
        Err(_) => new_entries.push((idx, amount)),
    }
}

fn normalize_indices_sparse_with_fuel(indices: &[u32]) -> (Vec<u32>, u64) {
    let mut counts = counts_from_indices(indices);
    let mut new_entries: Vec<(u32, u32)> = Vec::new();
    let mut fuel = 0usize;
    loop {
        fuel += 1;
        assert!(fuel < 100_000, "sparse normalization did not converge");
        let mut changed = false;

        new_entries.clear();
        let mut i = counts.len();
        while i > 0 {
            i -= 1;
            let (k, c) = counts[i];
            if c < 2 {
                continue;
            }
            changed = true;
            let pairs = c / 2;
            let rem = c % 2;
            counts[i].1 = rem;

            if k == 2 {
                add_existing_or_buffer(&mut counts, &mut new_entries, 3, pairs);
            } else if k == 3 {
                add_existing_or_buffer(&mut counts, &mut new_entries, 2, pairs);
                add_existing_or_buffer(&mut counts, &mut new_entries, 4, pairs);
            } else {
                add_existing_or_buffer(&mut counts, &mut new_entries, k - 2, pairs);
                add_existing_or_buffer(&mut counts, &mut new_entries, k + 1, pairs);
            }
        }
        counts.retain(|&(_, c)| c > 0);
        merge_new_entries_into(&mut counts, &mut new_entries);

        new_entries.clear();
        let mut j = 0usize;
        while j + 1 < counts.len() {
            let (k, ck) = counts[j];
            let (k1, ck1) = counts[j + 1];
            if k1 == k + 1 && ck > 0 && ck1 > 0 {
                let fuse = ck.min(ck1);
                counts[j].1 -= fuse;
                counts[j + 1].1 -= fuse;
                add_existing_or_buffer(&mut counts, &mut new_entries, k + 2, fuse);
                changed = true;
            }
            j += 1;
        }
        counts.retain(|&(_, c)| c > 0);
        merge_new_entries_into(&mut counts, &mut new_entries);

        if !changed {
            break;
        }
    }

    debug_assert!(
        canonical_sparse(&counts),
        "non-canonical sparse result: {counts:?}"
    );
    (indices_from_counts(&counts), fuel as u64)
}

pub fn normalize_indices_sparse(indices: &[u32]) -> Vec<u32> {
    normalize_indices_sparse_with_fuel(indices).0
}

fn sparse_ge(a: &[u32], b: &[u32]) -> bool {
    let mut ia = a.len();
    let mut ib = b.len();
    loop {
        match (ia.checked_sub(1), ib.checked_sub(1)) {
            (None, None) => return true,
            (Some(_), None) => return true,
            (None, Some(_)) => return false,
            (Some(pa), Some(pb)) => {
                let av = a[pa];
                let bv = b[pb];
                if av > bv {
                    return true;
                }
                if av < bv {
                    return false;
                }
                ia = pa;
                ib = pb;
            }
        }
    }
}

fn ensure_sparse_available(counts: &mut Vec<(u32, i32)>, idx: u32) -> bool {
    loop {
        match counts.binary_search_by_key(&idx, |&(k, _)| k) {
            Ok(pos) if counts[pos].1 > 0 => return true,
            _ => {}
        }
        let donor_pos = counts.iter().position(|&(k, c)| k > idx && c > 0);
        let Some(donor_pos) = donor_pos else {
            return false;
        };
        let donor_idx = counts[donor_pos].0;
        counts[donor_pos].1 -= 1;
        if counts[donor_pos].1 == 0 {
            counts.remove(donor_pos);
        }
        if donor_idx == 3 {
            upsert_signed(counts, 2, 2);
        } else if donor_idx >= 4 {
            upsert_signed(counts, donor_idx - 1, 1);
            upsert_signed(counts, donor_idx - 2, 1);
        } else {
            return false;
        }
    }
}

fn subtract_indices_sparse(a: &[u32], b: &[u32]) -> Option<Vec<u32>> {
    if !sparse_ge(a, b) {
        return None;
    }
    let mut counts: Vec<(u32, i32)> = a.iter().copied().map(|idx| (idx, 1)).collect();
    for &idx in b.iter().rev() {
        if !ensure_sparse_available(&mut counts, idx) {
            return None;
        }
        upsert_signed(&mut counts, idx, -1);
    }

    let mut expanded = Vec::new();
    for &(idx, count) in &counts {
        if count < 0 {
            return None;
        }
        for _ in 0..count {
            expanded.push(idx);
        }
    }
    Some(normalize_indices_sparse(&expanded))
}

pub fn inter_level_carry_native(level_indices: &[u32], level: u32) -> (Vec<u32>, u32) {
    let threshold = carry_threshold_indices(level);
    let mut current = normalize_indices_sparse(level_indices);

    let mut quotient = 0u32;
    while sparse_ge(&current, threshold) {
        let Some(next) = subtract_indices_sparse(&current, threshold) else {
            break;
        };
        current = next;
        quotient += 1;
        assert!(
            quotient < 1_000_000,
            "carry quotient runaway at level {level}"
        );
    }

    (current, quotient)
}

/// Fibonacci numbers F(0)..F(93) as u64. F(93) < 2^64.
#[rustfmt::skip]
const FIB64: [u64; 94] = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
    987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025,
    121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578,
    5702887, 9227465, 14930352, 24157817, 39088169, 63245986,
    102334155, 165580141, 267914296, 433494437, 701408733, 1134903170,
    1836311903, 2971215073, 4807526976, 7778742049, 12586269025,
    20365011074, 32951280099, 53316291173, 86267571272, 139583862445,
    225851433717, 365435296162, 591286729879, 956722026041,
    1548008755920, 2504730781961, 4052739537881, 6557470319842,
    10610209857723, 17167680177565, 27777890035288, 44945570212853,
    72723460248141, 117669030460994, 190392490709135, 308061521170129,
    498454011879264, 806515533049393, 1304969544928657,
    2111485077978050, 3416454622906707, 5527939700884757,
    8944394323791464, 14472334024676221, 23416728348467685,
    37889062373143906, 61305790721611591, 99194853094755497,
    160500643816367088, 259695496911122585, 420196140727489673,
    679891637638612258, 1100087778366101931, 1779979416004714189,
    2880067194370816120, 4660046610375530309, 7540113804746346429,
    12200160415121876738,
];

/// Carry threshold as u64 for levels 0..4. Level 4+ thresholds exceed u64.
const THRESHOLD_U64: [u64; 4] = [
    1000,              // level 0: weight(1)
    1000,              // level 1: weight(1)
    1_000_000,         // level 2: weight(2)
    1_000_000_000_000, // level 3: weight(3)
];

/// Zeckendorf decomposition of a u64 value (greedy algorithm).
fn zeckendorf_u64(mut n: u64) -> Vec<u32> {
    if n == 0 {
        return vec![];
    }
    let mut result = Vec::with_capacity(8);
    let mut k = FIB64.partition_point(|&f| f <= n) - 1;
    while n > 0 && k >= 2 {
        if FIB64[k] <= n {
            result.push(k as u32);
            n -= FIB64[k];
            k = k.saturating_sub(2);
        } else {
            k -= 1;
        }
    }
    result.sort_unstable();
    result
}

/// Check if indices are already in canonical Zeckendorf form
/// (sorted, no duplicates, no consecutive pairs).
fn is_canonical_indices(indices: &[u32]) -> bool {
    if indices.is_empty() {
        return true;
    }
    for w in indices.windows(2) {
        if w[0] >= w[1] || w[1] == w[0] + 1 {
            return false;
        }
    }
    true
}

/// Single-pass forward normalization using coefficient arithmetic for low
/// levels and sparse chip-firing for high levels.
///
/// - Levels 0-3: u64 arithmetic (all Fibonacci indices ≤ 52, coefficient fits u64).
///   Compute coefficient = Σ F(i), divide by threshold, Zeckendorf-decompose remainder.
/// - Levels 4+: sparse chip-firing on index values (u32). Skips already-canonical
///   levels. No carry occurs at these levels in practice.
fn normalize_flat_native_impl(flat: &mut FlatHybridNumber, mut profile: Option<&mut NormalizeProfile>) {
    for level in 0..MAX_LEVELS {
        if flat.levels[level].is_empty() {
            continue;
        }

        let input_len = flat.levels[level].len();
        let max_idx = flat.levels[level].iter().copied().max().unwrap_or(0);

        if level < 4 && (max_idx as usize) < FIB64.len() {
            if let Some(p) = profile.as_deref_mut() {
                p.note_u64_call(level, input_len);
            }
            // Fast u64 path: coefficient arithmetic with carry.
            let coeff: u64 = flat.levels[level].iter().map(|&i| FIB64[i as usize]).sum();
            let threshold = THRESHOLD_U64[level];
            let quotient = coeff / threshold;
            let remainder = coeff % threshold;
            let remainder_indices = zeckendorf_u64(remainder);
            if let Some(p) = profile.as_deref_mut() {
                p.note_set_call(level, remainder_indices.len());
            }
            flat.set_level(level, &remainder_indices);

            if quotient > 0 && level + 1 < MAX_LEVELS {
                if let Some(p) = profile.as_deref_mut() {
                    p.note_carry(level, quotient);
                }
                let q_indices = zeckendorf_u64(quotient);
                let mut next = flat.levels[level + 1].to_vec();
                next.extend_from_slice(&q_indices);
                if let Some(p) = profile.as_deref_mut() {
                    p.note_set_call(level + 1, next.len());
                }
                flat.set_level(level + 1, &next);
            }
        } else if is_canonical_indices(flat.levels[level].as_slice()) {
            // Already canonical — skip.
            if let Some(p) = profile.as_deref_mut() {
                p.note_canonical_skip(level, input_len);
            }
        } else {
            if let Some(p) = profile.as_deref_mut() {
                p.note_sparse_call(level, input_len);
            }
            // Sparse chip-firing: operates on index values (u32), not BigNum.
            // Faster than BigNum coefficient arithmetic at high levels where
            // Fibonacci values are multi-thousand-digit numbers.
            let (canonical, fuel_used) =
                normalize_indices_sparse_with_fuel(flat.levels[level].as_slice());
            if let Some(p) = profile.as_deref_mut() {
                p.levels[level].fuel_used += fuel_used;
            }
            if let Some(p) = profile.as_deref_mut() {
                p.note_set_call(level, canonical.len());
            }
            flat.set_level(level, &canonical);
        }
    }
}

pub fn normalize_flat_native(flat: &mut FlatHybridNumber) {
    normalize_flat_native_impl(flat, None);
}

pub fn profile_normalize_flat_native(flat: &mut FlatHybridNumber) -> NormalizeProfile {
    let mut profile = NormalizeProfile::default();
    normalize_flat_native_impl(flat, Some(&mut profile));
    profile
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::construct_sparse_hz;
    use crate::zeckendorf::{lazy_eval_fib, zeckendorf};
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    fn eval_digits(digits: &[u8]) -> rug::Integer {
        lazy_eval_fib(&digits_to_indices(digits))
    }

    #[test]
    fn normalize_indices_matches_legacy_zeckendorf() {
        let cases: &[&[u32]] = &[
            &[3, 3],
            &[3, 3, 5],
            &[4, 5],
            &[2, 2, 2, 2, 9, 10],
            &[6, 6, 6, 7, 9, 9, 12],
        ];
        for indices in cases {
            let native = normalize_indices(indices);
            let legacy = zeckendorf(&lazy_eval_fib(indices));
            assert_eq!(native, legacy, "indices={indices:?}");
        }
    }

    #[test]
    fn sparse_normalize_matches_dense_reference() {
        let cases: &[&[u32]] = &[
            &[3, 3],
            &[3, 3, 5],
            &[4, 5],
            &[2, 2, 2, 2, 9, 10],
            &[6, 6, 6, 7, 9, 9, 12],
        ];
        for indices in cases {
            let sparse = normalize_indices_sparse(indices);
            let dense = {
                let mut digits = indices_to_digits(indices);
                normalize_digits(&mut digits);
                digits_to_indices(&digits)
            };
            assert_eq!(sparse, dense, "indices={indices:?}");
        }
    }

    #[test]
    fn sparse_normalize_high_multiplicity() {
        let indices = vec![5; 10];
        let result = normalize_indices_sparse(&indices);
        let expected = zeckendorf(&lazy_eval_fib(&indices));
        assert_eq!(result, expected);
    }

    #[test]
    fn sparse_subtract_basic() {
        let a = vec![6];
        let b = vec![4];
        let result = subtract_indices_sparse(&a, &b).unwrap();
        assert_eq!(result, vec![5]);
    }

    #[test]
    fn sparse_ge_correctness() {
        assert!(sparse_ge(&[6], &[4]));
        assert!(!sparse_ge(&[4], &[6]));
        assert!(sparse_ge(&[4, 6], &[5]));
        assert!(sparse_ge(&[2, 6], &[2, 6]));
        assert!(!sparse_ge(&[2, 4], &[6]));
    }

    #[test]
    fn pass1_rule_020x() {
        let mut digits = vec![0, 2, 0, 0];
        pass1_fission(&mut digits);
        assert_eq!(digits, vec![1, 0, 0, 1]);
    }

    #[test]
    fn pass1_rule_030x() {
        let mut digits = vec![0, 3, 0, 0];
        pass1_fission(&mut digits);
        assert_eq!(digits, vec![1, 1, 0, 1]);
    }

    #[test]
    fn pass1_rule_021x() {
        let mut digits = vec![0, 2, 1, 0];
        pass1_fission(&mut digits);
        assert_eq!(digits, vec![1, 1, 0, 0]);
    }

    #[test]
    fn pass1_rule_012x() {
        let mut digits = vec![0, 1, 2, 0];
        pass1_fission(&mut digits);
        assert_eq!(digits, vec![1, 0, 1, 0]);
    }

    #[test]
    fn pass2_fusion_rule_011_to_100() {
        let mut digits = vec![0, 1, 1];
        pass2_fusion_rtl(&mut digits);
        assert_eq!(digits, vec![1, 0, 0]);
    }

    #[test]
    fn pass3_fusion_rule_011_to_100() {
        let mut digits = vec![0, 1, 1];
        pass3_fusion_ltr(&mut digits);
        assert_eq!(digits, vec![1, 0, 0]);
    }

    #[test]
    fn fib64_table_matches_reference() {
        for (idx, &value) in FIB64.iter().enumerate() {
            assert_eq!(
                rug::Integer::from(value),
                crate::zeckendorf::fib(idx as u32),
                "FIB64 mismatch at index {idx}"
            );
        }
    }

    #[test]
    fn normalize_at_carry_boundaries() {
        for level in 0..4usize {
            let threshold = THRESHOLD_U64[level];
            for delta in [0i64, -1, 1, -2, 2] {
                let target = (threshold as i64 + delta).max(0) as u64;
                if target == 0 {
                    continue;
                }
                let indices = zeckendorf_u64(target);
                let mut flat = FlatHybridNumber::empty();
                flat.set_level(level, &indices);
                let pre_eval = flat.eval();
                normalize_flat_native(&mut flat);
                let post_eval = flat.eval();
                assert_eq!(
                    pre_eval, post_eval,
                    "eval changed at level={level}, target={target}, delta={delta}"
                );
            }
        }
    }

    #[test]
    fn normalize_all_consecutive_indices() {
        for n in [5u32, 10, 20, 40] {
            let indices: Vec<u32> = (2..2 + n).collect();
            let expected = zeckendorf(&lazy_eval_fib(&indices));
            let result = normalize_indices_sparse(&indices);
            assert_eq!(result, expected, "all-consecutive n={n}");
        }
    }

    #[test]
    fn normalize_all_duplicate_indices() {
        for (idx, count) in [(5u32, 20u32), (10, 50), (20, 100)] {
            let indices: Vec<u32> = vec![idx; count as usize];
            let expected = zeckendorf(&lazy_eval_fib(&indices));
            let result = normalize_indices_sparse(&indices);
            assert_eq!(result, expected, "all-duplicate idx={idx} count={count}");
        }
    }

    #[test]
    fn canonical_skip_does_not_misfire() {
        assert!(!is_canonical_indices(&[2, 4, 6, 7]));
        assert!(!is_canonical_indices(&[2, 3, 6, 8]));
        assert!(!is_canonical_indices(&[2, 4, 4, 8]));
        assert!(!is_canonical_indices(&[4, 2, 6, 8]));

        assert!(is_canonical_indices(&[2, 4, 6, 8]));
        assert!(is_canonical_indices(&[3, 5, 7, 9]));
        assert!(is_canonical_indices(&[2]));
        assert!(is_canonical_indices(&[]));
        assert!(is_canonical_indices(&[2, 100, 5000]));
    }

    #[test]
    fn add_and_subtract_digits_preserve_value() {
        let a = indices_to_digits(&[2, 4, 6, 9]);
        let b = indices_to_digits(&[3, 7]);
        let sum = add_digits(&a, &b);
        let diff = subtract_digits(&sum, &b).expect("sum >= b");
        assert_eq!(eval_digits(&sum), lazy_eval_fib(&[2, 4, 6, 9, 3, 7]));
        assert_eq!(digits_to_indices(&diff), normalize_indices(&[2, 4, 6, 9]));
    }

    #[test]
    fn inter_level_carry_matches_integer_division() {
        for level in [0u32, 1, 2, 3] {
            let threshold = if level == 0 { weight(1) } else { weight(level) };
            let inputs = [
                zeckendorf(&(threshold.clone() - 1)),
                zeckendorf(&threshold),
                zeckendorf(&(threshold.clone() * 2 + 5)),
            ];
            for indices in inputs {
                let (remainder, quotient) = inter_level_carry_native(&indices, level);
                let value = lazy_eval_fib(&indices);
                let q = value.clone() / threshold.clone();
                let r = value % threshold.clone();
                assert_eq!(u32::try_from(q).ok(), Some(quotient));
                assert_eq!(lazy_eval_fib(&remainder), r);
            }
        }
    }

    #[test]
    fn threshold_table_correctness() {
        for level in 0..=12u32 {
            let threshold_indices = carry_threshold_indices(level);
            let expected_value = if level == 0 { weight(1) } else { weight(level) };
            assert_eq!(
                lazy_eval_fib(threshold_indices),
                expected_value,
                "threshold mismatch at level {level}"
            );
        }
    }

    #[test]
    fn threshold_table_high_level_smoke() {
        for level in [15u32, 17u32] {
            let threshold_indices = carry_threshold_indices(level);
            assert!(
                !threshold_indices.is_empty(),
                "empty threshold at level {level}"
            );
            assert!(
                threshold_indices
                    .windows(2)
                    .all(|w| w[0] < w[1] && w[1] != w[0] + 1),
                "non-canonical threshold at level {level}"
            );
        }
    }

    #[test]
    fn normalize_flat_matches_legacy_on_sparse_samples() {
        let mut rng = StdRng::seed_from_u64(0x5eed_cafe);
        for _ in 0..16 {
            let hz = construct_sparse_hz(20_000, 1e-3, &mut rng);
            let mut legacy = hz.clone();
            legacy.normalize_legacy();
            let mut flat = FlatHybridNumber::from_legacy(&hz);
            normalize_flat_native(&mut flat);
            assert_eq!(flat.eval(), legacy.eval());
        }
    }

    #[test]
    fn normalize_flat_matches_legacy_on_100k_sparse_samples() {
        let mut rng = StdRng::seed_from_u64(0xA11C_E5ED);
        for _ in 0..32 {
            let hz = construct_sparse_hz(100_000, 1e-4, &mut rng);
            let mut legacy = hz.clone();
            legacy.normalize_legacy();
            let mut flat = FlatHybridNumber::from_legacy(&hz);
            normalize_flat_native(&mut flat);
            assert_eq!(flat.eval(), legacy.eval());
        }
    }

    #[test]
    fn profile_normalize_matches_plain_normalize() {
        let mut rng = StdRng::seed_from_u64(0xFACE_FEED);
        for _ in 0..16 {
            let hz = construct_sparse_hz(100_000, 1e-5, &mut rng);
            let mut plain = FlatHybridNumber::from_legacy(&hz);
            let mut profiled = plain.clone();
            normalize_flat_native(&mut plain);
            let profile = profile_normalize_flat_native(&mut profiled);
            assert_eq!(plain.eval(), profiled.eval());
            assert!(profile
                .levels
                .iter()
                .any(|stats| stats.u64_calls + stats.sparse_calls + stats.canonical_skips > 0));
        }
    }

    #[test]
    fn native_normalize_correctness_1000() {
        let mut rng = StdRng::seed_from_u64(0x1234_5678_9abc_def0);
        for _ in 0..1000 {
            let hz = construct_sparse_hz(20_000, 1e-3, &mut rng);
            let mut legacy = hz.clone();
            legacy.normalize_legacy();
            let mut flat = FlatHybridNumber::from_legacy(&hz);
            normalize_flat_native(&mut flat);
            assert_eq!(flat.eval(), legacy.eval());
        }
    }

    #[test]
    fn sparse_normalize_1000_random() {
        let mut rng = StdRng::seed_from_u64(0x5A125ECA);
        for _ in 0..1000 {
            let hz = construct_sparse_hz(20_000, 1e-3, &mut rng);
            let mut legacy = hz.clone();
            legacy.normalize_legacy();
            let mut flat = FlatHybridNumber::from_legacy(&hz);
            normalize_flat_native(&mut flat);
            assert_eq!(flat.eval(), legacy.eval());
        }
    }

    #[test]
    fn random_multiset_normalization_matches_legacy_eval() {
        let mut rng = StdRng::seed_from_u64(0xabcddcba);
        for _ in 0..128 {
            let len = 1 + (rng.next_u32() % 48) as usize;
            let mut indices = Vec::with_capacity(len);
            for _ in 0..len {
                indices.push(2 + (rng.next_u32() % 120));
            }
            let native = normalize_indices(&indices);
            let legacy = zeckendorf(&lazy_eval_fib(&indices));
            assert_eq!(native, legacy);
        }
    }
}
