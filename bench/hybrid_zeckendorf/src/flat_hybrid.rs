//! Flat-array HybridNumber using inline storage.

use smallvec::SmallVec;
use std::array;

pub const MAX_LEVELS: usize = 32;
pub const INLINE_CAP: usize = 3;

pub type Payload = SmallVec<[u32; INLINE_CAP]>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchQuery {
    FullEval,
    LevelSum { level: usize },
    SupportCardinality,
    ActiveLevels,
}

#[derive(Debug, Clone)]
pub struct FlatHybridNumber {
    pub levels: Box<[Payload; MAX_LEVELS]>,
    pub active_mask: u32,
}

impl FlatHybridNumber {
    pub fn empty() -> Self {
        Self {
            levels: Box::new(array::from_fn(|_| Payload::new())),
            active_mask: 0,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.active_mask == 0
    }

    pub fn active_levels(&self) -> u32 {
        self.active_mask.count_ones()
    }

    pub fn support_card(&self) -> u32 {
        self.levels.iter().map(|payload| payload.len() as u32).sum()
    }

    pub fn set_level(&mut self, level: usize, indices: &[u32]) {
        assert!(
            level < MAX_LEVELS,
            "level {level} exceeds MAX_LEVELS={MAX_LEVELS}"
        );
        self.levels[level].clear();
        self.levels[level].extend_from_slice(indices);
        if indices.is_empty() {
            self.active_mask &= !(1u32 << level);
        } else {
            self.active_mask |= 1u32 << level;
        }
    }

    pub fn add_lazy(&self, other: &Self) -> Self {
        let mut result = self.clone();
        let mut mask = other.active_mask;
        while mask != 0 {
            let level = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            result.levels[level].extend_from_slice(&other.levels[level]);
            result.active_mask |= 1u32 << level;
        }
        result
    }

    pub fn add_lazy_mut(&mut self, other: &Self) {
        let mut mask = other.active_mask;
        while mask != 0 {
            let level = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            self.levels[level].extend_from_slice(&other.levels[level]);
            self.active_mask |= 1u32 << level;
        }
    }

    pub fn normalize_native(&mut self) {
        crate::zeckendorf_native::normalize_flat_native(self);
    }

    pub fn batch_readout(&mut self, queries: &[BatchQuery]) -> Vec<rug::Integer> {
        self.normalize_native();
        queries.iter().map(|query| self.eval_query(query)).collect()
    }

    fn eval_query(&self, query: &BatchQuery) -> rug::Integer {
        match *query {
            BatchQuery::FullEval => self.eval(),
            BatchQuery::LevelSum { level } => {
                use crate::fib_table::fib;
                if level >= MAX_LEVELS {
                    return rug::Integer::from(0);
                }
                self.levels[level].iter().map(|&i| fib(i)).sum()
            }
            BatchQuery::SupportCardinality => rug::Integer::from(self.support_card()),
            BatchQuery::ActiveLevels => rug::Integer::from(self.active_levels()),
        }
    }

    pub fn add_native(&self, other: &Self) -> Self {
        let mut out = self.add_lazy(other);
        out.normalize_native();
        out
    }

    pub fn eval(&self) -> rug::Integer {
        use crate::fib_table::fib;
        use crate::weight::weight;
        let mut sum = rug::Integer::from(0);
        let mut mask = self.active_mask;
        while mask != 0 {
            let level = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let coeff: rug::Integer = self.levels[level].iter().map(|&i| fib(i)).sum();
            sum += coeff * weight(level as u32);
        }
        sum
    }

    pub fn from_legacy(legacy: &crate::normalization::HybridNumber) -> Self {
        let mut flat = Self::empty();
        for (&level, payload) in &legacy.levels {
            assert!(
                level < MAX_LEVELS as u32,
                "legacy level {level} exceeds MAX_LEVELS={MAX_LEVELS}"
            );
            if !payload.is_empty() {
                flat.set_level(level as usize, payload);
            }
        }
        flat
    }

    pub fn to_legacy(&self) -> crate::normalization::HybridNumber {
        use std::collections::BTreeMap;

        let mut levels = BTreeMap::new();
        let mut mask = self.active_mask;
        while mask != 0 {
            let level = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            if !self.levels[level].is_empty() {
                levels.insert(level as u32, self.levels[level].to_vec());
            }
        }
        crate::normalization::HybridNumber { levels }
    }
}

impl Default for FlatHybridNumber {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn active_mask_tracks_payload_changes() {
        let mut h = FlatHybridNumber::empty();
        assert_eq!(h.active_mask, 0);
        h.set_level(3, &[2, 4, 6]);
        assert_eq!(h.active_mask, 0b1000);
        h.set_level(7, &[3, 5]);
        assert_eq!(h.active_mask, 0b10001000);
        h.set_level(3, &[]);
        assert_eq!(h.active_mask, 0b10000000);
    }

    #[test]
    fn flat_roundtrip_preserves_eval() {
        let mut levels = BTreeMap::new();
        levels.insert(0u32, vec![2, 4, 6]);
        levels.insert(2u32, vec![3, 7, 9]);
        let legacy = crate::normalization::HybridNumber { levels };
        let flat = FlatHybridNumber::from_legacy(&legacy);
        let roundtrip = flat.to_legacy();
        assert_eq!(legacy.eval(), flat.eval());
        assert_eq!(legacy.eval(), roundtrip.eval());
    }

    #[test]
    fn flat_add_lazy_matches_legacy_eval() {
        let a = crate::normalization::HybridNumber::from_u64(12_345);
        let b = crate::normalization::HybridNumber::from_u64(67_890);
        let flat_sum =
            FlatHybridNumber::from_legacy(&a).add_lazy(&FlatHybridNumber::from_legacy(&b));
        assert_eq!(flat_sum.eval(), a.eval() + b.eval());
    }

    #[test]
    fn flat_add_lazy_mut_matches_clone_path() {
        let a = crate::normalization::HybridNumber::from_u64(12_345);
        let b = crate::normalization::HybridNumber::from_u64(67_890);
        let flat_a = FlatHybridNumber::from_legacy(&a);
        let flat_b = FlatHybridNumber::from_legacy(&b);

        let clone_path = flat_a.add_lazy(&flat_b);
        let mut in_place = flat_a.clone();
        in_place.add_lazy_mut(&flat_b);

        assert_eq!(clone_path.eval(), in_place.eval());
        assert_eq!(clone_path.active_mask, in_place.active_mask);
    }

    #[test]
    fn flat_roundtrip_preserves_high_level_eval() {
        let mut levels = BTreeMap::new();
        levels.insert(0u32, vec![2, 4, 6]);
        levels.insert(19u32, vec![3, 7, 11]);
        let legacy = crate::normalization::HybridNumber { levels };
        let flat = FlatHybridNumber::from_legacy(&legacy);
        let roundtrip = flat.to_legacy();
        assert_eq!(legacy.eval(), flat.eval());
        assert_eq!(legacy.eval(), roundtrip.eval());
    }

    #[test]
    fn flat_hybrid_number_is_small() {
        assert!(
            std::mem::size_of::<FlatHybridNumber>() <= 1024,
            "FlatHybridNumber is {} bytes, must be <= 1024",
            std::mem::size_of::<FlatHybridNumber>()
        );
    }

    #[test]
    fn batch_readout_matches_normalized_single_readouts() {
        let a = crate::normalization::HybridNumber::from_u64(12_345);
        let b = crate::normalization::HybridNumber::from_u64(67_890);
        let mut batched =
            FlatHybridNumber::from_legacy(&a).add_lazy(&FlatHybridNumber::from_legacy(&b));
        let mut single = batched.clone();
        single.normalize_native();

        let queries = [
            BatchQuery::FullEval,
            BatchQuery::SupportCardinality,
            BatchQuery::ActiveLevels,
            BatchQuery::LevelSum { level: 0 },
        ];
        let batch_values = batched.batch_readout(&queries);
        let single_values = queries
            .iter()
            .map(|query| single.eval_query(query))
            .collect::<Vec<_>>();

        assert_eq!(batch_values, single_values);
        assert_eq!(batched.eval(), single.eval());
    }
}
