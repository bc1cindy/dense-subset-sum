//! Subset-sum sumset with saturating u8 counts (privacy decision is binary above thousands, so wider counts would be overkill).

use std::collections::HashMap;

/// `LowerBound` means counts undercount truth; `bounded` truncation introduces it and convolve propagates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Exact,
    LowerBound,
}

impl Bound {
    /// Absorbing element pattern: `LowerBound` taints any composition.
    pub fn join(self, other: Self) -> Self {
        match (self, other) {
            (Bound::Exact, Bound::Exact) => Bound::Exact,
            _ => Bound::LowerBound,
        }
    }
}

pub struct Sumset {
    counts: HashMap<u64, u8>,
    bound: Bound,
}

impl Sumset {
    pub fn empty() -> Self {
        let mut counts: HashMap<u64, u8> = HashMap::with_capacity(1);
        counts.insert(0, 1);
        Self {
            counts,
            bound: Bound::Exact,
        }
    }

    /// `x == 0` collapses both subsets to sum 0, so `count(0) == 2`.
    fn from_one_element(x: u64) -> Self {
        let mut counts: HashMap<u64, u8> = HashMap::with_capacity(2);
        counts.insert(0, 1);
        counts
            .entry(x)
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);
        Self {
            counts,
            bound: Bound::Exact,
        }
    }

    /// Multiset semantics: distinct positions are distinct subsets, so `powerset(&[1, 1])` enumerates 4 subsets and `count(1) == 2`.
    pub fn powerset(elements: &[u64]) -> Self {
        match elements {
            [] => Self::empty(),
            [single] => Self::from_one_element(*single),
            _ => {
                let mid = elements.len() / 2;
                let left = Self::powerset(&elements[..mid]);
                let right = Self::powerset(&elements[mid..]);
                Self::convolve(&left, &right)
            }
        }
    }

    /// Truncating at `fixed_degree < elements.len()` drops higher-degree contributions, so result is a lower bound.
    pub fn bounded(elements: &[u64], fixed_degree: usize) -> Self {
        let mut counts: HashMap<u64, u8> = HashMap::new();
        counts.insert(0, 1);
        walk_subsets_up_to(elements, fixed_degree, &mut |sum| {
            counts
                .entry(sum)
                .and_modify(|c| *c = c.saturating_add(1))
                .or_insert(1);
        });
        let bound = if fixed_degree < elements.len() {
            Bound::LowerBound
        } else {
            Bound::Exact
        };
        Self { counts, bound }
    }

    pub fn includes(&self, e: u64) -> bool {
        self.counts.contains_key(&e)
    }

    /// Returns 0 for unreachable sums; when `bound() == Bound::LowerBound`, the value undercounts truth.
    pub fn count(&self, e: u64) -> u8 {
        self.counts.get(&e).copied().unwrap_or(0)
    }

    pub fn sums(&self) -> impl Iterator<Item = u64> + '_ {
        self.counts.keys().copied()
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    pub fn bound(&self) -> Bound {
        self.bound
    }

    /// `true` is conservative when `other` is `LowerBound`; `false` may be a false negative under bounded enumeration.
    pub fn covers(&self, other: &Self, threshold: u8) -> bool {
        self.sums().all(|s| other.count(s) >= threshold)
    }

    /// Returns `true` iff some pair `(s_self, s_other)` satisfies `s_self − s_other = target`;
    /// iterates the smaller side and looks up complements in the larger.
    pub fn includes_balance(&self, other: &Self, target: i64) -> bool {
        if self.len() <= other.len() {
            any_complement(self, other, target)
        } else {
            any_complement(other, self, -target)
        }
    }

    /// Saturating count of pairs `(s_self, s_other)` with `s_self − s_other = target`,
    /// clamped at `u8::MAX = 255` since the privacy decision is binary above the threshold.
    /// Result undercounts when products saturate or when either operand is `Bound::LowerBound`.
    pub fn count_balance(&self, other: &Self, target: i64) -> u8 {
        self.sums()
            .filter_map(|s| {
                let s_signed = i64::try_from(s).ok()?;
                let needed = s_signed.checked_sub(target)?;
                if needed < 0 {
                    return None;
                }
                Some(self.count(s).saturating_mul(other.count(needed as u64)))
            })
            .fold(0u8, u8::saturating_add)
    }

    /// `(A * B)_k = Σ_{i+j=k} A_i · B_j`; bound joins via the absorbing element pattern.
    pub fn convolve(a: &Self, b: &Self) -> Self {
        let mut counts: HashMap<u64, u8> = HashMap::new();
        for (&x, &cx) in &a.counts {
            for (&y, &cy) in &b.counts {
                let sum = x.saturating_add(y);
                let contrib = cx.saturating_mul(cy);
                counts
                    .entry(sum)
                    .and_modify(|c| *c = c.saturating_add(contrib))
                    .or_insert(contrib);
            }
        }
        Self {
            counts,
            bound: a.bound.join(b.bound),
        }
    }
}

/// `target` is negated when iterating the smaller side from `other` so the equation
/// `s_self − s_other = target` mirrors to `s_other − s_self = −target`.
fn any_complement(iter_side: &Sumset, lookup_side: &Sumset, target: i64) -> bool {
    iter_side.sums().any(|s| {
        let Ok(s_signed) = i64::try_from(s) else {
            return false;
        };
        let Some(complement) = s_signed.checked_sub(target) else {
            return false;
        };
        complement >= 0 && lookup_side.includes(complement as u64)
    })
}

/// Excludes the empty subset; callers insert it explicitly so `count(0)` matches their merge semantics.
fn walk_subsets_up_to(elements: &[u64], fixed_degree: usize, f: &mut impl FnMut(u64)) {
    fn dfs(
        elements: &[u64],
        start: usize,
        depth: usize,
        fixed_degree: usize,
        sum: u64,
        f: &mut impl FnMut(u64),
    ) {
        if depth > 0 {
            f(sum);
        }
        if depth == fixed_degree {
            return;
        }
        for i in start..elements.len() {
            dfs(
                elements,
                i + 1,
                depth + 1,
                fixed_degree,
                sum.saturating_add(elements[i]),
                f,
            );
        }
    }
    dfs(elements, 0, 0, fixed_degree, 0, f);
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap as StdHashMap;

    fn brute_force(set: &[u64]) -> StdHashMap<u64, u8> {
        let n = set.len();
        let mut out: StdHashMap<u64, u8> = StdHashMap::new();
        for mask in 0u32..(1u32 << n) {
            let sum: u64 = (0..n)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| set[i])
                .sum();
            out.entry(sum)
                .and_modify(|c| *c = c.saturating_add(1))
                .or_insert(1);
        }
        out
    }

    fn brute_force_bounded(set: &[u64], fixed_degree: usize) -> StdHashMap<u64, u8> {
        let n = set.len();
        let mut out: StdHashMap<u64, u8> = StdHashMap::new();
        for mask in 0u32..(1u32 << n) {
            let size = mask.count_ones() as usize;
            if size > fixed_degree {
                continue;
            }
            let sum: u64 = (0..n)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| set[i])
                .sum();
            out.entry(sum)
                .and_modify(|c| *c = c.saturating_add(1))
                .or_insert(1);
        }
        out
    }

    fn assert_matches_brute_force(set: &[u64]) {
        let s = Sumset::powerset(set);
        let bf = brute_force(set);
        assert_eq!(s.len(), bf.len(), "set={:?}", set);
        for (&sum, &count) in &bf {
            assert_eq!(s.count(sum), count, "set={:?} sum={}", set, sum);
        }
    }

    #[test]
    fn empty_contains_zero_only() {
        let s = Sumset::empty();
        assert!(s.includes(0));
        assert_eq!(s.count(0), 1);
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
        assert!(!s.includes(1));
        assert_eq!(s.count(1), 0);
    }

    #[test]
    fn singleton_contains_zero_and_value() {
        let s = Sumset::from_one_element(7);
        assert!(s.includes(0));
        assert!(s.includes(7));
        assert_eq!(s.count(0), 1);
        assert_eq!(s.count(7), 1);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn singleton_zero_doubles_count_at_zero() {
        let s = Sumset::from_one_element(0);
        assert!(s.includes(0));
        assert_eq!(s.count(0), 2);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn powerset_empty() {
        let s = Sumset::powerset(&[]);
        assert!(s.includes(0));
        assert_eq!(s.count(0), 1);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn powerset_distinct_values() {
        assert_matches_brute_force(&[3, 5, 7]);
    }

    #[test]
    fn powerset_with_duplicates() {
        assert_matches_brute_force(&[1, 1, 2]);
    }

    #[test]
    fn convolve_with_empty_is_identity() {
        let a = Sumset::powerset(&[3, 5, 7]);
        let e = Sumset::empty();
        let convolved = Sumset::convolve(&a, &e);
        assert_eq!(convolved.len(), a.len());
        for sum in a.sums() {
            assert!(convolved.includes(sum));
            assert_eq!(convolved.count(sum), a.count(sum));
        }
    }

    #[test]
    fn convolve_is_commutative() {
        let a = Sumset::powerset(&[2, 3]);
        let b = Sumset::powerset(&[5, 7]);
        let ab = Sumset::convolve(&a, &b);
        let ba = Sumset::convolve(&b, &a);
        assert_eq!(ab.len(), ba.len());
        for sum in ab.sums() {
            assert_eq!(ab.count(sum), ba.count(sum));
        }
    }

    #[test]
    fn count_just_below_saturation() {
        let zeros = vec![0u64; 7];
        let s = Sumset::powerset(&zeros);
        assert_eq!(s.count(0), 128);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn count_saturates_at_u8_max() {
        let zeros = vec![0u64; 8];
        let s = Sumset::powerset(&zeros);
        assert_eq!(s.count(0), u8::MAX);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn bounded_max_zero_is_empty() {
        let s = Sumset::bounded(&[3, 5, 7], 0);
        assert_eq!(s.len(), 1);
        assert!(s.includes(0));
        assert_eq!(s.count(0), 1);
    }

    #[test]
    fn bounded_max_one_is_singletons_plus_empty() {
        let s = Sumset::bounded(&[3, 5, 7], 1);
        assert!(s.includes(0));
        assert!(s.includes(3));
        assert!(s.includes(5));
        assert!(s.includes(7));
        assert!(!s.includes(8));
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn bounded_full_matches_powerset() {
        let set = [1, 1, 2, 3];
        let bounded = Sumset::bounded(&set, set.len());
        let full = Sumset::powerset(&set);
        assert_eq!(bounded.len(), full.len());
        for sum in full.sums() {
            assert_eq!(bounded.count(sum), full.count(sum));
        }
    }

    #[test]
    fn exact_constructors_are_not_lower_bound() {
        assert_eq!(Sumset::empty().bound(), Bound::Exact);
        assert_eq!(Sumset::from_one_element(7).bound(), Bound::Exact);
        assert_eq!(Sumset::powerset(&[3, 5, 7]).bound(), Bound::Exact);
    }

    #[test]
    fn bounded_marks_lower_bound_when_truncating() {
        let set = [1, 2, 3, 4, 5];
        assert_eq!(Sumset::bounded(&set, 2).bound(), Bound::LowerBound);
        assert_eq!(Sumset::bounded(&set, set.len()).bound(), Bound::Exact);
        assert_eq!(Sumset::bounded(&set, set.len() + 10).bound(), Bound::Exact);
    }

    #[test]
    fn convolve_propagates_lower_bound() {
        let exact = Sumset::powerset(&[3, 5]);
        let bounded = Sumset::bounded(&[1, 2, 3, 4], 2);
        assert_eq!(Sumset::convolve(&exact, &exact).bound(), Bound::Exact);
        assert_eq!(Sumset::convolve(&exact, &bounded).bound(), Bound::LowerBound);
        assert_eq!(Sumset::convolve(&bounded, &exact).bound(), Bound::LowerBound);
        assert_eq!(Sumset::convolve(&bounded, &bounded).bound(), Bound::LowerBound);
    }

    #[test]
    fn bound_join_absorbs_lower_bound() {
        assert_eq!(Bound::Exact.join(Bound::Exact), Bound::Exact);
        assert_eq!(Bound::Exact.join(Bound::LowerBound), Bound::LowerBound);
        assert_eq!(Bound::LowerBound.join(Bound::Exact), Bound::LowerBound);
        assert_eq!(Bound::LowerBound.join(Bound::LowerBound), Bound::LowerBound);
    }

    fn brute_force_balance(pos: &[u64], neg: &[u64], target: i64) -> u8 {
        let mut total: u8 = 0;
        for p_mask in 0u32..(1u32 << pos.len()) {
            let s_pos: i64 = (0..pos.len())
                .filter(|i| p_mask & (1 << i) != 0)
                .map(|i| pos[i] as i64)
                .sum();
            for n_mask in 0u32..(1u32 << neg.len()) {
                let s_neg: i64 = (0..neg.len())
                    .filter(|i| n_mask & (1 << i) != 0)
                    .map(|i| neg[i] as i64)
                    .sum();
                if s_pos - s_neg == target {
                    total = total.saturating_add(1);
                }
            }
        }
        total
    }

    #[test]
    fn includes_balance_empty_sides_at_zero() {
        let e = Sumset::empty();
        assert!(e.includes_balance(&e, 0));
        assert!(!e.includes_balance(&e, 1));
        assert!(!e.includes_balance(&e, -1));
    }

    #[test]
    fn includes_balance_singleton_pair() {
        let p = Sumset::powerset(&[5]);
        let n = Sumset::powerset(&[3]);
        assert!(p.includes_balance(&n, 2));
        assert!(p.includes_balance(&n, 5));
        assert!(p.includes_balance(&n, -3));
        assert!(p.includes_balance(&n, 0));
        assert!(!p.includes_balance(&n, 100));
    }

    #[test]
    fn includes_balance_iteration_side_does_not_change_answer() {
        // Forces the iterate-other-side branch via len asymmetry.
        let p = Sumset::powerset(&[2, 3, 5, 7]);
        let n = Sumset::powerset(&[4]);
        assert_eq!(p.includes_balance(&n, 1), n.includes_balance(&p, -1));
    }

    #[test]
    fn count_balance_empty_at_zero() {
        let e = Sumset::empty();
        assert_eq!(e.count_balance(&e, 0), 1);
        assert_eq!(e.count_balance(&e, 1), 0);
    }

    #[test]
    fn count_balance_singleton_pair() {
        let p = Sumset::powerset(&[5]);
        let n = Sumset::powerset(&[3]);
        // {5}-{3}=2, {5}-{}=5, {}-{3}=-3, {}-{}=0; one pair per target.
        assert_eq!(p.count_balance(&n, 2), 1);
        assert_eq!(p.count_balance(&n, 5), 1);
        assert_eq!(p.count_balance(&n, -3), 1);
        assert_eq!(p.count_balance(&n, 0), 1);
        assert_eq!(p.count_balance(&n, 100), 0);
    }

    #[test]
    fn count_balance_saturates_at_u8_max() {
        // 256 subsets of zeros all sum to 0, so count(0) saturates at 255.
        // count_balance multiplies and sums saturating in u8, so the result also clamps at 255.
        let zeros = vec![0u64; 8];
        let s = Sumset::powerset(&zeros);
        assert_eq!(s.count(0), u8::MAX);
        assert_eq!(s.count_balance(&s, 0), u8::MAX);
    }

    proptest! {
        #[test]
        fn powerset_matches_brute_force(
            set in proptest::collection::vec(0u64..50, 0..10)
        ) {
            let s = Sumset::powerset(&set);
            let bf = brute_force(&set);
            prop_assert_eq!(s.len(), bf.len());
            for (sum, count) in bf {
                prop_assert_eq!(s.count(sum), count);
            }
        }

        #[test]
        fn bounded_matches_brute_force(
            set in proptest::collection::vec(0u64..30, 0..8),
            fixed_degree in 0usize..=8,
        ) {
            let s = Sumset::bounded(&set, fixed_degree);
            let bf = brute_force_bounded(&set, fixed_degree);
            prop_assert_eq!(s.len(), bf.len());
            for (sum, count) in bf {
                prop_assert_eq!(s.count(sum), count);
            }
        }

        #[test]
        fn convolve_is_associative(
            a in proptest::collection::vec(0u64..30, 0..5),
            b in proptest::collection::vec(0u64..30, 0..5),
            c in proptest::collection::vec(0u64..30, 0..5),
        ) {
            let sa = Sumset::powerset(&a);
            let sb = Sumset::powerset(&b);
            let sc = Sumset::powerset(&c);
            let ab_c = Sumset::convolve(&Sumset::convolve(&sa, &sb), &sc);
            let a_bc = Sumset::convolve(&sa, &Sumset::convolve(&sb, &sc));
            prop_assert_eq!(ab_c.len(), a_bc.len());
            for sum in ab_c.sums() {
                prop_assert_eq!(ab_c.count(sum), a_bc.count(sum));
            }
        }

        /// Inputs kept small enough that no per-sum count saturates at u8, so the
        /// proptest checks exact equality with the brute-force oracle.
        #[test]
        fn count_balance_matches_brute_force(
            pos in proptest::collection::vec(0u64..20, 0..6),
            neg in proptest::collection::vec(0u64..20, 0..6),
            target in -50i64..50,
        ) {
            let p = Sumset::powerset(&pos);
            let n = Sumset::powerset(&neg);
            prop_assert_eq!(
                p.count_balance(&n, target),
                brute_force_balance(&pos, &neg, target)
            );
        }

        #[test]
        fn includes_balance_matches_brute_force(
            pos in proptest::collection::vec(0u64..30, 0..6),
            neg in proptest::collection::vec(0u64..30, 0..6),
            target in -100i64..100,
        ) {
            let p = Sumset::powerset(&pos);
            let n = Sumset::powerset(&neg);
            prop_assert_eq!(
                p.includes_balance(&n, target),
                brute_force_balance(&pos, &neg, target) > 0
            );
        }
    }
}
