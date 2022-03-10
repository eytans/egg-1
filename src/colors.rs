use std::borrow::BorrowMut;
use std::cmp::{max, min};
pub use crate::{Id, EGraph, Language, Analysis, ColorId};
use crate::{Singleton, UnionFind};
use crate::util::JoinDisp;
use std::collections::{HashSet, HashMap};
use itertools::Itertools;
use std::fmt::Formatter;
use indexmap::IndexSet;
use log::{error, info, trace, warn};
use crate::egraph::DenseNodeColors;

pub type ColorParents = smallvec::SmallVec<[ColorId; 3]>;

global_counter!(COLOR_IDS, usize, usize::default());

#[derive(Clone, Default)]
pub struct Color {
    union_find: UnionFind,
    color_id: ColorId,
    /// Used for rebuilding uf
    pub(crate) dirty_unions: Vec<Id>,
    /// Maintain which classes in black are represented in colored class (including rep)
    union_map: HashMap<Id, HashSet<Id>>,
    children: Vec<ColorId>,
    base_set: Vec<ColorId>,
}

impl Color {
    pub(crate) fn new(union_find: &UnionFind, new_id: ColorId) -> Color {
        Color { union_find: union_find.clone(), color_id: new_id, dirty_unions: Default::default(), union_map: Default::default(), children: vec![], base_set: vec![new_id] }
    }

    pub fn get_id(&self) -> ColorId {
        self.color_id
    }

    pub fn children(&self) -> &[ColorId] {
        &self.children
    }

    pub fn find(&self, id: Id) -> Id {
        self.union_find.find(id)
    }

    /// Keep black ids up with current version of colored and black uf.
    /// `id1` Should be the id of "to" (after running find in black)
    /// `id2` Should be the id of "from" (after running find in black)
    /// `to` Should be the id of "to" (after running find in color)
    /// `from` Should be the id of "from" (after running find in color)
    /// `switched` if id1 is from and id2 is to
    fn update_union_map(&mut self, id1: Id, id2: Id, to: Id, from: Id) {
        // If to != from then the underlying assumption is that id1 != id2
        let rep = min(to, from);
        if to != from {
            let non_rep = max(to, from);
            let from_ids = self.union_map.remove(&non_rep).unwrap_or_else(|| HashSet::singleton(non_rep));
            let to_ids = self.union_map.entry(rep).or_insert_with(|| HashSet::singleton(rep));
            to_ids.retain(|id| !from_ids.contains(id));
            self.union_map.entry(rep).and_modify(|s| { s.remove(&max(id1, id2)); });
        } else if id1 != id2 {
            // We have to remove someone (because something was merged in black.
            // But sometimes `from` and `to` are merged in the color.
            // In this case, id2 might be equal to `to`, and in that case we should change
            // the equality class representative to be id1 (to reflect black).
            self.union_map.entry(rep).and_modify(|s| { s.remove(&max(id1, id2)); });
        }
        if self.union_map.get(&rep).map_or(false, |s| s.len() == 1) {
            debug_assert!(self.union_map.get(&rep).unwrap().contains(&rep), "We should always have the representative in the map");
            self.union_map.remove(&rep);
        }
    }

    pub fn add(&mut self, id: Id) {
        assert_eq!(self.union_find.make_set(), id);
    }

    fn union_impl(&mut self, id1: Id, id2: Id) -> (Id, Id, bool) {
        let (to, from) = self.union_find.union(id1, id2);
        debug_assert_eq!(to, self.union_find.find(id1));
        debug_assert_eq!(to, self.union_find.find(id2));
        let union = (to, from, to != from);
        if union.2 && cfg!(feature = "upward-merging") {
            unimplemented!("Upward merging not supported for colored graph");
            // self.process_unions();
        }
        union
    }

    /// Union to keep up with current version of black uf.
    /// `id1` Should be the id of "to" (after running find in black)
    /// `id2` Should be the id of "from" (after running find in black)
    pub fn black_union(&mut self, id1: Id, id2: Id) -> (Id, bool) {
        let orig_to = self.find(id1);
        let orig_from = self.find(id2);
        let (mut to, mut from, changed) = self.union_impl(id1, id2);
        self.update_union_map(id1, id2, orig_to, orig_from);
        (to, changed)
    }

    pub fn colored_union(&mut self, id1: Id, id2: Id) -> (Id, bool) {
        let (to, from, changed) = self.union_impl(id1, id2);
        if changed {
            self.dirty_unions.push(to);
            let from_ids = self.union_map.remove(&from).unwrap_or_else(|| HashSet::singleton(from));
            self.union_map.entry(to).or_insert_with(|| HashSet::singleton(to)).extend(from_ids);
        }
        (to, changed)
    }

    /// Reapply congruence closure for color.
    /// Returns which colors to remove from which edges.
    // TODO: When deleting an edge we need to remove it from: node, parents, dirty
    //       colors, and memo (parent can include other classes). Removing an edge
    //       from many classes may be expensive (search wise) so it might be better
    //       o do it during rebuild classes.
    pub fn cong_closure<L: Language, N: Analysis<L>>(&mut self, egraph: &mut EGraph<L, N>, black_merged: &[(Id, Id)]) {
        self.assert_black_ids(egraph);
        debug_assert!(black_merged.iter()
            .all(|(id1, id2)| self.find(*id1) == self.find(*id2)));


        let mut to_union = vec![];
        // We need to build memo ahead of time because even a single merge might miss needed unions.
        // Need to do some merging and initial color deletion here because we are deleting duplicate
        // edges.
        let mut memo: HashMap<L, (Id, DenseNodeColors)> = {
            let mut v = egraph.memo.iter()
                .filter(|(_, (_, cs))| cs.not_any() || cs[self.color_id.0])
                .map(|(orig, (e, cs))| {
                    let mut cloned = orig.clone();
                    cloned.update_children(|c| self.find(c));
                    (cloned, (self.find(*e), cs.clone()))
                }).sorted()
                .collect_vec();

            // TODO: Create dedup with side effects for iterator
            v.dedup_by(|(n1, (e1, cs1)), (n2, (e2, cs2))| {
                n1 == n2 && {
                    <EGraph<L, N>>::update_memoed(cs1, cs2);
                    to_union.push((*e1, *e2));
                    true
                }
            });
            v.into_iter().map(|(n, (e, cs))| (n, (e, cs))).collect()
        };

        for (id1, id2) in to_union.drain(..) {
            self.colored_union(id1, id2);
        }

        while !self.dirty_unions.is_empty() {
            // take the worklist, we'll get the stuff that's added the next time around
            // deduplicate the dirty list to avoid extra work
            let mut todo = std::mem::take(&mut self.dirty_unions);
            for id in todo.iter_mut() {
                *id = self.union_find.find(*id);
            }
            if cfg!(not(feature = "upward-merging")) {
                todo.sort_unstable();
                todo.dedup();
            }
            assert!(!todo.is_empty());

            // rep to all contained
            let all_groups = self.union_find.build_sets();
            for id in todo {
                let mut parents: Vec<(L, DenseNodeColors, Id)> = all_groups.get(&id)
                    .unwrap().iter()
                    .flat_map(|g| egraph[*g].parents.iter())
                    .filter(|(_, cs, _)| cs.not_any() || cs[self.color_id.0])
                    .map(|(n, cs, id)| {
                        memo.remove(n);
                        let mut res = n.clone();
                        res.update_children(|child| self.union_find.find(child));
                        (res, cs.clone(), (self.union_find.find(*id)))
                    }).collect();

                // We assume the edges are pretty sorted so sort will be faster then unstable_sort.
                parents.sort();
                // When two edges are equal, and one of them is colored, we may be able to remove
                // the edges dependence on the color. Because we can parallelize over the
                // different colors, we keep the egraph immutable. Later we will update memo for all
                // colors at once, and if some edges becomes colorless we can remove it during
                // `rebuild_classes`.
                parents.dedup_by(|
                    (n1, memoed1, e1),
                    (n2, memoed2, e2)| {
                    n1 == n2 && {
                        <EGraph<L, N>>::update_memoed(memoed1, memoed2);
                        to_union.push((*e1, *e2));
                        true
                    }
                });

                for (n, cs , e) in parents.iter_mut() {
                    // It is possible we don't have colors because parent was updated from different
                    // class.
                    let res = <EGraph<L, N>>::update_memo_from_parent(&mut memo, n, e, cs);
                    to_union.extend(res.into_iter());
                }
            }

            for (id1, id2) in to_union.drain(..) {
                self.colored_union(id1, id2);
            }
        }

        assert!(self.dirty_unions.is_empty(), "Dirty unions should be empty {}", self.dirty_unions.iter().join(", "));
        assert!(to_union.is_empty(), "to_union should be empty {}", to_union.iter().map(|x| format!("{}-{}", x.0, x.1)).join(", "));
        self.assert_black_ids(egraph);
    }

    pub fn black_ids(&self, id: Id) -> Option<&HashSet<Id>> {
        self.union_map.get(&self.union_find.find(id))
    }

    pub fn merge_ufs(&mut self, others: Vec<&mut Self>, new_id: ColorId, is_base: bool) -> Self {
        let mut res = self.clone();
        for other in others {
            res.dirty_unions.extend_from_slice(&other.dirty_unions);
            for (black_id, ids) in &other.union_map {
                for id in ids {
                    res.colored_union(*black_id, *id);
                }
            }
            other.children.push(new_id);
            res.base_set.extend(&other.base_set);
        }
        res.color_id = new_id;
        self.children.push(res.color_id);
        if is_base {
            res.base_set.push(new_id);
        }
        res.base_set = res.base_set.iter().sorted().dedup().copied().collect_vec();
        res
    }

    pub fn merge_uf(&mut self, other: &mut Self, new_id: ColorId) -> Self {
        self.merge_ufs(vec![other], new_id, false)
    }

    pub fn new_child(&mut self, new_id: ColorId) -> Self {
        let mut res = self.merge_ufs(vec![], new_id, true);
        res
    }

    pub fn assumptions(&self) -> &Vec<ColorId> {
        &self.base_set
    }

    pub fn assert_black_ids<L, N>(&self, egraph: &EGraph<L, N>)
        where L: Language, N: Analysis<L> {
        // Check that black ids are actually black representatives
        if cfg!(debug_assertions) {
            for (_, set) in &self.union_map {
                for id in set {
                    trace!("checking {:?} is black rep", id);
                    debug_assert!(egraph.find(*id) == *id, "black id {:?} is not black rep {:?}", id, egraph.find(*id));
                }
            }
        }
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Color(id={}, groups={})", self.color_id, self.union_map.iter().map(|(id, set)| format!("{} - {}", id, set.iter().sep_string(" "))).join(", "))
    }
}

#[cfg(test)]
mod test {
    use crate::colors::Color;
    use crate::{UnionFind, ColorId};

    #[test]
    fn test_black_union_alone() {
        let mut uf = UnionFind::default();
        let id1 = uf.make_set();
        let id2 = uf.make_set();
        let mut color = Color::new(&uf, ColorId::from(0));
        color.black_union(id1, id2);
        color.black_union(id1, id2);
        color.black_union(id1, id1);
        assert_eq!(color.find(id1), color.find(id2));
    }
}