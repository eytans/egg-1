use std::cmp::{max, min};
pub use crate::{Id, EGraph, Language, Analysis, ColorId};
use crate::{Singleton, UnionFind};
use crate::util::JoinDisp;
use itertools::Itertools;
use std::fmt::Formatter;
use indexmap::{IndexMap, IndexSet};
use log::{error, info, trace, warn};

pub type ColorParents = smallvec::SmallVec<[ColorId; 3]>;

global_counter!(COLOR_IDS, usize, usize::default());

#[derive(Clone, Default, Debug)]
pub struct Color {
    pub(crate) union_find: UnionFind,
    color_id: ColorId,
    /// Used for rebuilding uf
    pub(crate) dirty_unions: Vec<Id>,
    /// Maintain which classes in black are represented in colored class (including rep)
    pub(crate) union_map: IndexMap<Id, IndexSet<Id>>,
    pub(crate) black_colored_classes: IndexMap<Id, Id>,
    pub(crate) children: Vec<ColorId>,
    pub(crate) parents: Vec<ColorId>,
}

impl Color {
    pub(crate) fn new(union_find: &UnionFind, new_id: ColorId) -> Color {
        Color {
            union_find: union_find.clone(),
            color_id: new_id,
            dirty_unions: Default::default(),
            union_map: Default::default(),
            black_colored_classes: Default::default(),
            children: vec![],
            parents: vec![]
        }
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
            let from_ids = self.union_map.remove(&non_rep).unwrap_or_else(|| IndexSet::singleton(non_rep));
            let to_ids = self.union_map.entry(rep).or_insert_with(|| IndexSet::singleton(rep));
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

    fn union_impl(&mut self, id1: Id, id2: Id) -> (Id, Id, bool, Option<(Id, Id)>) {
        let (to, from) = self.union_find.union(id1, id2);
        let mut g_todo = None;
        if to != from {
            for colored_from in self.black_colored_classes.remove(&from) {
                let old_to = self.black_colored_classes.insert(to, colored_from);
                for colored_to in old_to {
                    g_todo = Some((colored_to, colored_from));
                }
            }
        }
        let union = (to, from, to != from, g_todo);
        if union.2 && cfg!(feature = "upward-merging") {
            unimplemented!("Upward merging not supported for colored graph");
            // self.process_unions();
        }
        union
    }

    /// Union to keep up with current version of black uf.
    /// `id1` Should be the id of "to" (after running find in black)
    /// `id2` Should be the id of "from" (after running find in black)
    pub fn black_union<L: Language, N: Analysis<L>>(&mut self, graph: &mut EGraph<L, N>, id1: Id, id2: Id) -> (Id, bool) {
        let (to, changed, to_union) = self.inner_black_union(id1, id2);
        if let Some((id1, id2)) = to_union {
            graph.union(id1, id2);
        }
        (to, changed)
    }

    pub(crate) fn inner_black_union(&mut self, id1: Id, id2: Id) -> (Id, bool, Option<(Id, Id)>) {
        let orig_to = self.find(id1);
        let orig_from = self.find(id2);
        let (mut to, mut from, changed, g_todo) = self.union_impl(id1, id2);
        if changed {
            self.dirty_unions.push(to);
        }
        self.update_union_map(id1, id2, orig_to, orig_from);
        (to, changed, g_todo)
    }

    pub fn colored_union<L: Language, N: Analysis<L>>(&mut self, graph: &mut EGraph<L, N>, id1: Id, id2: Id) -> (Id, bool) {
        let (to, changed, todo) = self.inner_colored_union(id1, id2);
        if let Some((id1, id2)) = todo {
            graph.union(id1, id2);
        }
        (to, changed)
    }

    pub(crate) fn inner_colored_union(&mut self, id1: Id, id2: Id) -> (Id, bool, Option<(Id, Id)>) {
        let (to, from, changed, g_todo) = self.union_impl(id1, id2);
        if changed {
            self.dirty_unions.push(to);
            let from_ids = self.union_map.remove(&from).unwrap_or_else(|| IndexSet::singleton(from));
            self.union_map.entry(to).or_insert_with(|| IndexSet::singleton(to)).extend(from_ids);
        }
        (to, changed, g_todo)
    }

    pub fn black_ids(&self, id: Id) -> Option<&IndexSet<Id>> {
        self.union_map.get(&self.union_find.find(id))
    }

    pub fn black_reps(&self) -> impl Iterator<Item=&Id> {
        self.union_map.keys().into_iter()
    }

    pub fn parents(&self) -> &Vec<ColorId> {
        &self.parents
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
    use crate::{UnionFind, ColorId, EGraph, SymbolLang};

    #[test]
    fn test_black_union_alone() {
        let mut g = EGraph::<SymbolLang, ()>::new(());
        let mut uf = UnionFind::default();
        let id1 = uf.make_set();
        let id2 = uf.make_set();
        let mut color = Color::new(&uf, ColorId::from(0));
        color.black_union(&mut g, id1, id2);
        color.black_union(&mut g, id1, id2);
        color.black_union(&mut g, id1, id1);
        assert_eq!(color.find(id1), color.find(id2));
    }
}
