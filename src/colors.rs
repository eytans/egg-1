use std::collections::BTreeSet;
pub use crate::{Id, EGraph, Language, Analysis, ColorId};
use crate::Singleton;
use crate::util::JoinDisp;
use invariants::dassert;
use itertools::Itertools;
use std::fmt::Formatter;
use indexmap::{IndexMap, IndexSet};
use crate::colored_union_find::ColoredUnionFind;

global_counter!(COLOR_IDS, usize, usize::default());

#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct Color<L: Language, N: Analysis<L>> {
    color_id: ColorId,
    /// Used for rebuilding uf
    pub(crate) dirty_unions: Vec<Id>,
    /// Maintain which classes in black are represented in colored class (including rep)
    pub(crate) equality_classes: IndexMap<Id, IndexSet<Id>>,
    /// Used to implement a union find. Opposite function of `equality_classes`.
    /// Supports removal of elements when they are not needed.
    union_find: ColoredUnionFind,
    /// Used to determine for each a colored equality class what is the black colored class.
    /// Relevant when a colored edge was added.
    pub(crate) black_colored_classes: IndexMap<Id, Id>,
    pub(crate) children: Vec<ColorId>,
    pub(crate) parent: Option<ColorId>,
    parents: Vec<ColorId>,
    phantom: std::marker::PhantomData<(L, N)>,
}

impl<L: Language, N: Analysis<L>> Color<L, N> {
    pub(crate) fn collect_decendents(&self, egraph: &EGraph<L, N>) -> Vec<ColorId> {
        let mut res = self.children.clone();
        for c in self.children.iter() {
            res.extend(egraph.get_color(*c).unwrap().collect_decendents(egraph));
        }
        res
    }
}

impl<L: Language, N: Analysis<L>> Color<L, N> {
    pub(crate) fn new(new_id: ColorId, parent: Option<ColorId>, graph: &EGraph<L, N>) -> Color<L, N> {
        let parents = parent.map_or_else(|| vec![], |p| {
            let mut res = graph.get_color(p).unwrap().parents.clone();
            res.push(p);
            res
        });
        Color {
            color_id: new_id,
            dirty_unions: Default::default(),
            equality_classes: Default::default(),
            union_find: Default::default(),
            black_colored_classes: Default::default(),
            children: vec![],
            parent,
            parents,
            phantom: Default::default(),
        }
    }

    pub fn get_id(&self) -> ColorId {
        self.color_id
    }

    pub fn children(&self) -> &[ColorId] {
        &self.children
    }

    pub fn parents(&self) -> &[ColorId] {
        &self.parents
    }

    pub(crate) fn verify_uf_minimal(&self, egraph: &EGraph<L, N>) {
        let mut parents: IndexMap<Id, usize> = IndexMap::default();
        for (k, _v) in self.union_find.iter() {
            let v = self.find(egraph, k);
            *parents.entry(v).or_default() += 1;
        }
        for (k, v) in parents {
            assert!(v >= 1, "Found {} parents for {}", v, k);
        }
    }

    pub fn find(&self, egraph: &EGraph<L, N>, id: Id) -> Id {
        let fixed = self.parent().map_or_else(|| egraph.find(id), |c_id| egraph.colored_find(c_id, id));
        self.union_find.find(&fixed).unwrap_or_else(|| {
            fixed
        })
    }

    pub fn is_dirty(&self) -> bool { !self.dirty_unions.is_empty() }

    /// Removes equality of `base_to` and `base_from` in the color.
    /// `base_to` Should be the id of "to" (after running find in the parent)
    /// `base_from` Should be the id of "from" (after running find in parent)
    /// `to` Should be the id of "to" (after running find in the current color)
    /// `from` Should be the id of "from" (after running find in the current color)
    fn remove_equality(&mut self, base_to: Id, base_from: Id, to: Id, from: Id) {
        // If to != from then the underlying assumption is that black_to != black_from
        if to != from {
            let from_ids = self.equality_classes.remove(&from).unwrap_or_else(|| IndexSet::singleton(from));
            let to_ids = self.equality_classes.entry(to).or_insert_with(|| IndexSet::singleton(to));
            // Remove everything that is no longer a representative
            // to_ids.retain(|id| !from_ids.contains(id));
            // Actually only one is no longer a rep
            to_ids.extend(from_ids);
            to_ids.remove(&base_from);
            // Remove the old from (no longer a representative)
            // self.equality_classes.entry(to).and_modify(|s| { s.remove(&black_from); });
            // Actually should have been in from_ids
            assert!(!to_ids.contains(&base_from));
        } else if base_to != base_from {
            // We have to remove someone (because something was merged in black).
            // In this case `black_from` =_c `black_to` as they are merged in the color (`to` == `from`).
            // Black from is no longer a rep so remove it.
            self.equality_classes.entry(to).and_modify(|s| { s.remove(&base_from); });
        }
        if self.equality_classes.get(&to).map_or(false, |s| s.len() == 1) {
            dassert!(self.equality_classes.get(&to).unwrap().contains(&to), "We should always have the representative in the map");
            self.equality_classes.remove(&to);
        }
    }

    // Assumed id1 and id2 are canonized to the colors ids
    fn update_black_classes(&mut self, to: Id, from: Id) -> Option<(Id, Id)> {
        let mut g_todo = None;
        if to != from {
            if let Some(colored_from) = self.black_colored_classes.remove(&from) {
                let old_to = self.black_colored_classes.insert(to, colored_from);
                if let Some(colored_to) = old_to {
                    if colored_to < colored_from {
                        self.black_colored_classes.insert(to, colored_to);
                    }
                    g_todo = Some((colored_to, colored_from));
                }
            }
        }
        if to != from && cfg!(feature = "upward-merging") {
            unimplemented!("Upward merging not supported for colored graph");
            // self.process_unions();
        }
        g_todo
    }

    // Assumes to and from canonised to the base (parent, black or colored) and !=
    pub(crate) fn inner_base_union(&mut self, egraph: &mut EGraph<L, N>, base_to: Id, base_from: Id) -> Vec<(Id, Id)> {
        let orig_to = self.union_find.find(&base_to);
        let orig_from = self.union_find.find(&base_from);
        let from_existed = orig_from.is_some();

        let (colored_to, colored_from) = if orig_to.is_some() || orig_from.is_some() {
            // This part only needs to happen if one of the two is in the union find.
            let orig_to = orig_to.unwrap_or(base_to);
            let orig_from = orig_from.unwrap_or(base_from);
            self.union_find.insert(orig_to);
            self.union_find.insert(orig_from);
            self.union_find.union(&orig_to, &orig_from).unwrap()
        } else {
            (base_to, base_from)
        };

        // We need to update equalities.
        self.remove_equality(base_to, base_from, colored_to, colored_from);

        // In case both were not colored union_find.remove will not have any effect which is good.
        if colored_to != colored_from {
            if from_existed {
                // Only need to update children in "this" union find.
                let ids = self.equality_classes.get(&colored_to).map(|x| x.iter().copied().collect_vec());
                self.union_find.remove(&base_from, ids.map(|x| x.into_iter()));
            }
            self.dirty_unions.push(colored_to);

            // If both color classes existed it will update colored enodes classes.
            let mut todo_res = {
                let opt = self.update_black_classes(colored_to, colored_from);
                if let Some((id1, id2)) = opt {
                    vec![(id1, id2)]
                } else {
                    vec![]
                }
            };

            for c in self.children() {
                todo_res.extend(egraph.inner_base_union(*c, colored_to, colored_from).into_iter());
            }

            return todo_res;
        }

        return Default::default();
    }

    // Assumed id1 and id2 are parent canonized
    pub(crate) fn inner_colored_union(&mut self, id1: Id, id2: Id) -> (Id, Id, bool, Vec<(Id, Id)>) {
        // Parent classes will be updated in black union to come.
        self.union_find.insert(id1);
        self.union_find.insert(id2);
        let (to, from) = self.union_find.union(&id1, &id2).unwrap();
        let changed = to != from;
        let g_todo = self.update_black_classes(to, from).into_iter().collect_vec();
        if changed {
            self.dirty_unions.push(to);
            let from_ids = self.equality_classes.remove(&from).unwrap_or_else(|| IndexSet::singleton(from));
            self.equality_classes.entry(to).or_insert_with(|| IndexSet::singleton(to)).extend(from_ids);
        }
        (to, from, changed, g_todo)
    }

    pub fn base_equality_class(&self, egraph: &EGraph<L, N>, id: Id) -> Option<&IndexSet<Id>> {
        self.equality_classes.get(&self.find(egraph, id))
    }

    pub fn equality_class(&self, egraph: &EGraph<L, N>, id: Id) -> Box<dyn Iterator<Item = Id>> {
        let parent = self.parent();
        let fixed_id = self.find(egraph, id);
        let single = BTreeSet::singleton(fixed_id);
        let res = if let Some(ids) = self.equality_classes.get(&fixed_id) {
            if let Some(c_id) = parent {
                ids.into_iter().copied().flat_map(|id| egraph.get_color(c_id).unwrap().equality_class(egraph, id)).collect_vec()
            } else {
                ids.into_iter().copied().collect_vec()
            }
        } else {
            single.into_iter().collect_vec()
        };
        dassert!(res.len() == res.iter().unique().count());
        Box::new(res.into_iter())
    }

    /// Returns the black representative of the colored e-class of the current color only. Does not
    /// include the parents equality classes.
    pub fn current_black_reps(&self) -> impl Iterator<Item=&Id> {
        self.equality_classes.keys().into_iter()
    }

    pub fn black_colored_classes_size(&self) -> usize {
        self.black_colored_classes.len()
    }

    pub fn parent(&self) -> Option<ColorId> { self.parent }

    pub fn get_all_enodes(&self, id: Id, egraph: &EGraph<L, N>) -> Vec<L> {
        let mut res: IndexSet<L> = IndexSet::default();
        for cls in self.equality_class(egraph, id) {
            res.extend(egraph[cls].nodes.iter().map(|n: &L| egraph.colored_canonize(self.color_id, n)));
        }
        return res.into_iter().collect_vec();
    }

    pub fn assert_black_ids(&self, egraph: &EGraph<L, N>) {
        // Check that black ids are actually black representatives
        dassert!({
            for (_, set) in &self.equality_classes {
                for id in set {
                    dassert!(egraph.find(*id) == *id, "black id {:?} is not black rep {:?}", id, egraph.find(*id));
                }
            }
            true
        });
    }
}

impl<L, N> std::fmt::Display for Color<L, N> where L: Language, N: Analysis<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Color(id={}, groups={})", self.color_id, self.equality_classes.iter().map(|(id, set)| format!("{} - {}", id, set.iter().sep_string(" "))).join(", "))
    }
}

#[cfg(test)]
mod test {

    // #[test]
    // fn test_black_union_alone() {
    //     let mut g = EGraph::<SymbolLang, ()>::new(());
    //     let id1 = g.add_expr(&"1".parse().unwrap());
    //     let id2 = g.add_expr(&"2".parse().unwrap());
    //     let mut color = Color::new(ColorId::from(0));
    //     color.black_union(&mut g, id1, id2);
    //     color.black_union(&mut g, id1, id2);
    //     color.black_union(&mut g, id1, id1);
    //     assert_eq!(color.find(&g, id1), color.find(&g, id2));
    // }
}
