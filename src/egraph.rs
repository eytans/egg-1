use std::{
    borrow::BorrowMut,
    fmt::{self, Debug},
};

use indexmap::{IndexMap, IndexSet};
use invariants::{dassert, iassert, tassert, wassert, AssertConfig, AssertLevel};
use log::*;
use std::iter::FromIterator;
use std::rc::Rc;

use crate::{Analysis, Singleton};
use crate::AstSize;
use crate::Dot;
use crate::EClass;
use crate::Extractor;
use crate::Id;
use crate::Language;
use crate::Pattern;
use crate::RecExpr;
use crate::Searcher;
use crate::Subst;
use crate::UnionFind;
use crate::{OpId, SymbolLang};

pub use crate::colors::{Color, ColorId};
use itertools::Itertools;

use serde::{Deserialize, Serialize};

/** A data structure to keep track of equalities between expressions.

# What's an egraph?

An egraph ([/'igraf/][sound]) is a data structure to maintain equivalence

classes of expressions.
An egraph conceptually is a set of eclasses, each of which
contains equivalent enodes.
An enode is conceptually and operator with children, but instead of
children being other operators or values, the children are eclasses.

In `egg`, these are respresented by the [`EGraph`], [`EClass`], and
[`Language`] (enodes) types.


Here's an egraph created and rendered by [this example](struct.Dot.html).
As described in the documentation for [egraph visualization][dot] and
in the academic literature, we picture eclasses as dotted boxes
surrounding the equivalent enodes:

<img src="https://mwillsey.com/assets/simple-egraph.svg"/>

We say a term _t_ is _represented_ in an eclass _e_ if you can pick a
single enode from each eclass such that _t_ is in _e_.
A term is represented in the egraph if it's represented in any eclass.
In the image above, the terms `2 * a`, `a * 2`, and `a << 1` are all
represented in the same eclass and thus are equivalent.
The terms `1`, `(a * 2) / 2`, and `(a << 1) / 2` are represented in
the egraph, but not in the same eclass as the prior three terms, so
these three are not equivalent to those three.

Egraphs are useful when you have a bunch of very similar expressions,
some of which are equivalent, and you'd like a compactly store them.
This compactness allows rewrite systems based on egraphs to
efficiently "remember" the expression before and after rewriting, so
you can essentially apply all rewrites at once.
See [`Rewrite`] and [`Runner`] for more details about rewrites and
running rewrite systems, respectively.

# Invariants and Rebuilding

An egraph has two core operations that modify the egraph:
[`add`] which adds enodes to the egraph, and
[`union`] which merges two eclasses.
These operations maintains two key (related) invariants:

1. **Uniqueness of enodes**

   There do not exist two distinct enodes with equal operators and equal
   children in the eclass, either in the same eclass or different eclasses.
   This is maintained in part by the hashconsing performed by [`add`],
   and by deduplication performed by [`union`] and [`rebuild`].

2. **Congruence closure**

   An egraph maintains not just an [equivalence relation] over
   expressions, but a [congruence relation].
   So as the user calls [`union`], many eclasses other than the given
   two may need to merge to maintain congruence.

   For example, suppose terms `a + x` and `a + y` are represented in
   eclasses 1 and 2, respectively.
   At some later point, `x` and `y` become
   equivalent (perhaps the user called [`union`] on their containing
   eclasses).
   Eclasses 1 and 2 must merge, because now the two `+`
   operators have equivalent arguments, making them equivalent.

`egg` takes a delayed approach to maintaining these invariants.
Specifically, the effects of calling [`union`] (or applying a rewrite,
which calls [`union`]) may not be reflected immediately.
To restore the egraph invariants and make these effects visible, the
user *must* call the [`rebuild`] method.

`egg`s choice here allows for a higher performance implementation.
Maintaining the congruence relation complicates the core egraph data
structure and requires an expensive traversal through the egraph on
every [`union`].
`egg` chooses to relax these invariants for better performance, only
restoring the invariants on a call to [`rebuild`].
See the [`rebuild`] documentation for more information.
Note also that [`Runner`]s take care of this for you, calling
[`rebuild`] between rewrite iterations.

# egraphs in `egg`

In `egg`, the main types associated with egraphs are
[`EGraph`], [`EClass`], [`Language`], and [`Id`].

[`EGraph`] and [`EClass`] are all generic over a
[`Language`], meaning that types actually floating around in the
egraph are all user-defined.
In particular, the enodes are elements of your [`Language`].
[`EGraph`]s and [`EClass`]es are additionally parameterized by some
[`Analysis`], abritrary data associated with each eclass.

Many methods of [`EGraph`] deal with [`Id`]s, which represent eclasses.
Because eclasses are frequently merged, many [`Id`]s will refer to the
same eclass.

[`EGraph`]: struct.EGraph.html
[`EClass`]: struct.EClass.html
[`Rewrite`]: struct.Rewrite.html
[`Runner`]: struct.Runner.html
[`Language`]: trait.Language.html
[`Analysis`]: trait.Analysis.html
[`Id`]: struct.Id.html
[`add`]: struct.EGraph.html#method.add
[`union`]: struct.EGraph.html#method.union
[`rebuild`]: struct.EGraph.html#method.rebuild
[equivalence relation]: https://en.wikipedia.org/wiki/Equivalence_relation
[congruence relation]: https://en.wikipedia.org/wiki/Congruence_relation
[dot]: struct.Dot.html
[extract]: struct.Extractor.html
[sound]: https://itinerarium.github.io/phoneme-synthesis/?w=/'igraf/
 **/
#[derive(Clone, Serialize, Deserialize)]
pub struct EGraph<L: Language, N: Analysis<L>> {
    /// The `Analysis` given when creating this `EGraph`.
    pub analysis: N,
    pub(crate) memo: IndexMap<L, Id>,
    unionfind: UnionFind,
    classes: SparseVec<EClass<L, N::Data>>,
    dirty_unions: Vec<Id>,
    repairs_since_rebuild: usize,
    pub(crate) classes_by_op: IndexMap<OpId, IndexSet<Id>>,

    /// To be used as a mechanism for case splitting.
    /// Need to rebuild these, but can probably use original memo for that purpose.
    /// For each inner vector of union finds, if there is a union common to all of them then it will
    /// be applied on the main union find (case split mechanism). Not true for UnionFinds of size 1.
    #[cfg(feature = "colored")]
    colors: Vec<Option<Color<L, N>>>,
    #[cfg(feature = "colored")]
    pub(crate) colored_memo: IndexMap<ColorId, IndexMap<L, Id>>,
    /// For each id in the egraph, what are the ids that are equivalent to it, and in what color.
    #[cfg(feature = "colored")]
    pub colored_equivalences: IndexMap<Id, IndexSet<(ColorId, Id)>>,
    /// A filter function used when creating dots.
    #[serde(skip_serializing, skip_deserializing)]
    pub filterer: Option<Rc<dyn Fn(&EGraph<L, N>, Id) -> bool + 'static>>,
    /// What operations are not allowed to be equal (for vacuity check).
    pub vacuity_ops: Vec<Vec<OpId>>,
    #[cfg(feature = "keep_splits")]
    /// All the case splits that happened for this EGraph. They can be recursive for deeper cases.
    /// Will be updated by the user. The runner will run on all of them.
    pub all_splits: Vec<EGraph<L, N>>,
    /// Collect how many e-nodes were deleted during rebuilds.
    pub deleted_enodes: usize,
}

impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    pub(crate) fn inner_new(
        uf: UnionFind,
        classes: Vec<Option<Box<EClass<SymbolLang, ()>>>>,
        memo: IndexMap<SymbolLang, Id>,
    ) -> EGraph<SymbolLang, ()> {
        for c in classes
            .iter()
            .filter(|c| c.is_some())
            .map(|c| c.as_ref().unwrap())
        {
            for n in &c.nodes {
                // Check children are canonized
                for c in n.children.iter() {
                    assert_eq!(uf.find(*c), *c);
                }
                // Check all parents exist as expected
                for ch in n.children.iter() {
                    assert!(classes[ch.0 as usize]
                        .as_ref()
                        .unwrap()
                        .parents
                        .iter()
                        .any(|(n1, _id1)| n1 == n));
                }
                // Check memo
                assert_eq!(memo.get(n), Some(&c.id));
            }
            // No lingerings in memo:
            assert_eq!(
                memo.len(),
                classes
                    .iter()
                    .filter(|c| c.is_some())
                    .map(|c| c.as_ref().unwrap().nodes.len())
                    .sum::<usize>()
            );
        }
        EGraph {
            analysis: (),
            memo,
            unionfind: uf,
            classes,
            dirty_unions: vec![],
            repairs_since_rebuild: 0,
            classes_by_op: IndexMap::new(),
            #[cfg(feature = "colored")]
            colors: vec![],
            #[cfg(feature = "colored")]
            colored_memo: IndexMap::new(),
            #[cfg(feature = "colored")]
            colored_equivalences: IndexMap::new(),
            filterer: None,
            vacuity_ops: Default::default(),
            #[cfg(feature = "keep_splits")]
            all_splits: vec![],
            deleted_enodes: 0,
        }
    }
}

impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    /// Return a new map of `OpId` (operation id) to `Id` (eclass id) for all eclasses.
    pub fn classes_by_op_id(&self) -> IndexMap<OpId, IndexSet<Id>> {
        self.classes_by_op.clone()
    }
}

type SparseVec<T> = Vec<Option<Box<T>>>;

impl<L: Language, N: Analysis<L> + Default> Default for EGraph<L, N> {
    fn default() -> Self {
        Self::new(N::default())
    }
}

// manual debug impl to avoid L: Language bound on EGraph defn
impl<L: Language, N: Analysis<L>> Debug for EGraph<L, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("EGraph")
            .field("memo", &self.memo)
            .field("classes", &self.classes)
            .finish()
    }
}

impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    /// Creates a new, empty `EGraph` with the given `Analysis`
    pub fn new(analysis: N) -> Self {
        Self {
            analysis,
            memo: Default::default(),
            classes: Default::default(),
            unionfind: Default::default(),
            colors: Default::default(),
            dirty_unions: Default::default(),
            classes_by_op: IndexMap::default(),
            repairs_since_rebuild: 0,
            colored_memo: Default::default(),
            colored_equivalences: Default::default(),
            filterer: None,
            vacuity_ops: Default::default(),
            #[cfg(feature = "keep_splits")]
            all_splits: vec![],
            deleted_enodes: 0,
        }
    }

    /// Return total number of ids.
    pub fn id_len(&self) -> usize {
        self.classes.len()
    }

    /// Returns an iterator over the eclasses in the egraph.
    pub fn classes(&self) -> impl Iterator<Item = &EClass<L, N::Data>> {
        self.classes
            .iter()
            .filter_map(Option::as_ref)
            .map(AsRef::as_ref)
    }

    /// Returns an mutating iterator over the eclasses in the egraph.
    pub fn classes_mut(&mut self) -> impl Iterator<Item = &mut EClass<L, N::Data>> {
        self.classes
            .iter_mut()
            .filter_map(Option::as_mut)
            .map(AsMut::as_mut)
    }

    /// Returns `true` if the egraph is empty
    /// # Example
    /// ```
    /// use egg::{*, SymbolLang as S};
    /// let mut egraph = EGraph::<S, ()>::default();
    /// assert!(egraph.is_empty());
    /// egraph.add(S::leaf("foo"));
    /// assert!(!egraph.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.memo.is_empty()
    }

    /// Returns the number of enodes in the `EGraph`.
    ///
    /// Actually returns the size of the hashcons index.
    /// # Example
    /// ```
    /// use egg::{*, SymbolLang as S};
    /// let mut egraph = EGraph::<S, ()>::default();
    /// let x = egraph.add(S::leaf("x"));
    /// let y = egraph.add(S::leaf("y"));
    /// // only one eclass
    /// egraph.union(x, y);
    /// egraph.rebuild();
    ///
    /// assert_eq!(egraph.total_size(), 2);
    /// assert_eq!(egraph.number_of_classes(), 1);
    /// ```
    pub fn total_size(&self) -> usize {
        self.memo.len()
    }

    /// Iterates over the classes, returning the total number of nodes.
    pub fn total_number_of_nodes(&self) -> usize {
        self.classes().map(|c| c.len()).sum()
    }

    /// Returns the number of eclasses in the egraph.
    pub fn number_of_classes(&self) -> usize {
        self.classes().count()
    }

    /// Canonicalizes an eclass id.
    ///
    /// This corresponds to the `find` operation on the egraph's
    /// underlying unionfind data structure.
    ///
    /// # Example
    /// ```
    /// use egg::{*, SymbolLang as S};
    /// let mut egraph = EGraph::<S, ()>::default();
    /// let x = egraph.add(S::leaf("x"));
    /// let y = egraph.add(S::leaf("y"));
    /// assert_ne!(egraph.find(x), egraph.find(y));
    ///
    /// egraph.union(x, y);
    /// assert_eq!(egraph.find(x), egraph.find(y));
    /// ```
    pub fn find(&self, id: Id) -> Id {
        self.unionfind.find(id)
    }

    /// Creates a [`Dot`] to visualize this egraph. See [`Dot`].
    ///
    /// [`Dot`]: struct.Dot.html
    pub fn dot(&self) -> Dot<L, N> {
        Dot {
            egraph: self,
            color: None,
            print_color: "blue".to_string(),
            pred: None,
        }
    }

    #[allow(missing_docs)]
    pub fn filtered_dot(&self, filter: impl Fn(&EGraph<L, N>, Id) -> bool + 'static) -> Dot<L, N> {
        Dot {
            egraph: self,
            color: None,
            pred: Some(Rc::new(filter)),
            print_color: "blue".to_string(),
        }
    }

    #[allow(missing_docs)]
    pub fn colored_dot(&self, color: ColorId) -> Dot<L, N> {
        Dot {
            egraph: self,
            color: Some(color),
            print_color: "blue".to_string(),
            pred: None,
        }
    }

    #[allow(missing_docs)]
    pub fn colored_filtered_dot(
        &self,
        color: ColorId,
        filter: impl Fn(&EGraph<L, N>, Id) -> bool + 'static,
    ) -> Dot<L, N> {
        Dot {
            egraph: self,
            color: Some(color),
            pred: Some(Rc::new(filter)),
            print_color: "blue".to_string(),
        }
    }
}

impl<L: Language, N: Analysis<L>> std::ops::Index<Id> for EGraph<L, N> {
    type Output = EClass<L, N::Data>;
    fn index(&self, id: Id) -> &Self::Output {
        let id = self.find(id);
        self.classes[usize::from(id)]
            .as_ref()
            .unwrap_or_else(|| panic!("Invalid id {}", id))
    }
}

impl<L: Language, N: Analysis<L>> std::ops::IndexMut<Id> for EGraph<L, N> {
    fn index_mut(&mut self, id: Id) -> &mut Self::Output {
        let id = self.find(id);
        self.classes[usize::from(id)]
            .as_mut()
            .unwrap_or_else(|| panic!("Invalid id {}", id))
    }
}

impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    /// Adds a [`RecExpr`] to the [`EGraph`].
    ///
    /// # Example
    /// ```
    /// use egg::{*, SymbolLang as S};
    /// let mut egraph = EGraph::<S, ()>::default();
    /// let x = egraph.add(S::leaf("x"));
    /// let y = egraph.add(S::leaf("y"));
    /// let plus = egraph.add(S::new("+", vec![x, y]));
    /// let plus_recexpr = "(+ x y)".parse().unwrap();
    /// assert_eq!(plus, egraph.add_expr(&plus_recexpr));
    /// ```
    ///
    /// [`EGraph`]: struct.EGraph.html
    /// [`RecExpr`]: struct.RecExpr.html
    /// [`add_expr`]: struct.EGraph.html#method.add_expr
    pub fn add_expr(&mut self, expr: &RecExpr<L>) -> Id {
        self.add_expr_rec(expr.as_ref(), None)
    }

    fn add_expr_rec(&mut self, expr: &[L], color: Option<ColorId>) -> Id {
        log::trace!("Adding expr {:?}", expr);
        let e = expr.last().unwrap().clone().map_children(|i| {
            let child = &expr[..usize::from(i) + 1];
            self.add_expr_rec(child, color)
        });
        let id = if let Some(c) = color {
            self.colored_add(c, e)
        } else {
            self.add(e)
        };
        log::trace!("Added!! expr {:?}", expr);
        id
    }

    /// Lookup the eclass of the given enode.
    ///
    /// You can pass in either an owned enode or a `&mut` enode,
    /// in which case the enode's children will be canonicalized.
    ///
    /// # Example
    /// ```
    /// # use egg::*;
    /// let mut egraph: EGraph<SymbolLang, ()> = Default::default();
    /// let a = egraph.add(SymbolLang::leaf("a"));
    /// let b = egraph.add(SymbolLang::leaf("b"));
    /// let c = egraph.add(SymbolLang::leaf("c"));
    ///
    /// // lookup will find this node if its in the egraph
    /// let mut node_f_ac = SymbolLang::new("f", vec![a, c]);
    /// assert_eq!(egraph.lookup(node_f_ac.clone()), None);
    /// let id = egraph.add(node_f_ac.clone());
    /// assert_eq!(egraph.lookup(node_f_ac.clone()), Some(id));
    ///
    /// // if the query node isn't canonical, and its passed in by &mut instead of owned,
    /// // its children will be canonicalized
    /// egraph.union(b, c);
    /// egraph.rebuild();
    /// assert_eq!(egraph.lookup(&mut node_f_ac), Some(id));
    /// assert_eq!(node_f_ac, SymbolLang::new("f", vec![a, b]));
    /// ```
    pub fn lookup<B>(&self, mut enode: B) -> Option<Id>
    where
        B: BorrowMut<L>,
    {
        let enode = enode.borrow_mut();
        enode.update_children(|id| self.find(id));
        self.memo.get(enode).map(|id| self.find(*id))
    }

    /// Looks up a [`L`] from the [`EGraph`]. This works with equivalences defined in `color`.
    pub fn colored_lookup<B>(&self, color: ColorId, mut enode: B) -> Option<Id>
    where
        B: BorrowMut<L>,
    {
        let enode = enode.borrow_mut();
        enode.update_children(|id| self.find(id));
        self.memo.get(enode).map(|id| self.find(*id)).or_else(|| {
            enode.update_children(|id| self.colored_find(color, id));
            // We need to find the black representative of the colored edge (yes, confusing).
            self.colored_memo[&color]
                .get(&*enode)
                .map(|id| self.find(*id))
        })
    }

    /// Adds an enode to the [`EGraph`].
    ///
    /// When adding an enode, to the egraph, [`add`] it performs
    /// _hashconsing_ (sometimes called interning in other contexts).
    ///
    /// Hashconsing ensures that only one copy of that enode is in the egraph.
    /// If a copy is in the egraph, then [`add`] simply returns the id of the
    /// eclass in which the enode was found.
    /// Otherwise
    ///
    /// [`EGraph`]: struct.EGraph.html
    /// [`EClass`]: struct.EClass.html
    /// [`add`]: struct.EGraph.html#method.add
    pub fn add(&mut self, mut enode: L) -> Id {
        if let Some(id) = self.lookup(&mut enode) {
            id
        } else {
            let id = self.inner_create_class(&mut enode, None);
            dassert!(
                enode == self.canonize(&enode),
                "Enode should be canonized after lookup."
            );
            // add this enode to the parent lists of its children
            enode.children().iter().copied().unique().for_each(|child| {
                let tup = (enode.clone(), id);
                self[child].parents.push(tup);
            });
            assert!(self.memo.insert(enode, id).is_none());

            N::modify(self, id);
            id
        }
    }

    /// Adds a [`RecExpr`] to the [`EGraph`].
    /// Like [`add_expr`], but under a color.
    ///
    /// [`EGraph`]: struct.EGraph.html
    /// [`RecExpr`]: struct.RecExpr.html
    /// [`add_expr`]: struct.EGraph.html#method.add_expr
    /// [`colored_add_expr`]: struct.EGraph.html#method.colored_add_expr
    pub fn colored_add_expr(&mut self, color: ColorId, expr: &RecExpr<L>) -> Id {
        self.add_expr_rec(expr.as_ref(), Some(color))
    }

    /// Adds an enode to the [`EGraph`], but only for a specific color.
    pub fn colored_add(&mut self, color: ColorId, mut enode: L) -> Id {
        if cfg!(feature = "colored_no_cmemo") {
            return self.add(enode);
        }
        return if let Some(id) = self.colored_lookup(color, &mut enode) {
            id
        } else {
            let id = self.inner_create_class(&mut enode, Some(color));
            self.get_color_mut(color)
                .unwrap()
                .black_colored_classes
                .insert(id, id);
            enode.children().iter().copied().unique().for_each(|child| {
                self[child]
                    .colored_parents
                    .entry(color)
                    .or_default()
                    .push((enode.clone(), id));
            });

            assert!(self.colored_memo[&color].insert(enode, id).is_none());
            N::modify(self, id);
            id
        };
    }

    fn inner_create_class(&mut self, enode: &mut L, color: Option<ColorId>) -> Id {
        let id = self.unionfind.make_set();
        if let Some(c) = color {
            enode.update_children(|id| self.colored_find(c, id));
        } else {
            enode.update_children(|id| self.find(id));
        }

        log::trace!("  ...colored ({:?}) adding {:?} to {}", color, enode, id);
        let class = Box::new(EClass {
            id,
            nodes: vec![enode.clone()],
            data: N::make(self, &enode),
            parents: Default::default(),
            changed_parents: Default::default(),
            colored_parents: Default::default(),
            color,
            colord_changed_parents: Default::default(),
        });

        assert_eq!(self.classes.len(), usize::from(id));
        self.classes.push(Some(class));
        id
    }

    /// Checks whether two [`RecExpr`]s are equivalent.
    /// Returns a list of id where both expression are represented.
    /// In most cases, there will none or exactly one id.
    ///
    /// [`RecExpr`]: struct.RecExpr.html
    pub fn equivs(&self, expr1: &RecExpr<L>, expr2: &RecExpr<L>) -> Vec<Id> {
        let matches1 = Pattern::from(expr1.as_ref()).search(self);
        trace!("Matches1 ({:?}): {:?}", expr1, matches1);

        let matches2 = Pattern::from(expr2.as_ref()).search(self);
        trace!("Matches2 ({:?}): {:?}", expr2, matches2);

        let mut equiv_eclasses = Vec::new();

        for m1 in &matches1 {
            if m1.substs.iter().all(|s| s.color.is_some()) {
                continue;
            }
            for m2 in &matches2 {
                if m2.substs.iter().all(|s| s.color.is_some()) {
                    continue;
                }
                if self.find(m1.eclass) == self.find(m2.eclass) {
                    equiv_eclasses.push(m1.eclass)
                }
            }
        }

        equiv_eclasses
    }

    /// Panic if the given eclass doesn't contain the given patterns
    ///
    /// Useful for testing.
    pub fn check_goals(&self, id: Id, goals: &[Pattern<L>]) {
        let (cost, best) = Extractor::new(self, AstSize).find_best(id);
        println!("End ({}): {}", cost, best.pretty(80));

        for (i, goal) in goals.iter().enumerate() {
            println!("Trying to prove goal {}: {}", i, goal.pretty(40));
            let matches = goal.search_eclass(&self, id);
            if matches.is_none() {
                let best = Extractor::new(&self, AstSize).find_best(id).1;
                panic!(
                    "Could not prove goal {}:\n{}\nBest thing found:\n{}",
                    i,
                    goal.pretty(40),
                    best.pretty(40),
                );
            }
        }
    }

    #[inline]
    fn union_impl(&mut self, id1: Id, id2: Id) -> (Id, bool) {
        fn concat<T>(to: &mut Vec<T>, mut from: Vec<T>) {
            if to.len() < from.len() {
                std::mem::swap(to, &mut from)
            }
            to.extend(from);
        }

        let (to, from) = self.unionfind.union(id1, id2);
        let changed = to != from;
        tassert!(to == self.find(id1));
        tassert!(to == self.find(id2));
        if changed {
            if cfg!(feature = "colored") {
                // warn!("union: {} {}", to, from);
                // An unsafe hack that is fine because we only use the union find of the egraph in inner
                // black union:
                let mut colors = std::mem::take(&mut self.colors);
                let todo = colors.iter_mut()
                    .filter_map(|c| c.as_mut())
                    .filter_map(|color| color.inner_black_union(self, to, from))
                    .collect_vec();
                self.colors = colors;
                for (id1, id2) in todo {
                    self.union(id1, id2);
                }
            }

            #[cfg(feature = "colored_no_cmemo")]
            iassert!(self[to].color().is_none());
            #[cfg(feature = "colored_no_cmemo")]
            iassert!(self[from].color().is_none());
            if let Some(c) = self[to].color {
                iassert!(self[from].color == self[to].color);
                iassert!(self.colored_find(c, to) == self.colored_find(c, from));
                let colored_to = self.colored_find(c, to);
                let old = self
                    .get_color_mut(c)
                    .unwrap()
                    .black_colored_classes
                    .insert(colored_to, to);
                if let Some(o) = old {
                    self.union(o, to);
                }
            } else {
                self.dirty_unions.push(to);
            }

            // update the classes data structure
            let from_class = self.classes[usize::from(from)].take().unwrap();
            let to_class = self.classes[usize::from(to)].as_mut().unwrap();
            dassert!(from_class.color == to_class.color);

            self.analysis.merge(&mut to_class.data, from_class.data);
            concat(&mut to_class.nodes, from_class.nodes);
            concat(&mut to_class.parents, from_class.parents);
            concat(&mut to_class.changed_parents, from_class.changed_parents);
            from_class
                .colored_parents
                .into_iter()
                .for_each(|(k, v)| to_class.colored_parents.entry(k).or_default().extend(v));
            N::modify(self, to);
        }
        (to, changed)
    }

    /// Unions two eclasses given their ids.
    ///
    /// The given ids need not be canonical.
    /// The returned `bool` indicates whether a union was done,
    /// so it's `false` if they were already equivalent.
    /// Both results are canonical.
    pub fn union(&mut self, id1: Id, id2: Id) -> (Id, bool) {
        let union = self.union_impl(id1, id2);
        if union.1 && cfg!(feature = "upward-merging") {
            // let merged = self.process_unions().iter()
            //     // .filter(|&tup| tup.2.is_none())
            //     .map(|(id1, id2, color)| (*id1, *id2)).collect_vec();
            let _merged = self.process_unions();
            if cfg!(feature = "colored") {
                if !self.colors.is_empty() {
                    self.process_colored_unions();
                    assert!(
                        false,
                        "Need to clean colors returned by process_colored_unions"
                    );
                }
            }
        }
        self.memo_classes_agree();
        union
    }

    /// Returns a more debug-able representation of the egraph.
    ///
    /// [`EGraph`]s implement [`Debug`], but it ain't pretty. It
    /// prints a lot of stuff you probably don't care about.
    /// This method returns a wrapper that implements [`Debug`] in a
    /// slightly nicer way, just dumping enodes in each eclass.
    ///
    /// [`Debug`]: https://doc.rust-lang.org/stable/std/fmt/trait.Debug.html
    /// [`EGraph`]: struct.EGraph.html
    pub fn dump<'a>(&'a self) -> impl Debug + 'a {
        EGraphDump(self)
    }
}

// All the rebuilding stuff
impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    #[inline(never)]
    fn rebuild_classes(&mut self) -> usize {
        let mut classes_by_op = std::mem::take(&mut self.classes_by_op);
        classes_by_op.values_mut().for_each(|ids| ids.clear());

        let mut trimmed = 0;

        let keys = self.classes().map(|e| e.id).collect_vec();
        let mut classes = std::mem::take(&mut self.classes);
        for key in keys {
            let old_len = classes[key.0 as usize].as_mut().unwrap().len();
            let c_id = classes[key.0 as usize].as_mut().unwrap().color;
            classes[key.0 as usize]
                .as_mut()
                .unwrap()
                .nodes
                .iter_mut()
                .for_each(|enode| {
                    enode.update_children(|id| {
                        if let Some(c) = c_id.as_ref() {
                            self.get_color(*c).unwrap().find(self, id)
                        } else {
                            self.unionfind.find(id)
                        }
                    });
                });
            // Prevent comparing colors. Black should be first for better dirty color application.
            classes[key.0 as usize]
                .as_mut()
                .unwrap()
                .nodes
                .sort_unstable();
            if let Some(c) = classes[key.0 as usize].as_mut().unwrap().color.clone() {
                let mut nodes =
                    std::mem::take(&mut classes[key.0 as usize].as_mut().unwrap().nodes);
                nodes.dedup_by(|a, b| {
                    dassert!(&self.colored_canonize(c, a) == a);
                    // This is a colored class so if the canonized node is not in the memo
                    // it needs deleting.
                    a == b || ((!cfg!(feature = "colored_no_cremove")) && (!self.colored_memo[&c].contains_key(a)) && {
                        trace!("Removing node {a:?} from class {key} @ color {c}. Normal memo contains: {}", self.memo.contains_key(a));
                        // Side effect for removing value due to being in black memo (so no need
                        // for a colored node):
                        self.update_deleted_enode(&mut classes, &c, a);
                        true
                    })
                });
                classes[key.0 as usize].as_mut().unwrap().nodes = nodes;
            } else {
                classes[key.0 as usize].as_mut().unwrap().nodes.dedup();
            }
            // There might be unused colors in it, use them.
            // TODO: make sure that a class will not be empty once we remove edges by color.
            dassert!(!classes[key.0 as usize].as_mut().unwrap().nodes.is_empty());

            trimmed += old_len - classes[key.0 as usize].as_mut().unwrap().nodes.len();

            // TODO this is the slow version, could take advantage of sortedness
            // maybe
            let mut add = |n: &L| classes_by_op.entry(n.op_id()).or_default().insert(key);

            // we can go through the ops in order to dedup them, becaue we
            // just sorted them
            if classes[key.0 as usize].as_mut().unwrap().nodes.len() > 0 {
                let first = &classes[key.0 as usize].as_mut().unwrap().nodes[0];
                let mut op_id = first.op_id();
                add(&first);
                for n in &classes[key.0 as usize].as_mut().unwrap().nodes[1..] {
                    if op_id != n.op_id() {
                        add(n);
                        op_id = n.op_id();
                    }
                }
            }
        }
        self.classes = classes;

        self.classes_by_op = classes_by_op;
        self.colored_equivalences.clear();
        let colors = std::mem::take(&mut self.colors);
        for c in colors.iter().filter_map(|x| x.as_ref()) {
            for (_, ids) in &c.equality_classes {
                dassert!(ids.len() > 1);
                for id in ids {
                    for id1 in ids {
                        if id1 == id {
                            continue;
                        }
                        self.colored_equivalences
                            .entry(*id)
                            .or_default()
                            .insert((c.get_id(), *id1));
                    }
                }
            }
        }
        self.colors = colors;
        trimmed
    }

    fn update_deleted_enode(
        &mut self,
        classes: &mut SparseVec<EClass<L, <N as Analysis<L>>::Data>>,
        c: &ColorId,
        a: &mut L,
    ) {
        #[cfg(feature = "stats")] {
            self.deleted_enodes += 1;
        }
        for id in a.children() {
            if let Some(ids) = self.get_color(*c).unwrap().black_ids(&self, *id).cloned() {
                for id in ids {
                    let mut was_empty = false;
                    if let Some(parents) = classes[id.0 as usize]
                        .as_mut()
                        .unwrap()
                        .colored_parents
                        .get_mut(c)
                    {
                        parents.retain(|(n, _e)| n != a);
                        if parents.is_empty() {
                            was_empty = true;
                        }
                    }
                    // Remove the color from the colored parents if it was the last one.
                    if was_empty {
                        classes[id.0 as usize]
                            .as_mut()
                            .unwrap()
                            .colored_parents
                            .remove(c);
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    #[inline(never)]
    fn check_memo(&self) -> bool {
        let mut test_memo = IndexMap::new();

        for (id, class) in self.classes.iter().enumerate() {
            let id = Id::from(id);
            let class = match class.as_ref() {
                Some(class) => class,
                None => continue,
            };
            if class.color.is_some() {
                continue;
            }
            // TODO: also work with colored classes and memo
            assert_eq!(class.id, id);
            for node in &class.nodes {
                if let Some(old) = test_memo.insert(node, id) {
                    assert_eq!(
                        self.find(old),
                        self.find(id),
                        "Found unexpected equivalence for {:?}\n{:?}\nvs\n{:?}",
                        node,
                        self[self.find(id)].nodes,
                        self[self.find(old)].nodes,
                    );
                }
            }
        }

        for (n, e) in test_memo {
            assert_eq!(e, self.find(e));
            assert_eq!(
                Some(e),
                self.memo.get(n).map(|id| self.find(*id)),
                "Entry for {:?} at {} in test_memo was incorrect",
                n,
                e
            );
        }

        true
    }

    #[inline(never)]
    fn process_unions(&mut self) -> Vec<Id> {
        let mut res = self.dirty_unions.clone();
        let mut to_union = vec![];

        while !self.dirty_unions.is_empty() {
            // take the worklist, we'll get the stuff that's added the next time around
            // deduplicate the dirty list to avoid extra work
            let mut todo = std::mem::take(&mut self.dirty_unions);
            todo.iter_mut().for_each(|id| *id = self.find(*id));
            if cfg!(not(feature = "upward-merging")) {
                todo.sort_unstable();
                todo.dedup();
            }
            assert!(!todo.is_empty());

            for id in todo {
                self.repairs_since_rebuild += 1;
                for (n, p_id) in std::mem::take(&mut self[id].changed_parents) {
                    if let Some(m_id) = self.memo.remove(&n) {
                        // We might have already unioned these two, so we need to check
                        dassert!({
                            let fixed_m_id = self.find(m_id);
                            let fixed_p_id = self.find(p_id);
                            if !(to_union.contains(&(fixed_p_id, fixed_m_id))
                                || to_union.contains(&(fixed_m_id, fixed_p_id)))
                            {
                                assert_eq!(fixed_m_id, fixed_p_id,
                                           "Found unexpected non-equivalence for {:?}(memo)!={:?}(changed_parent) for enode {:?}",
                                           fixed_m_id, fixed_p_id, n);
                            }
                            true
                        });
                    }
                }
                let mut parents = std::mem::take(&mut self[id].parents)
                    .into_iter()
                    .map(|(n, e)| {
                        self.memo.remove(&n);
                        (n, e)
                    })
                    .collect_vec();
                parents.iter_mut().for_each(|(n, p_id)| {
                    n.update_children(|child| self.find(child));
                    let old_p_id = *p_id;
                    *p_id = self.find(*p_id);
                    dassert!(self[*p_id].color.is_none());
                    trace!("Updating parent {:?} of {:?} to {:?}", n, old_p_id, *p_id);
                    for child in n.children().iter().filter(|c| **c != id) {
                        self[*child].changed_parents.push((n.clone(), *p_id));
                    }
                });
                parents.sort_unstable();
                parents.dedup_by(|(n1, e1), (n2, e2)| {
                    n1 == n2 && {
                        trace!("Adding union from parent dedup {:?} and {:?}", e1, e2);
                        to_union.push((*e1, *e2));
                        true
                    }
                });

                for (n, e) in parents.iter_mut() {
                    let e = self.find(*e);
                    let temp = Self::update_memo_from_parent(&mut self.memo, n, &e);
                    to_union.extend(temp.into_iter());
                }

                self.propagate_metadata(&parents[..]);

                self[id].parents = parents;
                N::modify(self, id);
            }

            for (id1, id2) in to_union.drain(..) {
                let (to, did_something) = self.union_impl(id1, id2);
                if did_something {
                    res.push(to);
                    self.dirty_unions.push(to);
                }
            }
        }
        assert!(self.dirty_unions.is_empty());
        assert!(to_union.is_empty());
        res
    }

    // pub(crate) fn update_from_parent_access_fn<'a, 'b, K>(mut memo: &mut HashMap<L, K>, accessor: impl FnMut(&'a mut HashMap<L, K>, L) -> Option<(Id, DenseNodeColors, &'a mut DenseNodeColors)>,
    //                                                n: &L, e: &Id, cs: &DenseNodeColors) -> Option<(Id, Id)>
    // {
    //     let old_memo = accessor(memo, n.clone());
    //     if let Some((old, old_cs, mut new_cs)) = old_memo {
    //         if (old_cs.not_any()) || cs.not_any() {
    //             new_cs.set_elements(0);
    //             return Some((old, *e));
    //         } else {
    //             new_cs.bitor_assign(&old_cs);
    //         }
    //     }
    //     None
    // }

    /// Updates the memo table with a new parent. Should be used in the context of processing unions
    pub fn update_memo_from_parent(memo: &mut IndexMap<L, Id>, n: &L, e: &Id) -> Option<(Id, Id)> {
        trace!("Updating memo from parent {:?} to {:?}", n, e);
        if let Some(old) = memo.insert(n.clone(), *e) {
            trace!("Adding memo union {:?} and {:?}", old, e);
            return Some((old, *e));
        }
        None
    }

    /// Restores the egraph invariants of congruence and enode uniqueness.
    ///
    /// As mentioned [above](struct.EGraph.html#invariants-and-rebuilding),
    /// `egg` takes a lazy approach to maintaining the egraph invariants.
    /// The `rebuild` method allows the user to manually restore those
    /// invariants at a time of their choosing. It's a reasonably
    /// fast, linear-ish traversal through the egraph.
    ///
    /// # Example
    /// ```
    /// use egg::{*, SymbolLang as S};
    /// let mut egraph = EGraph::<S, ()>::default();
    /// let x = egraph.add(S::leaf("x"));
    /// let y = egraph.add(S::leaf("y"));
    /// let ax = egraph.add_expr(&"(+ a x)".parse().unwrap());
    /// let ay = egraph.add_expr(&"(+ a y)".parse().unwrap());
    ///
    /// // The effects of this union aren't yet visible; ax and ay
    /// // should be equivalent by congruence since x = y.
    /// egraph.union(x, y);
    /// // Classes: [x y] [ax] [ay] [a]
    /// # #[cfg(not(feature = "upward-merging"))]
    /// assert_eq!(egraph.number_of_classes(), 4);
    /// # #[cfg(not(feature = "upward-merging"))]
    /// assert_ne!(egraph.find(ax), egraph.find(ay));
    ///
    /// // Rebuilding restores the invariants, finding the "missing" equivalence
    /// egraph.rebuild();
    /// // Classes: [x y] [ax ay] [a]
    /// assert_eq!(egraph.number_of_classes(), 3);
    /// assert_eq!(egraph.find(ax), egraph.find(ay));
    /// ```
    pub fn rebuild(&mut self) -> usize {
        let old_hc_size = self.memo.len();
        let old_n_eclasses = self.number_of_classes();

        let start = instant::Instant::now();

        // Verify colors on nodes and in memo only differ by dirty colors
        self.memo_classes_agree();

        while self.colors().any(|c| c.is_dirty()) || !self.dirty_unions.is_empty() {
            let _merged = self.process_unions();
            self.memo_black_canonized();
            self.process_colored_unions();
        }

        let n_unions = std::mem::take(&mut self.repairs_since_rebuild);
        let trimmed_nodes = self.rebuild_classes();
        self.memo_all_canonized();
        self.memo_black_canonized();
        let elapsed = start.elapsed();
        info!(
            concat!(
                "REBUILT! in {}.{:03}s\n",
                "  Old: hc size {}, eclasses: {}\n",
                "  New: hc size {}, eclasses: {}\n",
                "  unions: {}, trimmed nodes: {}"
            ),
            elapsed.as_secs(),
            elapsed.subsec_millis(),
            old_hc_size,
            old_n_eclasses,
            self.memo.len(),
            self.number_of_classes(),
            n_unions,
            trimmed_nodes,
        );

        dassert!(self.no_two_colored_classes_in_ec());
        assert!(self.dirty_unions.is_empty());
        iassert!(self.colors().all(|c| !c.is_dirty()));

        // debug_assert!(self.check_memo());
        n_unions
    }

    pub(crate) fn colored_update_node(&self, color: ColorId, e: &mut L) {
        e.update_children(|e| self.colored_find(color, e));
    }

    /// Return a canonized version of `e` under color `color`.
    pub fn colored_canonize(&self, color: ColorId, e: &L) -> L {
        let mut res = e.clone();
        self.colored_update_node(color, &mut res);
        res
    }

    /// canonize node in place
    pub fn update_node(&self, e: &mut L) {
        e.update_children(|e| self.find(e));
    }

    /// Return a canonized version of `e`.
    pub fn canonize(&self, e: &L) -> L {
        let mut res = e.clone();
        self.update_node(&mut res);
        res
    }

    /// Reapply congruence closure for color.
    /// Returns which colors to remove from which edges.
    pub fn colored_cong_closure(&mut self, c_id: ColorId) {
        self.get_color(c_id).unwrap().assert_black_ids(self);

        let mut to_union = vec![];
        // We need to build memo ahead of time because even a single merge might miss needed unions.
        // Need to do some merging and initial color deletion here because we are deleting duplicate
        // edges.
        let mut memo: IndexMap<L, Id> = {
            let colored_memo_canonized = self.colored_memo[&c_id]
                .iter()
                .map(|(node, id)| (self.colored_canonize(c_id, node), *id))
                .collect_vec();
            let mut v = self
                .memo
                .iter()
                .map(|(orig, e)| (self.colored_canonize(c_id, orig), self.find(*e)))
                .chain(colored_memo_canonized)
                .collect_vec();
            v.sort_unstable();
            // TODO: Create dedup with side effects for iterator
            v.dedup_by(|(n1, e1), (n2, e2)| {
                n1 == n2 && {
                    to_union.push((*e1, *e2));
                    true
                }
            });
            v.into_iter().collect()
        };

        for (id1, id2) in to_union.drain(..) {
            if self[id1].color.is_some() && self[id2].color.is_some() {
                dassert!(self[id1].color == self[id2].color);
                dassert!(self[id1].color.unwrap() == c_id);
            }
            self.colored_union(c_id, id1, id2);
        }
        while !self.get_color(c_id).unwrap().dirty_unions.is_empty() {
            // take the worklist, we'll get the stuff that's added the next time around
            // deduplicate the dirty list to avoid extra work
            let mut todo = std::mem::take(&mut self.get_color_mut(c_id).unwrap().dirty_unions);
            for id in todo.iter_mut() {
                *id = self.colored_find(c_id, *id);
            }
            if cfg!(not(feature = "upward-merging")) {
                todo.sort_unstable();
                todo.dedup();
            }
            assert!(!todo.is_empty());

            // rep to all contained
            // let all_groups: IndexMap<Id, IndexSet<Id>> =
            //     self.get_color(c_id).unwrap().union_find.build_sets();
            for id in todo {
                let single_id: IndexSet<Id> = IndexSet::singleton(id);
                let allids = self.get_color(c_id).unwrap().black_ids(&self, id)
                    .unwrap_or(&single_id).iter().copied().collect_vec();
                // I need to build parents while aware what is a colored edge
                // Colored edges might be deleted, and they need to be updated in colored_memo if not
                let mut parents: Vec<(L, bool, Id, Option<Id>)> = vec![];
                for g in allids {
                    for (p, id) in &self[g].parents {
                        let canoned = self.colored_canonize(c_id, p);
                        let fixed_id = self.find(*id);
                        if let Some(memo_id) = memo.remove(&canoned) {
                            to_union.push((fixed_id, memo_id));
                        }
                        // I need bool and option for sorting
                        parents.push((canoned, true, fixed_id, None));
                    }
                    for (mut p, id) in self[g].colored_parents.remove(&c_id).unwrap_or(vec![]) {
                        dassert!(self[id].color.unwrap() == c_id, "Color mismatch");
                        let fixed_id = self.find(id);
                        if let Some(memo_id) = memo.remove(&p) {
                            to_union.push((fixed_id, memo_id));
                        }
                        self.colored_memo[&c_id].remove(&p);
                        self.colored_update_node(c_id, &mut p);
                        if let Some(memo_id) = memo.remove(&p) {
                            to_union.push((fixed_id, memo_id));
                        }
                        parents.push((p, false, fixed_id, Some(g)));
                    }
                }
                if let Some(pars) = self[id].colord_changed_parents.remove(&c_id) {
                    #[cfg(debug_assertions)]
                    let fixed_to_union: IndexSet<(Id, Id)> = to_union
                        .iter()
                        .map(|(a, b)| (self.colored_find(c_id, *a), self.colored_find(c_id, *b)))
                        .collect();
                    for (n, p_id) in pars {
                        trace!("Removing colored parent from memo: {:?} {:?}", n, p_id);
                        #[allow(unused_variables)]
                        let opt_old = self.colored_memo[&c_id].remove(&n);
                        #[cfg(debug_assertions)]
                        if let Some(m_id) = opt_old {
                            let fixed_m_id = self.colored_find(c_id, m_id);
                            let fixed_p_id = self.colored_find(c_id, p_id);
                            if !(fixed_to_union.contains(&(fixed_p_id, fixed_m_id))
                                || fixed_to_union.contains(&(fixed_m_id, fixed_p_id)))
                            {
                                assert_eq!(fixed_m_id, fixed_p_id,
                                               "Found unexpected non-equivalence for {:?}(memo)!={:?}(changed_parent) for enode {:?}",
                                               fixed_m_id, fixed_p_id, n);
                            }
                        }
                    }
                }
                for (n, _, p_id, _) in parents.iter() {
                    for child in n.children().iter().filter(|c| **c != id) {
                        dassert!(self.colored_find(c_id, *child) == *child);
                        trace!(
                            "Adding colored parent to changed parents: {:?} {:?}",
                            n,
                            p_id
                        );
                        self[*child]
                            .colord_changed_parents
                            .entry(c_id)
                            .or_default()
                            .push((n.clone(), *p_id));
                    }
                }
                // TODO: we might be able to prevent parent recollection by memoization.
                // There is a bit of a trick here. We are sorting such that black nodes will be
                // last. The reason is that we will delete all colored nodes that have a black rep.
                // Of course we need to act differently for [colored_no_cremove]. Dedup goes in
                // reverse order, so we need to reverse the check.
                if cfg!(feature = "colored_no_cremove") {
                    parents
                        .iter_mut()
                        .for_each(|(_n1, is_black_1, _e1, _opt1)| *is_black_1 = !*is_black_1);
                }
                parents.sort_unstable();
                if cfg!(feature = "colored_no_cremove") {
                    parents
                        .iter_mut()
                        .for_each(|(_n1, is_black_1, _e1, _opt1)| *is_black_1 = !*is_black_1);
                }
                parents.dedup_by(|(n1, is_black_1, e1, _opt1), (n2, is_black_2, e2, _opt2)| {
                    n1 == n2 && {
                        if cfg!(feature = "colored_no_cremove") {
                            dassert!(*is_black_2 || !*is_black_1)
                        } else {
                            dassert!(*is_black_1 || !*is_black_2)
                        }
                        to_union.push((*e1, *e2));
                        true
                    }
                });

                #[allow(unused_variables)]
                for (n, is_black, e, orig_class) in parents {
                    if let Some(old) = memo.insert(n.clone(), e) {
                        to_union.push((old, e));
                    }

                    #[allow(unused_attributes)]
                    #[allow(unreachable_code)]
                    if !is_black {
                        #[cfg(feature = "colored_no_cmemo")]
                        panic!("Should not have colored enodes in no_cmemo mode");
                        dassert!(
                            self.colored_canonize(c_id, &n) == n,
                            "Colored canonize should be idempotent {:?} {:?}",
                            self.colored_canonize(c_id, &n),
                            n
                        );
                        // if let Some(b_ids) = self.get_color(c_id).unwrap().black_ids(e) {
                        //     dassert!(b_ids.iter().any(|b_id| self[*b_id].nodes.iter().any(|n1| &self.colored_canonize(c_id, n1) == &n)));
                        // }
                        trace!(
                            "Adding colored parent to memo: {:?} {:?} @ color {c_id}",
                            n,
                            e
                        );
                        let old = self.colored_memo[&c_id].insert(n.clone(), e);
                        if let Some(old) = old {
                            to_union.push((old, e));
                        }
                        let orig_class = orig_class.unwrap();
                        self[orig_class]
                            .colored_parents
                            .entry(c_id)
                            .or_default()
                            .push((n, e));
                    }
                }
            }

            for (id1, id2) in to_union.drain(..) {
                self.colored_union(c_id, id1, id2);
            }
        }

        assert!(
            self.get_color(c_id).unwrap().dirty_unions.is_empty(),
            "Dirty unions should be empty {}",
            self.get_color(c_id).unwrap().dirty_unions.iter().join(", ")
        );
        assert!(
            to_union.is_empty(),
            "to_union should be empty {}",
            to_union
                .iter()
                .map(|x| format!("{}-{}", x.0, x.1))
                .join(", ")
        );
        self.get_color(c_id).unwrap().assert_black_ids(self);
    }

    fn memo_black_canonized(&self) {
        debug_assert!(self
            .memo
            .keys()
            .all(|n| self.memo.contains_key(&self.canonize(n))));
    }

    fn colored_memo_canonized(&self) {
        if cfg!(debug_assertions) {
            for (c, c_memo) in self.colored_memo.iter() {
                for (n, id) in c_memo {
                    let mut is_deleted: Option<bool> = None;
                    let canoned_n = self.colored_canonize(*c, n);
                    let deleted: fn(&EGraph<L, N>, n: &L, c: ColorId, &mut Option<bool>) -> bool =
                        |egraph: &EGraph<L, N>,
                         n: &L,
                         c: ColorId,
                         is_deleted: &mut Option<bool>| {
                            if let Some(is_deleted) = is_deleted.clone() {
                                return is_deleted;
                            }
                            let canoned_n = egraph.colored_canonize(c, n);
                            let res = egraph
                                .memo
                                .iter()
                                .any(|(n1, _e1)| canoned_n == egraph.colored_canonize(c, n1));
                            *is_deleted = Some(res);
                            res
                        };
                    tassert!(
                        {
                            self.colored_memo[c].contains_key(&self.colored_canonize(*c, n))
                                || deleted(self, n, *c, &mut is_deleted)
                        },
                        "Missing {:?} (orig: {:?}) in {} id (under color {})",
                        self.colored_canonize(*c, n),
                        n,
                        id,
                        c
                    );
                    dassert!(
                        ((is_deleted.is_none() || !is_deleted.as_ref().unwrap())
                            && self.colored_memo[c].contains_key(&canoned_n))
                            || deleted(self, n, *c, &mut is_deleted)
                    );
                    if n.children().len() > 0
                        && (is_deleted.is_none() || !*is_deleted.as_ref().unwrap())
                    {
                        dassert!(self.find(self.colored_memo[c][&canoned_n]) == self.find(*id) || deleted(self, n, *c, &mut is_deleted),
                            "Colored memo does not have correct id for {:?} in color {}. It is {} but should be {}", n, c, self.colored_memo[c][&canoned_n], self.find(*id));
                    }
                    // dassert!(&self.colored_canonize(*c, n) == n ||
                    //     self.memo.iter().any(|(n1, e1)| {
                    //         self.colored_canonize(*c, n) == self.colored_canonize(*c, n1)
                    //     }), "The node {:?} was not canonized to {:?} in {}", n, self.colored_canonize(*c, n), c);
                }
            }
        }
    }

    fn memo_all_canonized(&self) {
        self.memo_black_canonized();
        self.colored_memo_canonized();
        self.memo_enode_integrity();
    }

    fn memo_enode_integrity(&self) {
        dassert!({
            for (key, value) in &self.memo {
                // assert_eq!(value,  &self.find(*value), "Memo should point to canonized class {} <- {:?}. Canonized edge: {:?}. Uncanonized class is {:?}", *value, key, self.canonize(key), self.classes[value.0 as usize]);
                assert!(
                    self[*value].nodes.binary_search(key).is_ok(),
                    "Bad edge in class {} edge {:?} canonized {:?}\n Class nodes: {:?}",
                    *value,
                    key,
                    self.canonize(key),
                    self[*value].nodes
                );
            }
            for (c, c_memo) in &self.colored_memo {
                for (key, id) in c_memo {
                    assert!(
                        self[*id].color == Some(*c),
                        "Bad color in colored memo {} <- {:?} (color {})",
                        *id,
                        key,
                        c
                    );
                    let found = self[*id].nodes.binary_search(key).is_ok();
                    let fixed_colored_id = self.colored_find(*c, *id);
                    let fixed_id = self.find(*id);
                    if !found {
                        println!("Stop here! {}", self.memo.contains_key(key));
                        println!("Memo contains key: {:?}", self.memo.contains_key(key));
                        println!("Is canonized: {:?}", &self.colored_canonize(*c, key) == key);
                        if let Some(is) = self.colors[c.0 as usize].as_ref().unwrap()
                            .black_ids(self, *id) {
                            for id_i in is {
                                if self[*id_i].nodes.iter().find(|x| *x == key).is_some() {
                                    println!(
                                        "Found in black id {} (fixed: {})",
                                        id_i,
                                        self.find(*id_i)
                                    );
                                }
                            }
                        }
                        for class in self.classes() {
                            let fixed = class
                                .nodes
                                .iter()
                                .map(|x| self.colored_canonize(*c, x))
                                .collect_vec();
                            if fixed.iter().find(|x| *x == key).is_some() {
                                println!("Found in class {}", class.id);
                            }
                        }

                        println!("Colored memo is: {:?}", self.colored_memo[c]);
                        println!("Colored classes nodes are:");
                        for class in self.classes().filter(|x| x.color == Some(*c)) {
                            println!("Class {} nodes: {:?}", class.id, class.nodes);
                        }
                        // Create dot file
                        self.colored_dot(*c).to_dot("debug.dot").unwrap();
                    }
                    assert!(
                        found,
                        "Edge {:?}(={:?}) not in class id {}(={}) under color {}",
                        key,
                        self.colored_canonize(*c, key),
                        fixed_id,
                        fixed_colored_id,
                        c
                    );
                }
            }
            true
        });
    }

    fn memo_classes_agree(&self) {
        debug_assert!(
            self.memo
                .iter()
                .all(|(_n, id)| self[self.find(*id)].color.is_none())
                && self.colored_memo.iter().all(|(c, c_memo)| c_memo
                    .iter()
                    .all(|(_, id)| self[self.find(*id)].color == Some(*c)))
        );
    }

    /// Assert that the invariant that there is at most one colored EClass in a colored equality class.
    /// This is true because for each colored equality class, there is at most one colored EClass for the colored ENodes.
    pub fn no_two_colored_classes_in_ec(&self) -> bool {
        dassert!({
            for c in self.colors() {
                for res in self.classes().map(|e| e.id) {
                    c.equality_classes.get(&res).iter().for_each(|ids| {
                        dassert!(
                            ids.iter()
                                .map(|id| {
                                    if self[*id].color.is_some() && !self[*id].nodes.is_empty() {
                                        1
                                    } else {
                                        0
                                    }
                                })
                                .sum::<usize>()
                                <= 1,
                            "Color: {}, Ids: {}",
                            c,
                            c.equality_classes[&res].iter().join(", ")
                        )
                    });
                }
            }
            true
        });
        true
    }

    #[inline(never)]
    fn propagate_metadata(&mut self, parents: &[(L, Id)]) {
        for (n, e) in parents {
            let e = self.find(*e);
            let node_data = N::make(self, n);
            let class = self.classes[usize::from(e)].as_mut().unwrap();
            if self.analysis.merge(&mut class.data, node_data) {
                // self.dirty_unions.push(e); // NOTE: i dont think this is necessary
                let e_parents = std::mem::take(&mut class.parents);
                self.propagate_metadata(&e_parents);
                self[e].parents = e_parents;
                N::modify(self, e)
            }
        }
    }

    /// If every `Var` in self agrees with other and the colors match then return true
    pub fn subst_agrees(&self, s1: &Subst, s2: &Subst, allow_missing_vars: bool) -> bool {
        (s1.color == s2.color || s1.color.is_none() || s2.color.is_none())
            && s1.vec.iter().all(|(v, i1)| {
                s2.get(*v)
                    .map(|i2| {
                        self.opt_colored_find(s1.color, *i1) == self.opt_colored_find(s1.color, *i2)
                    })
                    .unwrap_or(allow_missing_vars)
            })
    }
}

// ***  Colored Implementation  ***
#[cfg(feature = "colored")]
impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    #[allow(missing_docs)]
    pub fn create_sub_color(&mut self, color: ColorId) -> ColorId {
        self.create_combined_color(vec![color])
    }

    #[allow(missing_docs)]
    fn process_colored_unions(&mut self) {
        for i in 0..self.colors.len() {
            self.colored_cong_closure(ColorId(i));
            assert!(!self.colors[i].as_ref().unwrap().is_dirty());
        }
    }

    #[allow(missing_docs)]
    pub fn create_color(&mut self) -> ColorId {
        self.colors.push(Some(Color::new(ColorId::from(self.colors.len()))));
        let c_id = self.colors.last().unwrap().as_ref().unwrap().get_id();
        self.colored_memo.insert(c_id, Default::default());
        return c_id;
    }

    /// Do not use yet
    pub fn delete_color(&mut self, c_id: ColorId) {
        assert!(self.dirty_unions.is_empty());
        // TODO: Remove from parents and children
        let color = std::mem::replace(&mut self.colors[c_id.0], None).unwrap();
        for (colored, black) in color.black_colored_classes {
            let class = std::mem::replace(&mut self.classes[black.0 as usize], None).unwrap();
            for n in &class.nodes {
                self.classes_by_op
                    .get_mut(&n.op_id())
                    .map(|x| x.remove(&class.id));
            }
            self.colored_equivalences[&black].remove(&(c_id, colored));
            if self.colored_equivalences[&black].is_empty() {
                self.colored_equivalences.remove(&black);
            }
            self.colored_equivalences[&colored].remove(&(c_id, black));
            if self.colored_equivalences[&colored].is_empty() {
                self.colored_equivalences.remove(&colored);
            }
        }
        self.colored_memo.remove(&c_id);
        dassert!(self.colored_equivalences.iter().all(|(id, ids)| ids
            .iter()
            .chain([(c_id, *id)].iter())
            .all(|(c_id, id)| self[*id].color.iter().all(|c| c != c_id))));
    }

    #[allow(missing_docs)]
    pub fn opt_colored_union(&mut self, color: Option<ColorId>, id1: Id, id2: Id) -> (Id, bool) {
        if let Some(color) = color {
            self.colored_union(color, id1, id2)
        } else {
            self.union(id1, id2)
        }
    }

    #[allow(missing_docs)]
    pub fn colored_union(&mut self, color: ColorId, id1: Id, id2: Id) -> (Id, bool) {
        let (id1, id2) = (self.find(id1), self.find(id2));
        let (to, from, changed, todo) = self
            .get_color_mut(color)
            .unwrap()
            .inner_colored_union(id1, id2);
        dassert!({
            let fixed = self.colored_find(color, id1);
            self[fixed].color.is_none() || self[fixed].color().unwrap() == color
        });
        dassert!({
            let fixed = self.colored_find(color, id2);
            self[fixed].color.is_none() || self[fixed].color().unwrap() == color
        });
        if changed {
            let from_cp = self[from]
                .colord_changed_parents
                .remove(&color)
                .unwrap_or_default();
            self[to]
                .colord_changed_parents
                .entry(color)
                .or_default()
                .extend(from_cp);
        }
        if let Some((id1, id2)) = todo {
            wassert!(self[id1].color().unwrap() == color);
            wassert!(
                self[id1].color().unwrap() == self[id2].color().unwrap(),
                "Todo returned from colored union should be two colored 'black' classes"
            );
            self.union(id1, id2);
        }
        // if changed {
        //     for child in self.get_color(color).unwrap().children().iter().copied().collect_vec() {
        //         // TODO: fix this. id1 and id2 might need to be translated
        //         let child_c = self.get_color(child).unwrap();
        //         self.colored_union(child, child_c.translate_from_base(id1), child_c.translate_from_base(id2));
        //     }
        // }
        tassert!(AssertConfig::new(AssertLevel::Trace); self.classes().filter(|x| x.color().is_some())
            .group_by(|x| (self.opt_colored_find(x.color(),  x.id), x.color().unwrap()))
            .into_iter().all(|(_c, classes)| classes.count() == 1), "A color has two colored classes for a single Id");
        (to, changed)
    }

    pub fn colored_find(&self, color: ColorId, id: Id) -> Id {
        self.get_color(color).unwrap().find(&self, id)
    }

    pub fn opt_colored_find(&self, color: Option<ColorId>, id: Id) -> Id {
        if let Some(color) = color {
            self.get_color(color).unwrap().find(&self, id)
        } else {
            self.find(id)
        }
    }

    pub fn colors(&self) -> impl Iterator<Item = &Color<L, N>> {
        self.colors.iter().filter_map(|x| x.as_ref())
    }

    pub fn colors_mut(&mut self) -> impl Iterator<Item = &mut Color<L, N>> {
        self.colors.iter_mut().filter_map(|x| x.as_mut())
    }

    pub fn get_color(&self, color: ColorId) -> Option<&Color<L, N>> {
        self.colors[usize::from(color)].as_ref()
    }

    pub fn get_color_mut(&mut self, color: ColorId) -> Option<&mut Color<L, N>> {
        self.colors[usize::from(color)].as_mut()
    }

    pub fn get_colored_equalities(&self, id: Id) -> Option<&IndexSet<(ColorId, Id)>> {
        self.colored_equivalences.get(&id)
    }

    pub fn color_sizes(&self) -> impl Iterator<Item = (ColorId, usize)> + '_ {
        self.colored_memo.iter().map(|(c, m)| (*c, m.len()))
    }

    #[cfg(feature = "colored")]
    pub fn detect_color_vacuity(&self) -> Vec<ColorId> {
        self.vacuity_ops
            .iter()
            .flat_map(|disjoints| {
                let empty = Default::default();
                let dis_classes = disjoints
                    .into_iter()
                    .map(|o_id| self.classes_by_op.get(o_id).unwrap_or(&empty))
                    .map(|x| x.iter().map(|x| (self[*x].color, *x)).into_group_map())
                    .collect_vec();
                dassert!({
                    let empty = vec![];
                    dis_classes
                        .iter()
                        .flat_map(|x| x.get(&None).unwrap_or(&empty))
                        .unique()
                        .count()
                        == dis_classes
                            .iter()
                            .map(|x| x.get(&None).map_or(0, |x| x.len()))
                            .sum::<usize>()
                });
                self.colors()
                    .filter(|c| {
                        let canonized = dis_classes
                            .iter()
                            .map(|map| {
                                map.get(&None)
                                    .map_or(vec![].iter(), |x| x.iter())
                                    .chain(
                                        map.get(&Some(c.get_id()))
                                            .map_or(vec![].iter(), |x| x.iter()),
                                    )
                                    .map(|x| self.colored_find(c.get_id(), *x))
                                    .collect::<IndexSet<Id>>()
                            })
                            .collect_vec();
                        canonized.iter().map(|x| x.len()).sum::<usize>()
                            != canonized.into_iter().flatten().unique().count()
                    })
                    .map(|c| c.get_id())
                    .collect_vec()
            })
            .unique()
            .collect()
    }

    pub fn detect_graph_vacuity(&self) -> bool {
        let empty = Default::default();
        self.vacuity_ops.iter().any(|disjoints| {
            assert!(disjoints.len() > 1);
            let folded = disjoints
                .into_iter()
                .map(|o_id| self.classes_by_op.get(o_id).unwrap_or(&empty))
                .fold((false, empty.clone()), |(res, acc), ids| {
                    if res {
                        (res, empty.clone())
                    } else {
                        let next_intersection = acc.intersection(ids).next();
                        let unioned = acc.union(ids).copied().collect();
                        (!next_intersection.is_none(), unioned)
                    }
                });
            folded.0
        })
    }

    #[allow(dead_code)]
    /// For `Deserialization`
    pub(crate) fn add_class<'a>(&'a mut self, id: Id) {
        let idx = usize::from(id);
        while self.classes.len() <= idx {
            self.classes.push(None)
        }
        if self.classes[idx].is_none() {
            let dummy_node = L::from_op_str("?", vec![]).unwrap();
            self.classes[idx] = Some(Box::new(EClass {
                id: id,
                nodes: vec![],
                data: N::make(self, &dummy_node),
                parents: vec![],
                color: None,
                colored_parents: Default::default(),
                changed_parents: Default::default(),
                colord_changed_parents: Default::default(),
            }));
            self.unionfind.make_set_at(id);
        }
    }

    #[allow(dead_code)]
    /// For `Deserialization`
    pub(crate) fn add_node(&mut self, eclass: Id, enode: L) -> &L {
        self.update_parents(eclass, &enode);
        EGraph::<L, ()>::update_memo_from_parent(&mut self.memo, &enode, &eclass);
        let class = self.classes[usize::from(eclass)].as_mut().unwrap();
        class.nodes.push(enode);
        return class.nodes.last().unwrap();
    }

    fn update_parents(&mut self, parent: Id, enode: &L) {
        enode.children().iter().for_each(|u| {
            self.classes[usize::from(*u)]
                .as_mut()
                .unwrap()
                .parents
                .push((enode.clone(), parent))
        });
    }
}

#[cfg(feature = "colored")]
impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    /// Create a new color which is based on given colors. This should be used only if the new color
    /// has no assumptions of it's own (i.e. it is only a combination of existing assumptions).
    pub fn create_combined_color(&mut self, colors: Vec<ColorId>) -> ColorId {
        // First check if assumptions exist
        let new_c_id = self.create_color();
        assert!(colors.len() > 0);
        iassert!(self.dirty_unions.is_empty());
        iassert!(colors
            .iter()
            .all(|c| !self.get_color(*c).unwrap().is_dirty()));
        let mut todo = vec![];
        let mut by_color = vec![];
        for c in colors.iter().copied() {
            let union_map = self.collect_colored_equality_classes(&c);
            let id_changer = self.initialize_id_changer(&union_map);
            let new_classes = self.create_new_colored_classes(new_c_id, &id_changer);
            #[cfg(feature = "colored_no_cmemo")]
            iassert!(new_classes.len() == 0);
            iassert!(
                IndexSet::<Id>::from_iter(id_changer.values().copied())
                    == IndexSet::<Id>::from_iter(new_classes.iter().copied())
            );
            for new_class_id in new_classes.iter().copied() {
                // Already did first edge in create_new_colored_classes
                let the_one = self.create_data_for_nodes(new_class_id);
                self[new_class_id].data = the_one;

                let nodes = self[new_class_id].nodes.clone();
                for n in nodes {
                    let old = self.colored_memo[&new_c_id].insert(n, new_class_id);
                    if let Some(old) = old {
                        todo.push((old, new_class_id));
                    }
                }
            }
            for (root, ids) in &union_map {
                for id in ids {
                    todo.push((
                        *id_changer.get(root).unwrap_or(root),
                        *id_changer.get(id).unwrap_or(id),
                    ));
                }
                let colored_class = self.get_color(c).unwrap().black_colored_classes.get(root);
                dassert!(ids.iter().filter(|e| self[**e].color().is_some()).count() <= 1);
                // Can't have two colored eclasses for the id because I did not merge anything yet
                if let Some(new_class_id) = colored_class.map(|e| id_changer.get(e)).flatten() {
                    assert_eq!(
                        None,
                        self.get_color_mut(new_c_id)
                            .unwrap()
                            .black_colored_classes
                            .insert(*new_class_id, *new_class_id)
                    );
                }
            }
            by_color.push((new_classes, id_changer, union_map));
        }

        for (c, (new_classes, id_changer, _union_map)) in colors.iter().zip(by_color) {
            // for (black_id, ids) in union_map {
            //     for id in ids.iter() {
            //         todo.extend(self.get_color_mut(new_c_id).unwrap().inner_colored_union(black_id, *id_changer.get(id).unwrap_or(id)).2);
            //     }
            // }

            // Assert id changer points to correct colors
            iassert!(id_changer
                .iter()
                .all(|(k, v)| self[*k].color().unwrap() == *c
                    && self[*v].color().unwrap() == new_c_id));
            // Now fix nodes, and create data, and put in the parents with id_translation.
            for id in new_classes {
                let mut parents_to_add = vec![];
                for n in self[id].iter() {
                    for ch in n.children() {
                        iassert!(
                            self[*ch].color().is_none() || self[*ch].color().unwrap() == new_c_id
                        );
                        parents_to_add.push((*ch, n.clone(), id));
                    }
                }
                for (ch, n, id) in parents_to_add {
                    self[ch]
                        .colored_parents
                        .entry(new_c_id)
                        .or_default()
                        .push((n, id));
                }
                N::modify(self, id);
            }

            self.get_color_mut(*c).unwrap().children.push(new_c_id);
            self.get_color_mut(new_c_id).unwrap().parents.push(*c);
            self.get_color_mut(new_c_id)
                .unwrap()
                .parents_classes
                .push(id_changer.into_iter().collect());
        }

        for (id1, id2) in todo {
            self.colored_union(new_c_id, id1, id2);
        }
        // TODO: Is it necessary or can it wait for other colors? It probably doesnt matter.
        self.rebuild();
        self.no_two_colored_classes_in_ec();
        new_c_id
    }

    fn create_data_for_nodes(&mut self, new_class_id: Id) -> <N as Analysis<L>>::Data {
        let mut datas = self[new_class_id]
            .nodes
            .iter()
            .map(|n| N::make(self, &n))
            .collect_vec();
        let mut the_one = datas.pop().unwrap();
        for d in datas {
            self.analysis.merge(&mut the_one, d);
        }
        the_one
    }

    fn create_new_colored_classes(
        &mut self,
        new_c_id: ColorId,
        id_changer: &IndexMap<Id, Id>,
    ) -> Vec<Id> {
        let mut new_classes = vec![];
        for (key, value) in id_changer.iter() {
            // First I have to create some dumb classes to prevent bad finds on new class node.
            let mut n = self[*key].nodes[0].clone();
            let new_class_id = self.inner_create_class(&mut n, Some(new_c_id));
            dassert!(new_class_id.0 == value.0);
            new_classes.push(new_class_id);
        }
        for (key, value) in id_changer.iter() {
            // Assert class not empty
            iassert!(
                !self[*key].is_empty(),
                "Expecting that all defined classes will have edges"
            );
            let nodes = self[*key]
                .nodes
                .iter()
                .cloned()
                .map(|n| n.map_children(|id| *id_changer.get(&id).unwrap_or(&id)))
                .collect_vec();
            self[*value].nodes.pop();
            self[*value].nodes.extend(nodes);
            self[*value].data = N::make(self, &self[*value].nodes[0]);
        }
        new_classes
    }

    fn initialize_id_changer(&mut self, union_map: &Vec<(Id, IndexSet<Id>)>) -> IndexMap<Id, Id> {
        let mut id_changer = IndexMap::new();
        union_map.iter().for_each(|(black_id, ids)| {
            dassert!(ids.contains(black_id));
            dassert!(
                ids.iter()
                    .map(
                        |id| if self[*id].color.is_some() && !self[*id].nodes.is_empty() {
                            1
                        } else {
                            0
                        }
                    )
                    .sum::<usize>()
                    <= 1,
                "Ids: {}",
                ids.iter().join(", ")
            );
            let colored = ids.iter().find(|id| self[**id].color.is_some());
            colored.copied().map(|id| {
                id_changer.insert(id, Id((self.classes.len() + id_changer.len()) as u32))
            });
        });
        id_changer
    }

    fn collect_colored_equality_classes(&mut self, c: &ColorId) -> Vec<(Id, IndexSet<Id>)> {
        let c = *c;
        let mut union_map = self
            .get_color(c)
            .unwrap()
            .equality_classes
            .clone()
            .into_iter()
            .collect_vec();
        for class in self.classes() {
            if self
                .get_color(c)
                .unwrap()
                .equality_classes
                .contains_key(&self.colored_find(c, class.id))
            {
                dassert!(
                    self.get_color(c).unwrap().equality_classes[&self.colored_find(c, class.id)]
                        .contains(&class.id)
                );
                continue;
            }
            if class.color.is_none() || class.color == Some(c) {
                union_map.push((class.id, IndexSet::from_iter([class.id].iter().copied())));
            }
        }
        let union_map = union_map;
        union_map
    }
}

struct EGraphDump<'a, L: Language, N: Analysis<L>>(&'a EGraph<L, N>);

impl<'a, L: Language, N: Analysis<L>> Debug for EGraphDump<'a, L, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ids: Vec<Id> = self.0.classes().map(|c| c.id).collect();
        ids.sort();
        for id in ids {
            let mut nodes = self.0[id].nodes.clone();
            nodes.sort();
            writeln!(f, "{}: {:?}", id, nodes)?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::rewrite::*;
    use crate::*;
    use itertools::Itertools;
    use log::*;
    use std::str::FromStr;

    #[test]
    fn simple_add() {
        use SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        let x = egraph.add(S::leaf("x"));
        let x2 = egraph.add(S::leaf("x"));
        let _plus = egraph.add(S::new("+", vec![x, x2]));

        let y = egraph.add(S::leaf("y"));

        egraph.union(x, y);
        egraph.rebuild();

        egraph.dot().to_dot("target/foo.dot").unwrap();

        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn color_congruent() {
        use SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        // black x, f(x)
        // black y, f(y)
        // blue: x = y => f(x) = f(y)

        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));
        let fx = egraph.add_expr(&RecExpr::from_str("(f x)").unwrap());
        let fy = egraph.add_expr(&RecExpr::from_str("(f y)").unwrap());

        let color = egraph.create_color();
        egraph.colored_union(color, x, y);
        egraph.rebuild();

        assert_eq!(
            egraph.colored_find(color, fx),
            egraph.colored_find(color, fy)
        );
    }

    #[test]
    fn black_merge_color_congruent() {
        use SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        // black x, f(x)
        // black y, f(y)
        // blue: w = x
        // black w = y => blue x = y => blue f(x) = f(y)

        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));
        let w = egraph.add(S::leaf("w"));
        let fx = egraph.add_expr(&RecExpr::from_str("(f x)").unwrap());
        let fy = egraph.add_expr(&RecExpr::from_str("(f y)").unwrap());

        let color = egraph.create_color();
        let c = &egraph.get_color(color).unwrap();
        c.assert_black_ids(&egraph);
        egraph.colored_union(color, w, x);
        let c = &egraph.get_color(color).unwrap();
        c.assert_black_ids(&egraph);
        egraph.union(w, y);
        let c = &egraph.get_color(color).unwrap();
        warn!("{:?}", c.find(&egraph, w));
        c.assert_black_ids(&egraph);
        egraph.rebuild();

        assert_eq!(
            egraph.colored_find(color, fx),
            egraph.colored_find(color, fy)
        );
    }

    #[test]
    fn unroll_list_rev_concat() {
        let rev = rewrite!("reverse-base"; "(rev nil)" <=> "nil");
        let rev2 = rewrite!("reverse-ind"; "(rev (cons ?x ?l))" <=> "(cons (rev ?l) ?x)");
        let app = rewrite!("app-base"; "(app nil ?x)" => "nil");
        let app2 = rewrite!("app-ind"; "(app (cons ?x ?l) ?y)" <=> "(cons ?x (app ?l ?y))");
        let mut rules = vec![];
        rules.extend_from_slice(&rev);
        rules.extend_from_slice(&rev2);
        rules.extend_from_slice(&app2);
        rules.push(app);

        use SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        egraph.add_expr(&"(rev nil)".parse().unwrap());
        egraph.add_expr(&"(app nil l)".parse().unwrap());
        let a = egraph.add_expr(&"(rev (cons x (cons y nil)))".parse().unwrap());
        egraph.add_expr(&"(app (cons x (cons y nil)) l)".parse().unwrap());
        egraph.add_expr(&"(app (rev (cons x (cons y nil))) (rev l))".parse().unwrap());
        egraph.add_expr(&"(rev (app (cons x (cons y nil)) l))".parse().unwrap());

        let mut runner = Runner::default().with_egraph(egraph).with_iter_limit(8);
        println!("{:#?}", runner.egraph.total_size());
        runner = runner.run(&rules);
        println!("{:#?}", runner.egraph.total_size());
        assert!(runner.egraph[a].nodes.len() > 1);
    }

    #[test]
    fn union_maps_changes_after_unions() {
        use SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        let ex1 = egraph.add_expr(&"x".parse().unwrap());
        let ex2 = egraph.add_expr(&"y".parse().unwrap());
        let ex3 = egraph.add_expr(&"z".parse().unwrap());
        let ex4 = egraph.add_expr(&"a".parse().unwrap());
        let ex5 = egraph.add_expr(&"s".parse().unwrap());
        let ex6 = egraph.add_expr(&"d".parse().unwrap());

        let c = egraph.create_color();
        egraph.colored_union(c, ex1, ex2);
        egraph.colored_union(c, ex1, ex3);
        egraph.colored_union(c, ex1, ex4);
        egraph.colored_union(c, ex5, ex6);
        let (to, _) = egraph.colored_union(c, ex1, ex5);
        assert_eq!(
            egraph.get_color(c).unwrap().black_ids(&egraph, to).map(|x| x.len()),
            Some(6)
        );

        egraph.union(ex5, ex6);
        egraph.union(ex1, ex5);
        println!("{:#?}", egraph.get_color(c).unwrap().black_ids(&egraph, to));
        assert_eq!(
            egraph.get_color(c).unwrap().black_ids(&egraph, to).map(|x| x.len()),
            Some(4)
        );
    }

    #[test]
    fn color_hierarchy_union() {
        use SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        let ex1 = egraph.add_expr(&"x".parse().unwrap());
        let ex2 = egraph.add_expr(&"y".parse().unwrap());
        let ex3 = egraph.add_expr(&"z".parse().unwrap());
        let ex4 = egraph.add_expr(&"a".parse().unwrap());
        let ex5 = egraph.add_expr(&"s".parse().unwrap());
        let ex6 = egraph.add_expr(&"d".parse().unwrap());

        let c1 = egraph.create_color();
        let c2 = egraph.create_color();
        let c3 = egraph.create_combined_color(vec![c1, c2]);

        egraph.colored_union(c3, ex1, ex2);
        assert_eq!(egraph.colored_find(c3, ex1), egraph.colored_find(c3, ex2));
        egraph.colored_union(c3, ex3, ex4);
        egraph.colored_union(c3, ex1, ex3);
        assert_eq!(egraph.colored_find(c3, ex1), egraph.colored_find(c3, ex3));
        egraph.colored_union(c3, ex5, ex6);
        assert_eq!(egraph.colored_find(c3, ex5), egraph.colored_find(c3, ex6));
        assert_eq!(egraph.colored_find(c3, ex3), egraph.colored_find(c3, ex4));
        let (to, _) = egraph.colored_union(c3, ex1, ex5);
        egraph.rebuild();
        assert_eq!(
            egraph.get_color(c3).unwrap().black_ids(&egraph, to).map(|x| x.len()),
            Some(6)
        );
    }

    #[test]
    fn color_congruence_closure() {
        use SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));
        let w = egraph.add(S::leaf("w"));
        let fx = egraph.add_expr(&RecExpr::from_str("(f x)").unwrap());
        let fy = egraph.add_expr(&RecExpr::from_str("(f y)").unwrap());

        let color1 = egraph.create_color();
        let color2 = egraph.create_color();
        egraph.colored_union(color1, w, x);
        egraph.colored_union(color2, w, y);
        egraph.rebuild();
        assert_ne!(
            egraph.colored_find(color2, fx),
            egraph.colored_find(color2, fy)
        );
        assert_ne!(
            egraph.colored_find(color1, fx),
            egraph.colored_find(color1, fy)
        );
        let color3 = egraph.create_combined_color(vec![color1, color2]);
        egraph.rebuild();
        assert_eq!(
            egraph.colored_find(color3, fx),
            egraph.colored_find(color3, fy)
        );
    }

    #[test]
    fn color_new_child_unions() {
        use SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));
        let z = egraph.add(S::leaf("z"));
        let w = egraph.add(S::leaf("w"));

        let color1 = egraph.create_color();
        egraph.colored_union(color1, y, x);
        let color2 = egraph.create_color();
        egraph.colored_union(color2, w, z);
        egraph.rebuild();
        let child = egraph.create_combined_color(vec![color1, color2]);
        egraph.colored_union(child, x, z);
        println!("{}", egraph.get_color(child).unwrap());
        egraph.rebuild();
        println!("{}", egraph.get_color(child).unwrap());
        assert_eq!(egraph.colored_find(child, w), egraph.colored_find(child, y));
    }

    #[test]
    fn colored_drop_take() {
        use crate::SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        let nil = egraph.add_expr(&"nil".parse().unwrap());
        let consx = egraph.add_expr(&"(cons x nil)".parse().unwrap());
        let consxy = egraph.add_expr(&"(cons y (cons x nil))".parse().unwrap());
        let ex0 = egraph.add_expr(&"(append (take i nil) (drop i nil))".parse().unwrap());
        let ex1 = egraph.add_expr(
            &"(append (take i (cons x nil)) (drop i (cons x nil)))"
                .parse()
                .unwrap(),
        );
        let ex2 = egraph.add_expr(
            &"(append (take i (cons y (cons x nil))) (drop i (cons y (cons x nil))))"
                .parse()
                .unwrap(),
        );
        info!("Starting first rebuild");
        egraph.rebuild();
        let bad_rws =
            rewrite!("rule10"; "(take (succ ?x7) (cons ?y8 ?z))" <=> "(cons ?y8 (take ?x7 ?z))");
        info!("Done first rebuild");
        let mut rules = vec![
            rewrite!("rule2"; "(append nil ?x)" => "?x"),
            rewrite!("rule5"; "(drop ?x3 nil)" => "nil"),
            rewrite!("rule6"; "(drop zero ?x)" => "?x"),
            rewrite!("rule7"; "(drop (succ ?x4) (cons ?y5 ?z))" => "(drop ?x4 ?z)"),
            rewrite!("rule8"; "(take ?x3 nil)" => "nil"),
            rewrite!("rule9"; "(take zero ?x)" => "nil"),
        ];
        // rules.extend(rewrite!("rule0"; "(leq ?__x0 ?__y1)" <=> "(or (= ?__x0 ?__y1) (less ?__x0 ?__y1))"));
        rules
            .extend(rewrite!("rule3"; "(append (cons ?x2 ?y) ?z)" <=> "(cons ?x2 (append ?y ?z))"));
        rules.extend(bad_rws.clone());

        egraph = Runner::default()
            .with_iter_limit(8)
            .with_node_limit(400000)
            .with_egraph(egraph)
            .run(&rules)
            .egraph;
        info!("Done eq reduction");
        egraph.rebuild();
        assert_eq!(egraph.find(nil), egraph.find(ex0));
        assert_ne!(egraph.find(consx), egraph.find(ex1));
        let color_z = egraph.create_color();
        let color_s_p = egraph.create_color();
        let color_s_z = egraph.create_color();
        let i = egraph.add_expr(&"i".parse().unwrap());
        let zero = egraph.add_expr(&"zero".parse().unwrap());
        let succ_p_n = egraph.add_expr(&"(succ param_n_1)".parse().unwrap());
        let succ_z = egraph.add_expr(&"(succ zero)".parse().unwrap());
        egraph.colored_union(color_z, i, zero);
        egraph.colored_union(color_s_p, i, succ_p_n);
        egraph.colored_union(color_s_z, i, succ_z);
        egraph.rebuild();
        egraph = Runner::default()
            .with_iter_limit(8)
            .with_node_limit(400000)
            .with_egraph(egraph)
            .run(&rules)
            .egraph;
        egraph.rebuild();

        for x in egraph.colors() {
            warn!("{}", x);
            x.assert_black_ids(&egraph);
        }

        egraph.dot().to_dot("graph.dot").unwrap();

        let take_i_nil = egraph.add_expr(&"(take i nil)".parse().unwrap());
        warn!(
            "take i nil - {} - {}",
            take_i_nil,
            egraph.colored_find(color_z, take_i_nil)
        );
        let take_i_consx = egraph.add_expr(&"(take i (cons x nil))".parse().unwrap());
        warn!(
            "take i (cons x nil) - {} - {}",
            take_i_consx,
            egraph.colored_find(color_z, take_i_consx)
        );
        let drop_i_nil = egraph.add_expr(&"(drop i nil)".parse().unwrap());
        warn!(
            "drop i nil - {} - {}",
            drop_i_nil,
            egraph.colored_find(color_z, drop_i_nil)
        );
        let drop_i_consx = egraph.add_expr(&"(drop i (cons x nil))".parse().unwrap());
        warn!(
            "drop i (cons x nil) - {} - {}",
            drop_i_consx,
            egraph.colored_find(color_z, drop_i_consx)
        );

        assert_eq!(
            egraph.colored_find(color_z, consx),
            egraph.colored_find(color_z, ex1)
        );
        assert_eq!(
            egraph.colored_find(color_z, consxy),
            egraph.colored_find(color_z, ex2)
        );
        assert_eq!(
            egraph.colored_find(color_s_p, consx),
            egraph.colored_find(color_s_p, ex1)
        );
        assert_eq!(
            egraph.colored_find(color_s_z, consxy),
            egraph.colored_find(color_s_z, ex2)
        );
    }

    #[test]
    fn colored_plus_succ() {
        use crate::SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        egraph.rebuild();
        let rules = vec![
            rewrite!("rule2"; "(plus Z ?x)" => "?x"),
            rewrite!("rule5"; "(plus (succ ?x) ?y)" => "(succ (plus ?x ?y))"),
        ];

        let init = egraph.add_expr(&"(plus x (succ y))".parse().unwrap());

        let color_z = egraph.create_color();
        // let color_s_p = egraph.create_color();
        let x = egraph.add_expr(&"x".parse().unwrap());
        let zero = egraph.add_expr(&"Z".parse().unwrap());
        egraph.colored_union(color_z, x, zero);
        let res_z = egraph.add_expr(&"(succ y)".parse().unwrap());

        let color_succ = egraph.create_color();
        // let color_s_p = egraph.create_color();
        let succ_z = egraph.add_expr(&"(succ Z)".parse().unwrap());
        // let succ_p_n = egraph.add_expr(&"(succ param_n_1)".parse().unwrap());
        egraph.colored_union(color_succ, x, succ_z);
        let res_succ_z = egraph.add_expr(&"(succ (succ y))".parse().unwrap());
        egraph.rebuild();
        egraph = Runner::default()
            .with_iter_limit(8)
            .with_node_limit(400000)
            .with_egraph(egraph)
            .run(&rules)
            .egraph;
        egraph.rebuild();
        egraph.dot().to_dot("graph.dot").unwrap();

        assert_eq!(
            egraph.colored_find(color_z, init),
            egraph.colored_find(color_z, res_z),
            "Black ids for color_z:\n  {}",
            egraph.get_color(color_z).unwrap().to_string()
        );
        rules[0].search(&egraph).iter().for_each(|x| {
            println!("{}", x);
        });
        assert_eq!(
            egraph.colored_find(color_succ, init),
            egraph.colored_find(color_succ, res_succ_z),
            "Black ids for color_succ:\n  {}",
            egraph.get_color(color_succ).unwrap().to_string()
        );
    }

    #[test]
    fn color_true_eq_false() {
        use crate::SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        // rules for:
        // and (true ?x) => ?x
        // and (false ?x) => false
        // or (true ?x) => true
        // or (false ?x) => ?x
        // not true => false
        // not false => true
        let rules = vec![
            rewrite!("rule2"; "(eq ?x ?y)" => "(eq ?y ?x)"),
            rewrite!("rule3"; "(and true ?x)" => "?x"),
            rewrite!("rule4"; "(and false ?x)" => "false"),
            rewrite!("rule5"; "(or true ?x)" => "true"),
            rewrite!("rule6"; "(or false ?x)" => "?x"),
            rewrite!("rule7"; "(not true)" => "false"),
            rewrite!("rule8"; "(not false)" => "true"),
        ];

        // Add many boolean expressions like "true", "false", and "(and x (or y true))"
        let exprs = vec![
            "true",
            "false",
            "(and true true)",
            "(and x false)",
            "(and false true)",
            "(and false false)",
            "(or y true)",
            "(or true false)",
            "(or false true)",
            "(or false false)",
            "(not true)",
            "(not false)",
            "(not (and z true))",
            "(not (and true false))",
            "(not (and false true))",
            "(not (and false false))",
            "(not (or true true))",
            "(not (or true false))",
            "(not (or false z))",
            "(not (or false false))",
        ];

        let mut ids = vec![];
        for exp in exprs {
            ids.push(egraph.add_expr(&exp.parse().unwrap()));
        }

        egraph.rebuild();
        egraph = Runner::default()
            .with_iter_limit(8)
            .with_node_limit(400000)
            .with_egraph(egraph)
            .run(&rules)
            .egraph;
        egraph.rebuild();

        let t_id = egraph.add_expr(&"true".parse().unwrap());
        let f_id = egraph.add_expr(&"false".parse().unwrap());

        let color_tf = egraph.create_color();
        egraph.colored_union(color_tf, t_id, f_id);

        egraph.rebuild();
        egraph = Runner::default()
            .with_iter_limit(8)
            .with_node_limit(400000)
            .with_egraph(egraph)
            .run(&rules)
            .egraph;
        egraph.rebuild();

        assert_eq!(
            egraph.colored_find(color_tf, ids[0]),
            egraph.colored_find(color_tf, ids[2])
        );
        assert_eq!(
            egraph.colored_find(color_tf, ids[4]),
            egraph.colored_find(color_tf, ids[5])
        );
    }

    fn choose<T: Clone>(mut from: Vec<Vec<T>>, amount: usize) -> Vec<Vec<T>> {
        if from.len() < amount || amount == 0 {
            return vec![];
        }
        if amount == 1 {
            return from
                .clone()
                .into_iter()
                .flatten()
                .map(|v| vec![v])
                .collect_vec();
        }
        let cur = from.pop().unwrap();
        let rec_res = choose(from.clone(), amount - 1);
        let mut new_res = vec![];
        for res in rec_res {
            for u in cur.clone() {
                let mut new = res.clone();
                new.push(u);
                new_res.push(new);
            }
        }
        let other_rec = choose(from, amount);
        new_res.extend(other_rec);
        new_res
    }

    #[test]
    fn multi_level_colored_filter() {
        crate::init_logger();

        let (mut egraph, rules, expr_id, lv1_colors) = initialize_filter_tests();
        let mut lv2_colors = vec![];
        for color_vec in choose(lv1_colors.clone(), 2) {
            lv2_colors.push(egraph.create_combined_color(color_vec));
        }
        let mut lv3_colors = vec![];
        for color_vec in choose(lv1_colors.clone(), 3) {
            lv3_colors.push(egraph.create_combined_color(color_vec));
        }

        let egraph = Runner::default().with_egraph(egraph).run(&rules).egraph;
        for c in lv3_colors {
            println!("Doing something");
            assert!(egraph
                .get_color(c)
                .unwrap()
                .black_ids(&egraph, expr_id)
                .map(|x| x.clone())
                .unwrap_or([egraph.colored_find(c, expr_id)].iter().copied().collect())
                .iter()
                .any(|id| egraph[*id].nodes.iter().any(|n| {
                    let op = format!("{}", n.display_op());
                    op == "nil" || op == "cons"
                })));
        }
    }

    #[test]
    fn multi_level_colored_bad_filter() {
        crate::init_logger();

        let (mut egraph, rules, _expr_id, lv1_colors) = initialize_filter_tests();
        let mut lv2_colors = vec![];
        for c1 in lv1_colors.iter().flatten() {
            for c2 in lv1_colors.iter().flatten() {
                lv2_colors.push(egraph.create_combined_color(vec![*c1, *c2]));
            }
        }
        let mut lv3_colors = vec![];
        for c1 in lv1_colors.iter().flatten() {
            for c2 in lv2_colors.iter() {
                lv3_colors.push(egraph.create_combined_color(vec![*c1, *c2]));
            }
        }

        let mut egraph = Runner::default().with_egraph(egraph).run(&rules).egraph;
        egraph.rebuild();
        egraph.check_memo();
        egraph.memo_all_canonized();
    }

    #[test]
    fn colored_bad_filter() {
        crate::init_logger();

        let (mut egraph, rules, _expr_id, lv1_colors) = initialize_filter_tests();
        let _bad_color = egraph.create_combined_color(lv1_colors[2].iter().copied().collect_vec());

        let mut egraph = Runner::default().with_egraph(egraph).run(&rules).egraph;
        egraph.rebuild();
        egraph.check_memo();
        egraph.memo_all_canonized();
    }

    fn initialize_filter_tests() -> (
        EGraph<SymbolLang, ()>,
        Vec<Rewrite<SymbolLang, ()>>,
        Id,
        Vec<Vec<ColorId>>,
    ) {
        use crate::SymbolLang as S;

        let mut egraph = EGraph::<S, ()>::default();

        let rules: Vec<Rewrite<SymbolLang, ()>> = vec![
            rewrite!("rule1"; "(ite true ?x ?y)" => "?x"),
            rewrite!("rule2"; "(ite false ?x ?y)" => "?y"),
            rewrite!("rule3"; "(and true ?x)" => "?x"),
            rewrite!("rule4"; "(and false ?x)" => "false"),
            rewrite!("rule5"; "(or true ?x)" => "true"),
            rewrite!("rule6"; "(or false ?x)" => "?x"),
            rewrite!("rule7"; "(not true)" => "false"),
            rewrite!("rule8"; "(not false)" => "true"),
            rewrite!("rule9"; "(filter p (cons ?x ?xs))" => "(ite (p ?x) (cons x (filter p ?xs)) (filter p ?xs))"),
            rewrite!("rule10"; "(filter p nil)" => "nil"),
        ];

        let expr_id = egraph.add_expr(
            &"(filter p (cons x1 (cons x2 (cons x3 nil))))"
                .parse()
                .unwrap(),
        );
        let vars = [
            egraph.add_expr(&"(p x1)".parse().unwrap()),
            egraph.add_expr(&"(p x2)".parse().unwrap()),
            egraph.add_expr(&"(p x3)".parse().unwrap()),
        ];
        let tru = egraph.add_expr(&"true".parse().unwrap());
        let fals = egraph.add_expr(&"false".parse().unwrap());
        let lv1_colors = vars
            .iter()
            .map(|id| {
                let color_true = egraph.create_color();
                let color_false = egraph.create_color();
                egraph.colored_union(color_true, *id, tru);
                egraph.colored_union(color_false, *id, fals);
                vec![color_true, color_false]
            })
            .collect_vec();
        egraph.rebuild();
        (egraph, rules, expr_id, lv1_colors)
    }

    #[test]
    fn test_memo_nodes_agree() {
        /*
        I need to recreate the example from my notes:
            unions: (1, 3), (5,6), (8, 9)
            parents (1): [+] \rightarrow [3, 4]
                [+] \rightarrow [1,4]
            parents (5): [-] \rightarrow [5]@2 \quad [-] \rightarrow [6]@4
                [-] \rightarrow [5]@2 -> union(2,4)
            parents (8): [-] \rightarrow [8]@4 \quad [-] \rightarrow [9]@7
                [-] \rightarrow [8]@2 -> union(4,7)
            unions: (2, 4), (4, 7)
            parents (2): [+] \rightarrow [3, 7]
                [+] \rightarrow [1,2]
        */
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::new(());
        let w = RecExpr::from_str("id1").unwrap();
        let x = RecExpr::from_str("id3").unwrap();
        let y = RecExpr::from_str("id4").unwrap();
        let z = RecExpr::from_str("id5").unwrap();
        let k = RecExpr::from_str("id6").unwrap();
        let q = RecExpr::from_str("id7").unwrap();
        let x2 = RecExpr::from_str("id8").unwrap();
        let y2 = RecExpr::from_str("id9").unwrap();
        let id1 = egraph.add_expr(&w);
        let id3 = egraph.add_expr(&x);
        let id4 = egraph.add_expr(&y);
        let id5 = egraph.add_expr(&z);
        let id6 = egraph.add_expr(&k);
        let id7 = egraph.add_expr(&q);
        let id8 = egraph.add_expr(&x2);
        let id9 = egraph.add_expr(&y2);
        let idp37 = egraph.add(SymbolLang::from_op_str("+", vec![id3, id7]).unwrap());
        let id2 = egraph.add(SymbolLang::from_op_str("-", vec![id5]).unwrap());
        let _id2 = egraph.union(id2, idp37).0;
        egraph.rebuild();
        let _idp34 = egraph.add(SymbolLang::from_op_str("+", vec![id3, id4]).unwrap());
        let _temp = egraph.add(SymbolLang::from_op_str("-", vec![id6]).unwrap());
        let temp1 = egraph.add(SymbolLang::from_op_str("-", vec![id8]).unwrap());
        let temp2 = egraph.add(SymbolLang::from_op_str("-", vec![id9]).unwrap());
        let id4 = egraph.union(id4, temp1).0;
        let id7 = egraph.union(id7, temp2).0;
        let _id8 = egraph.union(id8, id9).0;
        let _id5 = egraph.union(id5, id6).0;
        let _id1 = egraph.union(id1, id3).0;
        let _id4 = egraph.union(id4, id7).0;
        println!("All op strings: {:?}", util::get_strings());
        // {0: "id1", 1: "id3", 2: "id4", 3: "id5", 4: "id6", 5: "id7", 6: "+", 7: "-"}
        egraph.rebuild();
        egraph.dot().to_dot("test.dot").unwrap();
    }

    #[test]
    #[cfg(feature = "colored")]
    fn test_colored_vacuity_check_sanity() {
        init_logger();

        // Create an egraph with both Zero and (Succ x y)
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::new(());
        let zero = RecExpr::from_str("Zero").unwrap();
        let succ = RecExpr::from_str("(Succ x y)").unwrap();
        let zero_id = egraph.add_expr(&zero);
        let succ_id = egraph.add_expr(&succ);

        // Create color where they are combined
        let color = egraph.create_color();
        egraph.colored_union(color, zero_id, succ_id);
        egraph.rebuild();

        // Create a color with (Succ z k)
        let color2 = egraph.create_color();
        let succ2 = RecExpr::from_str("(Succ z k)").unwrap();
        let succ2_id = egraph.add_expr(&succ2);
        egraph.colored_union(color2, succ_id, succ2_id);
        egraph.rebuild();

        // Check vacuity (with additonal op options) only returns color
        let zero_op = SymbolLang::leaf("Zero".to_string()).op_id();
        let succ_op = SymbolLang::leaf("Succ".to_string()).op_id();
        let another_op = SymbolLang::leaf("Another".to_string()).op_id();

        // two other sets
        let k_op = SymbolLang::leaf("k".to_string()).op_id();
        let z_op = SymbolLang::leaf("z".to_string()).op_id();
        let x_op = SymbolLang::leaf("x".to_string()).op_id();
        let y_op = SymbolLang::leaf("y".to_string()).op_id();

        egraph.vacuity_ops = vec![
            vec![k_op, z_op],
            vec![zero_op, succ_op, another_op],
            vec![x_op, y_op],
        ];
        let vacs = egraph.detect_color_vacuity();
        assert_eq!(vacs.len(), 1);
        assert_eq!(vacs[0], color);
    }

    #[test]
    fn test_graph_vacuity_check_sanity() {
        init_logger();

        // Create an egraph with both Zero and (Succ x y)
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::new(());
        let zero = RecExpr::from_str("Zero").unwrap();
        let succ = RecExpr::from_str("(Succ x y)").unwrap();
        let zero_id = egraph.add_expr(&zero);
        let succ_id = egraph.add_expr(&succ);

        egraph.rebuild();

        // Check vacuity (with additonal op options) only returns color
        let zero_op = SymbolLang::leaf("Zero".to_string()).op_id();
        let succ_op = SymbolLang::leaf("Succ".to_string()).op_id();
        let another_op = SymbolLang::leaf("Another".to_string()).op_id();

        // two other sets
        let k_op = SymbolLang::leaf("k".to_string()).op_id();
        let z_op = SymbolLang::leaf("z".to_string()).op_id();
        let x_op = SymbolLang::leaf("x".to_string()).op_id();
        let y_op = SymbolLang::leaf("y".to_string()).op_id();

        egraph.vacuity_ops = vec![
            vec![k_op, z_op],
            vec![zero_op, succ_op, another_op],
            vec![x_op, y_op],
        ];

        assert!(!egraph.detect_graph_vacuity());

        let succ2 = RecExpr::from_str("(Succ z k)").unwrap();
        let succ2_id = egraph.add_expr(&succ2);
        egraph.union(succ_id, succ2_id);
        egraph.rebuild();

        assert!(!egraph.detect_graph_vacuity());

        // Create color where they are combined
        egraph.union(zero_id, succ_id);
        egraph.rebuild();

        assert!(egraph.detect_graph_vacuity());
    }
}
