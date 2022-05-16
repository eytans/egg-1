use std::collections::{HashMap, HashSet};
use std::{
    borrow::BorrowMut,
    fmt::{self, Debug},
};
use std::cmp::Ordering;
use std::fmt::Alignment::Left;
use std::iter::Peekable;
use std::ops::{BitOr, BitOrAssign, BitXor, BitXorAssign};
use std::thread::current;
use std::vec::IntoIter;
use bitvec::macros::internal::funty::Fundamental;
use bitvec::vec::BitVec;

use indexmap::IndexMap;
use log::*;

use crate::{Analysis, AstSize, Dot, EClass, Extractor, Id, Language, Pattern, RecExpr, Searcher, UnionFind, Runner, Subst, Singleton, OpId, SparseNodeColors};

pub use crate::colors::{Color, ColorParents, ColorId};
use itertools::{assert_equal, Either, iproduct, Itertools};
use itertools::Either::Right;
use smallvec::smallvec;

#[cfg(test)]
use proptest::collection::vec;

pub type DenseNodeColors = BitVec;

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
#[derive(Clone)]
pub struct EGraph<L: Language, N: Analysis<L>> {
    /// The `Analysis` given when creating this `EGraph`.
    pub analysis: N,
    pub(crate) memo: HashMap<L, (Id, DenseNodeColors)>,
    unionfind: UnionFind,
    classes: SparseVec<EClass<L, N::Data>>,
    dirty_unions: Vec<Id>,
    repairs_since_rebuild: usize,
    pub(crate) classes_by_op: IndexMap<OpId, indexmap::IndexSet<Id>>,

    #[cfg(feature = "colored")]
    /// To be used as a mechanism of case splitting.
    /// Need to rebuild these, but can probably use original memo for that purpose.
    /// For each inner vector of union finds, if there is a union common to all of them then it will
    /// be applied on the main union find (case split mechanism). Not true for UnionFinds of size 1.
    colors: Vec<Color>,
    #[cfg(feature = "colored")]
    color_hierarchy: HashMap<Vec<ColorId>, ColorId>,
}

const MAX_COLORS: usize = 1000;

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
            color_hierarchy: Default::default(),
        }
    }

    /// Returns an iterator over the eclasses in the egraph.
    pub fn classes(&self) -> impl Iterator<Item=&EClass<L, N::Data>> {
        self.classes
            .iter()
            .filter_map(Option::as_ref)
            .map(AsRef::as_ref)
    }

    /// Returns an mutating iterator over the eclasses in the egraph.
    pub fn classes_mut(&mut self) -> impl Iterator<Item=&mut EClass<L, N::Data>> {
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
        Dot { egraph: self }
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
        self.add_expr_rec(expr.as_ref())
    }

    fn add_expr_rec(&mut self, expr: &[L]) -> Id {
        log::trace!("Adding expr {:?}", expr);
        let e = expr.last().unwrap().clone().map_children(|i| {
            let child = &expr[..usize::from(i) + 1];
            self.add_expr_rec(child)
        });
        let id = self.add(e);
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
        self.inner_lookup(enode).filter(|(id, cs)| !cs.any()).map(|(id, _)| id)
    }

    fn inner_lookup(&self, enode: &mut L) -> Option<(Id, &DenseNodeColors)> {
        enode.update_children(|id| self.find(id));
        let memoed = self.memo.get(enode);
        memoed.map(|(id, cs)| (self.find(*id), cs))
    }

    /// Looks up a [`L`] from the [`EGraph`]. This works with equivalences defined in `color`.
    /// Updates the enode's children to the relevant node, otherwise undefined.
    /// returns eclass id, node colors, and whether the node was in the correct color.
    fn inner_colored_lookup(&self, c_id: &ColorId, mut_enode: &mut L) -> Option<(Id, &DenseNodeColors, bool)> {
        mut_enode.update_children(|id| self.find(id));
        let mut memoed = self.memo.get(mut_enode);
        let color = self.get_color(*c_id).unwrap();
        let mut temp = None;
        if memoed.is_some() {
            let (id, cs) = memoed.unwrap();
            let id = self.find(*id);
            if cs[c_id.0] || !cs.any() {
                return Some((id, cs, true));
            }
            temp = Some((id, cs, false, mut_enode.clone()));
        }
        let enode = mut_enode.clone();
        // We maybe found an edge, but not in the correct color.
        // Now go through all possibilities in the correct color and find one of the correct color.
        // If no one is in the correct color try to find one in the wrong color.
        let children_options = enode.children().iter()
            .map(|id| color.black_ids(*id)
                .map(|hs| Either::Left(hs.into_iter()))
                .unwrap_or_else(|| Either::Right(std::iter::once(id))));
        for combination in children_options.into_iter().multi_cartesian_product() {
            let mut it = combination.iter();
            mut_enode.update_children(|id| **it.next().unwrap());
            if let Some((id, cs)) = self.inner_lookup(mut_enode) {
                if cs[color.get_id().0] || !cs.any() {
                    return Some((id, cs, true));
                } else {
                    temp.get_or_insert((id, cs, false, mut_enode.clone()));
                }
            }
        }

        if let Some((id, cs, b, n)) = temp {
            *mut_enode = n;
            Some((id, cs, b))
        } else {
             None
        }
    }

    pub fn colored_lookup<B>(&self, color: &ColorId, mut enode: B) -> Option<Id>
        where
            B: BorrowMut<L>,
    {
        self.inner_colored_lookup(color, enode.borrow_mut())
            .filter(|(_, _, b)| *b).map(|(id, _, _)| id)
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
        let inner_res = self.inner_lookup(&mut enode);
        let colored = inner_res.as_ref().map(|(_, cs)| cs.any());
        // can't blacken an edge because it might dirty the black color
        if inner_res.is_some() && inner_res.as_ref().unwrap().1.not_any() {
            let (id, _) = inner_res.unwrap();
            id
        } else {
            let id = self.unionfind.make_set();
            if cfg!(feature = "colored") {
                for c in &mut self.colors {
                    c.add(id);
                }
            }
            log::trace!("  ...adding to {}", id);
            let class = Box::new(EClass {
                id,
                nodes: vec![(enode.clone(), vec![])],
                data: N::make(self, &enode),
                parents: Default::default(),
                colored_parents: Default::default(),
                dirty_colors: Default::default(),
            });

            // add this enode to the parent lists of its children
            enode.children().iter().copied().unique().for_each(|child| {
                let tup = (enode.clone(), id);
                self[child].parents.push(tup);
            });
            assert_eq!(self.classes.len(), usize::from(id));
            self.classes.push(Some(class));
            assert!(self.memo.insert(enode, (id, Self::init_color_vec())).is_none());

            self.memo_parents_agree();

            N::modify(self, id);
            id
        }
    }

    pub(crate) fn init_color_vec() -> BitVec {
        let mut colors = BitVec::repeat(false, MAX_COLORS);
        colors
    }

    pub fn colored_add(&mut self, color: &ColorId, mut enode: L) -> Id {
        let inner_res = self.inner_colored_lookup(color, &mut enode);
        if inner_res.is_some() {
            let (id, cs, b) = inner_res.unwrap();
            if !b {
                // enode is colored now. Check if it is, and if it matters to these ops.
                self.memo.get_mut(&enode).iter_mut()
                    .for_each(|(id, cs)| cs.set(color.0, true));
                debug_assert!(self[id].nodes.iter().find(|x| x.0 == enode).is_some());
                debug_assert!(self[id].nodes.len() == 1);
                self[id].dirty_colors.entry(enode).or_insert_with(|| Self::init_color_vec()).set(color.0, true);
            }
            self.memo_parents_agree();
            id
        } else {
            enode.update_children(|id| self.find(id));

            // look for the node with the colored equivalence relation
            let id = self.unionfind.make_set();
            if cfg!(feature = "colored") {
                for c in &mut self.colors {
                    c.add(id);
                }
            }

            log::trace!("  ...colored ({}) adding to {}", color, id);
            let mut colors = Self::init_color_vec();
            colors.set(color.0, true);
            let class = Box::new(EClass {
                id,
                nodes: vec![(enode.clone(), vec![color.clone()])],
                data: N::make(self, &enode),
                parents: Default::default(),
                colored_parents: Default::default(),
                dirty_colors: Default::default()
            });


            // add this enode to the parent lists of its children
            enode.children().iter().copied().unique().for_each(|child| {
                let tup = (enode.clone(), id, colors.clone());
                debug_assert!(colors.iter_ones().all(|c| self.colors().len() > c));
                self[child].colored_parents.push(tup);
            });

            assert_eq!(self.classes.len(), usize::from(id));
            self.classes.push(Some(class));
            assert!(self.memo.insert(enode, (id, colors)).is_none());
            self.memo_parents_agree();
            N::modify(self, id);
            id
        }
    }

    /// Checks whether two [`RecExpr`]s are equivalent.
    /// Returns a list of id where both expression are represented.
    /// In most cases, there will none or exactly one id.
    ///
    /// [`RecExpr`]: struct.RecExpr.html
    pub fn equivs(&self, expr1: &RecExpr<L>, expr2: &RecExpr<L>) -> Vec<Id> {
        let matches1 = Pattern::from(expr1.as_ref()).search(self);
        trace!("Matches1: {:?}", matches1);

        let matches2 = Pattern::from(expr2.as_ref()).search(self);
        trace!("Matches2: {:?}", matches2);

        let mut equiv_eclasses = Vec::new();

        for m1 in &matches1 {
            for m2 in &matches2 {
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

        self.memo_parents_agree();
        let (to, from) = self.unionfind.union(id1, id2);
        let changed = to != from;
        self.memo_parents_agree();
        if cfg!(feature = "colored") {
            // warn!("union: {} {}", to, from);
            for color in self.colors.iter_mut() {
                color.black_union(to, from);
            }
        }
        debug_assert_eq!(to, self.find(id1));
        debug_assert_eq!(to, self.find(id2));
        self.memo_parents_agree();
        if changed {
            self.dirty_unions.push(to);

            // update the classes data structure
            let from_class = self.classes[usize::from(from)].take().unwrap();
            let to_class = self.classes[usize::from(to)].as_mut().unwrap();

            self.analysis.merge(&mut to_class.data, from_class.data);
            concat(&mut to_class.nodes, from_class.nodes);
            concat(&mut to_class.parents, from_class.parents);
            concat(&mut to_class.colored_parents, from_class.colored_parents);
            from_class.dirty_colors.into_iter().for_each(|(k, v)|
                to_class.dirty_colors.entry(k).or_default().extend(v));
            N::modify(self, to);
            self.memo_parents_agree();
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
        self.memo_parents_agree();
        let union = self.union_impl(id1, id2);
        if union.1 && cfg!(feature = "upward-merging") {
            // let merged = self.process_unions().iter()
            //     // .filter(|&tup| tup.2.is_none())
            //     .map(|(id1, id2, color)| (*id1, *id2)).collect_vec();
            let merged = self.process_unions();
            if cfg!(feature = "colored") {
                if !self.colors.is_empty() {
                    let to_remove = self.process_colored_unions(merged);
                    assert!(false, "Need to clean colors returned by process_colored_unions");
                }
            }
        }
        self.memo_nodes_agree();
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
    fn rebuild_classes(&mut self, to_remove: HashMap<L, SparseNodeColors>) -> usize {
        // for (color, nodes) in &to_remove {
        //     for n in nodes.iter() {
        //         debug_assert!(self.memo[n].1.any(), "Node {:?} should have the color {} being removed", n, color);
        //         self.memo[n].1.set(color.0, false);
        //         if self.memo[n].1.not_any() {
        //             todo!("remove from everywhere");
        //         } else {
        //             let id = self.memo[n].0;
        //             self[id].colored_parents.iter_mut()
        //                 .filter(|(x, y, z)| x == n)
        //                 .for_each(|(x, y, z)| z.set(color.0, false));
        //         }
        //     }
        // }
        self.memo_parents_agree();
        let mut classes_by_op = std::mem::take(&mut self.classes_by_op);
        classes_by_op.values_mut().for_each(|ids| ids.clear());

        let mut trimmed = 0;

        let uf = &self.unionfind;
        for class in self.classes.iter_mut().filter_map(Option::as_mut) {
            let mut todo: HashMap<L, DenseNodeColors> = {
                let old: HashMap<L, DenseNodeColors> = std::mem::take(&mut class.dirty_colors);
                old.into_iter()
                    .map(|(mut k, v)| {
                        k.update_children(|id| uf.find(id));
                        (k, v)
                    })
                    .collect()
            };


            let old_len = class.len();
            class
                .nodes
                .iter_mut()
                .for_each(|(n, cs)| n.update_children(|id| uf.find(id)));

            // Prevent comparing colors. Black should be first for better dirty color application.
            class.nodes.sort_unstable_by_key(|(n, cs)| (n.clone(), !cs.is_empty()));

            debug_assert!(todo.iter().all(|(n, _)| class.nodes.iter().any(|(n2, _)| n == n2)));
            // Hack: Using dedup_by to update colors on the remaining node. This should work because
            //       the Rust documentation states which element (a) in the dedup_by is removed.
            class.nodes.dedup_by(|a, b| {
                a.0 == b.0 && {
                    // Anything done here should have been already done on parents (and memo).
                    if a.1.is_empty() || b.1.is_empty() {
                        b.1.clear();
                    } else {
                        // It is ok to take because we are dudping a.
                        for x in std::mem::take(&mut a.1) {
                            if !b.1.contains(&x) {
                                b.1.push(x);
                            }
                        }
                    }
                    true
                }
            });
            // There might be unused colors in it, use them.
            // TODO: make sure that a class will not be empty once we remove edges by color.
            debug_assert!(!class.nodes.is_empty());
            class.nodes.iter_mut().for_each(|(n, cs)| {
               if let Some(colors) = todo.remove(n) {
                   if !cs.is_empty() {
                       for x in colors.iter_ones().map(|x| ColorId(x)) {
                           if !cs.contains(&x) {
                               cs.push(x);
                           }
                       }
                   }
               }
            });
            debug_assert!(todo.is_empty(), "All color changes should have been applied");
            Self::check_colored_edges(&self.memo, class);

            trimmed += old_len - class.nodes.len();

            // TODO this is the slow version, could take advantage of sortedness
            // maybe
            let mut add = |n: &L| {
                    classes_by_op
                    .entry(n.op_id())
                    .or_default()
                    .insert(class.id)
            };

            // we can go through the ops in order to dedup them, becaue we
            // just sorted them
            if class.nodes.len() > 0 {
                let (first, colors) = &class.nodes[0];
                let mut op_id = first.op_id();
                add(&first);
                for (n, colors) in &class.nodes[1..] {
                    if op_id != n.op_id() {
                        add(n);
                        op_id = n.op_id();
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        for ids in classes_by_op.values_mut() {
            let unique: indexmap::IndexSet<Id> = ids.iter().copied().collect();
            assert_eq!(ids.len(), unique.len());
        }

        self.classes_by_op = classes_by_op;
        self.memo_parents_agree();
        trimmed
    }

    fn check_colored_edges(memo: &HashMap<L, (Id, DenseNodeColors)>, class: &EClass<L, N::Data>) {
        if cfg!(debug_assertions) {
            fn into_dense(cs: &SparseNodeColors) -> DenseNodeColors {
                let mut dense = DenseNodeColors::repeat(false, MAX_COLORS);
                for c in cs.iter() {
                    dense.set(c.0, true);
                }
                dense
            }
            for (n, cs) in class.nodes.iter() {
                let dense = &memo.get(n).unwrap().1;
                if into_dense(cs).bitxor(dense).any() {
                    warn!("class nodes len {:?}", class.nodes.len());
                    assert!(false, "Dense and sparse colors (node {:?}; sparse {}; dense {}) don't \
                    match in class: \n{:?}", n, cs.iter().map(|c_id| c_id.0).join(", "), dense.iter_ones().join(", "), class)

                }
            }
        }
    }

    #[inline(never)]
    fn check_memo(&self) -> bool {
        let mut test_memo = IndexMap::new();

        for (id, class) in self.classes.iter().enumerate() {
            let id = Id::from(id);
            let class = match class.as_ref() {
                Some(class) => class,
                None => continue,
            };
            assert_eq!(class.id, id);
            for (node, cs) in &class.nodes {
                if let Some(old) = test_memo.insert(node, id) {
                    assert_eq!(
                        self.find(old),
                        self.find(id),
                        "Found unexpected equivalence for {:?}\n{:?}\nvs\n{:?} (colored: {})",
                        node,
                        self[self.find(id)].nodes,
                        self[self.find(old)].nodes,
                        self.memo[&node].1.any(),
                    );
                }
            }
        }

        for (n, e) in test_memo {
            assert_eq!(e, self.find(e));
            assert_eq!(
                Some(e),
                self.memo.get(n).map(|(id, cs)| self.find(*id)),
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
                let mut parents = std::mem::take(&mut self[id].parents)
                    .into_iter().map(|(n, e)| {
                    self.memo.remove(&n);
                    (n, e)
                }).collect_vec();
                parents.iter_mut().for_each(|(n, id)| {
                    n.update_children(|child| self.find(child));
                    *id = self.find(*id);
                });
                // We are using cs.any() to make sure black edges are first.
                parents.sort_unstable();
                parents.dedup_by(|(n1, e1), (n2, e2)| {
                    n1 == n2 && {
                        to_union.push((*e1, *e2));
                        true
                        // Dedup removed 1 and keeps 2. Update 2's colors.
                        // if cs1.not_any() && cs2.not_any() {
                        //     Self::update_memoed(cs1, cs2);
                        //     to_union.push((*e1, *e2, None));
                        //     true
                        // } else if cs1.not_any() || cs2.not_any() {
                        //
                        //     for c in cs2.iter_ones().chain(cs1.iter_ones()) {
                        //         to_union.push((*e1, *e2, Some(ColorId(c))));
                        //     }
                        //     if cs1.not_any() {
                        //         *e2 = *e1;
                        //     }
                        //     cs2.set_elements(0);
                        //     true
                        // } else {
                        //     false
                        // }
                    }
                });

                let empty_colors = Self::init_color_vec();
                for (n, e) in parents.iter_mut() {
                    // It is possible we don't have colors because parent was updated from different
                    // class.
                    let temp = Self::update_memo_from_parent(&mut self.memo, n, e, &empty_colors);
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

                // if let Some(c_id) = color_opt {
                //     let (to, did_something) = self.colored_union(c_id, id1, id2);
                //     // We will deal with the dirty unions of the color later.
                // } else {
                //     let (to, did_something) = self.union_impl(id1, id2);
                //     if did_something {
                //         self.dirty_unions.push(to);
                //     }
                // }
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


    pub fn update_memo_from_parent(memo: &mut HashMap<L, (Id, DenseNodeColors)>,
                                   n: &L, e: &Id, cs: &DenseNodeColors) -> Option<(Id, Id)> {
        // let mut accessor = |x: (L, Id, DenseNodeColors)| {
        //     let (n, id, cs) = x;
        //     let old = memo.insert(n.clone(), (id, cs));
        //     old.map(|(id, cs)| (id, cs, memo.get_mut(&n).map(|(x, y)| y).unwrap()))
        // };
        // Self::update_from_parent_access_fn(&mut accessor, n, e, cs)
        if let Some((old, mut old_cs)) = memo.insert(n.clone(), (*e, cs.clone())) {
            for (_, new_cs) in memo.get_mut(n).iter_mut() {
                if (old_cs.not_any()) || cs.not_any() {
                    new_cs.set_elements(0);
                    return Some((old, *e));
                } else {
                    new_cs.bitor_assign(&old_cs);
                }
            }
        }
        None
    }

    /// Updated cs2 to contain all colors of cs1 or black if needed.
    pub fn update_memoed(cs1: &DenseNodeColors, cs2: &mut DenseNodeColors) {
        if cs1.not_any() || cs2.not_any() {
            cs2.set_elements(0);
        } else {
            // Sparse will have the same merge during rebuild_classes
            // TODO: will it though? Especially when this is a colored merge, but I think it doesn't
            //      matter in colroed merge.
            cs2.bitor_assign(&*cs1);
        }
    }

    /// Updated cs2 to contain all colors of cs1 or black if needed.
    pub fn update_optional_memoed(cs1: &mut Option<DenseNodeColors>, cs2: &mut Option<DenseNodeColors>) {
        if cs1.is_none() || cs2.is_none() {
            *cs2 = None;
        } else {
            Self::update_memoed(&cs1.as_ref().unwrap(), &mut cs2.as_mut().unwrap());
        }
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
        self.memo_nodes_agree();
        self.memo_parents_agree();

        let merged = self.process_unions();

        self.memo_parents_agree();
        self.memo_black_canonized();

        let to_remove = self.process_colored_unions(merged);
        self.memo_parents_agree();
        let n_unions = std::mem::take(&mut self.repairs_since_rebuild);
        let trimmed_nodes = self.rebuild_classes(to_remove);
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

        // debug_assert!(self.check_memo());
        n_unions
    }

    pub(crate) fn canonize(&self, e: &L) -> L {
        e.clone().map_children(|e| self.find(e))
    }

    fn memo_black_canonized(&self) {
        debug_assert!(self.memo.iter().all(|(n, cs)| cs.1.any() || n == &self.canonize(n)));
    }

    fn memo_all_canonized(&self) {
        if cfg!(debug_assertions) {
            for (n, (id, cs)) in self.memo.iter() {
                debug_assert!(n == &self.canonize(n), "{:?} with colors: {:?} @ {}", n, cs.iter_ones().collect_vec(), id);
            }
        }
    }

    fn memo_parents_agree(&self) {
        // if cfg!(debug_assertions) {
        //     for (n, (id, cs)) in self.memo.iter() {
        //         let (n, cs) = if n != &self.canonize(n) {
        //             let mut result_cs = cs.clone();
        //             let n = self.canonize(n);
        //             for (n2, (_, cs)) in self.memo.iter() {
        //                 if &self.canonize(n2) == &n {
        //                     result_cs.bitor_assign(&*cs);
        //                 }
        //             }
        //             (n, result_cs)
        //         } else {
        //             (n.clone(), cs.clone())
        //         };
        //         if n.len() != 0 {
        //             let mut bitv = n.children().iter().flat_map(|c| {
        //                 self[*c].parents.iter()
        //                     .map(|(n1, cs, _)| (self.canonize(n1), cs))
        //                     .filter(|(n1, cs)| { &n == n1 })
        //                     .map(|t| t.1.clone())
        //             }).fold1(|a, b| a.bitor(b)).unwrap();
        //             let empty = Self::init_color_vec();
        //             let dirty = self[*id].dirty_colors.get(&n).unwrap_or(&empty);
        //             bitv.bitor_assign(dirty);
        //             let res = bitv.clone().bitxor(&cs).not_any();
        //             assert!(res);
        //         }
        //     }
        // }
    }

    fn memo_nodes_agree(&self) {
        debug_assert!(self.memo.iter().all(|(n, (id, cs))| {
            // Find n in nodes of id
            // assert!(&self.canonize(n) == n);
            let (n2, cs2) = self[self.find(*id)].iter().find(|(n2, cs2)| n == n2).unwrap();
            let temp_hashset = Default::default();
            let dirty = self[*id].dirty_colors.get(n).unwrap_or(&temp_hashset);
            let mut new_dense = Self::init_color_vec();
            for c in cs2.iter() {
                new_dense.set(c.0, true);
            }
            new_dense.bitor_assign(dirty);
            new_dense.bitxor(cs).not_any()
        }));
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
    pub fn subst_agrees(&self, s1: &crate::Subst, s2: &crate::Subst) -> bool {
        s1.vec.iter().all(|(v, i1)| s2.get(*v)
            .map(|i2| {
                let s1_ids = self.gather_all_ids(s1, i1);
                let s2_ids = self.gather_all_ids(s2, i2);
                i1 == i2 || !s1_ids.unwrap_or(&HashSet::default()).is_disjoint(&s2_ids.unwrap_or(&HashSet::default()))
            }).unwrap_or(false))
    }

    fn gather_all_ids(&self, subs: &Subst, id: &Id) -> Option<&HashSet<Id>> {
        let s1_ids = subs.color.map(|c_id| self.colors()[c_id.0].black_ids(*id)).flatten();
        s1_ids
    }
}

// ***  Colored Implementation  ***
#[cfg(feature = "colored")]
impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    pub fn create_sub_color(&mut self, color: ColorId) -> ColorId {
        let new_color_id = self.create_color();
        let new_color = self.colors[color.0].new_child(new_color_id);
        self.colors.push(new_color);
        new_color_id
    }

    fn process_colored_unions(&mut self, mut black_merged: Vec<Id>) -> HashMap<L, SparseNodeColors> {
        let mut res: HashMap<L, SparseNodeColors> = Default::default();
        let mut next = black_merged.iter().unique().copied().collect_vec();
        let mut todo = vec![];
        while !next.is_empty() {
            todo = std::mem::take(&mut next);
            let mut to_union = vec![];
            for id in todo.iter().unique().copied() {
                let mut ps = std::mem::take(&mut self[id].colored_parents);
                for (n, e, cs) in ps.iter_mut() {
                    let old = self.memo.remove(n);
                    debug_assert!(old.iter().all(|(id, old_cs)| id == e && cs.any() == old_cs.any()));
                    n.update_children(|c| self.find(c));
                }
                // I need to dedup here. I should assert that there is only one edge in a colored class.
                // If it is true and there is always only one edge I can union the classes on all their
                // colors and union the color set into a single class. Also I might need to union with a
                // black edge from memo after canonization so be aware of that when inserting edges back.
                ps.sort_unstable_by_key(|(n, _, _)| n.clone());
                ps.dedup_by(|(n1, e1, cs1), (n2, e2, cs2)| {
                    assert!(cs1.any() && cs2.any());
                    n1 == n2 && {
                        assert!(1 == self[*e1].len() && 1 == self[*e2].len(),
                                "Colored eclasses should always have one edge (or be black)");
                        for c in cs1.iter_ones() {
                            debug_assert!(self.colors.len() > c);
                            to_union.push((Some(ColorId(c)), *e1, *e2));
                        }
                        cs2.bitor_assign(std::mem::take(cs1));
                        true
                    }
                });
                for (n, e, cs) in ps.iter() {
                    // add node back into memo and if old exists then union
                    let old = self.memo.insert(n.clone(), (*e, cs.clone()));
                    if let Some((old_e, old_cs)) = old {
                        if cs.any() {
                            for c in old_cs.iter_ones() {
                                debug_assert!(self.colors.len() > c);
                                to_union.push((Some(ColorId(c)), *e, old_e));
                            }
                            self.memo.get_mut(&n).unwrap().1.bitor_assign(&old_cs);
                            self[*e].dirty_colors.entry(n.clone()).and_modify(|d_cs| d_cs.bitor_assign(&old_cs)).or_insert(old_cs);
                        } else {
                            assert!(self[*e].parents.is_empty());
                            to_union.push((None, *e, old_e));
                            // Clear colors because edge should be black now.
                            self.memo.get_mut(n).unwrap().1.set_elements(0);
                        }
                    }
                }
                self[id].colored_parents = ps;
                for (c, e1, e2) in to_union.drain(..) {
                    if let Some(c) = c {
                        self.colored_union(c, e1, e2);
                    } else {
                        let (res, did_something) = self.union_impl(e1, e2);
                        // Because there are no parents, we don't need black congruence
                        // closure. We just need to update all relevant data structures locally.
                        // We will handle dirty colored unions in the next step, so it is not needed
                        // here (and union will handle that update).
                        if did_something {
                            assert_eq!(self.dirty_unions.len(), 0,
                                       "Should have been empty after process unions");
                            next.push(res);
                            black_merged.push(res);
                            // no parents so no **uncolored** node changes in memo.
                        }
                    }
                }
            }
        }

        self.memo_all_canonized();

        let dirty_color_map: HashMap<L, DenseNodeColors> = {
            let mut res: HashMap<L, DenseNodeColors> = Default::default();
            for c in self.classes() {
                for (n, cs) in &c.dirty_colors {
                    let mut cur = res.entry(self.canonize(n)).or_insert_with(|| DenseNodeColors::repeat(false, MAX_COLORS));
                    cur.bitor_assign(cs);
                }
            }
            res
        };
        let mut classes = std::mem::take(&mut self.classes);
        for c in classes.iter_mut().filter_map(|c| c.as_mut()) {
            for (n, e, cs) in c.colored_parents.iter_mut() {
                let dirty_cs = dirty_color_map.get(&self.canonize(n));
                if let Some(dirty_cs) = dirty_cs {
                    if cs.any() {
                        cs.bitor_assign(dirty_cs);
                    }
                }
            }
        }
        self.classes = classes;

        let mut colors = std::mem::take(&mut self.colors);
        for c in &mut colors {
            let to_remove = c.cong_closure(self, &black_merged);
            debug_assert!(to_remove.iter().all(|(n1, n2)| n1 == &self.canonize(n1) && n2 == &self.canonize(n2)));
            debug_assert!(to_remove.iter().all(|(n1, n2)| self.memo.contains_key(&n1) && self.memo.contains_key(&n2)));
            let mut mapping: HashMap<L, Id> = HashMap::default();
            let mut uf: UnionFind = Default::default();
            for (n1, n2) in to_remove {
                if !mapping.contains_key(&n1) {
                    mapping.insert(n1.clone(), uf.make_set());
                }
                if !mapping.contains_key(&n2) {
                    mapping.insert(n2.clone(), uf.make_set());
                }
                uf.union(mapping[&n1], mapping[&n2]);
            }
            debug_assert!(mapping.iter().all(|(n, i)| &self.canonize(n) == n && self.memo.contains_key(&n)));
            let mut groups = mapping.keys().into_iter()
                .map(|n| (uf.find(mapping[n]), n)).into_group_map();
            debug_assert!(groups.iter().all(|(_, ns)| ns.iter().all(|n| self.memo.contains_key(n))));
            for (_, vs) in groups.iter_mut() {
                let max: usize = vs.iter().position_max_by_key(|v| self.memo[**v].1.count_ones()).unwrap();
                for (i, v) in vs.into_iter().enumerate() {
                    // skip max
                    if i == max {
                        continue;
                    }
                    // get v's entry, use default if it doesn't exist.
                    // Add c to the entry.
                    res.entry(v.clone()).or_default().push(c.get_id());
                }
            }
        }
        self.colors = colors;
        res
    }

    pub fn create_color(&mut self) -> ColorId {
        self.colors.push(Color::new(&self.unionfind, ColorId::from(self.colors.len())));
        let res = ColorId::from(self.colors.len() - 1);
        self.color_hierarchy.insert(self.colors.last().unwrap().assumptions().clone(), res);
        res
    }

    /// Create a new color which is based on given colors. This should be used only if the new color
    /// has no assumptions of it's own (i.e. it is only a combination of existing assumptions).
    pub fn create_combined_color(&mut self, colors: Vec<ColorId>) -> ColorId {
        // First check if assumptions exist
        let assumptions = colors.iter()
            .flat_map(|c_id| self.colors[c_id.0].assumptions())
            .sorted().dedup()
            .copied().collect_vec();
        if self.color_hierarchy.contains_key(&assumptions) {
            println!("Skipping on color assumptions:");
            for a in &assumptions {
                println!("{}", a);
            }
            return *self.color_hierarchy.get(&assumptions).unwrap();
        }

        let new_id = ColorId::from(self.colors.len());
        assert!(colors.len() > 1);
        let mut needed_colors = {
            colors.iter().dropping(1).map(|c_id| std::mem::take(&mut self.colors[c_id.0]))
        }.collect_vec();
        let new_color = self.colors[colors[0].0].merge_ufs(needed_colors.iter_mut().collect_vec(), new_id, false);
        colors.iter().dropping(1).enumerate().for_each(|(i, c_id)| self.colors[c_id.0] = std::mem::take(needed_colors.get_mut(i).unwrap()));
        self.color_hierarchy.insert(new_color.assumptions().clone(), new_id);
        self.colors.push(new_color);
        new_id
    }

    pub fn colored_union(&mut self, color: ColorId, id1: Id, id2: Id) -> (Id, bool) {
        self.memo_parents_agree();
        let c = self.colors.get_mut(color.0).unwrap();
        let union = c.colored_union(id1, id2);
        if union.1 {
            for child in c.children().iter().copied().collect_vec() {
                self.colored_union(child, id1, id2);
            }
        }
        self.memo_parents_agree();
        union
    }

    pub fn colored_find(&self, color: ColorId, id: Id) -> Id {
        self.colors[usize::from(color)].find(id)
    }

    pub fn colors(&self) -> &[Color] {
        &self.colors
    }

    pub fn get_color(&self, color: ColorId) -> Option<&Color> {
        // TODO: when colors are deletable, check if color exists
        Some(&self.colors[usize::from(color)])
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
    use std::collections::HashSet;
    use crate::*;
    use std::str::FromStr;
    use log::*;
    use crate::rewrite::*;
    use itertools::Itertools;
    use crate::util::*;

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

        assert_eq!(egraph.colored_find(color, fx), egraph.colored_find(color, fy));
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
        let c = &egraph.colors()[color.0];
        c.assert_black_ids(&egraph);
        egraph.colored_union(color, w, x);
        let c = &egraph.colors()[color.0];
        c.assert_black_ids(&egraph);
        egraph.union(w, y);
        let c = &egraph.colors()[color.0];
        c.assert_black_ids(&egraph);
        egraph.rebuild();

        assert_eq!(egraph.colored_find(color, fx), egraph.colored_find(color, fy));
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
        assert_eq!(egraph.colors[c.0].black_ids(to).map(|x| x.len()), Some(6));

        egraph.union(ex5, ex6);
        egraph.union(ex1, ex5);
        println!("{:#?}", egraph.colors[c.0].black_ids(to));
        assert_eq!(egraph.colors[c.0].black_ids(to).map(|x| x.len()), Some(4));
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

        egraph.colored_union(c1, ex1, ex2);
        assert_eq!(egraph.colored_find(c3, ex1), egraph.colored_find(c3, ex2));
        egraph.colored_union(c1, ex3, ex4);
        egraph.colored_union(c2, ex1, ex3);
        assert_eq!(egraph.colored_find(c3, ex1), egraph.colored_find(c3, ex3));
        egraph.colored_union(c2, ex5, ex6);
        assert_eq!(egraph.colored_find(c3, ex5), egraph.colored_find(c3, ex6));
        assert_eq!(egraph.colored_find(c3, ex3), egraph.colored_find(c3, ex4));
        let (to, _) = egraph.colored_union(c2, ex1, ex5);
        egraph.rebuild();
        assert_eq!(egraph.colors[c3.0].black_ids(to).map(|x| x.len()), Some(6));
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
        let color3 = egraph.create_combined_color(vec![color1, color2]);

        egraph.colored_union(color1, w, x);
        egraph.colored_union(color2, w, y);
        assert_ne!(egraph.colored_find(color3, fx), egraph.colored_find(color3, fy));
        egraph.rebuild();
        assert_eq!(egraph.colored_find(color3, fx), egraph.colored_find(color3, fy));
    }

    #[test]
    fn color_new_child_unions() {
        use SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));

        let color = egraph.create_color();
        let child = egraph.create_sub_color(color);

        egraph.colored_union(color, y, x);
        assert_eq!(egraph.colored_find(child, x), egraph.colored_find(child, y));
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
        let ex1 = egraph.add_expr(&"(append (take i (cons x nil)) (drop i (cons x nil)))".parse().unwrap());
        let ex2 = egraph.add_expr(&"(append (take i (cons y (cons x nil))) (drop i (cons y (cons x nil))))".parse().unwrap());
        info!("Starting first rebuild");
        egraph.rebuild();
        let bad_rws = rewrite!("rule10"; "(take (succ ?x7) (cons ?y8 ?z))" <=> "(cons ?y8 (take ?x7 ?z))");
        info!("Done first rebuild");
        let mut rules = vec![
            rewrite!("rule2"; "(append nil ?x)" => "?x"),
            rewrite!("rule5"; "(drop ?x3 nil)" => "nil"),
            rewrite!("rule6"; "(drop zero ?x)" => "?x"),
            rewrite!("rule7"; "(drop (succ ?x4) (cons ?y5 ?z))" => "(drop ?x4 ?z)"),
            rewrite!("rule8"; "(take ?x3 nil)" => "nil"),
            rewrite!("rule9"; "(take zero ?x)" => "nil"), ];
        // rules.extend(rewrite!("rule0"; "(leq ?__x0 ?__y1)" <=> "(or (= ?__x0 ?__y1) (less ?__x0 ?__y1))"));
        rules.extend(rewrite!("rule3"; "(append (cons ?x2 ?y) ?z)" <=> "(cons ?x2 (append ?y ?z))"));
        rules.extend(bad_rws.clone());

        egraph = Runner::default().with_iter_limit(8).with_node_limit(400000).with_egraph(egraph).run(&rules).egraph;
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
        egraph = Runner::default().with_iter_limit(8).with_node_limit(400000).with_egraph(egraph).run(&rules).egraph;
        egraph.rebuild();

        for x in egraph.colors.iter() {
            warn!("{}", x);
            x.assert_black_ids(&egraph);
        }

        egraph.dot().to_dot("graph.dot");

        let take_i_nil = egraph.add_expr(&"(take i nil)".parse().unwrap());
        warn!("take i nil - {} - {}", take_i_nil, egraph.colored_find(color_z, take_i_nil));
        let take_i_consx = egraph.add_expr(&"(take i (cons x nil))".parse().unwrap());
        warn!("take i (cons x nil) - {} - {}", take_i_consx, egraph.colored_find(color_z, take_i_consx));
        let drop_i_nil = egraph.add_expr(&"(drop i nil)".parse().unwrap());
        warn!("drop i nil - {} - {}", drop_i_nil, egraph.colored_find(color_z, drop_i_nil));
        let drop_i_consx = egraph.add_expr(&"(drop i (cons x nil))".parse().unwrap());
        warn!("drop i (cons x nil) - {} - {}", drop_i_consx, egraph.colored_find(color_z, drop_i_consx));

        assert_eq!(egraph.colored_find(color_z, consx), egraph.colored_find(color_z, ex1));
        assert_eq!(egraph.colored_find(color_z, consxy), egraph.colored_find(color_z, ex2));
        assert_eq!(egraph.colored_find(color_s_p, consx), egraph.colored_find(color_s_p,ex1));
        assert_eq!(egraph.colored_find(color_s_z, consxy), egraph.colored_find(color_s_z,ex2));
    }

    #[test]
    fn colored_plus_succ() {
        use crate::SymbolLang as S;

        crate::init_logger();
        let mut egraph = EGraph::<S, ()>::default();

        egraph.rebuild();
        let mut rules = vec![
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
        egraph = Runner::default().with_iter_limit(8).with_node_limit(400000).with_egraph(egraph).run(&rules).egraph;
        egraph.rebuild();
        egraph.dot().to_dot("graph.dot");


        assert_eq!(egraph.colored_find(color_z, init), egraph.colored_find(color_z, res_z), "Black ids for color_z:\n  {}", egraph.colors[color_z.0].to_string());
        println!("Nodes from 7 with colors {}", egraph[Id(7)].nodes.iter().map(|x| format!("node {:?} - cs [{}]", x.0, x.1.iter().join(", "))).join(", "));
        println!("Matches from rule2:\n  {}", rules[0].search(&egraph).iter().map(|x| format!("{}", x)).join(", "));
        rules[0].search(&egraph).iter().for_each(|x| {
            println!("{}", x);
        });
        assert_eq!(egraph.colored_find(color_succ, init), egraph.colored_find(color_succ, res_succ_z), "Black ids for color_succ:\n  {}", egraph.colors[color_succ.0].to_string());
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
        let mut rules = vec![
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
        egraph = Runner::default().with_iter_limit(8).with_node_limit(400000).with_egraph(egraph).run(&rules).egraph;
        egraph.rebuild();

        let t_id = egraph.add_expr(&"true".parse().unwrap());
        let f_id = egraph.add_expr(&"false".parse().unwrap());

        let color_tf = egraph.create_color();
        egraph.colored_union(color_tf, t_id, f_id);

        egraph.rebuild();
        egraph = Runner::default().with_iter_limit(8).with_node_limit(400000).with_egraph(egraph).run(&rules).egraph;
        egraph.rebuild();

        assert_eq!(egraph.colored_find(color_tf, ids[0]), egraph.colored_find(color_tf, ids[2]));
        assert_eq!(egraph.colored_find(color_tf, ids[4]), egraph.colored_find(color_tf, ids[5]));
    }
}
