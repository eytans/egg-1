use std::{
    borrow::BorrowMut,
    fmt::{self, Debug},
};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::Alignment::Left;
use std::iter::Peekable;
use std::ops::{BitOr, BitOrAssign, BitXor, BitXorAssign};
use std::thread::current;
use std::vec::IntoIter;
use bitvec::macros::internal::funty::Fundamental;
use bitvec::vec::BitVec;
use std::iter::FromIterator;

use indexmap::{IndexMap, IndexSet};
use invariants::{dassert, iassert, tassert, wassert};
use log::*;

use crate::{Analysis, AstSize, Dot, EClass, Extractor, Id, Language, Pattern, RecExpr, Searcher, UnionFind, Runner, Subst, Singleton, OpId};

pub use crate::colors::{Color, ColorParents, ColorId};
use itertools::{assert_equal, Either, iproduct, Itertools};
use itertools::Either::Right;
use multimap::MultiMap;
use smallvec::smallvec;

#[cfg(test)]
use proptest::collection::vec;

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
    pub(crate) memo: IndexMap<L, Id>,
    unionfind: UnionFind,
    classes: SparseVec<EClass<L, N::Data>>,
    dirty_unions: Vec<Id>,
    repairs_since_rebuild: usize,
    pub(crate) classes_by_op: IndexMap<OpId, IndexSet<Id>>,

    #[cfg(feature = "colored")]
    /// To be used as a mechanism for case splitting.
    /// Need to rebuild these, but can probably use original memo for that purpose.
    /// For each inner vector of union finds, if there is a union common to all of them then it will
    /// be applied on the main union find (case split mechanism). Not true for UnionFinds of size 1.
    colors: Vec<Option<Color>>,
    #[cfg(feature = "colored")]
    pub(crate) colored_memo: IndexMap<L, IndexMap<ColorId, Id>>,
    #[cfg(feature = "colored")]
    pub(crate) colored_equivalences: IndexMap<Id, IndexSet<(ColorId, Id)>>,
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
            colored_memo: Default::default(),
            colored_equivalences: Default::default(),
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
        Dot { egraph: self, color: None , print_color: "blue".to_string() }
    }

    pub fn colored_dot(&self, color: ColorId) -> Dot<L, N> {
        Dot { egraph: self, color: Some(color), print_color: "blue".to_string() }
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
            self.colored_memo.get(enode).map(|colors|
                colors.get(&color).map(|id| self.find(*id))).flatten()
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

    pub(crate) fn init_color_vec() -> BitVec {
        let mut colors = BitVec::repeat(false, MAX_COLORS);
        colors
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

    pub fn colored_add(&mut self, color: ColorId, mut enode: L) -> Id {
        if let Some(id) = self.colored_lookup(color, &mut enode) {
            id
        } else {
            let id = self.inner_create_class(&mut enode, Some(color));
            self.get_color_mut(color).unwrap().black_colored_classes.insert(id, id);
            enode.children().iter().copied().unique().for_each(|child| {
                self[child].colored_parents.entry(color).or_default().push((enode.clone(), id));
            });

            let mut color_map = self.colored_memo.entry(enode).or_default();
            assert!(color_map.insert(color, id).is_none());
            N::modify(self, id);
            id
        }
    }

    fn inner_create_class(&mut self, mut enode: &mut L, color: Option<ColorId>) -> Id {
        let id = self.unionfind.make_set();
        if cfg!(feature = "colored") {
            for c in &mut self.colors {
                c.as_mut().unwrap().add(id);
            }
        }

        if let Some(c) = color {
            enode.update_children(|id| self.colored_find(c, id));
        } else {
            enode.update_children(|id| self.find(id));
        }

        log::trace!("  ...colored ({:?}) adding to {}", color, id);
        let class = Box::new(EClass {
            id,
            nodes: vec![enode.clone()],
            data: N::make(self, &enode),
            parents: Default::default(),
            colored_parents: Default::default(),
            color
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

        let (to, from) = self.unionfind.union(id1, id2);
        let changed = to != from;
        if cfg!(feature = "colored") {
            // warn!("union: {} {}", to, from);
            let todo = self.colors_mut().filter_map(|color|
                color.inner_black_union(to, from).2).collect_vec();
            for (id1, id2) in todo {
                self.union(id1, id2);
            }
        }
        tassert!(to == self.find(id1));
        tassert!(to == self.find(id2));
        if changed {
            if let Some(c) = self[to].color {
                iassert!(self.colored_find(c, to) == self.colored_find(c, from));
                let colored_to = self.colored_find(c ,to);
                self.get_color_mut(c).unwrap().black_colored_classes.insert(colored_to, to);
            } else {
                self.dirty_unions.push(to);
            }

            // update the classes data structure
            let from_class = self.classes[usize::from(from)].take().unwrap();
            let to_class = self.classes[usize::from(to)].as_mut().unwrap();
            debug_assert!(from_class.color == to_class.color);

            self.analysis.merge(&mut to_class.data, from_class.data);
            concat(&mut to_class.nodes, from_class.nodes);
            concat(&mut to_class.parents, from_class.parents);
            from_class.colored_parents.into_iter().for_each(|(k, v)|
                to_class.colored_parents.entry(k).or_default().extend(v));
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
            let merged = self.process_unions();
            if cfg!(feature = "colored") {
                if !self.colors.is_empty() {
                    let to_remove = self.process_colored_unions(merged);
                    assert!(false, "Need to clean colors returned by process_colored_unions");
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

        let mut classes = std::mem::take(&mut self.classes);
        for class in classes.iter_mut().filter_map(Option::as_mut) {
            let old_len = class.len();
            let uf = if let Some(c) = class.color {
                &self.get_color(c).unwrap().union_find
            } else {
                &self.unionfind
            };
            let mut n_updater: Box<dyn FnMut(&mut L) -> ()> = if let Some(c) = class.color {
                Box::new(|mut n: &mut L| {
                    n.update_children(|id| uf.find(id))
                })
            } else {
                Box::new(|mut n: &mut L| {
                    n.update_children(|id| uf.find(id))
                })
            };

            class
                .nodes
                .iter_mut()
                .for_each(n_updater);

            // Prevent comparing colors. Black should be first for better dirty color application.
            class.nodes.sort_unstable();
            if let Some(c) = class.color {
                class.nodes.dedup_by(|a, b| a == b
                    || self.colored_memo.get(a).map_or(false, |m| m.contains_key(&c)));
            } else {
                class.nodes.dedup();
            }
            // There might be unused colors in it, use them.
            // TODO: make sure that a class will not be empty once we remove edges by color.
            debug_assert!(!class.nodes.is_empty());

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
                let first = &class.nodes[0];
                let mut op_id = first.op_id();
                add(&first);
                for n in &class.nodes[1..] {
                    if op_id != n.op_id() {
                        add(n);
                        op_id = n.op_id();
                    }
                }
            }
        }
        self.classes = classes;

        #[cfg(debug_assertions)]
        for ids in classes_by_op.values_mut() {
            let unique: indexmap::IndexSet<Id> = ids.iter().copied().collect();
            assert_eq!(ids.len(), unique.len());
        }
        self.classes_by_op = classes_by_op;
        self.colored_equivalences.clear();
        let colors = std::mem::take(&mut self.colors);
        for c in colors.iter().filter_map(|x| x.as_ref()) {
            for (_, ids) in &c.union_map {
                dassert!(ids.len() > 1);
                for id in ids {
                    for id1 in ids {
                        if id1 == id {
                            continue;
                        }
                        self.colored_equivalences.entry(*id).or_default().insert((c.get_id(), *id1));
                    }
                }
            }
        }
        self.colors = colors;
        trimmed
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
                let mut parents = std::mem::take(&mut self[id].parents)
                    .into_iter().map(|(n, e)| {
                    self.memo.remove(&n);
                    (n, e)
                }).collect_vec();
                parents.iter_mut().for_each(|(n, id)| {
                    n.update_children(|child| self.find(child));
                    *id = self.find(*id);
                    debug_assert!(self[*id].color.is_none());
                });
                parents.sort_unstable();
                parents.dedup_by(|(n1, e1), (n2, e2)| {
                    n1 == n2 && {
                        to_union.push((*e1, *e2));
                        true
                    }
                });

                for (n, e) in parents.iter_mut() {
                    let temp = Self::update_memo_from_parent(&mut self.memo, n, e);
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


    pub fn update_memo_from_parent(memo: &mut IndexMap<L, Id>, n: &L, e: &Id) -> Option<(Id, Id)> {
        if let Some(old) = memo.insert(n.clone(), *e) {
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

        let merged = self.process_unions();

        self.memo_black_canonized();

        self.process_colored_unions(merged);
        let n_unions = std::mem::take(&mut self.repairs_since_rebuild);
        let trimmed_nodes = self.rebuild_classes();
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

    pub(crate) fn colored_update_node(&self, color: ColorId, e: &mut L) {
        e.update_children(|e| self.colored_find(color, e));
    }

    pub(crate) fn colored_canonize(&self, color: ColorId, e: &L) -> L {
        let mut res = e.clone();
        self.colored_update_node(color, &mut res);
        res
    }

    pub(crate) fn update_node(&self, e: &mut L) {
        e.update_children(|e| self.find(e));
    }

    pub(crate) fn canonize(&self, e: &L) -> L {
        let mut res = e.clone();
        self.update_node(&mut res);
        res
    }

    /// Reapply congruence closure for color.
    /// Returns which colors to remove from which edges.
    pub fn colored_cong_closure(&mut self, c_id: ColorId, black_merged: &[Id]) {
        // TODO: When we rebuild the colored_memo, we should point to the *black* representative id.
        self.get_color(c_id).unwrap().assert_black_ids(self);

        let mut to_union = vec![];
        // We need to build memo ahead of time because even a single merge might miss needed unions.
        // Need to do some merging and initial color deletion here because we are deleting duplicate
        // edges.
        let mut memo: IndexMap<L, Id> = {
            let mut v = self.memo.iter()
                .map(|(orig, e)| (self.colored_canonize(c_id, orig), self.find(*e)))
                .chain(self.colored_memo.iter()
                    .filter_map(|(x, map)|
                        map.get(&c_id).map(|c| (x.clone(), *c))))
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
                debug_assert!(self[id1].color == self[id2].color);
                debug_assert!(self[id1].color.unwrap() == c_id);
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
            let all_groups: IndexMap<Id, IndexSet<Id>> = self.get_color(c_id).unwrap().union_find.build_sets();
            for id in todo {
                // I need to build parents while aware what is a colored edge
                // Colored edges might be deleted, and they need to be updated in colored_memo if not
                let mut parents: Vec<(L, Id, bool, Option<Id>)> = vec![];
                for g in all_groups.get(&id).unwrap() {
                    for (p, id) in &self[*g].parents {
                        let canoned = self.colored_canonize(c_id, p);
                        memo.remove(&canoned);
                        // I need bool and option for sorting
                        parents.push((canoned, self.find(*id), true, None));
                    }
                    for (mut p, id) in self[*g].colored_parents.remove(&c_id).unwrap_or(vec![]) {
                        debug_assert!(self[id].color.unwrap() == c_id, "Color mismatch");
                        self.colored_memo.get_mut(&p).iter_mut().for_each(|x| {
                            x.remove(&c_id);
                        });
                        if self.colored_memo.get(&p).map_or(false, |x| x.is_empty()) {
                            self.colored_memo.remove(&p);
                        }
                        memo.remove(&p);
                        self.colored_update_node(c_id, &mut p);
                        parents.push((p, self.find(id), false, Some(*g)));
                    }
                }
                // TODO: we might be able to prevent parent recollection by memoization.
                parents.sort_unstable();
                parents.dedup_by(|(n1, e1, is_black_1, opt1),
                                  (n2, e2, is_black_2, opt2)| {
                    n1 == n2 && {
                        to_union.push((*e1, *e2));
                        true
                    }
                });

                for (n, e, is_black, orig_class) in parents {
                    if let Some(old) = memo.insert(n.clone(), e) {
                        to_union.push((old, e));
                    }
                    if !is_black {
                        let old = self.colored_memo.entry(n.clone()).or_default().insert(c_id, e);
                        if let Some(old) = old {
                            to_union.push((old, e));
                        }
                        let orig_class = orig_class.unwrap();
                        self[orig_class].colored_parents.entry(c_id).or_default().push((n, e));
                    }
                }
            }

            for (id1, id2) in to_union.drain(..) {
                self.colored_union(c_id, id1, id2);
            }
        }

        assert!(self.get_color(c_id).unwrap().dirty_unions.is_empty(), "Dirty unions should be empty {}", self.get_color(c_id).unwrap().dirty_unions.iter().join(", "));
        assert!(to_union.is_empty(), "to_union should be empty {}", to_union.iter().map(|x| format!("{}-{}", x.0, x.1)).join(", "));
        self.get_color(c_id).unwrap().assert_black_ids(self);
    }


    fn memo_black_canonized(&self) {
        debug_assert!(self.memo.keys().all(|n| self.memo.contains_key(&self.canonize(n))));
    }

    fn colored_memo_canonized(&self) {
        if cfg!(debug_assertions) {
            for (n, colors) in self.colored_memo.iter() {
                debug_assert!(!colors.is_empty());
                for (c, id) in colors {
                    let mut is_deleted: Option<bool> = None;
                    let deleted: fn(&EGraph<L, N>, n: &L, c: ColorId, &mut Option<bool>) -> bool = |egraph: &EGraph<L, N>, n: &L, c: ColorId, mut is_deleted: &mut Option<bool> | {
                        if let Some(is_deleted) = is_deleted.clone() {
                            return is_deleted;
                        }
                        let res = egraph.memo.iter().any(|(n1, e1)| {
                            egraph.colored_canonize(c, n) == egraph.colored_canonize(c, n1)
                        });
                        *is_deleted = Some(res);
                        res
                    };
                    tassert!({
                        self.colored_memo.contains_key(&self.colored_canonize(*c, n)) || deleted(self, n, *c, &mut is_deleted)
                    }, "Missing {:?} (orig: {:?}) in {} id (under color {})", self.colored_canonize(*c, n), n, id, c);
                    dassert!(((is_deleted.is_none() || !is_deleted.as_ref().unwrap()) && self.colored_memo[&self.colored_canonize(*c, n)].contains_key(c)) || deleted(self, n, *c, &mut is_deleted));
                    if n.children().len() > 0 && (is_deleted.is_none() || !*is_deleted.as_ref().unwrap()) {
                        dassert!(self.find(self.colored_memo[&self.colored_canonize(*c, n)][c]) == self.find(*id) || deleted(self, n, *c, &mut is_deleted), "Colored memo does not have correct id for {:?} in color {}. It is {} but should be {}", n, c, self.colored_memo[&self.colored_canonize(*c, n)][c], self.find(*id));
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
    }

    fn memo_classes_agree(&self) {
        debug_assert!(self.memo.iter().all(|(n, id)| self[self.find(*id)].color.is_none())
            && self.colored_memo.iter().all(|(_, cs)| cs.iter().all(|(c, id)|
                self[self.find(*id)].color == Some(*c))));
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
                i1 == i2 || !s1_ids.unwrap_or(&IndexSet::default()).is_disjoint(&s2_ids.unwrap_or(&IndexSet::default()))
            }).unwrap_or(false))
    }

    fn gather_all_ids(&self, subs: &Subst, id: &Id) -> Option<&IndexSet<Id>> {
        let s1_ids = subs.color.map(|c_id| self.get_color(c_id).unwrap().black_ids(*id)).flatten();
        s1_ids
    }
}

// ***  Colored Implementation  ***
#[cfg(feature = "colored")]
impl<L: Language, N: Analysis<L>> EGraph<L, N> {
    pub fn create_sub_color(&mut self, color: ColorId) -> ColorId {
        self.create_combined_color(vec![color])
    }

    fn process_colored_unions(&mut self, mut black_merged: Vec<Id>) {
        for i in 0..self.colors.len() {
            self.colored_cong_closure(ColorId(i), &mut black_merged);
        }
        self.memo_all_canonized();
    }

    pub fn create_color(&mut self) -> ColorId {
        self.colors.push(Some(Color::new(&self.unionfind, ColorId::from(self.colors.len()))));
        self.colors.last().unwrap().as_ref().unwrap().get_id()
    }

    /// Create a new color which is based on given colors. This should be used only if the new color
    /// has no assumptions of it's own (i.e. it is only a combination of existing assumptions).
    pub fn create_combined_color(&mut self, colors: Vec<ColorId>) -> ColorId {
        // First check if assumptions exist
        let new_c_id = self.create_color();
        assert!(colors.len() > 0);
        let mut todo = vec![];
        for c in colors {
            let mut new_classes = vec![];
            let old_dirty_unions = self.get_color(c).unwrap().dirty_unions.clone();
            self.get_color_mut(new_c_id).unwrap().dirty_unions.extend_from_slice(&old_dirty_unions);
            let mut union_map = self.get_color(c).unwrap().union_map.iter().map(|(x,y)| (*x, y.clone())).collect_vec();
            for class in self.classes() {
                if self.get_color(c).unwrap().union_map.contains_key(&self.colored_find(c, class.id)) {
                    dassert!(self.get_color(c).unwrap().union_map[&self.colored_find(c, class.id)].contains(&class.id));
                    continue;
                }
                if class.color.is_none() || class.color == Some(c) {
                    union_map.push((class.id, IndexSet::from_iter([class.id].into_iter().copied())));
                }
            }
            let union_map = union_map;
            let mut id_changer = IndexMap::new();
            union_map.iter().for_each(|(black_id, ids)| {
                dassert!(ids.contains(black_id));
                dassert!(ids.iter().map(|id| if self[*id].color.is_some() && !self[*id].nodes.is_empty() {1} else {0}).sum::<usize>() <= 1, "Ids: {}", ids.iter().join(", "));
                let colored = ids.iter().find(|id| self[**id].color.is_some());
                colored.copied().map(|id| {
                    id_changer.insert(id, Id((self.classes.len() + id_changer.len()) as u32))
                });
            });
            for (key, value) in id_changer.iter() {
                let mut node = self[*key].nodes[0].clone();
                let new_class_id = self.inner_create_class(&mut node, Some(new_c_id));
                dassert!(new_class_id.0 == value.0);
            }
            for (black_id, ids) in union_map {
                for id in ids.iter() {
                    if self[*id].color.is_some() && !self[*id].is_empty() {
                        let classes_len = self.classes.len();
                        let mut class = &mut self[*id];
                        wassert!(class.color == Some(c), "Color mismatch {:?} != {:?}", class.color, c);
                        let mut class_nodes = class.nodes.clone();
                        class_nodes.remove(0);
                        // Create a class, fix node and Analysis::Data
                        let new_class_id = id_changer[id];
                        let enode = &mut self[new_class_id].nodes[0];
                        enode.update_children(|id| *id_changer.get(&id).unwrap_or(&id));
                        let enode = enode.clone();
                        self[new_class_id].data = N::make(self, &enode);
                        let old = self.colored_memo.entry(enode).or_default().insert(new_c_id, new_class_id);
                        for old in old {
                            let res = self.colored_union(new_c_id, old, new_class_id);
                        }
                        for mut n in class_nodes {
                            n.update_children(|id| {
                                self.colored_find(new_c_id, *id_changer.get(&id).unwrap_or(&id))
                            });

                            let mut temp = {
                                let temp = N::make(self, &n);
                                std::mem::replace(&mut self[new_class_id].data, temp)
                            };
                            self.analysis.merge(&mut temp, N::make(self, &n));
                            self[new_class_id].data = temp;

                            // We are changing nodes later so this is actually a temporary value
                            self[new_class_id].nodes.push(n.clone());
                            let old = self.colored_memo.entry(n).or_default().insert(new_c_id, new_class_id);
                            for old in old {
                                let res = self.colored_union(new_c_id, old, new_class_id);
                            }
                        }
                        iassert!(self.classes[classes_len..].iter().map(|x| if x.is_none() {0} else {1}).sum::<usize>() <= 1);
                        let new_class_id = self.find(new_class_id);
                        new_classes.push(new_class_id);
                        self.get_color_mut(new_c_id).unwrap().black_colored_classes.insert(new_class_id, new_class_id);
                    }
                    todo.extend(self.get_color_mut(new_c_id).unwrap().inner_colored_union(black_id, *id_changer.get(id).unwrap_or(id)).2);
                }
            }

            // Now fix nodes, and create data, and put in the parents with id_translation.
            for id in new_classes {
                let mut parents_to_add = vec![];
                for n in self[id].iter() {
                    for ch in n.children() {
                        parents_to_add.push((*ch, n.clone(), id));
                    }
                }
                for (ch, n, id) in parents_to_add {
                    self[ch].colored_parents.entry(new_c_id).or_default().push((n, id));
                }
                N::modify(self, id);
            }

            self.get_color_mut(c).unwrap().children.push(new_c_id);
            let old_parents = self.get_color(c).unwrap().parents.clone();
            self.get_color_mut(new_c_id).unwrap().parents.push(c);
            self.get_color_mut(new_c_id).unwrap().parents.extend(old_parents);
            let old_unions = self.get_color(c).unwrap().dirty_unions.iter().map(|id| id_changer.get(id).unwrap_or(id)).copied().collect_vec();
            self.get_color_mut(new_c_id).unwrap().dirty_unions.extend(old_unions);
        }
        for (id1, id2) in todo {
            self.union(id1, id2);
        }
        new_c_id
    }

    pub fn delete_color(&mut self, c_id: ColorId) {
        assert!(self.dirty_unions.is_empty());
        let color = std::mem::replace(&mut self.colors[c_id.0], None).unwrap();
        for (colored, black) in color.black_colored_classes {
            let class = std::mem::replace(&mut self.classes[black.0 as usize], None).unwrap();
            for n in &class.nodes {
                self.classes_by_op.get_mut(&n.op_id()).map(|x| x.remove(&class.id));
                self.colored_memo.get_mut(n).map(|x| x.remove(&c_id));
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
        dassert!(self.colored_equivalences.iter().all(|(id, ids)|
            ids.iter().chain([(c_id, *id)].iter()).all(|(c_id, id)| self[*id].color.iter().all(|c| c != c_id))));
    }

    pub fn colored_union(&mut self, color: ColorId, id1: Id, id2: Id) -> (Id, bool) {
        let (to, changed, todo) = self.get_color_mut(color).unwrap().inner_colored_union(id1, id2);
        if let Some((id1, id2)) = todo {
            self.union(id1, id2);
        }
        if changed {
            for child in self.get_color(color).unwrap().children().iter().copied().collect_vec() {
                self.colored_union(child, id1, id2);
            }
        }
        (to, changed)
    }

    pub fn colored_find(&self, color: ColorId, id: Id) -> Id {
        self.get_color(color).unwrap().find(id)
    }

    pub fn colors(&self) -> impl Iterator<Item=&Color> {
        self.colors.iter().filter_map(|x| x.as_ref())
    }

    pub fn colors_mut(&mut self) -> impl Iterator<Item=&mut Color> {
        self.colors.iter_mut().filter_map(|x| x.as_mut())
    }

    pub fn get_color(&self, color: ColorId) -> Option<&Color> {
        self.colors[usize::from(color)].as_ref()
    }

    pub fn get_color_mut(&mut self, color: ColorId) -> Option<&mut Color> {
        self.colors[usize::from(color)].as_mut()
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
        let c = &egraph.get_color(color).unwrap();
        c.assert_black_ids(&egraph);
        egraph.colored_union(color, w, x);
        let c = &egraph.get_color(color).unwrap();
        c.assert_black_ids(&egraph);
        egraph.union(w, y);
        let c = &egraph.get_color(color).unwrap();
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
        assert_eq!(egraph.get_color(c).unwrap().black_ids(to).map(|x| x.len()), Some(6));

        egraph.union(ex5, ex6);
        egraph.union(ex1, ex5);
        println!("{:#?}", egraph.get_color(c).unwrap().black_ids(to));
        assert_eq!(egraph.get_color(c).unwrap().black_ids(to).map(|x| x.len()), Some(4));
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
        assert_eq!(egraph.get_color(c3).unwrap().black_ids(to).map(|x| x.len()), Some(6));
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
        let z = egraph.add(S::leaf("z"));
        let w = egraph.add(S::leaf("w"));

        let color1 = egraph.create_color();
        egraph.colored_union(color1, y, x);
        let color2 = egraph.create_color();
        egraph.colored_union(color2, w, z);
        let child = egraph.create_combined_color(vec![color1, color2]);
        egraph.colored_union(color2, x, z);
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

        for x in egraph.colors() {
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


        assert_eq!(egraph.colored_find(color_z, init), egraph.colored_find(color_z, res_z), "Black ids for color_z:\n  {}", egraph.get_color(color_z).unwrap().to_string());
        rules[0].search(&egraph).iter().for_each(|x| {
            println!("{}", x);
        });
        assert_eq!(egraph.colored_find(color_succ, init), egraph.colored_find(color_succ, res_succ_z), "Black ids for color_succ:\n  {}", egraph.get_color(color_succ).unwrap().to_string());
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

    fn choose<T: Clone>(mut from: Vec<Vec<T>>, amount: usize) -> Vec<Vec<T>> {
        if from.len() < amount || amount == 0 {
            return vec![];
        }
        if amount == 1 {
            return from.clone().into_iter().flatten().map(|v| vec![v]).collect_vec();
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
        use crate::SymbolLang as S;

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
            assert!(egraph.get_color(c).unwrap().black_ids(expr_id)
                .map(|x| x.clone())
                .unwrap_or([egraph.colored_find(c, expr_id)].iter().copied().collect())
                .iter().any(|id|
                    egraph[*id].nodes.iter().any(|n| {
                        let op = format!("{}", n.display_op());
                        op == "nil" || op == "cons"
                    })
            ));
        }
    }

    #[test]
    fn multi_level_colored_bad_filter() {
        use crate::SymbolLang as S;

        crate::init_logger();

        let (mut egraph, rules, expr_id, lv1_colors) = initialize_filter_tests();
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
        use crate::SymbolLang as S;

        crate::init_logger();

        let (mut egraph, rules, expr_id, lv1_colors) = initialize_filter_tests();
        let bad_color = egraph.create_combined_color(lv1_colors[2].iter().copied().collect_vec());

        let mut egraph = Runner::default().with_egraph(egraph).run(&rules).egraph;
        egraph.rebuild();
        egraph.check_memo();
        egraph.memo_all_canonized();
    }

    fn initialize_filter_tests() -> (EGraph<SymbolLang, ()>, Vec<Rewrite<SymbolLang, ()>>, Id, Vec<Vec<ColorId>>) {
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

        let expr_id = egraph.add_expr(&"(filter p (cons x1 (cons x2 (cons x3 nil))))".parse().unwrap());
        let vars = [egraph.add_expr(&"(p x1)".parse().unwrap()), egraph.add_expr(&"(p x2)".parse().unwrap()), egraph.add_expr(&"(p x3)".parse().unwrap())];
        let tru = egraph.add_expr(&"true".parse().unwrap());
        let fals = egraph.add_expr(&"false".parse().unwrap());
        let lv1_colors = vars.iter().map(|id| {
            let color_true = egraph.create_color();
            let color_false = egraph.create_color();
            egraph.colored_union(color_true, *id, tru);
            egraph.colored_union(color_false, *id, fals);
            vec![color_true, color_false]
        }).collect_vec();
        egraph.rebuild();
        (egraph, rules, expr_id, lv1_colors)
    }
}
