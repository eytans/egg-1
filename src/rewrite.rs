use std::fmt;
use std::rc::Rc;

use crate::{Analysis, EGraph, Id, Language, Pattern, SearchMatches, Subst, Var, ColorId, FromOp};
use std::fmt::{Formatter, Debug};
use std::marker::PhantomData;
use std::ops::Deref;
use itertools::Itertools;
use invariants::iassert;

/// A rewrite that searches for the lefthand side and applies the righthand side.
///
/// The [`rewrite!`] is the easiest way to create rewrites.
///
/// A [`Rewrite`] consists principally of a [`Searcher`] (the lefthand
/// side) and an [`Applier`] (the righthand side).
/// It additionally stores a name used to refer to the rewrite and a
/// long name used for debugging.
///
/// [`rewrite!`]: macro.rewrite.html
/// [`Searcher`]: trait.Searcher.html
/// [`Applier`]: trait.Applier.html
/// [`Condition`]: trait.Condition.html
/// [`ConditionalApplier`]: struct.ConditionalApplier.html
/// [`Rewrite`]: struct.Rewrite.html
/// [`Pattern`]: struct.Pattern.html
#[derive(Clone)]
#[non_exhaustive]
pub struct Rewrite<L: Language, N: Analysis<L>> {
    name: String,
    long_name: String,
    searcher: Rc<dyn Searcher<L, N>>,
    applier: Rc<dyn Applier<L, N>>,
}

impl<L, N> fmt::Debug for Rewrite<L, N>
    where
        L: Language + 'static,
        N: Analysis<L> + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct DisplayAsDebug<T>(T);
        impl<T: fmt::Display> fmt::Debug for DisplayAsDebug<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        use std::any::Any;

        let mut d = f.debug_struct("Rewrite");
        d.field("name", &self.name)
            .field("long_name", &self.long_name);

        if let Some(pat) = <dyn Any>::downcast_ref::<Pattern<L>>(&self.searcher) {
            d.field("searcher", &DisplayAsDebug(pat));
        } else {
            d.field("searcher", &"<< searcher >>");
        }

        if let Some(pat) = <dyn Any>::downcast_ref::<Pattern<L>>(&self.applier) {
            d.field("applier", &DisplayAsDebug(pat));
        } else {
            d.field("applier", &"<< applier >>");
        }

        d.finish()
    }
}

impl<L: Language, N: Analysis<L>> Rewrite<L, N> {
    /// Returns the name of the rewrite.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the long name of the rewrite which should only be used for
    /// debugging and displaying.
    pub fn long_name(&self) -> &str {
        &self.long_name
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> Rewrite<L, N> {
    /// Create a new [`Rewrite`]. You typically want to use the
    /// [`rewrite!`] macro instead.
    ///
    /// [`Rewrite`]: struct.Rewrite.html
    /// [`rewrite!`]: macro.rewrite.html
    pub fn old_new(
        name: impl Into<String>,
        long_name: impl Into<String>,
        searcher: impl Searcher<L, N> + 'static,
        applier: impl Applier<L, N> + 'static,
    ) -> Result<Self, String> {
        let name = name.into();
        let long_name = long_name.into();
        let searcher = Rc::new(searcher);
        let applier = Rc::new(applier);

        let bound_vars = searcher.vars();
        for v in applier.vars() {
            if !bound_vars.contains(&v) {
                return Err(format!("Rewrite {} refers to unbound var {}", name, v));
            }
        }

        Ok(Self {
            name,
            long_name,
            searcher,
            applier,
        })
    }

    /// Create a new [`Rewrite`]. You typically want to use the
    /// [`rewrite!`] macro instead.
    ///
    /// [`Rewrite`]: struct.Rewrite.html
    /// [`rewrite!`]: macro.rewrite.html
    pub fn new(
        name: impl Into<String>,
        // long_name: impl Into<String>,
        searcher: impl Searcher<L, N> + 'static,
        applier: impl Applier<L, N> + 'static,
    ) -> Result<Self, String> {
        let name = name.into();
        let long_name = name.clone();
        let searcher = Rc::new(searcher);
        let applier = Rc::new(applier);

        let bound_vars = searcher.vars();
        for v in applier.vars() {
            if !bound_vars.contains(&v) {
                return Err(format!("Rewrite {} refers to unbound var {}", name, v));
            }
        }

        Ok(Self {
            name,
            long_name,
            searcher,
            applier,
        })
    }

    /// Call [`search`] on the [`Searcher`].
    ///
    /// [`Searcher`]: trait.Searcher.html
    /// [`search`]: trait.Searcher.html#method.search
    pub fn search(&self, egraph: &EGraph<L, N>) -> Vec<SearchMatches> {
        self.searcher.search(egraph)
    }

    /// Call [`apply_matches`] on the [`Applier`].
    ///
    /// [`Applier`]: trait.Applier.html
    /// [`apply_matches`]: trait.Applier.html#method.apply_matches
    pub fn apply(&self, egraph: &mut EGraph<L, N>, matches: &[SearchMatches]) -> Vec<Id> {
        self.applier.apply_matches(egraph, matches)
    }

    /// This `run` is for testing use only. You should use things
    /// from the `egg::run` module
    #[cfg(test)]
    pub(crate) fn run(&self, egraph: &mut EGraph<L, N>) -> Vec<Id> {
        let start = instant::Instant::now();

        let matches = self.search(egraph);
        log::debug!("Found rewrite {} {} times", self.name, matches.len());

        let ids = self.apply(egraph, &matches);
        let elapsed = start.elapsed();
        log::debug!(
            "Applied rewrite {} {} times in {}.{:03}",
            self.name,
            ids.len(),
            elapsed.as_secs(),
            elapsed.subsec_millis()
        );

        egraph.rebuild();
        ids
    }
}

impl<L: Language, N: Analysis<L>> std::fmt::Display for Rewrite<L, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rewrite({}, {}, {})", self.name, self.searcher, self.applier)
    }
}

/// The lefthand side of a [`Rewrite`].
///
/// A [`Searcher`] is something that can search the egraph and find
/// matching substititions.
/// Right now the only significant [`Searcher`] is [`Pattern`].
///
/// [`Rewrite`]: struct.Rewrite.html
/// [`Searcher`]: trait.Searcher.html
/// [`Pattern`]: struct.Pattern.html
pub trait Searcher<L, N>: std::fmt::Display
    where
        L: Language,
        N: Analysis<L>,
{
    /// Search one eclass, returning None if no matches can be found.
    /// This should not return a SearchMatches with no substs.
    fn search_eclass(&self, egraph: &EGraph<L, N>, eclass: Id) -> Option<SearchMatches>;

    /// Search the whole [`EGraph`], returning a list of all the
    /// [`SearchMatches`] where something was found.
    /// This just calls [`search_eclass`] on each eclass.
    ///
    /// [`EGraph`]: struct.EGraph.html
    /// [`search_eclass`]: trait.Searcher.html#tymethod.search_eclass
    /// [`SearchMatches`]: struct.SearchMatches.html
    fn search(&self, egraph: &EGraph<L, N>) -> Vec<SearchMatches> {
        egraph
            .classes()
            .filter_map(|e| self.search_eclass(egraph, e.id))
            .collect()
    }

    /// Search one eclass with starting color
    /// This should also check all color-equal classes
    fn colored_search_eclass(&self, egraph: &EGraph<L, N>, eclass: Id, color: ColorId) -> Option<SearchMatches>;

    /// Returns a list of the variables bound by this Searcher
    fn vars(&self) -> Vec<Var>;

    /// Returns the number of matches in the e-graph
    fn n_matches(&self, egraph: &EGraph<L, N>) -> usize {
        self.search(egraph).iter().map(|m| m.substs.len()).sum()
    }
}

/// The righthand side of a [`Rewrite`].
///
/// An [`Applier`] is anything that can do something with a
/// substitition ([`Subst`]). This allows you to implement rewrites
/// that determine when and how to respond to a match using custom
/// logic, including access to the [`Analysis`] data of an [`EClass`].
///
/// Notably, [`Pattern`] implements [`Applier`], which suffices in
/// most cases.
/// Additionally, `egg` provides [`ConditionalApplier`] to stack
/// [`Condition`]s onto an [`Applier`], which in many cases can save
/// you from having to implement your own applier.
///
/// # Example
/// ```
/// use egg::{rewrite as rw, *};
/// use std::fmt::{Display, Formatter};
///
/// define_language! {
///     enum Math {
///         Num(i32),
///         "+" = Add([Id; 2]),
///         "*" = Mul([Id; 2]),
///         Symbol(Symbol),
///     }
/// }
///
/// type EGraph = egg::EGraph<Math, MinSize>;
///
/// // Our metadata in this case will be size of the smallest
/// // represented expression in the eclass.
/// #[derive(Default, Clone)]
/// struct MinSize;
/// impl Analysis<Math> for MinSize {
///     type Data = usize;
///     fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
///         merge_if_different(to, (*to).min(from))
///     }
///     fn make(egraph: &EGraph, enode: &Math) -> Self::Data {
///         let get_size = |i: Id| egraph[i].data;
///         AstSize.cost(enode, get_size)
///     }
/// }
///
/// let rules = &[
///     rw!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
///     rw!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
///     rw!("add-0"; "(+ ?a 0)" => "?a"),
///     rw!("mul-0"; "(* ?a 0)" => "0"),
///     rw!("mul-1"; "(* ?a 1)" => "?a"),
///     // the rewrite macro parses the rhs as a single token tree, so
///     // we wrap it in braces (parens work too).
///     rw!("funky"; "(+ ?a (* ?b ?c))" => { Funky {
///         a: "?a".parse().unwrap(),
///         b: "?b".parse().unwrap(),
///         c: "?c".parse().unwrap(),
///     }}),
/// ];
///
/// #[derive(Debug, Clone, PartialEq, Eq)]
/// struct Funky {
///     a: Var,
///     b: Var,
///     c: Var,
/// }
///
/// impl std::fmt::Display for Funky {
///     fn fmt(&self,f: &mut Formatter<'_>) -> std::fmt::Result {
///         write!(f, "Funky")
///     }
/// }
///
/// impl Applier<Math, MinSize> for Funky {
///     fn apply_one(&self, egraph: &mut EGraph, matched_id: Id, subst: &Subst) -> Vec<Id> {
///         let a: Id = subst[self.a];
///         // In a custom Applier, you can inspect the analysis data,
///         // which is powerful combination!
///         let size_of_a = egraph[a].data;
///         if size_of_a > 50 {
///             println!("Too big! Not doing anything");
///             vec![]
///         } else {
///             // we're going to manually add:
///             // (+ (+ ?a 0) (* (+ ?b 0) (+ ?c 0)))
///             // to be unified with the original:
///             // (+    ?a    (*    ?b       ?c   ))
///             let b: Id = subst[self.b];
///             let c: Id = subst[self.c];
///             let zero = egraph.add(Math::Num(0));
///             let a0 = egraph.add(Math::Add([a, zero]));
///             let b0 = egraph.add(Math::Add([b, zero]));
///             let c0 = egraph.add(Math::Add([c, zero]));
///             let b0c0 = egraph.add(Math::Mul([b0, c0]));
///             let a0b0c0 = egraph.add(Math::Add([a0, b0c0]));
///             // NOTE: we just return the id according to what we
///             // want unified with matched_id. The `apply_matches`
///             // method actually does the union, _not_ `apply_one`.
///             vec![a0b0c0]
///         }
///     }
/// }
///
/// let start = "(+ x (* y z))".parse().unwrap();
/// let mut runner = Runner::default().with_expr(&start);
/// runner.egraph.rebuild();
/// runner.run(rules);
/// ```
/// [`Pattern`]: struct.Pattern.html
/// [`EClass`]: struct.EClass.html
/// [`Rewrite`]: struct.Rewrite.html
/// [`ConditionalApplier`]: struct.ConditionalApplier.html
/// [`Subst`]: struct.Subst.html
/// [`Applier`]: trait.Applier.html
/// [`Condition`]: trait.Condition.html
/// [`Analysis`]: trait.Analysis.html
pub trait Applier<L, N>: std::fmt::Display
    where
        L: Language + 'static,
        N: Analysis<L> +'static,
{
    /// Apply many substititions.
    ///
    /// This method should call [`apply_one`] for each match and then
    /// unify the results with the matched eclass.
    /// This should return a list of [`Id`]s where the union actually
    /// did something.
    ///
    /// The default implementation does this and should suffice for
    /// most use cases.
    ///
    /// [`Id`]: struct.Id.html
    /// [`apply_one`]: trait.Applier.html#method.apply_one
    fn apply_matches(&self, egraph: &mut EGraph<L, N>, matches: &[SearchMatches]) -> Vec<Id> {
        let mut added = vec![];
        for mat in matches {
            for subst in &mat.substs {
                let ids = self
                    .apply_one(egraph, mat.eclass, subst)
                    .into_iter()
                    .filter_map(|id| {
                        if !cfg!(feature = "colored") {
                            iassert!(subst.color().is_none());
                        }
                        let (to, did_something) =
                            egraph.opt_colored_union(subst.color(), id, mat.eclass);
                        if did_something {
                            Some(to)
                        } else {
                            None
                        }
                    });
                added.extend(ids)
            }
        }
        added
    }

    /// Apply a single substitition.
    ///
    /// An [`Applier`] should only add things to the egraph here,
    /// _not_ union them with the id `eclass`.
    /// That is the responsibility of the [`apply_matches`] method.
    /// The `eclass` parameter allows the implementer to inspect the
    /// eclass where the match was found if they need to.
    ///
    /// This should return a list of [`Id`]s of things you'd like to
    /// be unioned with `eclass`. There can be zero, one, or many.
    ///
    /// [`Applier`]: trait.Applier.html
    /// [`Id`]: struct.Id.html
    /// [`apply_matches`]: trait.Applier.html#method.apply_matches
    fn apply_one(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> Vec<Id>;

    /// Returns a list of variables that this Applier assumes are bound.
    ///
    /// `egg` will check that the corresponding `Searcher` binds those
    /// variables.
    /// By default this return an empty `Vec`, which basically turns off the
    /// checking.
    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

/// An [`Applier`] that checks a [`Condition`] before applying.
///
/// A [`ConditionalApplier`] simply calls [`check`] on the
/// [`Condition`] before calling [`apply_one`] on the inner
/// [`Applier`].
///
/// See the [`rewrite!`] macro documentation for an example.
///
/// [`rewrite!`]: macro.rewrite.html
/// [`Applier`]: trait.Applier.html
/// [`apply_one`]: trait.Applier.html#method.apply_one
/// [`Condition`]: trait.Condition.html
/// [`check`]: trait.Condition.html#method.check
/// [`ConditionalApplier`]: struct.ConditionalApplier.html
#[derive(Clone, Debug)]
pub struct ConditionalApplier<C, A, L, N> {
    /// The [`Condition`] to [`check`] before calling [`apply_one`] on
    /// `applier`.
    ///
    /// [`apply_one`]: trait.Applier.html#method.apply_one
    /// [`Condition`]: trait.Condition.html
    /// [`check`]: trait.Condition.html#method.check
    pub condition: C,
    /// The inner [`Applier`] to call once `condition` passes.
    ///
    /// [`Applier`]: trait.Applier.html
    pub applier: A,
    pub phantom: PhantomData<(L, N)>,
}

impl<L, N, C, A> ConditionalApplier<C, A, L, N> where
    L: Language + 'static,
    N: Analysis<L> +'static,
    C: Condition<L, N>,
    A: Applier<L, N>,
{
    /// Create a new [`ConditionalApplier`].
    ///
    /// [`ConditionalApplier`]: struct.ConditionalApplier.html
    pub fn new(condition: C, applier: A) -> Self {
        ConditionalApplier {
            condition,
            applier,
            phantom: PhantomData::default(),
        }
    }

    fn conditioned_apply_one(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> Vec<(Option<ColorId>, Id)> {
        if cfg!(feature = "colored") {
            if let Some(v) = self.condition.check_colored(egraph, eclass, subst) {
                if v.is_empty() {
                    self.applier.apply_one(egraph, eclass, subst).into_iter().map(|id| (subst.color(), id)).collect()
                } else {
                    let mut res = vec![];
                    let mut colored_subst = subst.clone();
                    for c in v {
                        colored_subst.color = Some(c);
                        colored_subst.vec.iter_mut()
                            .for_each(|(_, id)| *id = egraph.colored_find(c, *id));
                        let ids = self.applier
                            .apply_one(egraph, eclass, &colored_subst)
                            .into_iter().map(|id| (Some(c), id));
                        res.extend(ids);
                    }
                    res
                }
            } else {
                vec![]
            }
        } else {
            if self.condition.check(egraph, eclass, subst) {
                self.applier.apply_one(egraph, eclass, subst).into_iter().map(|id| (None, id)).collect()
            } else {
                vec![]
            }
        }
    }
}

impl<L, N, C, A> std::fmt::Display for ConditionalApplier<C, A, L, N> where
    L: Language + 'static,
    N: Analysis<L> + 'static,
    C: Condition<L, N>,
    A: Applier<L, N>, {
        fn fmt( & self, f: & mut Formatter < '_ > ) -> std::fmt::Result {
          write ! (f, "if {:?} then apply {}", self.condition.describe(), self.applier)
    }
}

impl<C, A, N, L> Applier<L, N> for ConditionalApplier<C, A, L, N>
    where
        L: Language + 'static,
        C: Condition<L, N>,
        A: Applier<L, N>,
        N: Analysis<L> + 'static,
{
    // TODO: sort out an API for this
    fn apply_one(&self, _egraph: &mut EGraph<L, N>, _eclass: Id, _subst: &Subst) -> Vec<Id> {
        unimplemented!("ConditionalApplier::apply_one is not implemented. Instead we use conditioned_apply_one");
    }

    fn apply_matches(&self, egraph: &mut EGraph<L, N>, matches: &[SearchMatches]) -> Vec<Id> {
        let mut added = vec![];
        for mat in matches {
            for subst in &mat.substs {
                let ids = self
                    .conditioned_apply_one(egraph, mat.eclass, subst)
                    .into_iter()
                    .filter_map(|(opt_c, id)| {
                        let (to, did_something) =
                            egraph.opt_colored_union(opt_c, mat.eclass, id);
                        did_something.then(|| to)
                    });
                added.extend(ids)
            }
        }
        added
    }

    fn vars(&self) -> Vec<Var> {
        let mut vars = self.applier.vars();
        vars.extend(self.condition.vars());
        vars
    }
}

/// A condition to check in a [`ConditionalApplier`].
///
/// See the [`ConditionalApplier`] docs.
///
/// Notably, any function ([`Fn`]) that doesn't mutate other state
/// and matches the signature of [`check`] implements [`Condition`].
///
/// [`check`]: trait.Condition.html#method.check
/// [`Fn`]: https://doc.rust-lang.org/std/ops/trait.Fn.html
/// [`ConditionalApplier`]: struct.ConditionalApplier.html
/// [`Condition`]: trait.Condition.html
pub trait Condition<L, N>
    where
        L: Language,
        N: Analysis<L>,
{
    /// Check a condition.
    ///
    /// `eclass` is the eclass [`Id`] where the match (`subst`) occured.
    /// If this is true, then the [`ConditionalApplier`] will fire.
    ///
    /// [`Id`]: struct.Id.html
    /// [`ConditionalApplier`]: struct.ConditionalApplier.html
    fn check(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> bool;

    /// Check a condition and possibly add colored assumptions
    ///
    /// If the condition cannot be satisfied returns None. If the condition is satisfied in current
    /// color, returns an empty vector. Otherwise, it will return a vector of all the sub-colors in
    /// which the condition is satisfied (currently does not support sub-colors).
    /// This additional complexity is necessary because we cannot make the searcher return matches
    /// from all possible colors when it is not needed. Instead, the condition will check when it
    /// can be satisfied.
    ///
    /// Consider the following query that looks for a node that is false with the additional
    /// condition that it is the result of the function f: `false if match_root ~= (f ?a ?b)`.
    /// We will get the resulting Subst: `Subst { no_holes, color: None }`, but the condition check
    /// will fail. We need to be able to query the condition on additional colored conditions that
    /// might have been ignored by the pattern, but are necessary for the condition to be true.
    /// TODO: add support for hierarchical colors in the future.
    fn check_colored(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>>;

    /// Returns a list of variables that this Condition assumes are bound.
    ///
    /// `egg` will check that the corresponding `Searcher` binds those
    /// variables.
    /// By default this return an empty `Vec`, which basically turns off the
    /// checking.
    fn vars(&self) -> Vec<Var> {
        vec![]
    }

    /// Returns a string representing the condition that is checked
    fn describe(&self) -> String;
}

impl<L, N> Condition<L, N> for Box<dyn Condition<L, N>> where
    L: Language,
    N: Analysis<L>,
{
    fn check(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        self.as_ref().check(egraph, eclass, subst)
    }

    fn check_colored(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        self.as_ref().check_colored(egraph, eclass, subst)
    }

    fn vars(&self) -> Vec<Var> {
        self.as_ref().vars()
    }

    fn describe(&self) -> String {
        self.as_ref().describe()
    }
}

// pub struct FunctionCondition<L, N> where
//     L: Language,
//     N: Analysis<L>,
// {
//     f: Rc<dyn Fn(&mut EGraph<L, N>, Id, &Subst) -> bool>,
//     description: String,
//     phantom_l: PhantomData<L>,
//     phantom_n: PhantomData<N>,
// }
//
// impl<L, N> Condition<L, N> for FunctionCondition<L, N>
//     where
//         L: Language,
//         N: Analysis<L>,
// {
//     fn check(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
//         (self.f)(egraph, eclass, subst)
//     }
//
//     fn describe(&self) -> String {
//         self.description.clone()
//     }
// }
//
// impl<L, N> FunctionCondition<L, N>
//     where
//         L: Language,
//         N: Analysis<L>,
// {
//     pub fn new<S>(f: Rc<dyn Fn(&mut EGraph<L, N>, Id, &Subst) -> bool>, description: S) -> Self
//         where
//             S: Into<String>,
//     {
//         FunctionCondition {
//             f,
//             description: description.into(),
//             phantom_l: PhantomData,
//             phantom_n: PhantomData,
//         }
//     }
// }

/// A [`Condition`] that checks if two terms are equivalent.
///
/// This condition adds its two [`Applier`]s to the egraph and passes
/// if and only if they are equivalent (in the same eclass).
///
/// [`Applier`]: trait.Applier.html
/// [`Condition`]: trait.Condition.html
pub struct ConditionEqual<A1, A2>(pub A1, pub A2);

impl<L: Language + FromOp> ConditionEqual<Pattern<L>, Pattern<L>> {
    /// Create a ConditionEqual by parsing two pattern strings.
    ///
    /// This panics if the parsing fails.
    pub fn parse(a1: &str, a2: &str) -> Self {
        Self(a1.parse().unwrap(), a2.parse().unwrap())
    }
}

impl<L, N, A1, A2> Condition<L, N> for ConditionEqual<A1, A2>
    where
        L: Language + 'static,
        N: Analysis<L> + 'static,
        A1: Applier<L, N>,
        A2: Applier<L, N>,
{
    fn check(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        let a1 = self.0.apply_one(egraph, eclass, subst);
        let a2 = self.1.apply_one(egraph, eclass, subst);
        assert_eq!(a1.len(), 1);
        assert_eq!(a2.len(), 1);
        egraph.opt_colored_find(subst.color(), a1[0]) == egraph.opt_colored_find(subst.color(), a2[0])
    }

    fn check_colored(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        if self.check(egraph, eclass, subst) {
            return Some(vec![]);
        } else if subst.color().is_some() {
            // TODO: add support for hierarchical colors in the future.
            return None;
        }
        let a1 = self.0.apply_one(egraph, eclass, subst)[0];
        let a2 = self.1.apply_one(egraph, eclass, subst)[0];
        let eqs = egraph.get_colored_equalities(a1);
        eqs.map(|eqs| {
            let res = eqs.into_iter()
                .filter(|(c, id)| egraph.opt_colored_find(Some(*c), a2) == egraph.opt_colored_find(Some(*c), *id))
                .map(|(c, _)| c).collect_vec();
            (!res.is_empty()).then(|| res)
        }).flatten()
    }

    fn vars(&self) -> Vec<Var> {
        let mut vars = self.0.vars();
        vars.extend(self.1.vars());
        vars
    }

    fn describe(&self) -> String {
        format!("Check that {}", self.vars().iter().map(|v| v.to_string()).join(" = "))
    }
}

pub trait ToCondRc<L, N> where
    L: Language,
    N: Analysis<L>,
{
    fn into_rc(self) -> Rc<dyn ImmutableCondition<L, N>>
        where
            Self: Sized, Self: ImmutableCondition<L, N>, Self: 'static,
    {
        Rc::new(self)
    }
}

pub trait ImmutableCondition<L, N>: ToCondRc<L, N> where
    L: Language,
    N: Analysis<L>,
{
    fn check_imm<'a>(&'a self, egraph: &'a EGraph<L, N>, eclass: Id, subst: &'a Subst) -> bool;

    /// Check a condition and possibly add colored assumptions. For more information and a useful
    /// example see [`Condition::check_colored`].
    ///
    /// # Arguments
    ///
    /// * `egraph`:
    /// * `eclass`:
    /// * `subst`:
    ///
    /// returns: Option<Vec<ColorId, Global>>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    fn colored_check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>>;

    /// Returns a list of variables that this Condition assumes are bound.
    ///
    /// `egg` will check that the corresponding `Searcher` binds those
    /// variables.
    /// By default this return an empty `Vec`, which basically turns off the
    /// checking.
    fn vars(&self) -> Vec<Var> {
        vec![]
    }

    /// Returns a string representing the condition that is checked
    fn describe(&self) -> String;

    fn filter(&self, egraph: &EGraph<L, N>, sms: Vec<SearchMatches>) -> Vec<SearchMatches> {
        sms.into_iter().filter_map(|mut sm| {
            let eclass = sm.eclass;
            let substs = std::mem::take(&mut sm.substs);
            sm.substs = substs.into_iter()
                .filter_map(|s| {
                    self.colored_check_imm(egraph, eclass, &s).map(|x| (s, x)) })
                .flat_map(|(s, colors)| {
                    if colors.is_empty() {
                        vec![s]
                    } else {
                        colors.into_iter().map(|c| {
                            let mut s = s.clone();
                            s.color = Some(c);
                            s.vec.iter_mut().for_each(|(_, id)| *id = egraph.colored_find(c, *id));
                            s
                        }).collect_vec()
                    }
                }).collect_vec();
            if sm.substs.is_empty() {
                None
            } else {
                Some(sm)
            }
        }).collect_vec()
    }
}

pub type RcImmutableCondition<L, N> = Rc<dyn ImmutableCondition<L, N>>;

// pub struct ImmutableFunctionCondition<L, N> where
//     L: Language,
//     N: Analysis<L>,
// {
//     f: Rc<dyn Fn(&EGraph<L, N>, Id, &Subst) -> bool>,
//     description: String,
//     phantom_l: PhantomData<L>,
//     phantom_n: PhantomData<N>,
// }
//
// impl<L, N> ImmutableFunctionCondition<L, N>
//     where
//         L: Language,
//         N: Analysis<L>,
// {
//     pub fn new(f: Rc<dyn Fn(&EGraph<L, N>, Id, &Subst) -> bool>, desc: String) -> Self {
//         Self {
//             f, description: desc, phantom_l: Default::default(), phantom_n: Default::default(),
//         }
//     }
// }

// impl<L, N> ToCondRc<L, N> for ImmutableFunctionCondition<L, N> where L: Language, N: Analysis<L> {}

// impl<L, N> ImmutableCondition<L, N> for ImmutableFunctionCondition<L, N>
//     where
//         L: Language,
//         N: Analysis<L>,
// {
//     fn check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
//         (self.f)(egraph, eclass, subst)
//     }
//
//     fn describe(&self) -> String {
//         self.description.clone()
//     }
// }

impl<L, N> ToCondRc<L, N> for RcImmutableCondition<L, N> where L: Language, N: Analysis<L> {}

impl<L, N> ImmutableCondition<L, N> for RcImmutableCondition<L, N> where
    L: Language,
    N: Analysis<L>,
{
    fn check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        self.deref().check_imm(egraph, eclass, subst)
    }

    fn colored_check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        self.deref().colored_check_imm(egraph, eclass, subst)
    }

    fn vars(&self) -> Vec<Var> {
        self.deref().vars()
    }

    fn describe(&self) -> String {
        self.deref().describe()
    }
}

impl<L, N> Condition<L, N> for dyn ImmutableCondition<L, N>
    where
        L: Language,
        N: Analysis<L>,
{
    fn check(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        self.check_imm(egraph, eclass, subst)
    }

    fn check_colored(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        self.colored_check_imm(egraph, eclass, subst)
    }

    fn vars(&self) -> Vec<Var> {
        self.vars()
    }

    fn describe(&self) -> String {
        self.describe()
    }
}

#[cfg(test)]
mod tests {
    use crate::{SymbolLang as S, *};
    use std::str::FromStr;
    use std::fmt::Formatter;
    // use crate::rewrite::{ImmutableCondition, ImmutableFunctionCondition, RcImmutableCondition};
    use crate::rewrite::ImmutableCondition;
    use crate::searchers::{MatcherContainsCondition, ToRc, VarMatcher};

    type EGraph = crate::EGraph<S, ()>;

    #[test]
    fn conditional_rewrite() {
        crate::init_logger();
        let mut egraph = EGraph::default();

        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("2"));
        let mul = egraph.add(S::new("*", vec![x, y]));

        let true_pat: Pattern<SymbolLang> = Pattern::from_str("TRUE").unwrap();
        let true_id = egraph.add(S::leaf("TRUE"));

        let pow2b: Pattern<SymbolLang> = Pattern::from_str("(is-power2 ?b)").unwrap();
        let mul_to_shift = rewrite!(
            "mul_to_shift";
            "(* ?a ?b)" => "(>> ?a (log2 ?b))"
            if ConditionEqual(pow2b, true_pat)
        );

        println!("rewrite shouldn't do anything yet");
        egraph.rebuild();
        let apps = mul_to_shift.run(&mut egraph);
        assert!(apps.is_empty());

        println!("Add the needed equality");
        let two_ispow2 = egraph.add(S::new("is-power2", vec![y]));
        egraph.union(two_ispow2, true_id);

        println!("Should fire now");
        egraph.rebuild();
        let apps = mul_to_shift.run(&mut egraph);
        assert_eq!(apps, vec![egraph.find(mul)]);
    }

    #[test]
    fn fn_rewrite() {
        crate::init_logger();
        let mut egraph = EGraph::default();

        let start = RecExpr::from_str("(+ x y)").unwrap();
        let goal = RecExpr::from_str("xy").unwrap();

        let root = egraph.add_expr(&start);

        fn get(egraph: &EGraph, id: Id) -> Symbol {
            egraph[id].nodes[0].op
        }

        #[derive(Debug)]
        struct Appender;
        impl std::fmt::Display for Appender {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:?}", self)
            }
        }

        impl Applier<SymbolLang, ()> for Appender {
            fn apply_one(&self, egraph: &mut EGraph, _eclass: Id, subst: &Subst) -> Vec<Id> {
                let a: Var = "?a".parse().unwrap();
                let b: Var = "?b".parse().unwrap();
                let a = get(&egraph, subst[a]);
                let b = get(&egraph, subst[b]);
                let s = format!("{}{}", a, b);
                vec![egraph.add(S::leaf(&s))]
            }
        }

        let fold_add = rewrite!(
            "fold_add"; "(+ ?a ?b)" => { Appender }
        );

        egraph.rebuild();
        fold_add.run(&mut egraph);
        assert_eq!(egraph.equivs(&start, &goal), vec![egraph.find(root)]);
    }

    #[test]
    fn conditional_applier_respects_colors() {
        // This is a very specific case. It should be a case where the applier can be applied on a
        // black result, but the condition doesn't hold for black.
        // Then, we should add a color to the graph, and show the condition holds but only under
        // the new color.
        // Finally, we want to see that the application is done under the new color (create a new
        // colored class, and colored union classes).
        crate::init_logger();
        let mut egraph = EGraph::default();

        let matcher = VarMatcher::new(Var::from_str("?a").unwrap());
        // add x + y expression
        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));
        let add = egraph.add(S::new("+", vec![x, y]));
        egraph.rebuild();
        // ?x + ?y pattern
        let pat = Pattern::from_str("(+ ?a ?b)").unwrap();
        let sms = pat.search(&egraph);
        assert_eq!(sms.len(), 1);
        let sm = sms.first().unwrap().clone();
        assert_eq!(sm.substs.len(), 1);
        let subst = sm.substs[0].clone();
        // Check matcher condition doesn't hold
        let condition = MatcherContainsCondition::new(matcher.into_rc());
        assert!(!condition.check_imm(&mut egraph, add, &subst));
        assert!(condition.colored_check_imm(&mut egraph, add, &subst).is_none());
        // Add color, and merge add and x
        let color = egraph.create_color();
        egraph.colored_union(color, add, x);
        egraph.rebuild();
        // Check matcher colored condition holds, and only contains the color
        let cond_res = condition.colored_check_imm(&mut egraph, add, &subst);
        assert!(cond_res.is_some());
        assert_eq!(cond_res.unwrap()[0], color);

        // Create an applier for this condition and see that the application will merge the colored
        // classes.
        egraph.rebuild();
        let applier = rewrite!("applier"; "(+ ?a ?b)" => "(+ ?b ?a)" if {Box::new(condition) as Box<dyn Condition<SymbolLang, ()>>});
        let class_count = egraph.classes().count();
        let mut egraph = Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(1)
            .run(vec![&applier]).egraph;
        let z = egraph.add(S::leaf("z"));
        egraph.rebuild();
        assert_ne!(egraph.find(add), egraph.find(z));
        let new_class_count = egraph.classes().count();
        let yx = egraph.colored_add(color, S::new("+", vec![y, x]));
        assert_eq!(new_class_count, egraph.classes().count());
        assert_ne!(egraph.find(yx), egraph.find(add));
        assert_eq!(egraph.colored_find(color, yx), egraph.colored_find(color, add));
        // Added z and yx
        assert_eq!(egraph.classes().count(), class_count + 2);
        #[cfg(not(feature = "colored_no_cmemo"))]
        assert_eq!(egraph[egraph.find(yx)].color().unwrap(), color);
    }
}
