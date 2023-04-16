use std::iter::FromIterator;
use std::str::FromStr;

// use crate::{EGraph, Id, Pattern, Searcher, SearchMatches, Subst, SymbolLang, Var, Language, Analysis, Condition, ImmutableCondition, ENodeOrVar, ImmutableFunctionCondition, RcImmutableCondition, ToCondRc, ColorId};
use crate::{EGraph, Id, Pattern, Searcher, SearchMatches, Subst, SymbolLang, Var, Language, Analysis, Condition, ImmutableCondition, RcImmutableCondition, ToCondRc, ColorId};
use itertools::{Itertools, Either};

use crate::tools::tools::Grouped;
use crate::eggstentions::pretty_string::PrettyString;
use std::fmt::{Debug, Display};
use smallvec::alloc::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;

use std::rc::Rc;
use std::time::Instant;
use indexmap::{IndexMap, IndexSet};
use log::{trace, warn};


/// A trait for a matcher that can be used in a Searcher. Differs from condition, as it is more
/// general and can be seen as a subpattern.
pub trait Matcher<L: Language + 'static, N: Analysis<L> + 'static>: ToRc<L, N> {
    /// Returns ids of all roots that match this matcher, considering egraph and subst.
    /// Does not return colored ids for black subst!
    fn match_<'b>(&self, egraph: &'b EGraph<L, N>, subst: &'b Subst) -> IndexSet<Id>;

    /// Returns a string representation of the matcher.
    fn describe(&self) -> String;
}

impl<L: 'static + Language, N: 'static + Analysis<L>> std::fmt::Display for dyn Matcher<L, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}

/// A trait for a matcher that matches a single hole in a pattern.
pub struct VarMatcher<L: Language, N: Analysis<L>> {
    pub(crate) var: Var,
    phantom: PhantomData<(N, L)>,
}

impl<L: Language, N: Analysis<L>> VarMatcher<L, N> {
    /// Creates a new VarMatcher.
    pub fn new(var: Var) -> Self {
        VarMatcher {
            var,
            phantom: Default::default(),
        }
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> ToRc<L, N> for VarMatcher<L, N> {}

impl<L: Language + 'static, N: Analysis<L> + 'static> Matcher<L, N> for VarMatcher<L, N> {
    fn match_<'b>(&self, graph: &'b EGraph<L, N>, subst: &'b Subst) -> IndexSet<Id> {
        subst.get(self.var).copied().map(|v| graph.opt_colored_find(subst.color(), v))
            .into_iter().collect()
    }

    fn describe(&self) -> String {
        self.var.to_string()
    }
}

/// A trait for a matcher that matches a different pattern.
pub struct PatternMatcher<L: Language, N: Analysis<L>> {
    pub(crate) pattern: Pattern<L>,
    phantom: PhantomData<N>,
}

impl<L: Language, N: Analysis<L>> PatternMatcher<L, N> {
    /// Creates a new PatternMatcher.
    pub fn new(pattern: Pattern<L>) -> Self {
        PatternMatcher {
            pattern,
            phantom: Default::default(),
        }
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> ToRc<L, N> for PatternMatcher<L, N> {}

impl<L: Language + 'static, N: Analysis<L> + 'static> Matcher<L, N> for PatternMatcher<L, N> {
    fn match_<'b>(&self, graph: &'b EGraph<L, N>, subst: &'b Subst) -> IndexSet<Id> {
        let time = Instant::now();
        // TODO: Support hierarchical colors.
        let res = self.pattern.search(graph).into_iter().flat_map(|x| {
            let mut black_subs = None;
            let mut colored_subs = None;
            for s in x.substs.iter()
                .filter(|s| s.color().is_none() || (s.color() == subst.color())) {
                if graph.subst_agrees(s, subst, true) {
                    if s.color().is_some() {
                        colored_subs = Some(graph.colored_find(s.color().unwrap(), x.eclass));
                    } else {
                        black_subs = Some(graph.find(x.eclass));
                    }
                }
                if black_subs.is_some() && (colored_subs.is_some() || subst.color().is_none()) {
                    break;
                }
            }
            vec![black_subs, colored_subs].into_iter().filter_map(|s| s)
        }).collect();
        // warn!("res ({:?}) of PatternMatcher {} ({}): {:?}", subst.color(), self.describe(), graph.total_number_of_nodes(), res);
        if cfg!(debug_assertions) {
            if time.elapsed().as_secs() > 1 {
                warn!("Matcher Pattern search took {} seconds", time.elapsed().as_secs());
            }
        }
        res
    }

    fn describe(&self) -> String {
        self.pattern.to_string()
    }
}

/// Helper alias for wrapping a dynamically typed matcher in a smart pointer.
pub type RcMatcher<L, N> = Rc<dyn Matcher<L, N>>;

impl<L: Language + 'static, N: Analysis<L> + 'static> ToRc<L, N> for Rc<dyn Matcher<L, N>> {}

impl<L: Language + 'static, N: Analysis<L> + 'static> Matcher<L, N> for Rc<dyn Matcher<L, N>> {
    fn match_<'b>(&self, graph: &'b EGraph<L, N>, subst: &'b Subst) -> IndexSet<Id> {
        let res = self.deref().match_(graph, subst);
        res
    }

    fn describe(&self) -> String {
        self.deref().describe()
    }
}

/// A trait for a matcher that matches a pattern but ignores a different pattern.
pub struct DisjointMatcher<L: Language, N: Analysis<L>> {
    pub(crate) matcher1: Rc<dyn Matcher<L, N>>,
    pub(crate) matcher2: Rc<dyn Matcher<L, N>>,
    /// A description of the matcher.
    #[cfg(debug_assertions)]
    pub desc: String,
}

impl<L: Language + 'static, N: Analysis<L> + 'static> DisjointMatcher<L, N> {
    /// Creates a new DisjointMatcher.
    pub fn new(matcher1: Rc<dyn Matcher<L, N>>, matcher2: Rc<dyn Matcher<L, N>>) -> Self {
        #[cfg(debug_assertions)]
        let desc = format!("{} != {}", matcher1.describe(), matcher2.describe());
        DisjointMatcher {
            matcher1,
            matcher2,
            #[cfg(debug_assertions)]
            desc,
        }
    }

    /// Returns true if the two matchers are disjoint for this subst and graph.
    pub fn is_disjoint<'b>(&self, graph: &'b EGraph<L, N>, subst: &'b Subst) -> bool {
        let match_2 = self.matcher2.match_(graph, subst);
        let res = self.matcher1.match_(graph, subst).into_iter().all(|x| {
            !match_2.contains(&x)
        });
        res
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> ToRc<L, N> for DisjointMatcher<L, N> {}

impl<L: Language + 'static, N: Analysis<L> + 'static> Matcher<L, N> for DisjointMatcher<L, N> {
    fn match_<'b>(&self, graph: &'b EGraph<L, N>, subst: &'b Subst) -> IndexSet<Id> {
        let time = Instant::now();
        let res = self.matcher1.match_(graph, subst).into_iter().filter(|&x| {
            !self.matcher2.match_(graph, subst).contains(&x)
        }).collect();
        if cfg!(debug_assertions) {
            if time.elapsed().as_secs() > 1 {
                warn!("Matcher Disjoint search took {} seconds", time.elapsed().as_secs());
            }
        }
        res
    }

    fn describe(&self) -> String {
        format!("{} != {}", self.matcher1.describe(), self.matcher2.describe())
    }
}

/// Wrapper for two types of searchers.
pub struct EitherSearcher<L: Language, N: Analysis<L>, A: Searcher<L, N> + Debug, B: Searcher<L, N> + Debug> {
    node: Either<A, B>,
    phantom: PhantomData<(L, N)>,
}

impl<L: Language, N: Analysis<L>, A: Searcher<L, N> + Debug, B: Searcher<L, N> + Debug> EitherSearcher<L, N, A, B> {
    /// Creates a new EitherSearcher with the left type.
    pub fn left(a: A) -> EitherSearcher<L, N, A, B> {
        EitherSearcher { node: Either::Left(a), phantom: PhantomData::default() }
    }

    /// Creates a new EitherSearcher with the right type.
    pub fn right(b: B) -> EitherSearcher<L, N, A, B> {
        EitherSearcher { node: Either::Right(b), phantom: PhantomData::default() }
    }
}

impl<L: Language, N: Analysis<L>, A: Searcher<L, N> + Debug, B: Searcher<L, N> + Debug> Searcher<L, N> for EitherSearcher<L, N, A, B> {
    fn search_eclass(&self, egraph: &EGraph<L, N>, eclass: Id) -> Option<SearchMatches> {
        if self.node.is_left() {
            self.node.as_ref().left().unwrap().search_eclass(egraph, eclass)
        } else {
            self.node.as_ref().right().unwrap().search_eclass(egraph, eclass)
        }
    }

    fn search(&self, egraph: &EGraph<L, N>) -> Vec<SearchMatches> {
        if self.node.is_left() {
            self.node.as_ref().left().unwrap().search(egraph)
        } else {
            self.node.as_ref().right().unwrap().search(egraph)
        }
    }

    fn colored_search_eclass(&self, egraph: &EGraph<L, N>, eclass: Id, color: ColorId) -> Option<SearchMatches> {
        if self.node.is_left() {
            self.node.as_ref().left().unwrap().colored_search_eclass(egraph, eclass, color)
        } else {
            self.node.as_ref().right().unwrap().colored_search_eclass(egraph, eclass, color)
        }
    }

    fn vars(&self) -> Vec<Var> {
        if self.node.is_left() {
            self.node.as_ref().left().unwrap().vars()
        } else {
            self.node.as_ref().right().unwrap().vars()
        }
    }
}

impl<L: Language, N: Analysis<L>, A: Searcher<L, N> + Debug, B: Searcher<L, N> + Debug> std::fmt::Display for EitherSearcher<L, N, A, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.node {
            Either::Left(x) => { write!(f, "{}", x) }
            Either::Right(x) => { write!(f, "{}", x) }
        }
    }
}

impl<L: Language, N: Analysis<L>, A: Searcher<L, N> + Debug + Clone, B: Searcher<L, N> + Debug + Clone> Clone for EitherSearcher<L, N, A, B> {
    fn clone(&self) -> Self {
        if self.node.is_left() {
            Self::left(self.node.as_ref().left().unwrap().clone())
        } else {
            Self::right(self.node.as_ref().right().unwrap().clone())
        }
    }
}

impl<L: Language, N: Analysis<L>, A: Searcher<L, N> + Debug + PrettyString, B: Searcher<L, N> + Debug + PrettyString> PrettyString for EitherSearcher<L, N, A, B> {
    fn pretty_string(&self) -> String {
        if self.node.is_left() {
            self.node.as_ref().left().unwrap().pretty_string()
        } else {
            self.node.as_ref().right().unwrap().pretty_string()
        }
    }
}

impl<L: Language, N: Analysis<L>, A: Searcher<L, N> + Debug, B: Searcher<L, N> + Debug> Debug for EitherSearcher<L, N, A, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.node, f)
    }
}

fn count_common_vars(patterns: &mut Vec<impl Searcher<SymbolLang, ()>>) -> IndexMap<Var, usize> {
    let common_vars = patterns.iter().flat_map(|p| p.vars())
        .grouped(|v| v.clone()).iter()
        .filter_map(|(k, v)|
            if v.len() <= 1 { None } else { Some((*k, v.len())) })
        .collect::<IndexMap<Var, usize>>();

    common_vars
}

fn merge_substs(vars: &Vec<Var>, sub1: &Subst, sub2: &Subst) -> Subst {
    let mut res = Subst::colored_with_capacity(vars.len(), sub1.color().or_else(|| sub2.color()));
    for v in vars {
        let v1 = *v;
        let s1 = sub1.get(v1);
        let s2 = sub2.get(v1);
        if s1.is_some() || s2.is_some() {
            // TODO: Assert with egraph they agree on color
            // if s1.is_some() && s2.is_some() {
            //     assert_eq!(s1.as_ref().unwrap(), s2.as_ref().unwrap());
            // }
            res.insert(v1, *s1.unwrap_or_else(|| s2.unwrap()));
        }
    }
    res
}

// Aggregate product of equal common var substs
fn aggregate_substs(matches_by_subst: &[Vec<(Vec<Option<Id>>, Vec<(Id, Subst)>)>],
                    limits: Vec<Option<Id>>,
                    all_vars: &Vec<Var>) -> Vec<(Id, Subst)> {
    let current = matches_by_subst.first().unwrap();
    let matches = current.iter()
        .filter(|(keys, _)| limits.iter().zip(keys.iter())
            .all(|(lim, key)| lim.as_ref().map_or(true, |l| key.as_ref().map_or(true, |k| k == l))));
    if matches_by_subst.len() > 1 {
        let mut collected = Vec::new();
        for (key, val) in matches {
            let new_limit = limits.iter().zip(key)
                .map(|(l, k)| if l.is_some() { l } else { k })
                .cloned()
                .collect_vec();

            let rec_res = aggregate_substs(&matches_by_subst[1..],
                                           new_limit,
                                           all_vars);
            collected.extend(rec_res.iter().cartesian_product(val)
                .map(|((_, s1), (id, s2))| (*id, merge_substs(all_vars, s1, s2))));
        }
        collected
    } else {
        // TODO: I changed this from merge_substs(s, s). Might get an error later on missing vars.
        matches.flat_map(|(_, v)| v.iter().map(|(id, s)| (*id, s.clone()))).collect()
    }
}

/// Uses multiple searchers with results agreeing on vars but not on root
pub struct MultiDiffSearcher<A: Searcher<SymbolLang, ()>> {
    patterns: Vec<A>,
    common_vars_priorities: IndexMap<Var, usize>,
}

impl<A: Searcher<SymbolLang, ()>> MultiDiffSearcher<A> {
    /** 
     * Creates a new MultiDiffSearcher from a list of patterns.
     */
    pub fn new(mut patterns: Vec<A>) -> MultiDiffSearcher<A> {
        let common_vars = count_common_vars(&mut patterns);
        assert!(!patterns.is_empty());
        MultiDiffSearcher { patterns, common_vars_priorities: common_vars }
    }
}

impl<S: Searcher<SymbolLang, ()> + 'static> ToDyn<SymbolLang, ()> for MultiDiffSearcher<S> {
    fn into_rc_dyn(self) -> Rc<dyn Searcher<SymbolLang, ()>> {
        let dyn_s: Rc<dyn Searcher<SymbolLang, ()>> = Rc::new(self);
        dyn_s
    }
}

impl<S: Searcher<SymbolLang, ()> + 'static> std::fmt::Display for MultiDiffSearcher<S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.patterns.iter().map(|x| x.to_string()).join(" ||| "))
    }
}

// impl PrettyString for MultiDiffSearcher {
//     fn pretty_string(&self) -> String {
//         self.patterns.iter().map(|p| p.pretty_string()).intersperse(" |||| ".to_string()).collect()
//     }
// }

impl<A: 'static + Searcher<SymbolLang, ()>> Searcher<SymbolLang, ()> for MultiDiffSearcher<A> {
    fn search_eclass(&self, _: &EGraph<SymbolLang, ()>, _: Id) -> Option<SearchMatches> {
        unimplemented!()
    }

    fn search(&self, egraph: &EGraph<SymbolLang, ()>) -> Vec<SearchMatches> {
        if self.patterns.len() == 1 {
            return self.patterns[0].search(egraph);
        }

        let search_results =
            self.patterns.iter().map(|p| {
                // For each color collect all substitutions by common var assignments
                let mut res: IndexMap<Option<ColorId>, Vec<_>> = IndexMap::new();
                for m in p.search(egraph) {
                    let class = m.eclass;
                    let by_vars = {
                        let groups = m.substs.into_iter()
                            .map(|s|
                                (self.common_vars_priorities.keys().map(|v| s.get(*v)
                                    .map(|id| egraph.opt_colored_find(s.color(), *id))).collect_vec(),
                                 class,
                                 s))
                            .sorted()
                            .group_by(|(v, _c, s)| (s.color(), v.clone()));
                        groups.into_iter().map(|(k, v)| (k, v.collect_vec())).grouped(|x| x.0.0)
                    };
                    for (color, vars) in by_vars {
                        res.entry(color).or_default().extend(vars.into_iter().map(|((_c, vars), g)| {
                            (vars, g.into_iter().map(|(_var, c, s)| (c, s)).collect_vec())
                        }));
                    }
                }
                res
            }).collect_vec();

        // To reuse group_by_common_vars we will merge all results to a single searchmatches.
        // We don't really care which eclass we use so we will choose the first one.
        // It is a really stupid way to do it but we will run the grouping for each eclass from
        // the first one.
        egraph.colors().map(|c| Some(c.get_id()))
            .chain(std::iter::once(None)).flat_map(|c_id| {
            let empty = vec![];
            let collect_results: Box<dyn Fn(&IndexMap<Option<ColorId>, Vec<(Vec<Option<Id>>, Vec<(Id, Subst)>)>>)
                -> Vec<(Vec<Option<Id>>, Vec<(Id, Subst)>)>> = Box::new(|map: &IndexMap<Option<ColorId>, Vec<(Vec<Option<Id>>, Vec<(Id, Subst)>)>>| {
                let mut res = map.get(&c_id).unwrap_or_else(|| &empty).iter().map(|(vars, g)|
                    ((*vars).clone(), g.clone())).collect_vec();
                if c_id.is_some() {
                    for (vars, g) in map.get(&None).unwrap_or_else(|| &empty).iter() {
                        let new_vars = vars.iter()
                            .map(|opt_id| opt_id.map(|id|
                                egraph.opt_colored_find(c_id, id)))
                            .collect_vec();
                        res.push((new_vars, g.clone()));
                    }
                }
                res.sort_by(|(vars1, _), (vars2, _)| vars1.cmp(vars2));
                res.dedup_by(|(vars, g1), (vars2, g2)|
                    vars == vars2 && {
                        g2.extend(std::mem::take(g1));
                        true
                    });
                res
            });

            let all_combinations = search_results.iter().map(|res| (*collect_results)(res)).collect_vec();
            let initial_limits = self.common_vars_priorities.iter().map(|_| None).collect_vec();
            let res = aggregate_substs(&all_combinations[..], initial_limits, &self.vars());
            if res.is_empty() {
                vec![]
            } else {
                res.into_iter().group_by(|x| x.0).into_iter()
                    .map(|(id, s)| SearchMatches {
                        eclass: id,
                        substs: s.into_iter().map(|(_, s)| s).unique().collect(),
                    }).collect_vec()
            }
        }).collect()
    }

    fn colored_search_eclass(&self, _egraph: &EGraph<SymbolLang, ()>, _eclass: Id, _color: ColorId) -> Option<SearchMatches> {
        unimplemented!()
    }

    fn vars(&self) -> Vec<Var> {
        Vec::from_iter(self.patterns.iter().flat_map(|p| p.vars()).sorted().dedup())
    }
}

impl FromStr for MultiDiffSearcher<Pattern<SymbolLang>> {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let patterns = s.split("||||")
            .map(|p| Pattern::from_str(p).unwrap())
            .collect::<Vec<Pattern<SymbolLang>>>();
        if patterns.len() == 1 {
            Err(String::from("Need at least two patterns"))
        } else {
            Ok(MultiDiffSearcher::new(patterns))
        }
    }
}

impl<A: Searcher<SymbolLang, ()> + Clone> Clone for MultiDiffSearcher<A> {
    fn clone(&self) -> Self {
        MultiDiffSearcher::new(self.patterns.clone())
    }
}

impl<A: Searcher<SymbolLang, ()> + Debug> Debug for MultiDiffSearcher<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(&self.patterns).finish()
    }
}

impl<A: Searcher<SymbolLang, ()> + PrettyString> PrettyString for MultiDiffSearcher<A> {
    fn pretty_string(&self) -> String {
        itertools::Itertools::intersperse(self.patterns.iter().map(|p| p.pretty_string()), " ||| ".to_string()).collect()
    }
}

/**
 * A condition that is true for ids where the two disjoint matchers disagree.
 */
pub struct DisjointMatchCondition<L: Language, N: Analysis<L>> {
    disjointer: DisjointMatcher<L, N>,
    #[allow(dead_code)]
    #[cfg(debug_assertions)]
    desc: String,
}

impl<L: Language + 'static, N: Analysis<L> + 'static> DisjointMatchCondition<L, N> {
    /// Create a new disjoint matcher condition.
    pub fn new(disjointer: DisjointMatcher<L, N>) -> Self {
        #[cfg(debug_assertions)]
        let desc = disjointer.describe();
        DisjointMatchCondition {
            disjointer,
            #[cfg(debug_assertions)]
            desc,
        }
    }
}

impl<L: Language, N: Analysis<L>> ToCondRc<L, N> for DisjointMatchCondition<L, N> {}

impl<L: Language + 'static, N: Analysis<L> + 'static> ImmutableCondition<L, N> for DisjointMatchCondition<L, N> {
    fn check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        trace!("DisjointMatchCondition::{}({}, {}) - Start", self.describe(), eclass, subst);
        let res = self.disjointer.is_disjoint(egraph, subst);
        trace!("DisjointMatchCondition::{}({}, {}) - End - {}", self.describe(), eclass, subst, res);
        res
    }

    fn colored_check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        // I think this is always like check_imm because adding colored assumptions will just
        // create sets that are less disjoint.
        trace!("DisjointMatchCondition::colored::{}({}, {}) - Start", self.describe(), eclass, subst);
        let res = self.check_imm(egraph, eclass, subst)
            .then(|| subst.color().map(|c| vec![c]).unwrap_or(vec![]));
        trace!("DisjointMatchCondition::colored::{}({}, {}) - End - {:?}", self.describe(), eclass, subst, res);
        res
    }

    fn describe(&self) -> String {
        format!("{}", self.disjointer.describe())
    }
}

/**
 * A condition that is true when the matcher contains the id being checked.
 */
pub struct MatcherContainsCondition<L: Language + 'static, N: Analysis<L> + 'static> {
    matcher: Rc<dyn Matcher<L, N>>,
}

impl <L: Language + 'static, N: Analysis<L> + 'static> MatcherContainsCondition<L, N> {
    /// Create a new matcher contains condition.
    pub fn new(matcher: Rc<dyn Matcher<L, N>>) -> Self {
        MatcherContainsCondition { matcher }
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> ToCondRc<L, N> for MatcherContainsCondition<L, N> {}

impl<L: Language + 'static, N: Analysis<L> + 'static> ImmutableCondition<L, N> for MatcherContainsCondition<L, N> {
    fn check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        trace!("MatcherContainsCondition::{}({}, {}) - Start", ImmutableCondition::describe(self), eclass, subst);
        let fixed = egraph.opt_colored_find(subst.color(), eclass);
        let res = (self.matcher.match_(egraph, subst)).iter()
            .map(|id| egraph.opt_colored_find(subst.color(), *id))
            .any(|id| id == fixed);
        trace!("MatcherContainsCondition::{}({}, {}) - End - {}", ImmutableCondition::describe(self), eclass, subst, res);
        res
    }

    fn colored_check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        trace!("MatcherContainsCondition::colored::{}({}, {}) - Start", ImmutableCondition::describe(self), eclass, subst);
        let fixed = egraph.opt_colored_find(subst.color(), eclass);
        let mut colors = Vec::new();
        for id in self.matcher.match_(egraph, subst) {
            if egraph.opt_colored_find(subst.color(), id) == fixed {
                let res = Some(subst.color().map(|c| vec![c]).unwrap_or(vec![]));
                trace!("MatcherContainsCondition::colored::{}({}, {}) - End - {:?}", ImmutableCondition::describe(self), eclass, subst, res);
                return res;
            }
            if subst.color().is_none() {
                if let Some(eqs) = egraph.colored_equivalences.get(&id) {
                    for (c, id) in eqs {
                        if egraph.colored_find(*c, *id) == egraph.colored_find(*c, eclass) {
                            if !colors.contains(c) {
                                colors.push(*c);
                            }
                        }
                    }
                }
            }
        }
        let res = if colors.is_empty() {
            None
        } else {
            Some(colors)
        };
        trace!("MatcherContainsCondition::colored::{}({}, {}) - End - {:?}", ImmutableCondition::describe(self), eclass, subst, res);
        res
    }

    fn describe(&self) -> String {
        format!("({}).root.contains(subst_root)", self.matcher.describe())
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> Condition<L, N> for MatcherContainsCondition<L, N> {
    fn check(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        self.check_imm(egraph, eclass, subst)
    }

    fn check_colored(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        self.colored_check_imm(egraph, eclass, subst)
    }

    fn describe(&self) -> String {
        self.matcher.describe()
    }
}

/// Searcher that only returns results where given condition (`predicate`) is true.
#[derive(Clone)]
pub struct FilteringSearcher<L: Language, N: Analysis<L>> {
    searcher: Rc<dyn Searcher<L, N>>,
    predicate: RcImmutableCondition<L, N>,
    phantom_ln: PhantomData<(L, N)>,
}

impl<L: Language, N: Analysis<L>> PrettyString for FilteringSearcher<L, N> {
    fn pretty_string(&self) -> String {
        format!("{}[{}]", self.searcher, self.predicate.describe())
    }
}

impl<'a, L: Language + 'static, N: Analysis<L> + 'static> FilteringSearcher<L, N> {
    /// Create a new DisjointMatchCondition from two matchers.
    pub fn create_non_pattern_filterer(matcher: RcMatcher<L, N>,
                                       negator: RcMatcher<L, N>)
        -> RcImmutableCondition<L, N> {
        let dis_matcher = DisjointMatcher::new(matcher, negator);
        DisjointMatchCondition::new(dis_matcher).into_rc()
    }

    /// Create a new Pattern matcher condition that will check a pattern exists in the graph.
    pub fn create_exists_pattern_filterer(searcher: Pattern<L>) -> RcImmutableCondition<L, N> {
        // TODO: partially fill pattern and if not all vars have values then search by eclass
        //       In practice, create special searcher that will take the constant part from
        //       subst and check existence for each subpattern over eclasses found in subst
        let matcher = PatternMatcher::new(searcher);
        MatcherContainsCondition::new(matcher.into_rc()).into_rc()
    }

    /// Create a new FilteringSearcher.
    pub fn new(searcher: Rc<dyn Searcher<L, N>>,
               predicate: RcImmutableCondition<L, N>, ) -> Self {
        FilteringSearcher {
            searcher,
            predicate,
            phantom_ln: Default::default()
        }
    }

    /// Create a new FilteringSearcher from a searcher and a predicate.
    pub fn from<S: Searcher<L, N> + 'static>(s: S, predicate: RcImmutableCondition<L, N>) -> Self {
        let dyn_searcher: Rc<dyn Searcher<L, N>> = Rc::new(s);
        Self::new(dyn_searcher, predicate)
    }
}

impl FilteringSearcher<SymbolLang, ()> {
    /// Create a new FilteringSearcher that will filter out all EClasses that are not equal to `true`.
    pub fn searcher_is_true<S: Searcher<SymbolLang, ()> + 'static>(s: S) -> Self {
        Self::searcher_is_pattern(s, "true".parse().unwrap())
    }

    /// Create a new FilteringSearcher that will filter out all EClasses that are not equal to `false`.
    pub fn searcher_is_false<S: Searcher<SymbolLang, ()> + 'static>(s: S) -> Self {
        Self::searcher_is_pattern(s, "false".parse().unwrap())
    }

    /// Create a new FilteringSearcher that will filter out all EClasses also match with `p`.
    pub fn searcher_is_pattern<S: Searcher<SymbolLang, ()> + 'static>(s: S, p: Pattern<SymbolLang>) -> Self {
        FilteringSearcher::new(
            Rc::new(s),
            MatcherContainsCondition::new(PatternMatcher::new(p).into_rc()).into_rc()
        )
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> std::fmt::Display for FilteringSearcher<L, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} if {}", self.searcher, self.predicate.describe())
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> std::fmt::Debug for FilteringSearcher<L, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl<L: 'static + Language, N: 'static + Analysis<L>> Searcher<L, N> for FilteringSearcher<L, N> {
    fn search_eclass(&self, _egraph: &EGraph<L, N>, _eclass: Id) -> Option<SearchMatches> {
        unimplemented!()
    }

    fn search(&self, egraph: &EGraph<L, N>) -> Vec<SearchMatches> {
        trace!("FilteringSearcher::search({})", self.pretty_string());
        let origin = self.searcher.search(egraph);
        let res = self.predicate.filter(egraph, origin);
        res
    }

    fn colored_search_eclass(&self, _egraph: &EGraph<L, N>, _eclass: Id, _color: ColorId) -> Option<SearchMatches> {
        unimplemented!()
    }

    fn vars(&self) -> Vec<Var> {
        self.searcher.vars()
    }
}

/// Trait for converting a type to a dynamic type behind a Rc pointer.
pub trait ToDyn<L: Language, N: Analysis<L>> {
    /// Convert to a dynamic type behind a Rc pointer.
    fn into_rc_dyn(self) -> Rc<dyn Searcher<L, N>>;
}

impl<L: Language + 'static, N: Analysis<L> + 'static> ToDyn<L, N> for Pattern<L> {
    fn into_rc_dyn(self) -> Rc<dyn Searcher<L, N>> {
        let dyn_s: Rc<dyn Searcher<L, N>> = Rc::new(self);
        dyn_s
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> ToDyn<L, N> for FilteringSearcher<L, N> {
    fn into_rc_dyn(self) -> Rc<dyn Searcher<L, N>> {
        let dyn_s: Rc<dyn Searcher<L, N>> = Rc::new(self);
        dyn_s
    }
}

/// Trait for converting a type to a dynamic type behind a Rc pointer.
pub trait ToRc<L: Language + 'static, N: Analysis<L> + 'static> {
    /// Convert to a dynamic type behind a Rc pointer.
    fn into_rc(self) -> Rc<dyn Matcher<L, N>>
    where
        Self: Sized + Matcher<L, N> + 'static,
    {
        Rc::new(self)
    }
}

/// A searcher that wraps another searcher and returns the same result.
pub struct PointerSearcher<L: Language, N: Analysis<L>> {
    searcher: Rc<dyn Searcher<L, N>>,
}

impl<L: Language, N: Analysis<L>> PointerSearcher<L, N> {
    /// Create a new PointerSearcher.
    pub fn new(searcher: Rc<dyn Searcher<L, N>>) -> Self { PointerSearcher { searcher } }
}

impl<L: Language, N: Analysis<L>> std::fmt::Display for PointerSearcher<L, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.searcher)
    }
}

impl<L: Language, N: Analysis<L>> Searcher<L, N> for PointerSearcher<L, N> {
    fn search_eclass(&self, egraph: &EGraph<L, N>, eclass: Id) -> Option<SearchMatches> {
        self.searcher.search_eclass(egraph, eclass)
    }

    fn search(&self, egraph: &EGraph<L, N>) -> Vec<SearchMatches> {
        self.searcher.search(egraph)
    }

    fn colored_search_eclass(&self, egraph: &EGraph<L, N>, eclass: Id, color: ColorId) -> Option<SearchMatches> {
        self.searcher.colored_search_eclass(egraph, eclass, color)
    }

    fn vars(&self) -> Vec<Var> {
        self.searcher.vars()
    }
}


#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::{EGraph, RecExpr, Searcher, SymbolLang, Pattern, Var, ImmutableCondition, ToCondRc};

    use crate::eggstentions::searchers::{MultiDiffSearcher, FilteringSearcher, ToDyn, Matcher, PatternMatcher};
    use crate::searchers::{MatcherContainsCondition, ToRc, VarMatcher};
    // use crate::system_case_splits;

    #[test]
    fn eq_two_trees_one_common() {
        let matcher = FilteringSearcher::create_exists_pattern_filterer("(a ?c ?d)".parse().unwrap());
        let searcher = {
            let pattern: Pattern<SymbolLang> = "(a ?b ?c)".parse().unwrap();
            FilteringSearcher::new(pattern.into_rc_dyn(), matcher)
        };
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::default();
        let x = egraph.add_expr(&RecExpr::from_str("x").unwrap());
        let z = egraph.add_expr(&RecExpr::from_str("z").unwrap());
        let a = egraph.add_expr(&RecExpr::from_str("(a x y)").unwrap());
        egraph.add_expr(&RecExpr::from_str("(a z x)").unwrap());
        egraph.rebuild();
        assert_eq!(searcher.search(&egraph).len(), 0);
        let a2 = egraph.add(SymbolLang::new("a", vec![z, x]));
        egraph.union(a, a2);
        egraph.rebuild();
        assert_eq!(searcher.search(&egraph).len(), 1);
    }

    #[test]
    fn diff_two_trees_one_common() {
        let searcher = MultiDiffSearcher::from_str("(a ?b ?c) |||| (a ?c ?d)").unwrap();
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::default();
        let _x = egraph.add_expr(&RecExpr::from_str("x").unwrap());
        let _z = egraph.add_expr(&RecExpr::from_str("z").unwrap());
        let _a = egraph.add_expr(&RecExpr::from_str("(a x y)").unwrap());
        egraph.add_expr(&RecExpr::from_str("(a z x)").unwrap());
        egraph.rebuild();
        assert_eq!(searcher.search(&egraph).len(), 1);
    }

    #[test]
    fn find_ind_hyp() {
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::default();
        let full_pl = egraph.add_expr(&"(pl (S p0) Z)".parse().unwrap());
        let after_pl = egraph.add_expr(&"(S (pl p0 Z))".parse().unwrap());
        let sp0 = egraph.add_expr(&"(S p0)".parse().unwrap());
        let ind_var = egraph.add_expr(&"ind_var".parse().unwrap());
        egraph.union(ind_var, sp0);
        let _ltwf = egraph.add_expr(&"(ltwf p0 (S p0))".parse().unwrap());
        egraph.union(full_pl, after_pl);
        egraph.rebuild();
        let searcher = MultiDiffSearcher::from_str("(ltwf ?x ind_var) |||| (pl ?x Z)").unwrap();
        assert!(!searcher.search(&egraph).is_empty());
    }

    #[test]
    fn pattern_to_matcher_sanity() {
        let mut graph: EGraph<SymbolLang, ()> = EGraph::default();
        graph.add_expr(&RecExpr::from_str("(+ 1 (+ 2 3))").unwrap());
        graph.add_expr(&RecExpr::from_str("(+ 31 (+ 32 33))").unwrap());
        graph.add_expr(&RecExpr::from_str("(+ 21 (+ 22 23))").unwrap());
        graph.add_expr(&RecExpr::from_str("(+ 11 (+ 12 13))").unwrap());
        let p: Pattern<SymbolLang> = Pattern::from_str("(+ ?z (+ ?x ?y))").unwrap();
        let m = PatternMatcher::new(p.clone());
        let results = p.search(&graph);
        for sm in results {
            let eclass = sm.eclass;
            for sb in sm.substs {
                assert_eq!(m.match_(&graph, &sb).contains(&eclass), true);
            }
        }
    }

    #[test]
    fn filtering_searcher_finds_color() {
        // This is a very specific case, similar to conditional applier test in rewrite.rs.
        // It should be a case where the filtering searcher can not find a black result because the
        // condition doesn't hold for black. Then, we should add a color to the graph, and show the
        // condition holds, but only under the new color.
        crate::init_logger();
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::default();

        let matcher:VarMatcher<SymbolLang, ()> = VarMatcher::new(Var::from_str("?a").unwrap());
        // add x + y expression
        let x = egraph.add(SymbolLang::leaf("x"));
        let y = egraph.add(SymbolLang::leaf("y"));
        let add = egraph.add(SymbolLang::new("+", vec![x, y]));
        egraph.rebuild();
        // ?x + ?y pattern
        let pat = Pattern::from_str("(+ ?a ?b)").unwrap();
        let sms = pat.search(&egraph);
        assert_eq!(sms.len(), 1);
        let sm = sms.first().unwrap().clone();
        assert_eq!(sm.substs.len(), 1);
        let subst = sm.substs[0].clone();
        // Check matcher condition doesn't hold
        let condition = MatcherContainsCondition::new(matcher.into_rc()).into_rc();
        assert!(!condition.check_imm(&mut egraph, add, &subst));
        assert!(condition.colored_check_imm(&mut egraph, add, &subst).is_none());
        let searcher = FilteringSearcher::new(pat.into_rc_dyn(), condition.clone());
        assert_eq!(searcher.search(&egraph).len(), 0);
        // Add color, and merge add and x
        let color = egraph.create_color();
        egraph.colored_union(color, add, x);
        egraph.rebuild();
        // Check matcher colored condition holds, and only contains the color
        let cond_res = condition.colored_check_imm(&mut egraph, add, &subst);
        assert!(cond_res.is_some());
        assert_eq!(cond_res.unwrap()[0], color);

        // Check that the searcher now finds the match
        let results = searcher.search(&egraph);
        assert_eq!(results.len(), 1);
        let sm = results.first().unwrap().clone();
        assert_eq!(sm.substs.len(), 1);
        let subst = sm.substs[0].clone();
        assert!(subst.color().is_some());
        assert_eq!(subst.color().unwrap(), color);
    }

    // #[cfg(feature = "split_colored")]
    // #[test]
    // fn skip_vacuity_cases() {
    //     let mut graph: EGraph<SymbolLang, ()> = EGraph::default();
    //     graph.add_expr(&RecExpr::from_str("(ite x 1 2)").unwrap());
    //     graph.rebuild();
    //     let mut case_splitter = system_case_splits();
    //     let pattern: Pattern<SymbolLang> = Pattern::from_str("(ite ?z ?x ?y)").unwrap();
    //     println!("{:?}", pattern.search(&graph));
    //     let splitters = case_splitter.find_splitters(&mut graph);
    //     println!("{:?}", splitters);
    //     assert_eq!(splitters.len(), 1);
    //     let colors = splitters[0].create_colors(&mut graph);
    //     graph.rebuild();
    //     println!("{:?}", pattern.search(&graph));
    //     let new_splitters = case_splitter.find_splitters(&mut graph);
    //     println!("{:?}", new_splitters);
    //     assert_eq!(new_splitters.len(), 1);
    //
    // }
}