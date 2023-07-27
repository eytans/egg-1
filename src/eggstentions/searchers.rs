use crate::{EGraph, Id, Pattern, Searcher, SearchMatches, Subst, Var, Language, Analysis, ColorId};
use itertools::{Itertools};

use smallvec::alloc::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;

use std::rc::Rc;
use std::time::Instant;
use indexmap::IndexSet;
use log::warn;


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
            for (eclass, substs) in x.matches {
                for s in substs.iter()
                    .filter(|s| s.color().is_none() || (s.color() == subst.color())) {
                    if graph.subst_agrees(s, subst, true) {
                        if s.color().is_some() {
                            colored_subs = Some(graph.colored_find(s.color().unwrap(), eclass));
                        } else {
                            black_subs = Some(graph.find(eclass));
                        }
                    }
                    if black_subs.is_some() && (colored_subs.is_some() || subst.color().is_none()) {
                        break;
                    }
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

#[allow(unused)]
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
#[allow(unused)]
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

/**
 * A condition that is true for ids where the two disjoint matchers disagree.
 */
// pub struct DisjointMatchCondition<L: Language, N: Analysis<L>> {
//     disjointer: DisjointMatcher<L, N>,
//     #[allow(dead_code)]
//     #[cfg(debug_assertions)]
//     desc: String,
// }

/**
 * A condition that is true when the matcher contains the id being checked.
 */
// #[derive(Clone)]
// pub struct MatcherContainsCondition<L: Language + 'static, N: Analysis<L> + 'static> {
//     matcher: Rc<dyn Matcher<L, N>>,
// }

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

    fn search(&self, egraph: &EGraph<L, N>) -> Option<SearchMatches> {
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

    use crate::{EGraph, RecExpr, Searcher, SymbolLang, Pattern, MultiPattern, init_logger};

    use crate::eggstentions::searchers::{Matcher, PatternMatcher};
    // use crate::system_case_splits;

    #[test]
    fn eq_two_trees_one_common() {
        let searcher: MultiPattern<SymbolLang> =  "?x = (a ?b ?c), ?x = (a ?c ?d)".parse().unwrap();
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::default();
        let x = egraph.add_expr(&RecExpr::from_str("x").unwrap());
        let z = egraph.add_expr(&RecExpr::from_str("z").unwrap());
        let a = egraph.add_expr(&RecExpr::from_str("(a x y)").unwrap());
        egraph.add_expr(&RecExpr::from_str("(a z x)").unwrap());
        egraph.rebuild();
        assert!(searcher.search(&egraph).is_none());
        let a2 = egraph.add(SymbolLang::new("a", vec![z, x]));
        egraph.union(a, a2);
        egraph.rebuild();
        assert_eq!(searcher.search(&egraph).unwrap().len(), 1);
    }

    #[test]
    fn diff_two_trees_one_common() {
        init_logger();

        let searcher = MultiPattern::from_str("?v1 = (a ?b ?c), ?v2 = (a ?c ?d)").unwrap();
        let mut egraph: EGraph<SymbolLang, ()> = EGraph::default();
        let _x = egraph.add_expr(&RecExpr::from_str("x").unwrap());
        let _z = egraph.add_expr(&RecExpr::from_str("z").unwrap());
        let _a = egraph.add_expr(&RecExpr::from_str("(a x y)").unwrap());
        egraph.add_expr(&RecExpr::from_str("(a z x)").unwrap());
        egraph.rebuild();
        assert_eq!(searcher.search(&egraph).unwrap().len(), 1);
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
        let searcher = MultiPattern::from_str("?v1 = (ltwf ?x ind_var), ?v2 = (pl ?x Z)").unwrap();
        assert!(searcher.search(&egraph).is_some());
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
        if let Some(sm) = results {
            for (eclass, substs) in sm.matches {
                let eclass = eclass;
                for sb in substs {
                    assert_eq!(m.match_(&graph, &sb).contains(&eclass), true);
                }
            }
        }
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