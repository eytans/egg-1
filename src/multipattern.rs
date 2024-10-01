use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::rc::Rc;
use std::str::FromStr;
use indexmap::IndexSet;
use invariants::dassert;
use itertools::Itertools;
use thiserror::Error;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::*;
use crate::expression_ops::{IntoTree, Tree};
use crate::pretty_string::PrettyString;
use crate::searchers::ToDyn;

/// A set of open expressions bound to variables.
///
/// Multipatterns bind many expressions to variables,
/// allowing for simultaneous searching or application of many terms
/// constrained to the same substitution.
///
/// Multipatterns are good for writing graph rewrites or datalog-style rules.
///
/// You can create multipatterns via the [`MultiPattern::new`] function or the
/// [`multi_rewrite!`] macro.
///
/// [`MultiPattern`] implements both [`Searcher`] and [`Applier`].
/// When searching a multipattern, the result ensures that
/// patterns bound to the same variable are equivalent.
/// When applying a multipattern, patterns bound a variable occuring in the
/// searcher are unioned with that e-class.
///
/// Multipatterns currently do not support the explanations feature.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct MultiPattern<L> {
    asts: Vec<(Var, PatternAst<L>)>,
    or_asts: Vec<(Var, Vec<PatternAst<L>>)>,
    not_asts: Vec<(Var, PatternAst<L>)>,
    program: machine::Program<L>,
}

impl<L: Language> MultiPattern<L> {
    /// Creates a new multipattern, binding the given patterns to the corresponding variables.
    ///
    /// ```
    /// use easter_egg::*;
    ///
    /// let mut egraph = EGraph::<SymbolLang, ()>::default();
    /// egraph.add_expr(&"(f a a)".parse().unwrap());
    /// egraph.add_expr(&"(f a b)".parse().unwrap());
    /// egraph.add_expr(&"(g a a)".parse().unwrap());
    /// egraph.add_expr(&"(g a b)".parse().unwrap());
    /// egraph.rebuild();
    ///
    /// let f_pat: PatternAst<SymbolLang> = "(f ?x ?y)".parse().unwrap();
    /// let g_pat: PatternAst<SymbolLang> = "(g ?x ?y)".parse().unwrap();
    /// let v1: Var = "?v1".parse().unwrap();
    /// let v2: Var = "?v2".parse().unwrap();
    ///
    /// let multipattern = MultiPattern::new(vec![(v1, f_pat), (v2, g_pat)]);
    /// // you can also parse multipatterns
    /// assert_eq!(multipattern, "?v1 = (f ?x ?y), ?v2 = (g ?x ?y)".parse().unwrap());
    ///
    /// assert_eq!(multipattern.n_matches(&egraph), 2);
    /// ```
    pub fn new(asts: Vec<(Var, PatternAst<L>)>) -> Self {
        let program = machine::Program::compile_from_multi_pat(&asts, &vec![], &vec![]);
        Self { asts, or_asts: vec![], not_asts: vec![], program }
    }

    pub fn new_with_specials(
        asts: Vec<(Var, PatternAst<L>)>,
        or_asts: Vec<(Var, Vec<PatternAst<L>>)>,
        not_asts: Vec<(Var, PatternAst<L>)>,
    ) -> Self {
        let mut asts = asts;
        asts.sort_by(|(_v, p), (_v2, p2)| {
            let p_holes = p.into_tree().holes();
            if p_holes.len() == 0 {
                return Ordering::Less;
            }
            let p2_holes = p2.into_tree().holes();
            if p2_holes.len() == 0 {
                return Ordering::Greater;
            }
            return p2.into_tree().consts().len().cmp(&p.into_tree().consts().len());
        });
        let program = machine::Program::compile_from_multi_pat(&asts, &or_asts, &not_asts);
        Self { asts, or_asts, not_asts, program }
    }
}

#[derive(Error, Debug)]
/// An error raised when parsing a [`MultiPattern`]
pub enum MultiPatternParseError<E> {
    /// One of the patterns in the multipattern failed to parse.
    #[error(transparent)]
    PatternParseError(E),
    /// One of the clauses in the multipattern wasn't of the form `?var (= pattern)+`.
    #[error("Bad clause in the multipattern: `{0}`")]
    PatternAssignmentError(String),
    /// One of the variables failed to parse.
    #[error(transparent)]
    VariableError(<Var as FromStr>::Err),
}

impl<L: Language + FromOp> FromStr for MultiPattern<L> {
    type Err = MultiPatternParseError<<PatternAst<L> as FromStr>::Err>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use MultiPatternParseError::*;
        let mut asts = vec![];
        let mut or_asts: BTreeMap<Var, Vec<PatternAst<L>>> = BTreeMap::default();
        let mut not_asts = vec![];
        for split in s.trim().split(',') {
            let split = split.trim();
            if split.is_empty() {
                continue;
            }
            let regex = Regex::new(r"(\?[a-zA-Z0-9_/\-+*^%$#@!]+)\s*(\|=|!=|=)\s*(.+)").expect("bad regex");
            let parts = regex.captures(split)
                .ok_or_else(|| PatternAssignmentError(split.to_string()))?;

            let v: Var = parts[1].parse().map_err(VariableError)?;
            let pattern_ast: PatternAst<L> = parts[3].trim().parse()
                .map_err(PatternParseError)?;
            if &parts[2] == "!=" {
                not_asts.push((v, pattern_ast));
            } else if &parts[2] == "|=" {
                or_asts.entry(v).or_default().push(pattern_ast);
            } else {
                asts.push((v, pattern_ast));
            }
        }
        Ok(MultiPattern::new_with_specials(asts, or_asts.into_iter().collect_vec(), not_asts))
    }
}

impl<L: Language> Display for MultiPattern<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.asts.iter()
            .map(|(v, ast)| format!("{} = {}", v, ast)).chain(
            self.or_asts.iter().map(|(v, asts)| format!("{} |= {}", v, asts.iter().map(|p| p.to_string()).join(" | "))).chain(
            self.not_asts.iter().map(|(v, ast)| format!("{} != {}", v, ast))
            )).join(", "))
    }
}

impl<L: Language> PrettyString for MultiPattern<L> {
    fn pretty_string(&self) -> String {
        format!("{}", self)
    }
}

impl<L: Language + 'static, N: Analysis<L>> ToDyn<L, N> for MultiPattern<L> {
    fn into_rc_dyn(self) -> Rc<dyn Searcher<L, N>> {
        Rc::new(self)
    }
}

impl<L: Language, A: Analysis<L>> Searcher<L, A> for MultiPattern<L> {
    fn search_eclass_with_limit(&self, egraph: &EGraph<L, A>, eclass: Id, limit: usize) -> Option<SearchMatches> {
        self.program.colored_run_with_limit(egraph, eclass, None, limit)
    }

    fn colored_search_eclass(&self, egraph: &EGraph<L, A>, eclass: Id, color: ColorId) -> Option<SearchMatches> {
        self.colored_search_eclass_with_limit(egraph, eclass, color, usize::MAX)
    }

    fn colored_search_eclass_with_limit(&self, egraph: &EGraph<L, A>, eclass: Id, color: ColorId, mut limit: usize) -> Option<SearchMatches> {
        let todo = egraph.get_color(color).unwrap().equality_class(egraph, eclass);
        let matches = todo.into_iter().fold(None, |acc: Option<SearchMatches>, id| {
            if let Some(new_matches) = self.program.colored_run_with_limit(egraph, id, Some(color), limit) {
                limit -= new_matches.total_substs();
                if let Some(acc) = acc {
                    Some(acc.merge(new_matches))
                } else {
                    Some(new_matches)
                }
            } else {
                acc
            }
        });
        
        if let Some(ref matches) = matches {
            dassert!(matches.matches.values().all(|v| v.iter().all(|s| s.color == Some(color))));
        }
        
        matches
    }

    fn vars(&self) -> Vec<Var> {
        let mut vars = vec![];
        for (v, pat) in &self.asts {
            vars.push(*v);
            for n in pat.as_ref() {
                if let ENodeOrVar::Var(v) = n {
                    vars.push(*v)
                }
            }
        }
        vars.sort();
        vars.dedup();
        vars
    }
}

impl<L: Language + 'static, A: Analysis<L> + 'static> Applier<L, A> for MultiPattern<L> {
    fn apply_matches(
        &self,
        egraph: &mut EGraph<L, A>,
        matches: &Option<SearchMatches>,
    ) -> Vec<Id> {
        // TODO explanations?
        // the ids returned are kinda garbage
        let mut added = vec![];
        if let Some(mat) = matches {
            for subst in mat.substs() {
                let mut subst = subst.clone();
                for (i, (v, p)) in self.asts.iter().enumerate() {
                    let id1 = pattern::apply_pat(p.as_ref(), egraph, &subst);
                    if let Some(id2) = subst.insert(*v, id1) {
                        egraph.opt_colored_union(subst.color, id1, id2);
                    }
                    if i == 0 {
                        added.push(id1)
                    }
                }
            }
        }
        added
    }

    fn apply_one(
        &self,
        _egraph: &mut EGraph<L, A>,
        _eclass: Id,
        _subst: &Subst,
    ) -> Vec<Id> {
        panic!("Multipatterns do not support apply_one")
    }

    fn vars(&self) -> Vec<Var> {
        let mut bound_vars: IndexSet<&Var> = IndexSet::default();
        let mut vars = vec![];
        for (bv, pat) in &self.asts {
            for n in pat.as_ref() {
                if let ENodeOrVar::Var(v) = n {
                    // using vars that are already bound doesn't count
                    if !bound_vars.contains(v) {
                        vars.push(*v)
                    }
                }
            }
            bound_vars.insert(bv);
        }
        vars.sort();
        vars.dedup();
        vars
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use crate::{SymbolLang as S, *};
    use crate::multipattern::MultiPattern;

    type EGraph = crate::EGraph<S, ()>;

    impl EGraph {
        fn add_string(&mut self, s: &str) -> Id {
            self.add_expr(&s.parse().unwrap())
        }
    }

    #[test]
    #[should_panic = "unbound var ?z"]
    fn bad_unbound_var() {
        let _: Rewrite<S, ()> = multi_rewrite!("foo"; "?x  = (foo ?y)" => "?x = ?z");
    }

    #[test]
    fn ok_unbound_var() {
        let _: Rewrite<S, ()> = multi_rewrite!("foo"; "?x = (foo ?y)" => "?z = (baz ?y), ?x = ?z");
    }

    #[test]
    fn multi_patterns() {
        init_logger();
        let mut egraph = EGraph::default();
        let _ = egraph.add_expr(&"(f a a)".parse().unwrap());
        let ab = egraph.add_expr(&"(f a b)".parse().unwrap());
        let ac = egraph.add_expr(&"(f a c)".parse().unwrap());
        egraph.union(ab, ac);
        egraph.rebuild();

        let n_matches = |multipattern: &str| -> usize {
            let mp: MultiPattern<S> = multipattern.parse().unwrap();
            mp.n_matches(&egraph)
        };

        assert_eq!(n_matches("?x = (f a a),   ?y = (f ?c b)"), 1);
        assert_eq!(n_matches("?x = (f a a),   ?y = (f a b)"), 1);
        assert_eq!(n_matches("?x = (f a a),   ?y = (f a a)"), 1);
        assert_eq!(n_matches("?x = (f ?a ?b), ?y = (f ?c ?d)"), 9);
        assert_eq!(n_matches("?x = (f ?a a),  ?y = (f ?a b)"), 1);

        assert_eq!(n_matches("?x = (f a a), ?x = (f a c)"), 0);
        assert_eq!(n_matches("?x = (f a b), ?x = (f a c)"), 1);
    }

    #[test]
    fn unbound_rhs() {
        let mut egraph = EGraph::default();
        let _x = egraph.add_expr(&"(x)".parse().unwrap());
        let rules = vec![
            // Rule creates y and z if they don't exist.
            multi_rewrite!("rule1"; "?x = (x)" => "?y = (y), ?y = (z)"),
            // Can't fire without above rule. `y` and `z` don't already exist in egraph
            multi_rewrite!("rule2"; "?x = (x), ?y = (y), ?z = (z)" => "?y = (y), ?y = (z)"),
        ];
        let mut runner = Runner::default().with_egraph(egraph).run(&rules);
        println!("{}", runner.egraph.dot().to_string());
        let y = runner.egraph.add_expr(&"(y)".parse().unwrap());
        let z = runner.egraph.add_expr(&"(z)".parse().unwrap());
        assert_eq!(runner.egraph.find(y), runner.egraph.find(z));
    }

    #[test]
    fn ctx_transfer() {
        let mut egraph = EGraph::default();
        egraph.add_string("(lte ctx1 ctx2)");
        egraph.add_string("(lte ctx2 ctx2)");
        egraph.add_string("(lte ctx1 ctx1)");
        let x2 = egraph.add_string("(tag x ctx2)");
        let y2 = egraph.add_string("(tag y ctx2)");
        let z2 = egraph.add_string("(tag z ctx2)");

        let x1 = egraph.add_string("(tag x ctx1)");
        let y1 = egraph.add_string("(tag y ctx1)");
        let z1 = egraph.add_string("(tag z ctx2)");
        egraph.union(x1, y1);
        egraph.union(y2, z2);
        let rules = vec![multi_rewrite!("context-transfer"; 
                     "?x = (tag ?a ?ctx1) = (tag ?b ?ctx1),
                      ?t = (lte ?ctx1 ?ctx2), 
                      ?a1 = (tag ?a ?ctx2), 
                      ?b1 = (tag ?b ?ctx2)" 
                      =>
                      "?a1 = ?b1")];
        let runner = Runner::default().with_egraph(egraph).run(&rules);
        assert_eq!(runner.egraph.find(x1), runner.egraph.find(y1));
        assert_eq!(runner.egraph.find(y2), runner.egraph.find(z2));

        assert_eq!(runner.egraph.find(x2), runner.egraph.find(y2));
        assert_eq!(runner.egraph.find(x2), runner.egraph.find(z2));

        assert_ne!(runner.egraph.find(y1), runner.egraph.find(z1));
        assert_ne!(runner.egraph.find(x1), runner.egraph.find(z1));
    }

    #[test]
    fn multipattern_works_middle_colored() {
        init_logger();
        let mut egraph = EGraph::default();
        let l = egraph.add_expr(&"l".parse().unwrap());
        let y = egraph.add_expr(&"y".parse().unwrap());
        let f = egraph.add_expr(&"(f (p x) l)".parse().unwrap());
        let g = egraph.add_expr(&"(g y (and b a))".parse().unwrap());
        egraph.rebuild();

        // Going to test 2 cases:
        // 1. "Big" pattern is colored as sub of small one
        // 2. "Small" pattern is colored as sub of big one

        let pattern: MultiPattern<SymbolLang> = "?x = (f ?a ?b), ?b = (g ?c (and ?d ?k))".parse().unwrap();
        let sms = pattern.search(&egraph);
        assert!(sms.is_none());

        let small_big_color = egraph.create_color(None);
        egraph.colored_union(small_big_color, l, g);
        egraph.rebuild();

        let sms = pattern.search(&egraph);
        assert!(sms.is_some());
        let sms = sms.unwrap();
        assert_eq!(sms.len(), 1);
        let sm = sms.first().unwrap();
        assert_eq!(sm.substs.len(), 1);
        assert_eq!(sm.substs.first().unwrap().color(), Some(small_big_color));
        assert!(sm.substs.first().unwrap().get("?a".parse().unwrap()).is_some());
        assert!(sm.substs.first().unwrap().get("?b".parse().unwrap()).is_some());
        assert!(sm.substs.first().unwrap().get("?c".parse().unwrap()).is_some());
        assert!(sm.substs.first().unwrap().get("?d".parse().unwrap()).is_some());
        assert!(sm.substs.first().unwrap().get("?k".parse().unwrap()).is_some());

        let big_small_color = egraph.create_color(None);
        egraph.colored_union(big_small_color, y, f);
        egraph.rebuild();

        let pattern2: MultiPattern<SymbolLang> = "?x = (f (p ?a) ?l), ?b = (g ?x (and ?d ?k))".parse().unwrap();
        let sms = pattern2.search(&egraph);
        assert!(sms.is_some());
        let sms = sms.unwrap();
        assert_eq!(sms.len(), 1);
        let subst = sms.iter()
            .flat_map(|sm| sm.substs)
            .filter(|subst| subst.color() == Some(big_small_color))
            .collect::<Vec<_>>();
        assert_eq!(subst.len(), 1);
        assert_eq!(subst[0].color(), Some(big_small_color));
    }

    #[test]
    fn test_search_colored_nodes() {
        init_logger();
        let mut egraph = EGraph::default();

        let k = egraph.add_expr(&"k".parse().unwrap());
        let _f = egraph.add_expr(&"(f (p k) l)".parse().unwrap());

        let color = egraph.create_color(None);
        let t = egraph.colored_add_expr(color, &"true".parse().unwrap());
        egraph.colored_union(color, k, t);
        egraph.rebuild();
        egraph.verify_colored_uf_minimal();

        let pattern: MultiPattern<SymbolLang> = "?x = (f (p ?y) ?b), ?y = true".parse().unwrap();
        let pattern2: MultiPattern<SymbolLang> = "?x = (f (p true) ?b)".parse().unwrap();
        let sms = pattern.search(&egraph);
        let sms2 = pattern2.search(&egraph);
        assert!(sms.is_some());
        assert!(sms2.is_some());
        let sms = sms.unwrap();
        let sms2 = sms2.unwrap();
        assert_eq!(sms.len(), 1);
        assert_eq!(sms2.len(), 1);
        let sm = sms.first().unwrap();
        let sm2 = sms2.first().unwrap();
        assert_eq!(sm.substs.len(), 1);
        assert_eq!(sm2.substs.len(), 1);
        assert_eq!(sm.substs.first().unwrap().color(), Some(color));
        assert_eq!(sm2.substs.first().unwrap().color(), Some(color));
    }

    // After it is a "black" match no more colored matches
    #[test]
    fn test_black_only_match() {
        init_logger();
        let mut egraph = EGraph::default();
        let yz = egraph.add_expr(&"(y z)".parse().unwrap());
        let t = egraph.add_expr(&"true".parse().unwrap());
        egraph.add_expr(&"(f (g x) true)".parse().unwrap());
        let color = egraph.create_color(None);
        egraph.colored_union(color, yz, t);
        egraph.verify_colored_uf_minimal();
        egraph.rebuild();
        egraph.verify_colored_uf_minimal();

        let pattern: MultiPattern<SymbolLang> = "?x = (f (g ?y) ?z), ?z = true".parse().unwrap();
        let sms = pattern.search(&egraph);
        assert!(sms.is_some());
        let sms = sms.unwrap();
        assert_eq!(sms.len(), 1, "sms: {:?}", sms);
        let sm = &sms.first().unwrap();
        assert_eq!(sm.substs.len(), 1);
        assert_eq!(sm.substs.first().unwrap().color(), None);
    }

    #[test]
    fn test_apply_colored_multi_match() {
        init_logger();
        let mut egraph = EGraph::default();

        let searcher = MultiPattern::from_str("?x = true, ?g = (f ?l ?k), ?k = (foo ?x)").unwrap();
        let applier: MultiPattern<SymbolLang> = MultiPattern::from_str("?x = false, ?k = ?g").unwrap();

        let root = egraph.add_expr(&"(f l (foo b))".parse().unwrap());
        let g = egraph.add_expr(&"(foo b)".parse().unwrap());
        let b = egraph.add_expr(&"b".parse().unwrap());
        let color = egraph.create_color(None);
        let t = egraph.colored_add_expr(color, &"true".parse().unwrap());
        egraph.colored_union(color, b, t);
        egraph.verify_colored_uf_minimal();
        egraph.rebuild();

        let sms = searcher.search(&egraph);
        assert!(sms.is_some());
        let sms = sms.unwrap();
        assert_eq!(sms.len(), 1);
        let sm = &sms.first().unwrap();
        assert_eq!(sm.substs.len(), 1);
        let first_subs = sm.substs.first().unwrap();
        assert_eq!(first_subs.color(), Some(color));
        assert!(first_subs.get("?k".parse().unwrap()).is_some());
        assert!(first_subs.get("?g".parse().unwrap()).is_some());
        let _matches = applier.apply_matches(&mut egraph, &Some(sms));
        egraph.verify_colored_uf_minimal();
        egraph.rebuild();
        egraph.verify_colored_uf_minimal();
        let colored_false_class = egraph.colored_add_expr(color, &"false".parse().unwrap());
        assert_eq!(egraph.colored_find(color, colored_false_class),
                   egraph.colored_find(color, t));
        assert_eq!(egraph.colored_find(color, root), egraph.colored_find(color, g));
        assert_ne!(egraph.find(root), egraph.find(g));
        assert_ne!(egraph.add_expr(&"false".parse().unwrap()), egraph.add_expr(&"true".parse().unwrap()));
    }

    #[test]
    fn test_not_exists_pattern() {
        init_logger();

        let pattern: MultiPattern<S> = "?v1 = x, ?v2 = y, ?v3 != z, ?v3 != w, ?v3 != p".parse().unwrap();

        let mut egraph = EGraph::default();
        let _x = egraph.add_expr(&"x".parse().unwrap());
        let _y = egraph.add_expr(&"y".parse().unwrap());
        egraph.rebuild();

        assert!(pattern.search(&egraph).is_some());

        let _z = egraph.add_expr(&"z".parse().unwrap());
        egraph.rebuild();

        assert!(pattern.search(&egraph).is_none());

        let pattern: MultiPattern<S> = "?v1 = x, ?v2 = (f ?y), ?v2 != w".parse().unwrap();
        let color = egraph.create_color(None);
        egraph.add_expr(&"(f y)".parse().unwrap());
        let p = egraph.colored_add_expr(color, &"(f p)".parse().unwrap());
        let w = egraph.colored_add_expr(color, &"w".parse().unwrap());
        egraph.rebuild();

        let sms = pattern.search(&egraph);
        assert!(sms.is_some());
        let sms = sms.unwrap();
        assert_eq!(sms.len(), 1);
        assert_eq!(sms.first().unwrap().substs.len(), 2);

        egraph.colored_union(color, p, w);
        egraph.rebuild();

        let sms = pattern.search(&egraph);
        assert!(sms.is_some());
        let sms = sms.unwrap();
        assert_eq!(sms.len(), 1);
        assert_eq!(sms.first().unwrap().substs.len(), 1);
    }

    #[test]
    fn test_not_layered_match() {
        init_logger();

        let pattern: MultiPattern<S> = "?v1 = (f ?x (g ?z ?w)), ?w != (cons ?l)".parse().unwrap();
        let mut egraph = EGraph::default();
        egraph.add_expr(&"(f X (g Z W))".parse().unwrap());
        egraph.add_expr(&"(f X (g Z (cons L)))".parse().unwrap());
        egraph.add_expr(&"(f X (g Z (cons (f z))))".parse().unwrap());
        let color = egraph.create_color(None);
        egraph.rebuild();

        let sms = pattern.search(&egraph);
        assert!(sms.is_some());
        let sms = sms.unwrap();
        assert_eq!(1, sms.len());
        assert_eq!(sms.first().unwrap().substs.len(), 1);

        let w = egraph.add_expr(&"W".parse().unwrap());
        let cons = egraph.add_expr(&"(cons L)".parse().unwrap());
        egraph.union(w, cons);
        egraph.rebuild();

        let sms = pattern.search(&egraph);
        assert!(sms.is_none());

        let _colored_fl = egraph.colored_add_expr(color, &"(f X (g Z L))".parse().unwrap());
        egraph.rebuild();
        let colored_l = egraph.colored_add_expr(color, &"L".parse().unwrap());
        egraph.rebuild();

        let sms = pattern.search(&egraph);
        assert!(sms.is_some());
        let sms = sms.unwrap();
        assert_eq!(1, sms.len());
        let first_match = sms.first().unwrap();
        assert_eq!(first_match.substs.len(), 1);
        let first_subs = first_match.substs.first().unwrap();
        assert_eq!(first_subs.get("?w".parse().unwrap()), Some(&colored_l));
        assert!(first_subs.color().is_some());

        egraph.colored_union(color, colored_l, cons);
        egraph.rebuild();

        let sms = pattern.search(&egraph);
        assert!(sms.is_none());
    }

    #[test]
    fn test_second_or_matches() {
        init_logger();

        let mut egraph = EGraph::default();
        let pattern: MultiPattern<S> = "?v1 = (f ?x (g ?z ?w)), ?w |= (cons ?l), ?w |= nil".parse().unwrap();
        egraph.add_expr(&"(f X (g Z W))".parse().unwrap());
        egraph.rebuild();
        assert!(pattern.search(&egraph).is_none());

        egraph.add_expr(&"(f X (g Z nil))".parse().unwrap());
        egraph.rebuild();
        let sms = pattern.search(&egraph).expect("should have found a match");
        assert_eq!(sms.len(), 1);
        assert_eq!(sms.first().unwrap().substs.len(), 1);

        let color = egraph.create_color(None);
        let cons = egraph.colored_add_expr(color, &"(cons JJH)".parse().unwrap());
        egraph.rebuild();
        let sms = pattern.search(&egraph).expect("should have found a match");
        assert_eq!(sms.len(), 1);
        assert_eq!(sms.first().unwrap().substs.len(), 1);

        let w = egraph.add_expr(&"W".parse().unwrap());
        egraph.colored_union(color, w, cons);
        egraph.rebuild();

        assert_eq!(pattern.search(&egraph).map_or(0, |sms| sms.total_substs()), 2);
    }
}
