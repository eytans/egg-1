use std::fmt::{Display, Formatter};
use std::str::FromStr;
use indexmap::IndexSet;
use itertools::Itertools;
use thiserror::Error;

use crate::*;

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
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MultiPattern<L> {
    asts: Vec<(Var, PatternAst<L>)>,
    program: machine::Program<L>,
}

impl<L: Language> MultiPattern<L> {
    /// Creates a new multipattern, binding the given patterns to the corresponding variables.
    ///
    /// ```
    /// use egg::*;
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
        let program = machine::Program::compile_from_multi_pat(&asts);
        Self { asts, program }
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
        for split in s.trim().split(',') {
            let split = split.trim();
            if split.is_empty() {
                continue;
            }
            let mut parts = split.split('=');
            let vs: &str = parts
                .next()
                .ok_or_else(|| PatternAssignmentError(split.to_string()))?;
            let v: Var = vs.trim().parse().map_err(VariableError)?;
            let ps = parts
                .map(|p| p.trim().parse())
                .collect::<Result<Vec<PatternAst<L>>, _>>()
                .map_err(PatternParseError)?;
            if ps.is_empty() {
                return Err(PatternAssignmentError(split.to_string()));
            }
            asts.extend(ps.into_iter().map(|p| (v, p)))
        }
        Ok(MultiPattern::new(asts))
    }
}

impl<L: Language> Display for MultiPattern<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.to_string().fmt(f)
    }
}

impl<L: Language, A: Analysis<L>> Searcher<L, A> for MultiPattern<L> {
    fn search_eclass(
        &self,
        egraph: &EGraph<L, A>,
        eclass: Id,
    ) -> Option<SearchMatches> {
        let substs = self.program.colored_run(egraph, eclass, None);
        if substs.is_empty() {
            None
        } else {
            Some(SearchMatches {
                eclass,
                substs,
            })
        }
    }

    fn colored_search_eclass(&self, egraph: &EGraph<L, A>, eclass: Id, color: ColorId) -> Option<SearchMatches> {
        let eq_classes = egraph.get_color(color).unwrap().black_ids(egraph, eclass);
        let todo: Box<dyn Iterator<Item=Id>> = if let Some(ids) = eq_classes {
            Box::new(ids.iter().copied())
        } else {
            Box::new(std::iter::once(eclass))
        };
        let mut res = vec![];
        for id in todo {
            let substs = self.program.colored_run(egraph, id, Some(color));
            if !substs.is_empty() {
                res.extend(substs)
            }
        }
        let matches = SearchMatches { eclass: egraph.colored_find(color, eclass), substs: res.into_iter().unique().collect_vec() };
        (!matches.substs.is_empty()).then(|| matches)
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
        matches: &[SearchMatches],
    ) -> Vec<Id> {
        // TODO explanations?
        // the ids returned are kinda garbage
        let mut added = vec![];
        for mat in matches {
            for subst in &mat.substs {
                let mut subst = subst.clone();
                for (i, (v, p)) in self.asts.iter().enumerate() {
                    let id1 = crate::pattern::apply_pat(p.as_ref(), egraph, &subst);
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
        let _: Rewrite<S, ()> = multi_rewrite!("foo"; "?x = (foo ?y)" => "?x = ?z");
    }

    #[test]
    fn ok_unbound_var() {
        let _: Rewrite<S, ()> = multi_rewrite!("foo"; "?x = (foo ?y)" => "?z = (baz ?y), ?x = ?z");
    }

    #[test]
    fn multi_patterns() {
        crate::init_logger();
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
        assert!(sms.is_empty());

        let small_big_color = egraph.create_color();
        egraph.colored_union(small_big_color, l, g);
        egraph.rebuild();

        let sms = pattern.search(&egraph);
        assert_eq!(sms.len(), 1);
        let sm = &sms[0];
        assert_eq!(sm.substs.len(), 1);
        assert_eq!(sm.substs[0].color(), Some(small_big_color));

        let big_small_color = egraph.create_color();
        egraph.colored_union(big_small_color, y, f);
        egraph.rebuild();

        let pattern2: MultiPattern<SymbolLang> = "?x = (f (p ?a) ?l), ?b = (g ?x (and ?d ?k))".parse().unwrap();
        let sms = pattern2.search(&egraph);
        assert_eq!(sms.len(), 1);
        let subst = sms.into_iter()
            .flat_map(|sm| sm.substs)
            .filter(|subst| subst.color() == Some(big_small_color))
            .collect::<Vec<_>>();
        assert_eq!(subst.len(), 1);
        assert_eq!(subst[0].color(), Some(big_small_color));
    }

    // Tests to do:
    // multiPattern matches over colored nodes
    // After it is a "black" match no more colored matches
    // Colored Applier tests
}
