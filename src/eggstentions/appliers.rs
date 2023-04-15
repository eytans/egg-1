use crate::{Analysis, Applier, EGraph, Id, Language, Pattern, SearchMatches, Subst, SymbolLang, Var};
use itertools::Itertools;
use std::fmt::Formatter;

/// A wrapper around an Applier that applies the applier to all matches.
pub struct DiffApplier<T: Applier<SymbolLang, ()>> {
    applier: T
}

impl<T: Applier<SymbolLang, ()>> DiffApplier<T> {
    /// Create a new DiffApplier.
    pub fn new(applier: T) -> DiffApplier<T> {
        DiffApplier { applier }
    }
}

impl DiffApplier<Pattern<SymbolLang>> {
    /// Returns a string representation of the pattern.
    pub fn pretty(&self, width: usize) -> String {
        self.applier.pretty(width)
    }
}

impl<T: Applier<SymbolLang, ()>> std::fmt::Display for DiffApplier<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "-|> {}", self.applier)
    }
}

impl<T: Applier<SymbolLang, ()>> Applier<SymbolLang, ()> for DiffApplier<T> {
    fn apply_matches(&self, egraph: &mut EGraph<SymbolLang, ()>, matches: &[SearchMatches]) -> Vec<Id> {
        let added = vec![];
        for mat in matches {
            for subst in &mat.substs {
                let _ids = self.apply_one(egraph, mat.eclass, subst);
            }
        }
        added
    }

    fn apply_one(&self, egraph: &mut EGraph<SymbolLang, ()>, eclass: Id, subst: &Subst) -> Vec<Id> {
        self.applier.apply_one(egraph, eclass, subst)
    }
}

/// A special applier that will run union for `vars`.
pub struct UnionApplier {
    vars: Vec<Var>,
}

impl UnionApplier {
    /// Create a new UnionApplier.
    pub fn new(vars: Vec<Var>) -> UnionApplier {
        UnionApplier{vars}
    }
}

impl std::fmt::Display for UnionApplier {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "-> {}", self.vars.iter().map(|x| x.to_string()).join(" =:= "))
    }
}

impl<L: Language + 'static, N: Analysis<L> + 'static> Applier<L, N> for UnionApplier {
    fn apply_matches(&self, egraph: &mut EGraph<L, N>, matches: &[SearchMatches]) -> Vec<Id> {
        let mut added = vec![];
        for mat in matches {
            for subst in &mat.substs {
                let first = self.vars.first().unwrap();
                let ids = self.vars.iter().skip(1).filter_map(|v| {
                    let (to, did_something) = egraph.opt_colored_union(subst.color(), *subst.get(*first).unwrap(), *subst.get(*v).unwrap());
                    if did_something {
                        Some(to)
                    } else {
                        None
                    }
                    }).collect_vec();
                added.extend(ids)
            }
        }
        added
    }

    fn apply_one(&self, _egraph: &mut EGraph<L, N>, _eclass: Id, _subst: &Subst) -> Vec<Id> {
        unimplemented!()
    }


    fn vars(&self) -> Vec<Var> {
        self.vars.clone()
    }
}