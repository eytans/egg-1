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
    fn apply_matches(&self, egraph: &mut EGraph<SymbolLang, ()>, matches: &Option<SearchMatches>) -> Vec<Id> {
        let added = vec![];
        if let Some(mat) = matches {
            for (eclass, substs) in &mat.matches {
                for subst in substs {
                    let _ids = self.apply_one(egraph, *eclass, subst);
                }
            }
        }
        added
    }

    fn apply_one(&self, egraph: &mut EGraph<SymbolLang, ()>, eclass: Id, subst: &Subst) -> Vec<Id> {
        self.applier.apply_one(egraph, eclass, subst)
    }
}