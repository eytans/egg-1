use std::collections::HashSet;
use crate::{ImmutableCondition, Condition, EGraph, Var, Subst, Id, Language, Analysis, RcImmutableCondition, ToCondRc, ColorId};
use itertools::Itertools;
use std::fmt::Formatter;
use std::rc::Rc;
use crate::tools::tools::Grouped;

pub struct AndCondition<L: Language, N: Analysis<L>> {
    conditions: Vec<RcImmutableCondition<L, N>>
}

impl<L: Language, N: Analysis<L>> AndCondition<L, N> {
    pub fn new(conditions: Vec<RcImmutableCondition<L, N>>) -> AndCondition<L, N> {
        AndCondition {conditions}
    }
}

impl<L: Language, N: Analysis<L>> ToCondRc<L, N> for AndCondition<L, N> {}

impl<L: Language, N: Analysis<L>> ImmutableCondition<L, N> for AndCondition<L, N> {
    fn check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        self.conditions.iter().all(|c| c.check_imm(egraph, eclass, subst))
    }

    fn colored_check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        self.conditions.iter()
            .map(|c| c.colored_check_imm(egraph, eclass, subst))
            .fold1(|a, b| a.and_then(|x|
                b.and_then(|y| {
                    // If either is empty return the other. Otherwise, return the intersection.
                    if x.is_empty() {
                        Some(y)
                    } else if y.is_empty() {
                        Some(x)
                    } else {
                        Some(x.into_iter().chain(y.into_iter())
                            .grouped(|x| *x).into_iter()
                            .filter(|(c, v)| v.len() > 1)
                            .map(|(c, _)| c).collect_vec())
                    }
                }))).flatten()
    }

    fn vars(&self) -> Vec<Var> {
        self.conditions.iter().flat_map(|c| c.vars()).unique().collect_vec()
    }

    fn describe(&self) -> String {
        format!("{}", self.conditions.iter().map(|x| x.describe()).join(" && "))
    }
}

pub struct MutAndCondition<L: Language, N: Analysis<L>> {
    conditions: Vec<Box<dyn Condition<L, N>>>
}

impl<L: Language, N: Analysis<L>> MutAndCondition<L, N> {
    pub fn new(conditions: Vec<Box<dyn Condition<L, N>>>) -> MutAndCondition<L, N> {
        MutAndCondition {conditions}
    }
}

impl<L: Language, N: Analysis<L>> Condition<L, N> for MutAndCondition<L, N> {
    fn check(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        self.conditions.iter().all(|c| c.check(egraph, eclass, subst))
    }

    fn check_colored(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        self.conditions.iter()
            .map(|c| c.check_colored(egraph, eclass, subst))
            .fold1(|a, b| a.and_then(|x|
                b.and_then(|y| {
                    // If either is empty return the other. Otherwise, return the intersection.
                    if x.is_empty() {
                        Some(y)
                    } else if y.is_empty() {
                        Some(x)
                    } else {
                        Some(x.into_iter().chain(y.into_iter())
                            .grouped(|x| *x).into_iter()
                            .filter(|(c, v)| v.len() > 1)
                            .map(|(c, _)| c).collect_vec())
                    }
                }))).flatten()
    }

    fn vars(&self) -> Vec<Var> {
        self.conditions.iter().flat_map(|c| c.vars()).unique().collect_vec()
    }

    fn describe(&self) -> String {
        format!("{}", self.conditions.iter().map(|x| x.describe()).join(" && "))
    }
}

pub struct OrCondition<L: Language, N: Analysis<L>> {
    conditions: Vec<RcImmutableCondition<L, N>>
}

impl<L: Language, N: Analysis<L>> OrCondition<L, N> {
    pub fn new(conditions: Vec<RcImmutableCondition<L, N>>) -> OrCondition<L, N> {
        OrCondition {conditions}
    }
}

impl<L: Language, N: Analysis<L>> ToCondRc<L, N> for OrCondition<L, N> {}

impl<L: Language, N: Analysis<L>> ImmutableCondition<L, N> for OrCondition<L, N> {
    fn check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        self.conditions.is_empty() || self.conditions.iter()
            .any(|c| c.check_imm(egraph, eclass, subst))
    }

    fn colored_check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        if self.conditions.is_empty() {
            return Some(vec![]);
        }
        let mut collected = HashSet::<ColorId>::default();
        for r in self.conditions.iter().map(|c|
            c.colored_check_imm(egraph, eclass, subst)) {
            if let Some(v) = r {
                if (v.is_empty()) {
                    return Some(vec![]);
                }
                collected.extend(v);
            }
        }
        (!collected.is_empty()).then(|| collected.into_iter().collect_vec())
    }

    fn vars(&self) -> Vec<Var> {
        self.conditions.iter().flat_map(|c| c.vars()).unique().collect_vec()
    }

    fn describe(&self) -> String {
        format!("{}", self.conditions.iter().map(|x| x.describe()).join(" || "))
    }
}
