use std::collections::HashSet;
use crate::{ImmutableCondition, Condition, EGraph, Var, Subst, Id, Language, Analysis, RcImmutableCondition, ToCondRc, ColorId};
use itertools::Itertools;
use log::{trace};
use crate::tools::tools::Grouped;

/// A condition that is true if all of the conditions are true.
pub struct AndCondition<L: Language, N: Analysis<L>> {
    conditions: Vec<RcImmutableCondition<L, N>>,
    #[cfg(debug_assertions)]
    #[allow(dead_code)]
    names: Vec<String>,
}

impl<L: Language, N: Analysis<L>> AndCondition<L, N> {
    /// Create a new AndCondition.
    pub fn new(conditions: Vec<RcImmutableCondition<L, N>>) -> AndCondition<L, N> {
        #[cfg(debug_assertions)]
        let names = conditions.iter().map(|c| c.describe()).collect();
        AndCondition {conditions, #[cfg(debug_assertions)] names}
    }
}

impl<L: Language, N: Analysis<L>> ToCondRc<L, N> for AndCondition<L, N> {}

impl<L: Language, N: Analysis<L>> ImmutableCondition<L, N> for AndCondition<L, N> {
    fn check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        trace!("AndCondition::{}({:?}, {:?} - Start)", self.describe(), eclass, subst);
        let res = self.conditions.iter().all(|c| c.check_imm(egraph, eclass, subst));
        trace!("AndCondition::{}({:?}, {:?}) = {:?}", self.describe(), eclass, subst, res);
        res
    }

    fn colored_check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        trace!("AndCondition::colored::{}({:?}, {:?} - Start)", self.describe(), eclass, subst);
        let res = self.conditions.iter()
            .map(|c|
                c.colored_check_imm(egraph, eclass, subst))
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
                            .filter(|(_c, v)| v.len() > 1)
                            .map(|(c, _)| c).collect_vec())
                    }
                }))).flatten();
        trace!("AndCondition::colored::{}({:?}, {:?}) = {:?}", self.describe(), eclass, subst, res);
        res
    }

    fn vars(&self) -> Vec<Var> {
        self.conditions.iter().flat_map(|c| c.vars()).unique().collect_vec()
    }

    fn describe(&self) -> String {
        format!("{}", self.conditions.iter().map(|x| x.describe()).join(" && "))
    }
}

/// A condition that is true if any of the conditions are true.
pub struct MutAndCondition<L: Language, N: Analysis<L>> {
    conditions: Vec<Box<dyn Condition<L, N>>>
}

impl<L: Language, N: Analysis<L>> MutAndCondition<L, N> {
    /// Create a new MutAndCondition.
    pub fn new(conditions: Vec<Box<dyn Condition<L, N>>>) -> MutAndCondition<L, N> {
        MutAndCondition {conditions}
    }
}

impl<L: Language, N: Analysis<L>> Condition<L, N> for MutAndCondition<L, N> {
    fn check(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        trace!("MutAndCondition::{}({:?}, {:?} - Start)", self.describe(), eclass, subst);
        let res = self.conditions.iter().all(|c| c.check(egraph, eclass, subst));
        trace!("MutAndCondition::{}({:?}, {:?}) = {:?}", self.describe(), eclass, subst, res);
        res
    }

    fn check_colored(&self, egraph: &mut EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        trace!("MutAndCondition::colored::{}({:?}, {:?} - Start)", self.describe(), eclass, subst);
        let res = self.conditions.iter()
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
                            .filter(|(_c, v)| v.len() > 1)
                            .map(|(c, _)| c).collect_vec())
                    }
                }))).flatten();
        trace!("MutAndCondition::colored::{}({:?}, {:?}) = {:?}", self.describe(), eclass, subst, res);
        res
    }

    fn vars(&self) -> Vec<Var> {
        self.conditions.iter().flat_map(|c| c.vars()).unique().collect_vec()
    }

    fn describe(&self) -> String {
        format!("{}", self.conditions.iter().map(|x| x.describe()).join(" && "))
    }
}

/// A condition that is true if any of the conditions are true.
pub struct OrCondition<L: Language, N: Analysis<L>> {
    conditions: Vec<RcImmutableCondition<L, N>>
}

impl<L: Language, N: Analysis<L>> OrCondition<L, N> {
    /// Create a new OrCondition.
    pub fn new(conditions: Vec<RcImmutableCondition<L, N>>) -> OrCondition<L, N> {
        OrCondition {conditions}
    }
}

impl<L: Language, N: Analysis<L>> ToCondRc<L, N> for OrCondition<L, N> {}

impl<L: Language, N: Analysis<L>> ImmutableCondition<L, N> for OrCondition<L, N> {
    fn check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        trace!("OrCondition::{}({:?}, {:?} - Start)", self.describe(), eclass, subst);
        let res = self.conditions.is_empty() || self.conditions.iter()
            .any(|c| c.check_imm(egraph, eclass, subst));
        trace!("OrCondition::{}({:?}, {:?}) = {:?}", self.describe(), eclass, subst, res);
        res
    }

    fn colored_check_imm(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
        if self.conditions.is_empty() {
            return Some(vec![]);
        }
        trace!("OrCondition::colored::{}({:?}, {:?} - Start)", self.describe(), eclass, subst);
        let mut collected = HashSet::<ColorId>::default();
        for r in self.conditions.iter().map(|c|
            c.colored_check_imm(egraph, eclass, subst)) {
            if let Some(v) = r {
                if v.is_empty() {
                    return Some(vec![]);
                }
                collected.extend(v);
            }
        }
        let res = (!collected.is_empty()).then(|| collected.into_iter().collect_vec());
        trace!("OrCondition::colored::{}({:?}, {:?}) = {:?}", self.describe(), eclass, subst, res);
        res
    }

    fn vars(&self) -> Vec<Var> {
        self.conditions.iter().flat_map(|c| c.vars()).unique().collect_vec()
    }

    fn describe(&self) -> String {
        format!("{}", self.conditions.iter().map(|x| x.describe()).join(" || "))
    }
}

#[cfg(test)]
mod test {
    use log::info;
    use crate::conditions::{AndCondition, OrCondition};
    use crate::searchers::{MatcherContainsCondition, PatternMatcher, ToRc, VarMatcher};
    use crate::{ColorId, EGraph, ImmutableCondition, init_logger, Pattern, Searcher, SymbolLang, ToCondRc};
    use crate::reconstruct::reconstruct;

    #[test]
    fn test_simple_and_cond() {
        init_logger();

        let cond1 = MatcherContainsCondition::new(
            PatternMatcher::new("false".parse().unwrap()).into_rc()).into_rc();
        let cond2 = MatcherContainsCondition::new(
            PatternMatcher::new("true".parse().unwrap()).into_rc()).into_rc();
        let condb_var = MatcherContainsCondition::new(
            VarMatcher::new("?b".parse().unwrap()).into_rc()).into_rc();
        let condb = MatcherContainsCondition::new(
            PatternMatcher::new("b".parse().unwrap()).into_rc()).into_rc();
        let fallacy = AndCondition::new(vec![cond1.clone(), cond2.clone()]);
        let btrue = AndCondition::new(vec![condb_var.clone(), cond2.clone()]);

        let mut egraph = EGraph::default();
        let t = egraph.add_expr(&"true".parse().unwrap());
        let f = egraph.add_expr(&"false".parse().unwrap());
        let b = egraph.add_expr(&"b".parse().unwrap());
        let _a = egraph.add_expr(&"a".parse().unwrap());
        let _c = egraph.add_expr(&"c".parse().unwrap());
        let _and_exp1 = egraph.add_expr(&"(and false true)".parse().unwrap());
        let _and_exp2 = egraph.add_expr(&"(and false false)".parse().unwrap());
        let and_exp3 = egraph.add_expr(&"(and true true)".parse().unwrap());
        egraph.union(and_exp3, t);
        let and_exp4 = egraph.add_expr(&"(and a b)".parse().unwrap());
        let and_exp5 = egraph.add_expr(&"(and b b)".parse().unwrap());
        egraph.union(and_exp5, b);
        let or_exp1 = egraph.add_expr(&"(or false true)".parse().unwrap());
        let or_exp2 = egraph.add_expr(&"(or false false)".parse().unwrap());
        let or_exp3 = egraph.add_expr(&"(or true true)".parse().unwrap());
        egraph.union(or_exp3, t);
        egraph.union(or_exp1, t);
        egraph.union(or_exp2, f);
        let _or_exp4 = egraph.add_expr(&"(or a b)".parse().unwrap());
        let or_exp5 = egraph.add_expr(&"(or b b)".parse().unwrap());
        egraph.union(or_exp5, b);
        egraph.rebuild();

        let t_pattern: Pattern<SymbolLang> = "true".parse().unwrap();
        let f_pattern = "false".parse().unwrap();
        let and_pattern: Pattern<SymbolLang> = "(and ?a ?b)".parse().unwrap();

        // Black tests
        for sms in vec![&t_pattern, &f_pattern, &and_pattern].iter().flat_map(|p| p.search(&egraph)) {
            for sbst in sms.substs {
                assert!(!fallacy.check_imm(&egraph, sms.eclass, &sbst));
            }
        }
        for sms in and_pattern.search(&egraph) {
            for sbst in sms.substs {
                assert!(btrue.check_imm(&egraph, sms.eclass, &sbst)
                    || sbst.get("?b".parse().unwrap()).unwrap() != &sms.eclass
                    || sbst.get("?b".parse().unwrap()) != Some(&t));
            }
        }

        // Now colored tests. I will add a color for weird stuff, and check matching succeeds as
        // expected.

        // A very bad color
        let fallacy_color = egraph.create_color();
        egraph.colored_union(fallacy_color, t, f);
        egraph.rebuild();
        assert!(f_pattern.search(&egraph).iter()
            .any(|sms| sms.substs.iter()
                .any(|sbst| {
                    let res = fallacy.colored_check_imm(&egraph, sms.eclass, sbst);
                    res.is_some() && !res.unwrap().is_empty()
                })), "Expecting fallacy in fallacy color");
        for sms in vec![&t_pattern, &f_pattern, &and_pattern].iter()
            .flat_map(|p| p.search(&egraph)) {
            for sbst in sms.substs {
                let colored_checked = fallacy.colored_check_imm(&egraph, sms.eclass, &sbst);
                assert!(colored_checked.is_none() || !colored_checked.as_ref().unwrap().is_empty());
                assert!(colored_checked.is_none() || {
                    let colors = colored_checked.unwrap();
                    (!colors.is_empty()) && {
                        let color = colors[0];
                        let fixed_eclass = egraph.colored_find(colors[0], sms.eclass);
                        fixed_eclass == egraph.colored_find(color, t) || fixed_eclass == egraph.colored_find(color, f)
                    }
                });
            }
        }

        // A nothing changed color
        let nop_color = egraph.create_color();
        egraph.colored_union(nop_color, or_exp5, and_exp5);
        egraph.rebuild();
        for sms in vec![&t_pattern, &f_pattern, &and_pattern].iter()
            .flat_map(|p| p.search(&egraph)) {
            for sbst in sms.substs {
                let colored_fallacies = fallacy.colored_check_imm(&egraph, sms.eclass, &sbst);
                let colored_btrues = btrue.colored_check_imm(&egraph, sms.eclass, &sbst);
                assert!(colored_fallacies.is_none() ||
                    !colored_fallacies.as_ref().unwrap().contains(&nop_color));
                assert!(colored_btrues.is_none() ||
                    !colored_fallacies.as_ref().unwrap().contains(&nop_color));
            }
        }

        let bfalse_color = egraph.create_color();
        egraph.colored_union(bfalse_color, b, f);
        let btrue_color = egraph.create_color();
        egraph.colored_union(btrue_color, b, t);
        egraph.rebuild();

        let b_or_true = OrCondition::new(vec![condb.clone(), cond2.clone()]);
        let b_or_false = OrCondition::new(vec![condb.clone(), cond1.clone()]);
        all_results_agree_with_color(&mut egraph, &t_pattern, vec![fallacy_color, btrue_color], &b_or_false);
        all_results_agree_with_color(&mut egraph, &f_pattern, vec![fallacy_color, bfalse_color], &b_or_true);

        // Finally - check pattern matching works as expected in conditions. I can do this by
        // checking the pattern `and a b`.
        let and_pattern: Pattern<SymbolLang> = "(and a ?b)".parse().unwrap();
        let and_cond = MatcherContainsCondition::new(PatternMatcher::new(and_pattern).into_rc());
        let and_false_andp = AndCondition::new(vec![cond1.clone(), and_cond.clone().into_rc()]);
        let and_color = egraph.create_color();
        egraph.colored_union(and_color, and_exp4, f);
        egraph.rebuild();
        all_results_agree_with_color(&mut egraph, &f_pattern, vec![and_color], &cond1);
        all_results_agree_with_color(&mut egraph, &f_pattern, vec![and_color], &and_cond);
        all_results_agree_with_color(&mut egraph, &f_pattern, vec![and_color], &and_false_andp);

        let or_false_andp = OrCondition::new(vec![cond1.clone(), and_cond.clone().into_rc()]);
        let or_color = egraph.create_color();
        egraph.colored_union(or_color, and_exp4, t);
        egraph.rebuild();
        all_results_agree_with_color(&mut egraph, &t_pattern, vec![fallacy_color, or_color], &or_false_andp);
    }

    fn all_results_agree_with_color(egraph: &mut EGraph<SymbolLang, ()>,
                                    pattern: &Pattern<SymbolLang>,
                                    checked_colors: Vec<ColorId>,
                                    cond: &dyn ImmutableCondition<SymbolLang, ()>) {
        let pattern_results = pattern.search(&egraph);
        pattern_results.iter().for_each(|sms| sms.substs.iter()
            .for_each(|sbst| {
                // TODO: Why is f_pattern returning a substitution with a color?
                if let Some(id) = sbst.color() {
                    if !checked_colors.contains(&id) {
                        return;
                    }
                }
                info!("Checking {:?} on eclass represented by {:?}", sbst, reconstruct(&egraph, sms.eclass, 2));
                let colors = cond.colored_check_imm(&egraph, sms.eclass, sbst);
                assert!(colors.is_some(), "No colored in check of cond {}", cond.describe());
                if let Some(inner) = colors.as_ref() {
                    if inner.is_empty() || sbst.color().is_some() {
                        return;
                    }
                }
                for c in &checked_colors {
                    assert!(colors.as_ref().unwrap().contains(&c),
                        "Color {} not in check of cond {}", c, cond.describe());
                }
            }));
        assert!(pattern.search(&egraph).iter().any(|sms| sms.substs.iter()
            .any(|sbst| {
                let colors = cond.colored_check_imm(&egraph, sms.eclass, sbst);
                colors.is_some() && !colors.as_ref().unwrap().is_empty()
            })), "Sanity check to see at least one non-trivial result for pattern {} and cond {}", pattern, cond.describe());
    }
}