use crate::{EGraph, Id, Pattern, Searcher, SearchMatches, Subst, Var, Language, Analysis, ColorId};
use itertools::{Itertools};

use smallvec::alloc::fmt::Formatter;
use std::rc::Rc;

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

    use crate::{EGraph, RecExpr, Searcher, SymbolLang, MultiPattern, init_logger};
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