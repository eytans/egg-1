use std::collections::{BTreeMap, BTreeSet};
use log::*;
use std::convert::TryFrom;
use std::fmt;

use crate::{machine, Analysis, Applier, EGraph, Id, Language, RecExpr, Searcher, Subst, Var, OpId, ColorId, FromOp, RecExprParseError};
use std::fmt::Formatter;
use std::str::FromStr;
use invariants::dassert;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A pattern that can function as either a [`Searcher`] or [`Applier`].
///
/// A [`Pattern`] is essentially a for-all quantified expression with
/// [`Var`]s as the variables (in the logical sense).
///
/// When creating a [`Rewrite`], the most common thing to use as either
/// the left hand side (the [`Searcher`]) or the right hand side
/// (the [`Applier`]) is a [`Pattern`].
///
/// As a [`Searcher`], a [`Pattern`] does the intuitive
/// thing.
/// Here is a somewhat verbose formal-ish statement:
/// Searching for a pattern in an egraph yields substitutions
/// ([`Subst`]s) _s_ such that, for any _s'_—where instead of
/// mapping a variables to an eclass as _s_ does, _s'_ maps
/// a variable to an arbitrary expression represented by that
/// eclass—_p[s']_ (the pattern under substitution _s'_) is also
/// represented by the egraph.
///
/// As an [`Applier`], a [`Pattern`] performs the given substitution
/// and adds the result to the [`EGraph`].
///
/// Importantly, [`Pattern`] implements [`FromStr`] if the
/// [`Language`] does.
/// This is probably how you'll create most [`Pattern`]s.
///
/// ```
/// use egg::*;
/// define_language! {
///     enum Math {
///         Num(i32),
///         "+" = Add([Id; 2]),
///     }
/// }
///
/// let mut egraph = EGraph::<Math, ()>::default();
/// let a11 = egraph.add_expr(&"(+ 1 1)".parse().unwrap());
/// let a22 = egraph.add_expr(&"(+ 2 2)".parse().unwrap());
///
/// // use Var syntax (leading question mark) to get a
/// // variable in the Pattern
/// let same_add: Pattern<Math> = "(+ ?a ?a)".parse().unwrap();
///
/// // Rebuild before searching
/// egraph.rebuild();
///
/// // This is the search method from the Searcher trait
/// let matches = same_add.search(&egraph).unwrap();
/// let matched_eclasses: Vec<Id> = matches.iter().map(|m| m.eclass).collect();
/// assert_eq!(matched_eclasses, vec![a11, a22]);
/// ```
///
/// [`Pattern`]: struct.Pattern.html
/// [`Rewrite`]: struct.Rewrite.html
/// [`EGraph`]: struct.EGraph.html
/// [`Subst`]: struct.Subst.html
/// [`FromStr`]: https://doc.rust-lang.org/std/str/trait.FromStr.html
/// [`Var`]: struct.Var.html
/// [`Searcher`]: trait.Searcher.html
/// [`Applier`]: trait.Applier.html
/// [`Language`]: trait.Language.html
#[derive(Debug, PartialEq, Clone)]
pub struct Pattern<L> {
    /// The actual pattern as a [`RecExpr`](struct.RecExpr.html)
    pub ast: PatternAst<L>,
    program: machine::Program<L>,
}

/// A [`RecExpr`](struct.RecExpr.html) that represents a
/// [`Pattern`](struct.Pattern.html).
pub type PatternAst<L> = RecExpr<ENodeOrVar<L>>;

impl<L: Language> Pattern<L> {
    /// Returns a list of the [`Var`](struct.Var.html)s in this pattern.
    pub fn vars(&self) -> Vec<Var> {
        let mut vars = vec![];
        for n in self.ast.as_ref() {
            match n {
                ENodeOrVar::ENode(_, Some(n)) => {
                    let v = Var::from_str(n).unwrap();
                    if !vars.contains(&v) {
                        vars.push(v)
                    }
                }
                ENodeOrVar::Var(v) => {
                    if !vars.contains(v) {
                        vars.push(*v)
                    }
                }
                _ => {}
            }
        }
        vars
    }

    /// Pretty print this pattern as a sexp with the given width
    pub fn pretty(&self, width: usize) -> String {
        self.ast.pretty(width)
    }
}

/// The language of [`Pattern`]s.
///
/// [`Pattern`]: struct.Pattern.html
#[derive(Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord)]
pub enum ENodeOrVar<L> {
    /// An enode from the underlying [`Language`](trait.Language.html)
    ENode(L, Option<String>),
    /// A pattern variable
    Var(Var),
}

impl<L: Language> Language for ENodeOrVar<L> {
    fn op_id(&self) -> OpId {
        panic!("Should never call this")
    }

    fn children(&self) -> &[Id] {
        match self {
            ENodeOrVar::ENode(e, _) => e.children(),
            ENodeOrVar::Var(_) => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            ENodeOrVar::ENode(e, _) => e.children_mut(),
            ENodeOrVar::Var(_) => &mut [],
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Should never call this")
    }

    fn display_op(&self) -> &dyn std::fmt::Display {
        match self {
            ENodeOrVar::ENode(e, _) => e.display_op(),
            ENodeOrVar::Var(v) => v,
        }
    }

    fn from_op_str(op_str: &str, children: Vec<Id>) -> Result<Self, String> {
        if op_str.starts_with('?') && op_str.len() > 1 {
            if children.is_empty() {
                op_str
                    .parse()
                    .map(ENodeOrVar::Var)
                    .map_err(|err| format!("Failed to parse var: {}", err))
            } else {
                Err(format!(
                    "Tried to parse pattern variable '{}' in the op position",
                    op_str
                ))
            }
        } else if op_str.starts_with("|@|") && op_str.matches("|@|").count() >= 2 {
            let matches = op_str.match_indices("|@|").collect_vec();
            // name between first two |@|
            let name = &op_str[matches[0].0 + 3..matches[1].0];
            if !name.starts_with("?") {
                return Err(format!("Invalid name for pattern: {}. Name should start with ?", name));
            }
            let l = L::from_op_str(&op_str[matches[1].0 + 3..], children)?;
            Ok(ENodeOrVar::ENode(l, Some(name.to_string())))
        } else {
            L::from_op_str(op_str, children).map(|x| ENodeOrVar::ENode(x, None))
        }
    }
}
#[derive(Debug, Error)]
pub enum ENodeOrVarParseError<E> {
    #[error(transparent)]
    BadVar(<Var as FromStr>::Err),

    #[error("tried to parse pattern variable {0:?} as an operator")]
    UnexpectedVar(String),

    #[error(transparent)]
    BadOp(E),
}

impl<L: FromOp> FromOp for ENodeOrVar<L> {
    type Error = ENodeOrVarParseError<L::Error>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        use ENodeOrVarParseError::*;

        if op.starts_with('?') && op.len() > 1 {
            if children.is_empty() {
                op.parse().map(Self::Var).map_err(BadVar)
            } else {
                Err(UnexpectedVar(op.to_owned()))
            }
        } else {
            L::from_op(op, children).map(|x| ENodeOrVar::ENode(x, None)).map_err(BadOp)
        }
    }
}

impl<L: FromOp> std::str::FromStr for Pattern<L> {
    type Err = RecExprParseError<ENodeOrVarParseError<L::Error>>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PatternAst::from_str(s).map(Self::from)
    }
}

impl<'a, L: Language> From<&'a [L]> for Pattern<L> {
    fn from(expr: &'a [L]) -> Self {
        let nodes: Vec<_> = expr.iter().cloned().map(|x| ENodeOrVar::ENode(x, None)).collect();
        let ast = RecExpr::from(nodes);
        Self::from(ast)
    }
}

impl<'a, L: Language> From<PatternAst<L>> for Pattern<L> {
    fn from(ast: PatternAst<L>) -> Self {
        let program = machine::Program::compile_from_pat(&ast);
        Pattern { ast, program }
    }
}

impl<L: Language> TryFrom<Pattern<L>> for RecExpr<L> {
    type Error = Var;
    fn try_from(pat: Pattern<L>) -> Result<Self, Self::Error> {
        let nodes = pat.ast.as_ref().iter().cloned();
        let ns: Result<Vec<_>, _> = nodes
            .map(|n| match n {
                ENodeOrVar::ENode(n, _) => Ok(n),
                ENodeOrVar::Var(v) => Err(v),
            })
            .collect();
        ns.map(RecExpr::from)
    }
}

impl<L: Language> fmt::Display for Pattern<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ast)
    }
}

pub struct Matches<'a> {
    pub eclass: Id,
    pub substs: &'a BTreeSet<Subst>,
}

impl<'a> Matches<'a> {
    pub fn new(eclass: Id, substs: &'a BTreeSet<Subst>) -> Self {
        Self {eclass, substs}
    }
}

/// The result of searching a [`Searcher`] over one eclass.
///
/// Note that one [`SearchMatches`] can contain many found
/// substititions. So taking the length of a list of [`SearchMatches`]
/// tells you how many eclasses something was matched in, _not_ how
/// many matches were found total.
///
/// [`SearchMatches`]: struct.SearchMatches.html
/// [`Searcher`]: trait.Searcher.html
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchMatches {
    /// Mapping of eclasses to their matches.
    pub matches: BTreeMap<Id, BTreeSet<Subst>>,
}

impl SearchMatches {
    fn merge(self, other: Self) -> Self {
        let mut matches = self.matches;
        for (eclass, substs) in other.matches {
            matches.entry(eclass).or_default().extend(substs);
        }
        Self { matches }
    }

    pub fn substs(&self) -> impl Iterator<Item = &Subst> {
        self.matches.values().map(|s| s.iter()).flatten()
    }

    pub fn len(&self) -> usize {
        self.matches.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = Matches> {
        self.matches.iter().map(|(eclass, subs)| Matches::new(*eclass, subs))
    }

    pub fn first(&self) -> Option<Matches> {
        self.iter().next()
    }

    pub(crate) fn total_substs(&self) -> usize {
        self.matches.values().map(|x| x.len()).sum()
    }

    pub fn merge_matches(res: Vec<SearchMatches>) -> Option<SearchMatches> {
        res.into_iter().fold(None, |acc, x| {
            if let Some(acc) = acc {
                Some(acc.merge(x))
            } else {
                Some(x)
            }
        })
    }

    pub fn collect_matches<L: Language, A: Analysis<L>>(egraph: &EGraph<L, A>, eclass: Id, substs: Vec<Subst>) -> SearchMatches {
        let mut matches: BTreeMap<Id, BTreeSet<Subst>> = BTreeMap::default();
        for mut s in substs {
            s.fix(egraph);
            matches.entry(egraph.opt_colored_find(s.color(), eclass)).or_default().insert(s);
        }
        let sms = SearchMatches { matches };
        sms
    }
}

impl<L: Language, A: Analysis<L>> Searcher<L, A> for Pattern<L> {
    fn search_eclass(&self, egraph: &EGraph<L, A>, eclass: Id) -> Option<SearchMatches> {
        let substs = if cfg!(feature = "colored") {
            self.program.colored_run(egraph, eclass, None)
        }  else {
            self.program.run(egraph, eclass)
        };
        if substs.is_empty() {
            None
        } else {
            let sms = SearchMatches::collect_matches(egraph, eclass, substs);
            Some(sms)
        }
    }

    fn search(&self, egraph: &EGraph<L, A>) -> Option<SearchMatches> {
        let res = match self.ast.as_ref().last().unwrap() {
            ENodeOrVar::ENode(e, _) => {
                let key = e.op_id();
                match egraph.classes_by_op.get(&key) {
                    None => vec![],
                    Some(ids) => ids
                        .iter()
                        .filter_map(|&id| self.search_eclass(egraph, id))
                        .collect(),
                }
            }
            ENodeOrVar::Var(_) => egraph
                .classes()
                .filter_map(|e| self.search_eclass(egraph, e.id))
                .collect(),
        };
        SearchMatches::merge_matches(res)
    }

    /// Searches all equivalent EClasses under the colored assumption. Returns all results under
    /// the representative of eclass in color.
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
        let matches = SearchMatches::collect_matches(egraph, eclass, res);
        dassert!(matches.matches.values().all(|v| v.iter().all(|s| s.color == Some(color))));
        if matches.substs().next().is_some() {
            Some(matches)
        } else {
            None
        }
    }

    fn vars(&self) -> Vec<Var> {
        Pattern::vars(self)
    }
}

impl fmt::Display for SearchMatches {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "SearchMatches({})", self.matches.iter().map(|(class, subs)| format!("class {:?}: {}", class, subs.iter().join(", "))).join(", "))
    }
}
impl<L, A> Applier<L, A> for Pattern<L>
where
    L: Language + 'static,
    A: Analysis<L> + 'static,
{
    fn apply_one(&self, egraph: &mut EGraph<L, A>, _: Id, subst: &Subst) -> Vec<Id> {
        let id = apply_pat(self.ast.as_ref(), egraph, subst);
        vec![id]
    }

    fn vars(&self) -> Vec<Var> {
        Pattern::vars(self)
    }
}

pub(crate) fn apply_pat<L: Language, A: Analysis<L>>(
    pat: &[ENodeOrVar<L>],
    egraph: &mut EGraph<L, A>,
    subst: &Subst,
) -> Id {
    trace!("apply_rec {:2?} {:?}", pat, subst);

    let result = match pat.last().unwrap() {
        ENodeOrVar::Var(w) => egraph.opt_colored_find(subst.color, subst[*w]),
        ENodeOrVar::ENode(e, _) => {
            let n = e
                .clone()
                .map_children(|child| apply_pat(&pat[..usize::from(child) + 1], egraph, subst));
            trace!("adding: {:?}", n);
            if let Some(c) = subst.color {
                egraph.colored_add(c, n)
            } else {
                egraph.add(n)
            }
        }
    };

    trace!("result: {:?}", result);
    result
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use crate::{SymbolLang as S, *};
    use crate::multipattern::MultiPattern;

    type EGraph = crate::EGraph<S, ()>;

    #[test]
    fn simple_match() {
        crate::init_logger();
        let mut egraph = EGraph::default();

        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));
        let plus = egraph.add(S::new("+", vec![x, y]));

        let z = egraph.add(S::leaf("z"));
        let w = egraph.add(S::leaf("w"));
        let plus2 = egraph.add(S::new("+", vec![z, w]));

        egraph.union(plus, plus2);
        egraph.rebuild();

        let commute_plus = rewrite!(
            "commute_plus";
            "(+ ?a ?b)" => "(+ ?b ?a)"
        );

        let matches = commute_plus.search(&egraph);
        let n_matches: usize = matches.iter().map(|m| m.substs().count()).sum();
        assert_eq!(n_matches, 2, "matches is wrong: {:#?}", matches);

        let applications = commute_plus.apply(&mut egraph, &matches);
        egraph.rebuild();
        assert_eq!(applications.len(), 2);

        let actual_substs: Vec<Subst> = matches.as_ref().unwrap().substs().cloned().collect();

        println!("Here are the substs!");
        for m in &actual_substs {
            println!("substs: {:?}", m);
        }

        egraph.dot().to_dot("target/simple-match.dot").unwrap();

        use crate::extract::{AstSize, Extractor};

        let mut ext = Extractor::new(&egraph, AstSize);
        let (_, best) = ext.find_best(plus);
        eprintln!("Best: {:#?}", best);
    }

    #[test]
    #[cfg(feature = "colored")]
    fn single_colored_find() {
        crate::init_logger();
        let mut egraph = EGraph::default();

        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));
        let _plus = egraph.add(S::new("+", vec![x, y]));

        let z = egraph.add(S::leaf("z"));
        let w = egraph.add(S::leaf("w"));
        let _plus2 = egraph.add(S::new("+", vec![z, w]));

        let c = egraph.create_color();
        egraph.colored_union(c, y, z);
        egraph.rebuild();

        let commute_plus = rewrite!(
            "commute_plus";
            "(+ x z)" => "(+ x x)"
        );

        let matches = commute_plus.search(&egraph);
        assert!(matches.is_some());
        assert!(matches.unwrap().substs().all(|s| !s.color.is_none()));
    }

    #[test]
    fn named_subpattern_is_var() {
        crate::init_logger();
        let p: MultiPattern<SymbolLang> = MultiPattern::from_str("?root = (+ ?x ?y)").unwrap();
        assert_eq!(Searcher::<SymbolLang, ()>::vars(&p).len(), 3);
    }

    #[test]
    fn name_enode_matches_correctly() {
        crate::init_logger();
        let p: MultiPattern<SymbolLang> = "?root = (+ ?x ?y)".parse().unwrap();
        let mut egraph = EGraph::default();
        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));
        let plus = egraph.add(S::new("+", vec![x, y]));
        egraph.rebuild();
        let matches = p.search(&egraph);
        assert!(matches.is_some());
        let mut matches = matches.unwrap();
        assert_eq!(matches.len(), 1);
        let m = &matches.matches.pop_first().unwrap();
        assert_eq!(m.0, plus);
        assert_eq!(m.1.len(), 1);
        let s = &m.1.first().unwrap();
        assert_eq!(s[Var::from_str("?x").unwrap()], x);
        assert_eq!(s[Var::from_str("?y").unwrap()], y);
        assert_eq!(s[Var::from_str("?root").unwrap()], plus);
    }

    #[test]
    #[cfg(feature = "colored")]
    fn colored_eq_x_x() {
        crate::init_logger();
        let mut egraph = EGraph::default();
        let x = egraph.add(S::leaf("x"));
        let z = egraph.add(S::leaf("z"));
        let y = egraph.add(S::leaf("y"));
        let equ = egraph.add(S::new("=", vec![z, y]));
        let c = egraph.create_color();
        egraph.colored_union(c, x, y);
        egraph.colored_union(c, x, z);
        egraph.rebuild();
        let p = Pattern::from_str("(= ?x ?x)").unwrap();
        let matches = p.search(&egraph);
        assert!(matches.is_some());
        let matches = matches.unwrap();
        assert_eq!(matches.len(), 1);
        let m = &matches.first().unwrap();
        assert_eq!(m.eclass, equ);
        assert_eq!(m.substs.len(), 1);
    }

    #[test]
    #[cfg(feature = "colored")]
    fn colored_eclass_search_sanity() {
        // Create an egraph with x and colored f(x) merged with black y
        crate::init_logger();
        let mut egraph = EGraph::default();
        let x = egraph.add(S::leaf("x"));
        let y = egraph.add(S::leaf("y"));
        let c = egraph.create_color();
        let fx = egraph.colored_add(c, S::new("f", vec![x]));
        egraph.colored_union(c, fx, y);
        egraph.rebuild();

        // Search for f(?z) and find it!
        let p_f_z = Pattern::from_str("(f ?z)").unwrap();
        let matches = p_f_z.colored_search_eclass(&egraph, y, c);
        assert!(matches.is_some());
        let matches = matches.unwrap();
        assert_eq!(matches.len(), 1);
        let matches = matches.first().unwrap();
        assert_eq!(matches.substs.len(), 1);
        assert_eq!(matches.substs.first().unwrap()[Var::from_str("?z").unwrap()], x);
        assert_eq!(egraph.colored_find(c, matches.eclass), egraph.colored_find(c, y));
    }
}
