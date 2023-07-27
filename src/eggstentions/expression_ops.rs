use std::collections::BTreeSet;
use std::fmt::Formatter;
use std::hash::{Hash, Hasher};

use crate::{EGraph, Id, Language, RecExpr};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use smallvec::alloc::fmt::Display;

/// A wrapper arround RecExp t omake it easier to use.
#[derive(Clone, Debug)]
pub struct RecExpSlice<'a, L: Language> {
    index: usize,
    exp: &'a RecExpr<L>
}

impl<'a, L: Language> RecExpSlice<'a, L> {
    /// Create a new RecExpSlice.
    pub fn new(index: usize, exp: &'a RecExpr<L>) -> RecExpSlice<'a, L> {
        RecExpSlice{index, exp}
    }

    /// Adds expression to the EGraph `graph` and returns the root of the expression.
    pub fn add_to_graph(&self, graph: &mut EGraph<L, ()>) -> Id {
        graph.add_expr(&RecExpr::from(self.exp.as_ref()[..self.index+1].iter().cloned().collect_vec()))
    }

    /// Returns a string representation that is easier to parse.
    pub fn to_spaceless_string(&self) -> String {
        self.to_sexp_string()
            .replace(" ", "_")
            .replace("(", "PO")
            .replace(")", "PC")
            .replace("->", "fn")
    }

    /// Returns a sexp string representation of the expression.
    pub fn to_sexp_string(&self) -> String {
        if self.is_leaf() {
            format!("{}", self.root().display_op().to_string())
        } else {
            format!("({} {})", self.root().display_op().to_string(), 
            itertools::Itertools::intersperse(self.children().iter().map(|t| t.to_sexp_string()), " ".to_string()).collect::<String>())
        }
    }

    /// Recreates the expression from the slice, but without any dangling children.
    pub fn to_clean_exp(&self) -> RecExpr<L> {
        fn add_to_exp<'a, L: Language>(expr: &mut Vec<L>, child: &RecExpSlice<'a, L>) -> Id {
            let children = child.children();
            let mut rec_res = children.iter().map(|c| add_to_exp(expr, c));
            let mut root = child.root().clone();
            root.update_children(|_id| rec_res.next().unwrap());
            expr.push(root);
            Id::from(expr.len() - 1)
        }

        let mut exp = vec![];
        add_to_exp(&mut exp, self);
        debug_assert_eq!(exp.iter().flat_map(|x| x.children()).count(),
                         exp.iter().flat_map(|x| x.children()).unique().count());
        RecExpr::from(exp)
    }
}

impl<'a, L: Language> PartialEq for RecExpSlice<'a, L> {
    fn eq(&self, other: &Self) -> bool {
        self.root() == other.root() && self.children() == other.children()
    }
}

impl<'a, L: Language> Hash for RecExpSlice<'a, L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.root().hash(state), self.children().hash(state)).hash(state)
    }
}

impl<'a, L: Language> From<&'a RecExpr<L>> for RecExpSlice<'a, L> {
    fn from(expr: &'a RecExpr<L>) -> Self {
        RecExpSlice{index: expr.as_ref().len() - 1, exp: expr}
    }
}

impl<'a, L: Language + Clone> From<&'a RecExpSlice<'a, L>> for RecExpr<L> {
    fn from(expr: &'a RecExpSlice<'a, L>) -> Self {
        // Need to remove unneeded nodes because recexpr comparison works straigt on vec
        let mut nodes: Vec<RecExpSlice<L>> = vec![];
        nodes.push(expr.clone());
        let mut indices = IndexSet::new();
        while !nodes.is_empty() {
            let current = nodes.pop().unwrap();
            indices.insert(current.index);
            for n in current.children() {
                nodes.push(n);
            }
        }
        let mut res: Vec<L> = vec![];
        let mut id_trans: IndexMap<Id, Id> = IndexMap::new();
        for i in indices.iter().sorted() {
            id_trans.insert(Id::from(*i), Id::from(res.len()));
            res.push(expr.exp.as_ref()[*i].clone().map_children(|id| *id_trans.get(&id).unwrap()));
        }
        RecExpr::from(res)
    }
}

impl<'a, L: Language> Into<RecExpr<L>> for RecExpSlice<'a, L> {
    fn into(self) -> RecExpr<L> {
        RecExpr::from(self.exp.as_ref()[..self.index + 1].iter().cloned().collect_vec())
    }
}

/// Trait to wrap a RecExpr like object into a RecExpSlice.
pub trait IntoTree<'a, T: Language> {
    /// Wraps the object into a RecExpSlice.
    fn into_tree(&'a self) -> RecExpSlice<'a, T>;
}

impl<'a, T: Language> IntoTree<'a, T> for RecExpr<T> {
    fn into_tree(&'a self) -> RecExpSlice<'a, T> {
        RecExpSlice::from(self)
    }
}

/// A trait for objects that can be used as trees.
pub trait Tree<'a, T: 'a + Language> {
    /// Returns the root of the tree.
    fn root(&self) -> &'a T;

    /// Returns the children (subtrees) of the root of the tree.
    fn children(&self) -> Vec<RecExpSlice<'a, T>>;

    /// Returns true if the tree is a leaf.
    fn is_leaf(&self) -> bool {
        self.children().is_empty()
    }

    /// Returns true if the root of the tree is a hole. Decide if a hole is a hole by checking if the
    /// display op starts with a question mark.
    fn is_root_hole(&self) -> bool {
        self.root().display_op().to_string().starts_with("?")
    }

    /// Returns true if the root of the tree is not a hole.
    fn is_root_ident(&self) -> bool {
        !self.is_root_hole()
    }

    /// Return all holes in tree
    fn holes(&self) -> BTreeSet<T> {
        let mut res: BTreeSet<T> = self.children().into_iter().flat_map(|c| c.holes()).collect();
        if self.is_root_hole() {
            res.insert(self.root().clone());
        }
        res
    }

    /// Return all non constants
    fn consts(&self) -> Vec<T> {
        let mut res: Vec<T> = self.children().into_iter().flat_map(|c| c.consts()).collect();
        if self.is_root_ident() {
            res.push(self.root().clone());
        }
        res
    }
}

impl<'a ,L: Language> Tree<'a, L> for RecExpSlice<'a, L> {
    fn root(&self) -> &'a L {
        &self.exp.as_ref()[self.index]
    }

    fn children(&self) -> Vec<RecExpSlice<'a, L>> {
        self.exp.as_ref()[self.index].children().iter().map(|t|
            RecExpSlice::new(usize::from(*t), self.exp)).collect_vec()
    }
}

impl<'a, T: 'a + Language + Display> Display for RecExpSlice<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.to_sexp_string())
    }
}

