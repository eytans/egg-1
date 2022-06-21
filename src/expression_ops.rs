use std::fmt::Formatter;
use std::hash::{Hash, Hasher};
use indexmap::{IndexMap, IndexSet};

use itertools::Itertools;
use smallvec::alloc::fmt::Display;
use crate::{Language, RecExpr, EGraph, Id};

#[derive(Clone, Debug)]
pub struct RecExpSlice<'a, L: Language> {
    index: usize,
    exp: &'a RecExpr<L>
}

impl<'a, L: Language> RecExpSlice<'a, L> {
    pub fn new(index: usize, exp: &'a RecExpr<L>) -> RecExpSlice<'a, L> {
        RecExpSlice{index, exp}
    }

    pub fn add_to_graph(&self, graph: &mut EGraph<L, ()>) -> Id {
        graph.add_expr(&RecExpr::from(self.exp.as_ref()[..self.index+1].iter().cloned().collect_vec()))
    }

    pub fn to_spaceless_string(&self) -> String {
        self.to_sexp_string()
            .replace(" ", "_")
            .replace("(", "PO")
            .replace(")", "PC")
            .replace("->", "fn")
    }

    pub fn to_sexp_string(&self) -> String {
        if self.is_leaf() {
            format!("{}", self.root().display_op().to_string())
        } else {
            format!("({} {})", self.root().display_op().to_string(), self.children().iter().map(|t| t.to_sexp_string()).intersperse(" ".to_string()).collect::<String>())
        }
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

pub trait IntoTree<'a, T: Language> {
    fn into_tree(&'a self) -> RecExpSlice<'a, T>;
}

impl<'a, T: Language> IntoTree<'a, T> for RecExpr<T> {
    fn into_tree(&'a self) -> RecExpSlice<'a, T> {
        RecExpSlice::from(self)
    }
}

pub trait Tree<'a, T: 'a + Language> {
    fn root(&self) -> &'a T;

    fn children(&self) -> Vec<RecExpSlice<'a, T>>;

    fn is_leaf(&self) -> bool {
        self.children().is_empty()
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
