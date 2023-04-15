use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::{EGraph, Id, SymbolLang};
use itertools::{Itertools, max};
use symbolic_expressions::Sexp;

macro_rules! bail {
    ($s:literal $(,)?) => {
        return Err($s.into())
    };
    ($s:literal, $($args:expr),+) => {
        return Err(format!($s, $($args),+).into())
    };
}

type ROption<T> = Rc<Option<T>>;

/// A term tree with a root and subtrees
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Tree {
    /// The root of the term 
    pub root: String,
    /// The subtrees of the term
    pub subtrees: Vec<Tree>,
    /// The type of the term
    pub typ: ROption<Tree>,
}

impl Tree {
    /// Create a new single node tree
    pub fn leaf(op: String) -> Tree {
        Tree { root: op, subtrees: Vec::new(), typ: Rc::new(None) }
    }

    /// Create a new single node tree with a type
    pub fn tleaf(op: String, typ: Option<Tree>) -> Tree {
        Tree { root: op, subtrees: Vec::new(), typ: Rc::new(typ) }
    }

    /// Create a new tree with a root and subtrees
    pub fn branch(op: String, subtrees: Vec<Tree>) -> Tree {
        Tree { root: op, subtrees, typ: Rc::new(None) }
    }

    #[allow(missing_docs)]
    pub fn depth(&self) -> usize {
        return max(self.subtrees.iter().map(|x| x.depth())).unwrap_or(0) + 1
    }

    #[allow(missing_docs)]
    pub fn size(&self) -> usize {
        return self.subtrees.iter().map(|x| x.size()).sum::<usize>() + 1
    }

    // pub fn to_rec_expr(&self, op_res: Option<RecExpr<SymbolLang>>) -> (Id, RecExpr<SymbolLang>) {
    //     let mut res = if op_res.is_none() { RecExpr::default() } else { op_res.unwrap() };
    //     return if self.is_leaf() {
    //         (res.add(SymbolLang::leaf(&self.root)), res)
    //     } else {
    //         let mut ids = Vec::default();
    //         for s in &self.subtrees {
    //             let (id, r) = s.to_rec_expr(Some(res));
    //             res = r;
    //             ids.insert(0, id);
    //         }
    //         (res.add(SymbolLang::new(&self.root, ids)), res)
    //     };
    // }

    /// Add this term to the egraph
    pub fn add_to_graph(&self, graph: &mut EGraph<SymbolLang, ()>) -> Id {
        let mut children = Vec::new();
        for t in &self.subtrees {
            children.push(t.add_to_graph(graph));
        };
        graph.add(SymbolLang::new(self.root.clone(), children))
    }

    #[allow(missing_docs)]
    pub fn is_leaf(&self) -> bool {
        self.subtrees.is_empty()
    }

    #[allow(missing_docs)]
    pub fn to_sexp_string(&self) -> String {
        if self.is_leaf() {
            self.root.clone()
        } else {
            format!("({} {})", self.root.clone(), itertools::Itertools::intersperse(self.subtrees.iter().map(|t| t.to_string()), " ".parse().unwrap()).collect::<String>())
        }
    }

    /// Lexicographic ordering for trees, by root symbol and then by subtree ordering.
    pub fn tree_lexicographic_ordering(t1: &Tree, t2: &Tree) -> Ordering {
        match t1.root.cmp(&t2.root ) {
            Less => Less,
            Equal => {
                t1.subtrees.iter().zip_longest(&t2.subtrees).find_map(|x| {
                    if !x.has_left() {
                        Some(Less)
                    } else if !x.has_right() {
                        Some(Greater)
                    } else {
                        let l = *x.as_ref().left().unwrap();
                        let r = *x.as_ref().right().unwrap();
                        let rec_res = Self::tree_lexicographic_ordering(l, r);
                        rec_res.is_eq().then(|| rec_res)
                    }
                }).unwrap_or(Equal)
            },
            Greater => Greater
        }
    }

    /// Ordering for trees, by depth and then by size and then by lexicographic ordering.
    pub fn tree_size_ordering(t1: &Tree, t2: &Tree) -> Ordering {
        match t1.depth().cmp(&t2.depth()) {
            Less => Less,
            Equal => match t1.size().cmp(&t2.size()) {
                Less => Less,
                // Oh the horror of string semantics (but I am not going to implement a full recursive
                // check here)
                Equal => Self::tree_lexicographic_ordering(t1, t2),
                Greater => Greater
            },
            Greater => Greater
        }
    }
}

impl Display for Tree {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.to_sexp_string())
    }
}

impl std::str::FromStr for Tree {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        fn parse_sexp_tree(sexp: &Sexp) -> Result<Tree, String> {
            match sexp {
                Sexp::Empty => Err("Found empty s-expression".into()),
                Sexp::String(s) => {
                    Ok(Tree::leaf(s.clone()))
                }
                Sexp::List(list) if list.is_empty() => Err("Found empty s-expression".into()),
                Sexp::List(list) => match &list[0] {
                    Sexp::Empty => unreachable!("Cannot be in head position"),
                    // TODO: add apply
                    Sexp::List(l) => bail!("Found a list in the head position: {:?}", l),
                    // Sexp::String(op) if op == "typed" => {
                    //     let mut tree = parse_sexp_tree(&list[1])?;
                    //     let types = parse_sexp_tree(&list[2])?;
                    //     tree.typ = Box::new(Some(types));
                    //     Ok(tree)
                    // }
                    Sexp::String(op) => {
                        let arg_ids = list[1..].iter().map(|s| parse_sexp_tree(s).expect("Parsing should succeed")).collect::<Vec<Tree>>();
                        let node = Tree::branch(op.clone(), arg_ids);
                        Ok(node)
                    }
                },
            }
        }

        let sexp = symbolic_expressions::parser::parse_str(s.trim()).map_err(|e| e.to_string())?;
        Ok(parse_sexp_tree(&sexp)?)
    }
}
