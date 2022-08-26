use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::{EGraph, Id, SymbolLang};
use itertools::Itertools;
use symbolic_expressions::Sexp;

macro_rules! bail {
    ($s:literal $(,)?) => {
        return Err($s.into())
    };
    ($s:literal, $($args:expr),+) => {
        return Err(format!($s, $($args),+).into())
    };
}


#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Tree {
    pub root: String,
    pub subtrees: Vec<Rc<Tree>>,
    pub typ: Rc<Option<Tree>>,
}

impl Tree {
    pub fn leaf(op: String) -> Tree {
        Tree { root: op, subtrees: Vec::new(), typ: Rc::new(None) }
    }

    pub fn tleaf(op: String, typ: Rc<Option<Tree>>) -> Tree {
        Tree { root: op, subtrees: Vec::new(), typ }
    }

    pub fn branch(op: String, subtrees: Vec<Rc<Tree>>) -> Tree {
        Tree { root: op, subtrees, typ: Rc::new(None) }
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

    pub fn add_to_graph(&self, graph: &mut EGraph<SymbolLang, ()>) -> Id {
        let mut children = Vec::new();
        for t in &self.subtrees {
            children.push(t.add_to_graph(graph));
        };
        graph.add(SymbolLang::new(self.root.clone(), children))
    }

    pub fn is_leaf(&self) -> bool {
        self.subtrees.is_empty()
    }

    pub fn to_sexp_string(&self) -> String {
        if self.is_leaf() {
            self.root.clone()
        } else {
            format!("({} {})", self.root.clone(), self.subtrees.iter().map(|t| t.to_string()).intersperse(" ".parse().unwrap()).collect::<String>())
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
                        let arg_ids = list[1..].iter().map(|s| Rc::new(parse_sexp_tree(s).expect("Parsing should succeed"))).collect::<Vec<Rc<Tree>>>();
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
