use std::io::{Read, Write, BufReader, Result, BufRead};
use indexmap::IndexMap;
use itertools::Itertools;
use std::str::FromStr;
use crate::{EGraph, Analysis, Language, Id, EClass, SymbolLang, RecExpr, ColorId};
use crate::unionfind::UnionFind;

/// A trait for EGraphs that can be serialized.
pub trait Serialization {
    /// Exports graph data as a set of tuples. The format is `(op, eclass, children...)`.
    fn to_tuples(&self) -> Vec<(String, Vec<Id>)>;
    /// Exports graph data as a set of tuples. The format is one tuple per line,
    /// space-separated: `op eclass children`.
    fn to_tuples_text(&self, palette: &ColorPalette, out: &mut impl Write) -> Result<()>;
}

/// A trait for deserializing EGraphs from previously serialized content.
pub trait Deserialization {
    /// Constructs a graph from tuple data (eclass ids as `Id`).
    fn from_tuples<'a>(tuples: impl Iterator<Item=&'a (impl ToString + 'a, Vec<Id>)>) -> (Self, ColorPalette) where Self: Sized;
    /// Constructs a graph from tuple data (eclass ids as numbers).
    fn from_tuples_int<'a>(tuples: impl Iterator<Item=&'a (impl ToString + 'a, Vec<usize>)>) -> (Self, ColorPalette) where Self: Sized;
    /// Parses an constructs a graph from textual representation.
    fn from_tuples_text(in_: &mut impl Read) -> Result<(Self, ColorPalette)> where Self: Sized;
}

#[derive(Debug, Clone)]
pub struct ColorPalette {
    colors: IndexMap<Id, ColorId>
}

impl ColorPalette {
    pub fn new() -> Self {
        ColorPalette { colors: Default::default() }
    }

    pub fn get<L: Language, N: Analysis<L>>(&mut self, g: &mut EGraph<L, N>, color: Id) -> ColorId {
        self.get_hier(g, color, None)
    }

    pub fn get_sub<L: Language, N: Analysis<L>>(&mut self, g: &mut EGraph<L, N>, color: Id, parent: ColorId) -> ColorId {
        self.get_hier(g, color, Some(parent))
    }

    pub fn get_hier<L: Language, N: Analysis<L>>(&mut self, g: &mut EGraph<L, N>, color: Id, parent: Option<ColorId>) -> ColorId {
        match self.colors.get(&color) {
           Some(cid) => *cid,
           None => {
               let v = g.create_color(parent);
               self.colors.insert(color, v);
               v
           }
        }
    }

    pub fn get_named<N: Analysis<SymbolLang>>(&mut self, g: &mut EGraph<SymbolLang, N>,
                                              name: &str) -> ColorId {
        self.get_named_hier(g, name, None)
    }
    pub fn get_named_sub<N: Analysis<SymbolLang>>(&mut self, g: &mut EGraph<SymbolLang, N>,
                                                  name: &str, parent: ColorId) -> ColorId {
        self.get_named_hier(g, name, Some(parent))
    }

    pub fn get_named_hier<N: Analysis<SymbolLang>>(&mut self, g: &mut EGraph<SymbolLang, N>,
                                                   name: &str, parent: Option<ColorId>) -> ColorId {
        let cid = g.add_expr(&RecExpr::from_str(name).unwrap());
        self.get_hier(g, cid, parent)
    }
}

impl Default for ColorPalette {
    fn default() -> Self { ColorPalette::new() }
}

impl Serialization for EGraph<SymbolLang, ()> {
    fn to_tuples(&self) -> Vec<(String, Vec<Id>)> {
        self.classes().flat_map(
            |ec| {
                let target = ec.id;
                ec.nodes.iter().map(
                    move |u| (u.op.to_string(), [vec![target], u.children.clone()].concat()))
            }).collect::<Vec<_>>()
    }

    fn to_tuples_text(&self, palette: &ColorPalette, out: &mut impl Write) -> Result<()>{
        // Write edges
        for (op, ids) in self.to_tuples() {
            writeln!(out, "{op} {ids}", ids = ids.iter().join(" "))?;
        }
        // Write colors
        let id_start = self.classes().map(|e| e.id).max().unwrap();
        for (i, color) in self.colors().enumerate() {
            let _color_id =
                if let Some(color_entry) = palette.colors.iter().find(|e| e.1 == &color.get_id()) {
                    usize::from(*color_entry.0)
                }
                else {
                    let color_id = usize::from(id_start) + i + 1;
                    writeln!(out, "clr#{i} {color_id}")?;
                    color_id
                };
            //todo!("Not supported with hierarchies");
            #[allow(unreachable_code)]
            for id in color.current_black_reps() {
                writeln!(out, "?~ {_color_id} {members}",
                         members = color.equality_class(self, *id).join(" "))?;
            }
        }
        Ok(())
    }
}

fn line_to_tuple(line: &str) -> (String, Vec<Id>) {
    let v: Vec<_> = line.split(' ').filter(|s| s.len() > 0).collect();
    (v[0].into(), v[1..].iter().map(
        |u| Id::from(u.parse::<usize>().unwrap())).collect())
}

impl Deserialization for EGraph<SymbolLang, ()> {

    fn from_tuples<'a>(tuples: impl Iterator<Item=&'a (impl ToString + 'a, Vec<Id>)>) -> (Self, ColorPalette) {
        let mut builder = EGraphBuilder::default();
        let mut leaf_ops = Vec::new();
        let mut color_unions = Vec::new();
        for (op, vertices) in tuples {
            let op = op.to_string();
            if op == "?~" { color_unions.push(vertices.clone()); continue; }
            for id in vertices { builder.add_class(*id); }
            let target = vertices[0];
            let sources = vertices[1..].to_vec();
            let enode = SymbolLang::new(op, sources);
            if enode.is_leaf() { leaf_ops.push(enode.op_id()) }
            builder.add_node(target, enode);
        }
        let mut g = builder.build();

        // Performed deferred color merges
        let mut palette = ColorPalette::new();
        for vertices in color_unions {
            let vertices = builder.gids(vertices);
            let v0 = g.find(vertices[0]);
            let c = palette.get(&mut g, v0);
            let u = vertices[1];
            for v in &vertices[2..] { g.colored_union(c, u, *v); }
        }
        g.rebuild();
        (g, palette)
    }

    fn from_tuples_int<'a>(tuples: impl Iterator<Item=&'a (impl ToString + 'a, Vec<usize>)>) -> (Self, ColorPalette) {
        EGraph::<SymbolLang, ()>::from_tuples(
            tuples.map(|(op, members)|
                (op.to_string(), members.iter().map(|v| Id::from(*v)).collect::<Vec<Id>>()))
                .collect::<Vec<_>>().iter())
    }

    fn from_tuples_text(in_: &mut impl Read) -> Result<(Self, ColorPalette)> {
        Ok(EGraph::<SymbolLang, ()>::from_tuples(BufReader::new(in_).lines()
            .map(|r| r.unwrap().trim().to_string())
            .filter(|s| s.len() > 0)
            .map(|ln| line_to_tuple(&ln)
        ).collect::<Vec<_>>().iter() ))
    }
}

#[derive(Default)]
struct EGraphBuilder {
    egraph: EGraph<SymbolLang, ()>,
    map: IndexMap<Id, Id>
}

impl EGraphBuilder {
    pub(crate) fn add_class(&mut self, id: Id) {
        // temp hack
        let gid = self.egraph.add(SymbolLang::new(format!("_{id}"), vec![]));
        self.map.insert(id, gid);
    }

    pub(crate) fn add_node(&mut self, eclass: Id, enode: SymbolLang) {
        let enode = SymbolLang::new(enode.op, enode.children.iter().map(
            |id|self.gid(*id)).collect());
        let gid = self.egraph.add(enode);
        self.egraph.union(gid, *self.map.get(&eclass).unwrap());
        self.egraph.rebuild();
    }

    pub(crate) fn build(&self) -> EGraph<SymbolLang, ()> { self.egraph.clone() }

    pub(crate) fn gid(&self, id: Id) -> Id { *self.map.get(&id).unwrap() }
    pub(crate) fn gids(&self, ids: Vec<Id>) -> Vec<Id> {
        ids.iter().map(|id| self.gid(*id)).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::str::FromStr;
    use crate::{EGraph, RecExpr, Serialization, Deserialization, SymbolLang, rewrite, Language, Runner};
    use crate::ser::ColorPalette;

    #[test]
    fn basic_term_ser() {
        let mut g = EGraph::<SymbolLang, ()>::new(());
        let exp1: RecExpr<SymbolLang> = RecExpr::from_str("(:: y (rev l))").unwrap();
        let exp2: RecExpr<SymbolLang> = RecExpr::from_str("(rev (:+ l y))").unwrap();
        let u1 = g.add_expr(&exp1);
        let u2 = g.add_expr(&exp2);

        // Serialize to text
        let mut v = Vec::<u8>::new();
        g.to_tuples_text(&ColorPalette::new(), &mut v)
            .unwrap();

        // Deserialize
        let (mut g, _pal) =
            EGraph::<SymbolLang, ()>::from_tuples_text(
                &mut io::Cursor::new(v)).unwrap();
        // Check that original exprs are still there
        assert_eq!(g.add_expr(&exp2), u2);
        assert_eq!(g.add_expr(&exp1), u1);
    }

    #[test]
    fn ser_after_rewrite() {
        // Define rewrites for addition
        let mut rws = vec![];
        rws.extend(rewrite!("add-base"; "(+ Z x)" <=> "x"));
        rws.extend(rewrite!("add-ind"; "(+ (S x) y)" <=> "(S (+ y x))"));
        rws.extend(rewrite!("add-comm"; "(+ x y)" <=> "(+ y x)"));

        let mut g = EGraph::<SymbolLang, ()>::new(());
        // Add the number 10 from successors
        let num10_exp = RecExpr::from_str("(S (S (S (S (S (S (S (S (S (S Z))))))))))").unwrap();
        let num10 = g.add_expr(&num10_exp);
        let exp1_exp = RecExpr::from_str("(+ (S Z) (S Z))").unwrap();
        let exp1 = g.add_expr(&exp1_exp);
        g.add(SymbolLang::from_op_str("+", vec![num10, exp1]).unwrap());
        g = Runner::default().with_egraph(g).run(&rws).egraph;

        // Serialize to text
        let mut v = Vec::<u8>::new();
        g.to_tuples_text(&ColorPalette::new(), &mut v)
            .unwrap();

        // Deserialize
        let (mut g, _pal) =
            EGraph::<SymbolLang, ()>::from_tuples_text(
                &mut io::Cursor::new(v)).unwrap();
        // Check that original exprs are still there
        assert_eq!(g.add_expr(&num10_exp), num10);
        assert_eq!(g.add_expr(&exp1_exp), exp1);
    }
}
