use std::collections::HashMap;
use std::io::{Read, Write, BufReader, Result, BufRead};
use itertools::Itertools;
use crate::{EGraph, Analysis, Language, Id, EClass, SymbolLang, ColorId};

/// A trait for EGraphs that can be serialized.
pub trait Serialization {
    /// Exports graph data as a set of tuples. The format is `(op, eclass, children...)`.
    fn to_tuples(&self) -> Vec<(String, Vec<Id>)>;
    /// Exports graph data as a set of tuples. The format is one tuple per line,
    /// space-separated: `op eclass children`.
    fn to_tuples_text(&self, out: &mut impl Write) -> Result<()>;
}

/// A trait for deserializing EGraphs from previously serialized content.
pub trait Deserialization {
    /// Constructs a graph from tuple data (eclass ids as `Id`).
    fn from_tuples<'a>(tuples: impl Iterator<Item=&'a (impl ToString + 'a, Vec<Id>)>) -> Self;
    /// Constructs a graph from tuple data (eclass ids as numbers).
    fn from_tuples_int<'a>(tuples: impl Iterator<Item=&'a (impl ToString + 'a, Vec<usize>)>) -> Self;
    /// Parses an constructs a graph from textual representation.
    fn from_tuples_text(in_: &mut impl Read) -> Result<Self> where Self: Sized;
}

trait DeserializationHelper<L: Language> {
    fn add_class<'a>(&'a mut self, id: Id);
    fn add_node(&mut self, eclass: Id, enode: L) -> &L;
    fn update_parents(&mut self, parent: Id, enode: &L);
    fn line_to_tuple(line: &str) -> (String, Vec<Id>);
}

struct ColorPalette {
    colors: HashMap<Id, ColorId>
}

impl ColorPalette {
    fn new() -> Self {
        ColorPalette { colors: HashMap::default() }
    }

    fn get<L: Language, N: Analysis<L>>(&mut self, g: &mut EGraph<L, N>, color: Id) -> ColorId {
        match self.colors.get(&color) {
           Some(cid) => *cid,
           None => {
               let v = g.create_color();
               self.colors.insert(color, v);
               v
           }
        }
    }
}

impl<N: Analysis<SymbolLang>> Serialization for EGraph<SymbolLang, N> {
    fn to_tuples(&self) -> Vec<(String, Vec<Id>)> {
        self.classes().flat_map(
            |ec| {
                let target = ec.id;
                ec.nodes.iter().map(
                    move |u| (u.op.to_string(), [vec![target], u.children.clone()].concat()))
            }).collect::<Vec<_>>()
    }

    fn to_tuples_text(&self, out: &mut impl Write) -> Result<()>{
        // Write edges
        for (op, ids) in self.to_tuples() {
            writeln!(out, "{op} {ids}", ids = ids.iter().join(" "))?;
        }
        // Write colors
        let id_start = self.classes().map(|e| e.id).max().unwrap();
        for (i, color) in self.colors().iter().enumerate() {
            let color_id = usize::from(id_start) + i + 1;
            writeln!(out, "clr#{i} {color_id}")?;
            for id in color.black_reps() {
                writeln!(out, "?~ {color_id} {members}",
                         members = color.black_ids(*id).unwrap().iter().join(" "))?;
            }
        }
        Ok(())
    }
}

impl<L: Language, N: Analysis<L>> DeserializationHelper<L> for EGraph<L, N> {
    fn add_class<'a>(&'a mut self, id: Id) {
        let idx = usize::from(id);
        while self.classes.len() <= idx { self.classes.push(None) }
        if self.classes[idx].is_none() {
            let dummy_node = L::from_op_str("?", vec![]).unwrap();
            self.classes[idx] = Some(Box::new(EClass {
                id: id,
                nodes: vec![],
                data: N::make(self, &dummy_node),
                parents: vec![],
                color: None,
                colored_parents: Default::default()
            }));
            self.unionfind.make_set_at(id);
        }
    }

    fn add_node(&mut self, eclass: Id, enode: L) -> &L {
        self.update_parents(eclass, &enode);
        EGraph::<L, ()>::update_memo_from_parent(&mut self.memo, &enode, &eclass);
        let class = self.classes[usize::from(eclass)].as_mut().unwrap();
        class.nodes.push(enode);
        return class.nodes.last().unwrap();
    }

    fn update_parents(&mut self, parent: Id, enode: &L) {
        enode.children().iter().for_each(|u| self.classes[usize::from(*u)].as_mut().unwrap()
            .parents.push((enode.clone(), parent)));
    }

    fn line_to_tuple(line: &str) -> (String, Vec<Id>) {
        let v: Vec<_> = line.split(' ').filter(|s| s.len() > 0).collect();
        (v[0].into(), v[1..].iter().map(
            |u| Id::from(u.parse::<usize>().unwrap())).collect())
    }
}

impl Deserialization for EGraph<SymbolLang, ()> {

    fn from_tuples<'a>(tuples: impl Iterator<Item=&'a (impl ToString + 'a, Vec<Id>)>) -> EGraph<SymbolLang, ()> {
        let mut g = EGraph::<SymbolLang, ()>::default();
        let mut leaf_ops = Vec::new();
        let mut color_unions = Vec::new();
        for (op, vertices) in tuples {
            let op = op.to_string();
            if op == "?~" { color_unions.push(vertices.clone()); continue; }
            for id in vertices { g.add_class(*id); }
            let target = vertices[0];
            let sources = vertices[1..].to_vec();
            let enode = g.add_node(target, SymbolLang::new(op, sources));
            if enode.is_leaf() { leaf_ops.push(enode.op_id()) }
        }
        //g.rebuild_classes();
        g.rebuild();
        for op in leaf_ops {
            let eclass = g.classes_by_op.get(&op).unwrap();
            let min = *eclass.iter().min().unwrap();
            let others = eclass.iter().filter_map(|u| { if *u != min { Some(*u) } else { None } })
                               .collect::<Vec<Id>>();
            others.iter().for_each(|u| { g.union(min, *u); });
        }
        g.rebuild();
        // Performed deferred color merges
        let mut palette = ColorPalette::new();
        for vertices in color_unions {
            let v0 = g.find(vertices[0]);
            let c = palette.get(&mut g, v0);
            let u = vertices[1];
            for v in &vertices[2..] { g.colored_union(c, u, *v); }
        }
        g.rebuild();
        g
    }

    fn from_tuples_int<'a>(tuples: impl Iterator<Item=&'a (impl ToString + 'a, Vec<usize>)>) -> EGraph<SymbolLang, ()> {
        EGraph::<SymbolLang, ()>::from_tuples(
            tuples.map(|(op, members)|
                (op.to_string(), members.iter().map(|v| Id::from(*v)).collect::<Vec<Id>>()))
                .collect::<Vec<_>>().iter())
    }

    fn from_tuples_text(in_: &mut impl Read) -> Result<Self> {
        Ok(EGraph::<SymbolLang, ()>::from_tuples(BufReader::new(in_).lines()
            .map(|r| r.unwrap().trim().to_string())
            .filter(|s| s.len() > 0)
            .map(|ln| EGraph::<SymbolLang, ()>::line_to_tuple(&ln)
        ).collect::<Vec<_>>().iter() ))
    }
}
