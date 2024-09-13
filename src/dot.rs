/*!
EGraph visualization with [GraphViz]

Use the [`Dot`] struct to visualize an [`EGraph`]

[`Dot`]: struct.Dot.html
[`EGraph`]: struct.EGraph.html
[GraphViz]: https://graphviz.gitlab.io/
!*/

use multimap::MultiMap;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::fmt::{self, Debug, Display, Formatter};
use std::io::{Error, ErrorKind, Result, Write};
use std::path::Path;
use std::rc::Rc;

use crate::{egraph::EGraph, Analysis, ColorId, EClass, Id, Language};

/**
A wrapper for an [`EGraph`] that can output [GraphViz] for
visualization.

The [`EGraph::dot`](struct.EGraph.html#method.dot) method creates `Dot`s.

# Example

```
use egg::{*, rewrite as rw};

let rules = &[
    rw!("mul-commutes"; "(* ?x ?y)" => "(* ?y ?x)"),
    rw!("mul-two";      "(* ?x 2)" => "(<< ?x 1)"),
];

let mut egraph: EGraph<SymbolLang, ()> = Default::default();
egraph.add_expr(&"(/ (* 2 a) 2)".parse().unwrap());
let egraph = Runner::default().with_egraph(egraph).run(rules).egraph;

// Dot implements std::fmt::Display
println!("My egraph dot file: {}", egraph.dot());

// create a Dot and then compile it assuming `dot` is on the system
egraph.dot().to_svg("target/foo.svg").unwrap();
// egraph.dot().to_png("target/foo.png").unwrap();
// egraph.dot().to_pdf("target/foo.pdf").unwrap();
// egraph.dot().to_dot("target/foo.dot").unwrap();
```

Note that self-edges (from an enode to its containing eclass) will be
rendered improperly due to a deficiency in GraphViz.
So the example above will render with an from the "+" enode to itself
instead of to its own eclass.

[`EGraph`]: struct.EGraph.html
[GraphViz]: https://graphviz.gitlab.io/
**/
pub struct Dot<'a, L: Language, N: Analysis<L>> {
    pub(crate) egraph: &'a EGraph<L, N>,
    pub(crate) color: Option<ColorId>,
    pub(crate) print_color: String,
    pub(crate) pred: Option<Rc<dyn Fn(&EGraph<L, N>, Id) -> bool + 'static>>,
}

impl<'a, L, N> Dot<'a, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    /// Sets the color that will be used for the colored ENodes and EClasses.
    pub fn set_print_color(mut self, text: String) -> Self {
        // Oh no! a security problem!
        assert!(text.len() < 20);
        self.print_color = text;
        self
    }

    /// Writes the `Dot` to a .dot file with the given filename.
    /// Does _not_ require a `dot` binary.
    pub fn to_dot(&self, filename: impl AsRef<Path>) -> Result<()> {
        let mut file = std::fs::File::create(filename)?;
        write!(file, "{}", self)?;
        Ok(())
    }

    /// Renders the `Dot` to a .png file with the given filename.
    /// Requires a `dot` binary to be on your `$PATH`.
    pub fn to_png(&self, filename: impl AsRef<Path>) -> Result<()> {
        self.run_dot(&["-Tpng".as_ref(), "-o".as_ref(), filename.as_ref()])
    }

    /// Renders the `Dot` to a .svg file with the given filename.
    /// Requires a `dot` binary to be on your `$PATH`.
    pub fn to_svg(&self, filename: impl AsRef<Path>) -> Result<()> {
        self.run_dot(&["-Tsvg".as_ref(), "-o".as_ref(), filename.as_ref()])
    }

    /// Renders the `Dot` to a .pdf file with the given filename.
    /// Requires a `dot` binary to be on your `$PATH`.
    pub fn to_pdf(&self, filename: impl AsRef<Path>) -> Result<()> {
        self.run_dot(&["-Tpdf".as_ref(), "-o".as_ref(), filename.as_ref()])
    }

    /// Invokes `dot` with the given arguments, piping this formatted
    /// `Dot` into stdin.
    pub fn run_dot<S, I>(&self, args: I) -> Result<()>
    where
        S: AsRef<OsStr>,
        I: IntoIterator<Item = S>,
    {
        self.run("dot", args)
    }

    /// Invokes some program with the given arguments, piping this
    /// formatted `Dot` into stdin.
    ///
    /// Can be used to run a different binary than `dot`:
    /// ```no_run
    /// # use egg::*;
    /// # let mut egraph: EGraph<SymbolLang, ()> = Default::default();
    /// egraph.dot().run(
    ///     "/path/to/my/dot",
    ///     &["arg1", "-o", "outfile"]
    /// ).unwrap();
    /// ```
    pub fn run<S1, S2, I>(&self, program: S1, args: I) -> Result<()>
    where
        S1: AsRef<OsStr>,
        S2: AsRef<OsStr>,
        I: IntoIterator<Item = S2>,
    {
        use std::process::{Command, Stdio};
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.as_mut().expect("Failed to open stdin");
        write!(stdin, "{}", self)?;
        match child.wait()?.code() {
            Some(0) => Ok(()),
            Some(e) => Err(Error::new(
                ErrorKind::Other,
                format!("dot program returned error code {}", e),
            )),
            None => Err(Error::new(
                ErrorKind::Other,
                "dot program was killed by a signal",
            )),
        }
    }
}

impl<'a, L: Language, N: Analysis<L>> Debug for Dot<'a, L, N> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Dot({:?})", self.egraph)
    }
}

// gives back the appropriate label and anchor
fn edge(i: usize, len: usize) -> (String, String) {
    assert!(i < len);
    let s = |s: &str| s.to_string();
    match (len, i) {
        (1, 0) => (s(""), s("")),
        (2, 0) => (s(":sw"), s("")),
        (2, 1) => (s(":se"), s("")),
        (3, 0) => (s(":sw"), s("")),
        (3, 1) => (s(":s"), s("")),
        (3, 2) => (s(":se"), s("")),
        (_, _) => (s(""), format!("label={}", i)),
    }
}

impl<'a, L, N> Display for Dot<'a, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let pred: Rc<dyn Fn(&EGraph<L, N>, Id) -> bool> = if let Some(b) = self.pred.clone() {
            b
        } else {
            Rc::new(|_, _| true)
        };

        let mut dropped: HashSet<Id> = self
            .egraph
            .classes()
            .filter(|c| !pred(&self.egraph, c.id))
            .map(|c| c.id)
            .collect();
        let mut dropped_nodes: MultiMap<Id, L> = MultiMap::new();
        let mut changed = !dropped.is_empty();
        while changed {
            changed = false;
            for c in self.egraph.classes() {
                if dropped.contains(&c.id) {
                    continue;
                }
                for n in &c.nodes {
                    if n.children().iter().any(|&id| dropped.contains(&id)) {
                        dropped_nodes.insert(c.id, n.clone());
                    }
                }
                if c.nodes.len() == dropped_nodes.get_vec(&c.id).map_or(0, |x| x.len()) {
                    dropped.insert(c.id);
                    changed = true;
                }
            }
        }

        writeln!(f, "digraph egraph {{")?;

        // set compound=true to enable edges to clusters
        writeln!(f, "  compound=true")?;
        writeln!(f, "  clusterrank=local")?;

        let empty = vec![];
        // define all the nodes, clustered by eclass
        if let Some(c_id) = self.color {
            let color = self.egraph.get_color(c_id).unwrap();
            let mut done = HashSet::new();
            // Collect all groups to put in subgraphs and filter classes using self.filter
            let mut groups = Vec::new();
            for (black_id, ids) in &color.equality_classes {
                let mut group = Vec::new();
                for class in ids.iter().map(|id| &self.egraph[*id]) {
                    if !dropped.contains(&class.id) {
                        group.push(class.id);
                    }
                }
                if !group.is_empty() {
                    groups.push((black_id, group));
                }
            }
            for (black_id, ids) in groups {
                writeln!(f, "  subgraph cluster_colored_{} {{", black_id)?;
                writeln!(f, "    color={}", self.print_color)?;
                for class in ids.iter().map(|id| &self.egraph[*id]) {
                    Self::format_class(
                        f,
                        class,
                        &self.print_color,
                        dropped_nodes.get_vec(&class.id).unwrap_or(&empty),
                    )?;
                }
                writeln!(f, "  }}")?;
                done.extend(ids.iter().copied());
            }
            for c in self.egraph.classes().filter(|c| !dropped.contains(&c.id)) {
                if done.contains(&c.id) || c.color().iter().any(|c1| c1 != &c_id) {
                    continue;
                }
                Self::format_class(
                    f,
                    c,
                    &self.print_color,
                    dropped_nodes.get_vec(&c.id).unwrap_or(&empty),
                )?;
            }
        } else {
            for c in self
                .egraph
                .classes()
                .filter(|c| c.color.is_none() && !dropped.contains(&c.id))
            {
                Self::format_class(
                    f,
                    c,
                    &self.print_color,
                    dropped_nodes.get_vec(&c.id).unwrap_or(&empty),
                )?
            }
        }
        for class in self
            .egraph
            .classes()
            .filter(|c| (c.color.is_none() || c.color == self.color) && !dropped.contains(&c.id))
        {
            let color_text = if class.color.is_some() {
                ", color=".to_owned() + &self.print_color.to_owned()
            } else {
                "".to_string()
            };
            for (i_in_class, node) in class
                .iter()
                .filter(|n| {
                    dropped_nodes
                        .get_vec(&class.id)
                        .map_or(true, |x| !x.contains(n))
                })
                .enumerate()
            {
                for (arg_i, child) in node.children().iter().enumerate() {
                    // write the edge to the child, but clip it to the eclass with lhead
                    let (anchor, label) = edge(arg_i, node.len());
                    let child_leader = self.egraph.find(*child);

                    if child_leader == class.id {
                        writeln!(
                            f,
                            // {}.0 to pick an arbitrary node in the cluster
                            "  {}.{}{} -> {}.{}:n [lhead = cluster_{}{}, {}]",
                            class.id,
                            i_in_class,
                            anchor,
                            class.id,
                            i_in_class,
                            class.id,
                            color_text,
                            label
                        )?;
                    } else {
                        writeln!(
                            f,
                            // {}.0 to pick an arbitrary node in the cluster
                            "  {}.{}{} -> {}.0 [lhead = cluster_{}{}, {}]",
                            class.id, i_in_class, anchor, child, child_leader, color_text, label
                        )?;
                    }
                }
            }
        }

        write!(f, "}}")
    }
}

impl<'a, L, N> Dot<'a, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    fn format_class(
        f: &mut Formatter,
        class: &EClass<L, <N as Analysis<L>>::Data>,
        print_color: &String,
        dropped_nodes: &Vec<L>,
    ) -> fmt::Result {
        let color_text = if class.color.is_some() {
            ", color=".to_owned() + print_color
        } else {
            "".to_string()
        };
        writeln!(f, "  subgraph cluster_{} {{", class.id.0)?;
        writeln!(
            f,
            "    style=dotted color=black label=\"{}\"",
            class.id.to_string()
        )?;
        for (i, node) in class
            .iter()
            .filter(|n| !dropped_nodes.contains(*n))
            .enumerate()
        {
            writeln!(
                f,
                "    {}.{}[label = \"{}\"{}]",
                class.id,
                i,
                node.display_op(),
                color_text,
            )?;
        }
        writeln!(f, "  }}")
    }
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    use crate::{SymbolLang, EGraph, RecExpr, init_logger};

    #[test]
    fn draw_if_xy_then_a_else_b() {
        init_logger();

        let mut egraph: EGraph<SymbolLang, ()> = EGraph::new(());
        let if_statement = egraph.add_expr(&RecExpr::from_str("(if (< x y) hello world)").unwrap());
        egraph.dot().to_dot("if.dot").unwrap();
        let color = egraph.create_color(None);
        let cond = egraph.add_expr(&RecExpr::from_str("(< x y)").unwrap());
        let tru = egraph.colored_add_expr(color, &RecExpr::from_str("true").unwrap());
        let hello = egraph.add_expr(&RecExpr::from_str("hello").unwrap());
        egraph.colored_union(color, cond, tru);
        egraph.colored_union(color, if_statement, hello);
        egraph
            .colored_dot(color)
            .set_print_color("red".to_string())
            .to_dot("if_tru.dot")
            .unwrap();
    }

    #[test]
    fn draw_max() {
        init_logger();

        let mut egraph: EGraph<SymbolLang, ()> = EGraph::new(());
        let max_st = egraph.add_expr(&RecExpr::from_str("(max x y)").unwrap());
        let min_st = egraph.add_expr(&RecExpr::from_str("(min x y)").unwrap());
        let minus = egraph.add_expr(&RecExpr::from_str("(- (max x y) (min x y))").unwrap());
        let _abs = egraph.add_expr(&RecExpr::from_str("(abs (- x y))").unwrap());
        egraph.dot().to_dot("maxmin.dot").unwrap();
        let mut smaller_egraph = egraph.clone();
        let smaller_then = smaller_egraph.add_expr(&RecExpr::from_str("(< x y)").unwrap());
        let tru = smaller_egraph.add_expr(&RecExpr::from_str("true").unwrap());
        smaller_egraph.union(smaller_then, tru);
        smaller_egraph.dot().to_dot("smaller_maxmin.dot").unwrap();
        let x = smaller_egraph.add_expr(&RecExpr::from_str("x").unwrap());
        let y = smaller_egraph.add_expr(&RecExpr::from_str("y").unwrap());
        smaller_egraph.union(x, min_st);
        smaller_egraph.union(y, max_st);
        smaller_egraph.rebuild();
        smaller_egraph
            .dot()
            .to_dot("smaller_maxmin_rw.dot")
            .unwrap();
        let smaller_abs = smaller_egraph.add_expr(&RecExpr::from_str("(abs (- x y))").unwrap());
        smaller_egraph.union(smaller_abs, minus);
        smaller_egraph.rebuild();
        smaller_egraph.dot().to_dot("smaller_final.dot").unwrap();
        let c_smaller = egraph.create_color(None);
        let smaller_then =
            egraph.colored_add_expr(c_smaller, &RecExpr::from_str("(< x y)").unwrap());
        let tru = egraph.colored_add_expr(c_smaller, &RecExpr::from_str("true").unwrap());
        egraph.colored_union(c_smaller, smaller_then, tru);
        egraph
            .colored_dot(c_smaller)
            .set_print_color("blue".to_string())
            .to_dot("smaller_no_rw.dot")
            .unwrap();
        let x = egraph.add_expr(&RecExpr::from_str("x").unwrap());
        let y = egraph.add_expr(&RecExpr::from_str("y").unwrap());
        egraph.colored_union(c_smaller, x, min_st);
        egraph.colored_union(c_smaller, y, max_st);
        egraph.rebuild();
        let c_smaller_abs =
            egraph.colored_add_expr(c_smaller, &RecExpr::from_str("(abs (- x y))").unwrap());
        egraph.colored_union(c_smaller, c_smaller_abs, minus);
        egraph.rebuild();
        egraph
            .colored_dot(c_smaller)
            .set_print_color("blue".to_string())
            .to_dot("c_smaller_final.dot")
            .unwrap();
        let c_bigger = egraph.create_color(None);
        let bigger_then = egraph.colored_add_expr(c_bigger, &RecExpr::from_str("(> x y)").unwrap());
        let tru = egraph.colored_add_expr(c_bigger, &RecExpr::from_str("true").unwrap());
        egraph.colored_union(c_bigger, bigger_then, tru);
        egraph
            .colored_dot(c_bigger)
            .set_print_color("red".to_string())
            .to_dot("bigger_no_rw.dot")
            .unwrap();
        egraph.colored_union(c_bigger, y, min_st);
        egraph.colored_union(c_bigger, x, max_st);
        egraph.rebuild();
        let c_smaller_abs =
            egraph.colored_add_expr(c_bigger, &RecExpr::from_str("(abs (- x y))").unwrap());
        egraph.colored_union(c_bigger, c_smaller_abs, minus);
        egraph.rebuild();
        egraph
            .colored_dot(c_bigger)
            .set_print_color("red".to_string())
            .to_dot("c_bigger_final.dot")
            .unwrap();
    }

    #[test]
    fn filter_edges() {
        init_logger();

        let mut egraph: EGraph<SymbolLang, ()> = EGraph::new(());
        let _max_st = egraph.add_expr(&RecExpr::from_str("(max x y)").unwrap());
        let _min_st = egraph.add_expr(&RecExpr::from_str("(min x y)").unwrap());
        let _minus = egraph.add_expr(&RecExpr::from_str("(- (max x y) (min x y))").unwrap());
        let _abs = egraph.add_expr(&RecExpr::from_str("(abs (- x y))").unwrap());
        let _tester = egraph.add_expr(&RecExpr::from_str("(tester (abs (- x y)))").unwrap());
        let filtered_dot =
            egraph.filtered_dot(|e, id| e[id].nodes.iter().any(|x| x.op.to_string() != "abs"));
        let dot_string = filtered_dot.to_string();
        filtered_dot.to_dot("filtered.dot").unwrap();
        assert!(!dot_string.contains("abs"));
        assert!(!dot_string.contains("tester"));
    }
}
