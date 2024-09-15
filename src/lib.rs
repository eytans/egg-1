
/*!

`egg` (**e**-**g**raphs **g**ood) is a e-graph library optimized for equality saturation.

This is the API documentation.

The [tutorial](tutorials/index.html) is a good starting point if you're new to
e-graphs, equality saturation, or Rust.

The [tests](https://github.com/mwillsey/egg/tree/master/tests)
on Github provide some more elaborate examples.

There is also a [paper](https://arxiv.org/abs/2004.03082)
describing `egg` and some of its technical novelties.

!*/

/* needs to be public for trait `GetOp` */
pub mod macros;

#[macro_use]
extern crate global_counter;

pub mod tutorials;

mod dot;
mod eclass;
mod egraph;
mod extract;
mod language;
mod machine;
mod pattern;
mod rewrite;
mod run;
mod ser;
mod subst;
mod unionfind;
mod util;

/// A key to identify [`EClass`](struct.EClass.html)es within an
/// [`EGraph`](struct.EGraph.html).
#[derive(Clone, Copy, Default, Ord, PartialOrd, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Id(pub u32);


#[derive(Clone, Copy, Default, Ord, PartialOrd, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ColorId(usize);


impl From<usize> for Id {
    fn from(n: usize) -> Id {
        Id(n as u32)
    }
}

impl From<Id> for usize {
    fn from(id: Id) -> usize {
        id.0 as usize
    }
}

impl Into<Id> for u32 {
    fn into(self) -> Id {
        Id(self)
    }
}

impl Into<Id> for i32 {
    fn into(self) -> Id {
        Id(self as u32)
    }
}

impl std::fmt::Debug for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for ColorId {
    fn from(n: usize) -> ColorId {
        ColorId(n as usize)
    }
}

impl From<ColorId> for usize {
    fn from(id: ColorId) -> usize {
        id.0 as usize
    }
}

impl std::fmt::Debug for ColorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for ColorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub(crate) use unionfind::SimpleUnionFind;

pub use {
    dot::Dot,
    eclass::{EClass},
    egraph::EGraph,
    extract::*,
    language::*,
    pattern::{ENodeOrVar, Pattern, PatternAst, SearchMatches},
    multipattern::MultiPattern,
    rewrite::{Applier, Rewrite, Searcher},
    run::*,
    ser::{Serialization, Deserialization},
    subst::{Subst, Var},
    util::*,
    eggstentions::*,
};

#[cfg(test)]
fn init_logger() {
    invariants::set_max_level(log::LevelFilter::Trace);
    let _ = env_logger::builder().is_test(true).filter_level(log::LevelFilter::Debug).try_init();
}

#[doc(hidden)]
pub mod test;
mod colors;
mod eggstentions;
pub mod tools;
pub mod colored_union_find;
mod multipattern;

