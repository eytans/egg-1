#![allow(unused_imports)]
use derive_new::new;
use crate::{unionfind::UnionFindWrapper, Analysis, Id, Language};

// /// Explanation object collects all information needed to explain existance and equality of enodes in the egraph.
// /// It holds a mapping from an Id to the node added when said Id was first added to the egraph.
// /// It also holds a union-find structure to query equivalencies between nodes that isn't merged upwards.
// #[derive(Debug, new)]
// pub struct Explanation<L: Language, N: Analysis<L>> {
//     /// Just handle everything with a non collapsing union find.
//     uf: UnionFindWrapper<(), Id>,
// }