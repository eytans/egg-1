use std::fmt::Debug;
use std::iter::ExactSizeIterator;

use crate::{ColorId, Id, Language};
use crate::egraph::DenseNodeColors;

pub type SparseNodeColors = Vec<ColorId>;

/// An equivalence class of enodes.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct EClass<L, D> {
    /// This eclass's id.
    pub id: Id,
    /// The equivalent enodes in this equivalence class with their associated colors.
    /// No color set means it is a black edge (which will never be removed).
    pub nodes: Vec<(L, SparseNodeColors)>,
    /// The analysis data associated with this eclass.
    pub data: D,
    pub(crate) parents: Vec<(L, DenseNodeColors, Id)>,
    /// Some colors not yet added to 'nodes', or clear if none.
    pub(crate) dirty_colors: Vec<(L, ColorId)>,
}

impl<L, D> EClass<L, D> {
    /// Returns `true` if the `eclass` is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the number of enodes in this eclass.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Iterates over the enodes in this eclass.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &(L, SparseNodeColors)> {
        self.nodes.iter()
    }
}

impl<L: Language, D> EClass<L, D> {
    /// Iterates over the childless enodes in this eclass.
    pub fn leaves(&self) -> impl Iterator<Item = &L> {
        self.nodes.iter().map(|(n, cs)| n).filter(|n| n.is_leaf())
    }

    /// Asserts that the childless enodes in this eclass are unique.
    pub fn assert_unique_leaves(&self)
    where
        L: Language,
    {
        let mut leaves = self.leaves();
        if let Some(first) = leaves.next() {
            assert!(
                leaves.all(|l| l == first),
                "Different leaves in eclass {}: {:?}",
                self.id,
                self.leaves().collect::<indexmap::IndexSet<_>>()
            );
        }
    }
}
