use std::cmp::Ordering;

use crate::{CostFunction, Id, SymbolLang};
use itertools::Itertools;

/// A cost function that minimizes the depth, number of nodes and maximizes variables in the egraph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepOrder {
    vars: Vec<String>,
    depth: usize,
    size: usize,
}

impl RepOrder {
    /// Returns the current depth.
    pub fn get_depth(&self) -> usize {
        self.depth
    }

    fn count_ph1(it: &Vec<String>) -> usize {
        it.iter().filter(|x| x.ends_with("1")).count()
    }

    fn compare_vars(&self, other: &Self) -> Option<Ordering> {
        match self.vars.iter().unique().count().partial_cmp(&other.vars.iter().unique().count()) {
            None => { Self::count_ph1(&self.vars).partial_cmp(&Self::count_ph1(&other.vars)) }
            Some(ord) => { match ord {
                Ordering::Less => { Some(Ordering::Less) }
                Ordering::Equal => { Self::count_ph1(&self.vars).partial_cmp(&Self::count_ph1(&other.vars)) }
                Ordering::Greater => { Some(Ordering::Greater) }
            }}
        }

    }
}

impl PartialOrd for RepOrder {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.size.partial_cmp(&other.size) {
            None => { other.compare_vars(self) }
            Some(x) => {
                match x {
                    Ordering::Less => {Some(Ordering::Less)}
                    Ordering::Equal => { other.compare_vars(self) }
                    Ordering::Greater => {Some(Ordering::Greater)}
                }
            }
        }
    }
}

impl Ord for RepOrder {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// A struct to create a cost function out of RepOrder.
pub struct MinRep;

impl CostFunction<SymbolLang> for MinRep {
    /// The `Cost` type. It only requires `PartialOrd` so you can use
    /// floating point types, but failed comparisons (`NaN`s) will
    /// result in a panic.
    /// We choose to count
    type Cost = RepOrder;

    /// Calculates the cost of an enode whose children are `Cost`s.
    ///
    /// For this to work properly, your cost function should be
    /// _monotonic_, i.e. `cost` should return a `Cost` greater than
    /// any of the child costs of the given enode.
    fn cost<C>(&mut self, enode: &SymbolLang, mut costs: C) -> Self::Cost where
        C: FnMut(Id) -> Self::Cost {
        let current_depth = enode.children.iter().map(|i| costs(*i).depth).max().unwrap_or(0);
        let current_size = enode.children.iter().map(|i| costs(*i).size).sum1().unwrap_or(0);
        let mut vars = enode.children.iter().flat_map(|i| costs(*i).vars).collect_vec();
        if enode.op.as_str().starts_with("ts_ph") {
            vars.push(enode.op.to_string());
        }
        RepOrder{depth: current_depth + 1, size: current_size + 1, vars}
    }
}

#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use crate::eggstentions::costs::RepOrder;

    #[test]
    fn compare_two_different_sizes() {
        assert!(RepOrder{vars: Vec::new(), depth: 0, size: 1} < RepOrder{vars: Vec::new(), depth: 0, size: 2});
        assert!(RepOrder{vars: Vec::from_iter(vec![":".to_string(), "a".to_string(), "b".to_string()]), depth: 0, size: 1} < RepOrder{vars: Vec::new(), depth: 0, size: 2});
    }

    #[test]
    fn compare_two_different_vars() {
        assert!(RepOrder{vars: Vec::from_iter(vec![":".to_string(), "a".to_string(), "b".to_string()]), depth: 0, size: 2} < RepOrder{vars:  Vec::from_iter(vec![":".to_string(), "a".to_string()]), depth: 0, size: 2});
    }
}