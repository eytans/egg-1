use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::atomic::Ordering::Relaxed;
use indexmap::IndexMap;
use itertools::Itertools;
use crate::Id;

type AtomicId = AtomicU32;

/// A type that can be used as an id in a union-find data structure.
///
/// This trait is implemented for hashable types, as a way to have a single object unionfind on complex data.
///
/// # Examples
///
/// ```
/// use colored_union_find::ColoredUnionFind;
/// use egg::Id;
///
/// let n = 10;
///
/// let mut uf = ColoredUnionFind::default();
/// for i in 0..n {
/// uf.insert(Id(i));
/// }
///
/// // build up one set
/// uf.union(&Id(0), &Id(1));
/// uf.union(&Id(0), &Id(2));
/// uf.union(&Id(0), &Id(3));
///
/// // build up another set
/// uf.union(&Id(6), &Id(7));
/// uf.union(&Id(6), &Id(8));
/// uf.union(&Id(6), &Id(9));
///
/// // indexes:         0, 1, 2, 3, 4, 5, 6, 7, 8, 9
/// let expected = vec![0, 0, 0, 0, 4, 5, 6, 6, 6, 6].into_iter().map(Id).collect::<Vec<_>>();
/// for i in 0..n {
/// assert_eq!(uf.find(&Id(i)).unwrap(), expected[i as usize]);
/// }
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct ColoredUnionFind {
    // The parents of each node. The index is T and we keep the maybe updated leader + rank.
    parents: IndexMap<Id, AtomicId>,
}

impl ColoredUnionFind {
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.parents.len()
    }

    // Create a new set from the element t.
    pub fn insert(&mut self, t: Id) {
        if self.parents.contains_key(&t) {
            return;
        }
        self.parents.insert(t, AtomicU32::new(t.0));
    }

    fn inner_find(&self, current: &Id) -> Option<u32> {
        // If the current node is not in the map, it is not in the union-find.
        // All other cases node will point to parent or itself.
        if !self.parents.contains_key(current) {
            return None;
        }

        let mut old = *current;
        let mut current = self.parents[&old].load(Relaxed);
        let mut to_update = vec![];
        while current != old.0 {
            to_update.push(old.clone());
            old = Id(current);
            current = self.parents[&old].load(Relaxed)
        }

        let current = current;
        for u in to_update {
            self.parents[&u].store(current, Ordering::Relaxed);
        }

        Some(current)
    }

    // Find the leader of the set that t is in. This is amortized to O(log*(n))
    pub fn find(&self, current: &Id) -> Option<Id> {
        self.inner_find(current).map(|leader| leader).map(Id)
    }

    /// Given two ids, unions the two eclasses making the bigger class the leader.
    /// If one of the items is missing returns None, otherwize return Some(to, from).
    pub fn union(&mut self, x: &Id, y: &Id) -> Option<(Id, Id)> {
        let mut x = self.inner_find(x)?;
        let mut y = self.inner_find(y)?;
        if x == y {
            return Some((Id(x), Id(y)));
        }
        if x > y {
            std::mem::swap(&mut x, &mut y);
        }
        let x = Id(x);
        let y = Id(y);
        let new_x_res = self.parents[&x].load(Ordering::Relaxed);
        self.parents[&y].store(new_x_res, Ordering::Relaxed);
        self.parents[&x].store(new_x_res, Ordering::Relaxed);
        Some((x, y))
    }

    /// Remove a node from the union-find. It will not remove the group, but it will remove a single node.
    /// Fails if the node is a leader.
    pub fn remove(&mut self, t: &Id, keys_to_check: Option<impl Iterator<Item = Id>>) -> Option<()> {
        let leader = self.inner_find(t)?;
        if leader == t.0 {
            return None;
        }
        if let Some(keys) = keys_to_check {
            for k in keys {
                let inner = self.inner_find(&Id(self.parents[&k].load(Relaxed))).unwrap();
                assert!(inner == t.0 || inner == leader);
                self.parents[&k].store(leader, Relaxed);
            }
        } else {
            let keys = self.parents.keys().filter(|k| k.0 == t.0).copied().collect_vec();
            for k in keys {
                self.parents[&k].store(leader, Relaxed);
            }
        }
        self.parents.remove(t);
        Some(())
    }
}

impl Clone for ColoredUnionFind {
    fn clone(&self) -> Self {
        Self {
            parents: self.parents.iter().map(|(k, v)|
                (k.clone(), AtomicU32::new(v.load(Relaxed)))).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_find() {
        let n = 10;

        let mut uf = ColoredUnionFind::default();
        for i in 0..n {
            uf.insert(Id(i));
        }

        // build up one set
        uf.union(&Id(0), &Id(1));
        uf.union(&Id(0), &Id(2));
        uf.union(&Id(0), &Id(3));

        // build up another set
        uf.union(&Id(6), &Id(7));
        uf.union(&Id(6), &Id(8));
        uf.union(&Id(6), &Id(9));

        // indexes:         0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        let expected = vec![0, 0, 0, 0, 4, 5, 6, 6, 6, 6].into_iter().map(Id).collect::<Vec<_>>();
        for i in 0..n {
            assert_eq!(uf.find(&Id(i)).unwrap(), expected[i as usize]);
        }
    }

    #[test]
    fn test_on_str() {
        let mut uf = ColoredUnionFind::default();
        let a = Id(0);
        let b = Id(1);
        let c = Id(2);
        let d = Id(3);
        let e = Id(4);
        let x = Id(5);
        uf.insert(a);
        uf.insert(b);
        uf.insert(c);
        uf.insert(d);
        uf.insert(e);

        uf.union(&a, &b);
        uf.union(&b, &c);

        uf.union(&d, &e);

        assert_eq!(None, uf.union(&x, &a));
        assert_eq!(None, uf.union(&a, &x));
        assert_eq!(None, uf.find(&x));

        assert_eq!(uf.find(&a), uf.find(&c));
        assert_ne!(uf.find(&a), uf.find(&d));

        uf.union(&a, &d);

        assert_eq!(uf.find(&a), uf.find(&e));
        assert_eq!(a, uf.find(&a).unwrap());
    }
}
