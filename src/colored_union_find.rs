use std::borrow::BorrowMut;
use std::cell::Cell;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::atomic::Ordering::Relaxed;
use indexmap::IndexMap;
use itertools::Itertools;
use crate::Id;

#[cfg(feature = "concurrent_cufind")]
type AtomicId = AtomicU32;
#[cfg(not(feature = "concurrent_cufind"))]
type AtomicId = Cell<u32>;

#[inline(always)]
fn load_id(id: &AtomicId) -> u32 {
    #[cfg(feature = "concurrent_cufind")]
        return id.load(Relaxed);
    #[cfg(not(feature = "concurrent_cufind"))]
        return id.get();
}

#[inline(always)]
fn store_id(id: &AtomicId, new: u32) {
    #[cfg(feature = "concurrent_cufind")]
    id.store(new, Relaxed);
    #[cfg(not(feature = "concurrent_cufind"))]
    {
        id.replace(new);
    }
}

#[inline(always)]
fn new_id(id: u32) -> AtomicId {
    #[cfg(feature = "concurrent_cufind")]
    return AtomicU32::new(id);
    #[cfg(not(feature = "concurrent_cufind"))]
    return Cell::new(id);
}

/// A type that can be used as an id in a union-find data structure.
///
/// This trait is implemented for hashable types, as a way to have a single object unionfind on complex data.
///
/// # Examples
///
/// ```
/// use egg::colored_union_find::ColoredUnionFind;
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
/// let expected = vec![0, 0, 0, 0, 4, 5, 6, 6, 6, 6].into_iter().map(|x| Id::from(x)).collect::<Vec<_>>();
/// for i in 0..n {
/// assert_eq!(uf.find(&Id(i)).unwrap(), expected[i as usize]);
/// }
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ColoredUnionFind {
    // The parents of each node. The index is T and we keep the maybe updated leader + rank.
    parents: IndexMap<Id, AtomicId>,
}

impl ColoredUnionFind {
    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, Id)> + '_ {
        self.parents.iter().map(|(k, v)| (k.clone(), Id(load_id(v))))
    }
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
        self.parents.insert(t, new_id(t.0));
    }

    fn inner_find(&self, current: &Id) -> Option<u32> {
        // If the current node is not in the map, it is not in the union-find.
        // All other cases node will point to parent or itself.
        if !self.parents.contains_key(current) {
            return None;
        }

        let mut old = *current;
        let mut current = load_id(&self.parents[&old]);
        let mut to_update = vec![];
        while current != old.0 {
            to_update.push(old.clone());
            old = Id(current);
            current = load_id(&self.parents[&old]);
        }

        let current = current;
        for u in to_update {
            store_id(&self.parents[&u], current);
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
        let new_x_res = load_id(&self.parents[&x]);
        store_id(&self.parents[&y],new_x_res);
        store_id(&self.parents[&x],new_x_res);
        Some((x, y))
    }

    /// Remove a node from the union-find. It will not remove the group, but it will remove a single node.
    /// Fails if the node is a leader.
    pub fn remove(&mut self, t: &Id, keys_to_check: Option<impl IntoIterator<Item = Id>>) -> Option<()> {
        let leader = self.inner_find(t)?;
        if leader == t.0 {
            return None;
        }
        if let Some(keys) = keys_to_check {
            for k in keys {
                let inner = self.inner_find(&Id(load_id(&self.parents[&k]))).unwrap();
                assert!(inner == t.0 || inner == leader);
                store_id(&self.parents[&k], leader);
            }
        } else {
            let keys = self.parents.iter()
                .filter(|(_k, v)| load_id(v) == t.0)
                .map(|(k, _v)| k)
                .copied()
                .collect_vec();
            for k in keys {
                store_id(&mut self.parents[&k], leader);
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
                (k.clone(), new_id(load_id(v)))).collect(),
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
