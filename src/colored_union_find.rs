#[allow(unused_imports)]
use std::borrow::BorrowMut;
#[allow(unused_imports)]
use std::cell::Cell;
use std::fmt::Debug;
#[allow(unused_imports)]
use std::sync::atomic::{AtomicU32, Ordering};
#[allow(unused_imports)]
use std::sync::atomic::Ordering::Relaxed;
use bimap::BiBTreeMap;
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

/// A type that can be used as an id in a union-find data structure.
///
/// This trait is implemented for hashable types, as a way to have a single object unionfind on complex data.
///
/// # Examples
///
/// ```
/// use easter_egg::colored_union_find::ColoredUnionFind;
/// use easter_egg::Id;
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
#[cfg_attr(not(feature = "concurrent_cufind"), derive(Clone))]
pub struct ColoredUnionFind {
    translation: BiBTreeMap<Id, usize>,
    // The parents of each node. The index is T and we keep the maybe updated leader + rank.
    parents: Vec<AtomicId>,
}

impl ColoredUnionFind {
    /// This should only be used for debug assertions and debugging
    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id, Id)> + '_ {
        let mut parents = vec![];
        for p in &self.parents {
            parents.push(load_id(p));
        }
        self.translation.iter().map(move |(k, v)| (*k, Id(load_id(&parents[*v].into()))))
    }
}

#[cfg(feature = "concurrent_cufind")]
impl Clone for ColoredUnionFind {
    fn clone(&self) -> Self {
        let mut parents = vec![];
        for p in &self.parents {
            parents.push(p.load(Relaxed));
        }
        Self {
            translation: self.translation.clone(),
            parents: parents.into_iter().map(|x| AtomicU32::new(x)).collect(),
        }
    }
    
}

impl ColoredUnionFind {
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.translation.len()
    }

    // Create a new set from the element t.
    pub fn insert(&mut self, t: Id) {
        if self.translation.contains_left(&t) {
            return;
        }
        self.translation.insert(t, self.parents.len());
        
        self.parents.push(Self::into_wrapped(self.parents.len()));
    }
    
    fn into_wrapped(x: usize) -> AtomicId {
        #[cfg(feature = "concurrent_cufind")]
        return AtomicU32::new(x as u32);
        #[cfg(not(feature = "concurrent_cufind"))]
        return Cell::new(x as u32);
    }

    fn inner_find(&self, current: usize) -> usize {
        let mut old = current;
        let mut current = load_id(&self.parents[old]) as usize;
        let mut to_update = vec![];
        while current != old {
            to_update.push(old);
            old = current;
            current = load_id(&self.parents[old]) as usize;
        }

        for u in to_update {
            store_id(&self.parents[u], current as u32);
        }

        current
    }

    // Find the leader of the set that t is in. This is amortized to O(log*(n))
    pub fn find(&self, current: &Id) -> Option<Id> {
        self.translation.get_by_left(current)
            .map(|x| self.inner_find(*x))
            .map(|x| self.translation.get_by_right(&x)).flatten().copied()
    }

    /// Given two ids, unions the two eclasses making the bigger class the leader.
    /// If one of the items is missing returns None, otherwize return Some(to, from).
    pub fn union(&mut self, x: &Id, y: &Id) -> Option<(Id, Id)> {
        let mut x_key = *self.translation.get_by_left(x)?;
        let mut y_key = *self.translation.get_by_left(y)?;
        x_key = self.inner_find(x_key);
        y_key = self.inner_find(y_key);
        let mut x_res = self.translation.get_by_right(&x_key).unwrap();
        let mut y_res = self.translation.get_by_right(&y_key).unwrap();
        if x_res > y_res {
            std::mem::swap(&mut x_key, &mut y_key);
            std::mem::swap(&mut x_res, &mut y_res);
        }
        if x_key != y_key {
            store_id(&self.parents[y_key], x_key as u32);
            store_id(&self.parents[x_key], x_key as u32);
        }
        return Some((*x_res, *y_res));
    }

    /// Remove a node from the union-find. It will not remove the group, but it will remove a single node.
    /// Fails if the node is a leader.
    pub fn remove(&mut self, t: &Id, keys_to_check: Option<impl IntoIterator<Item = Id>>) -> Option<()> {
        let t_i = *self.translation.get_by_left(t)?;
        let leader = self.inner_find(t_i);
        if leader == t_i {
            return None;
        }
        if let Some(keys) = keys_to_check {
            for k in keys {
                let k_i = *self.translation.get_by_left(&k).unwrap();
                let inner = self.inner_find(k_i);
                assert!(inner == t_i || inner == leader);
                store_id(&self.parents[k_i], leader as u32);
            }
        } else {
            let keys = self.parents.iter()
                .filter(|k| load_id(*k) as usize == t_i)
                .collect_vec();
            for k in keys {
                store_id(&self.parents[load_id(k) as usize], leader as u32);
            }
        }
        self.translation.remove_by_left(t);
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use crate::init_logger;
    use super::*;

    #[test]
    fn union_find() {
        init_logger();

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
        init_logger();

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
