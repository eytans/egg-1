use crate::Id;
use std::fmt::Debug;
use bimap::BiBTreeMap;
use itertools::Itertools;

pub trait UnionFind {
    /// Creates a new set with a single element.
    fn make_set(&mut self) -> Id;

    /// Returns the number of elements in the union find.
    fn size(&self) -> usize;

    /// Finds the leader of the set that `current` is in.
    fn find(&self, current: Id) -> Id;

    /// Finds the leader of the set that `current` is in.
    /// This version updates the parents to compress the path.
    fn find_mut(&mut self, current: Id) -> Id;

    /// Given two leader ids, unions the two eclasses.
    /// This should run find to compress paths
    /// Returns (new leader, other id found).
    fn union(&mut self, root1: Id, root2: Id) -> (Id, Id);
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SimpleUnionFind {
    parents: Vec<Id>,
}

impl SimpleUnionFind {
    pub(crate) fn parent(&self, query: Id) -> Id {
        self.parents[usize::from(query)]
    }

    pub(crate) fn parent_mut(&mut self, query: Id) -> &mut Id {
        &mut self.parents[usize::from(query)]
    }

    /// This is needed for deserialization
    pub(crate) fn make_set_at(&mut self, id: Id) -> Id {
        while self.parents.len() <= usize::from(id) { self.make_set(); }
        id
    }
}

impl UnionFind for SimpleUnionFind {
    fn make_set(&mut self) -> Id {
        let id = Id::from(self.parents.len());
        self.parents.push(id);
        id
    }
    fn size(&self) -> usize {
        self.parents.len()
    }
    fn find(&self, mut current: Id) -> Id {
        while current != self.parent(current) {
            current = self.parent(current)
        }
        current
    }
    fn find_mut(&mut self, mut current: Id) -> Id {
        let mut collected = vec![];
        while current != self.parent(current) {
            collected.push(current);
            current = self.parent(current);
        }
        for c in collected {
            *self.parent_mut(c) = current;
        }
        current
    }
    /// Given two leader ids, unions the two eclasses making root1 the leader.
    fn union(&mut self, root1: Id, root2: Id) -> (Id, Id) {
        let root1 = self.find_mut(root1);
        let root2 = self.find_mut(root2);
        if root1 > root2 {
            return self.union(root2, root1);
        }
        *self.parent_mut(root2) = root1;
        (root1, root2)
    }
}

impl SimpleUnionFind {
    pub(crate) fn union_no_swap(&mut self, root1: Id, root2: Id) -> (Id, Id) {
        let root1 = self.find_mut(root1);
        let root2 = self.find_mut(root2);
        *self.parent_mut(root2) = root1;
        (root1, root2)
    }
}

/// Data inside the union find wrapper should implement a merge function.
pub trait Merge {
    fn merge(&mut self, other: Self);
}

impl Merge for () {
    fn merge(&mut self, _: Self) {}
}

/// A wrapper for other union find implementations [U].
/// This "translates" keys [K] to the internal representation so that external api can use any key.
/// It also holds an object [T] for each equivalence class which is unioned with the [merge] function.
/// It won't implement the union find api right now because I don't want to change it at the moment
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UnionFindWrapper<T: Merge, K: Clone + Ord> {
    uf: SimpleUnionFind,
    data: Vec<Option<T>>,
    trns: BiBTreeMap<K, usize>
}

// Implement an iteration over inner and outer keys
impl<T: Merge, K: Clone + Ord> UnionFindWrapper<T, K> {
    pub fn iter(&self) -> impl Iterator<Item = (&K, &usize)> {
        self.trns.iter()
    }
}

impl<T: Merge, K: Clone + Ord> UnionFindWrapper<T, K> {
    fn get(&self, key: &K) -> Option<&T> {
        let res = self.trns.get_by_left(&key)
            .map(|&idx| &self.data[self.uf.find(idx.into()).0 as usize]);
        assert!(res.is_some() || self.trns.get_by_left(&key).is_none());
        res?.as_ref()
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&T> {
        let idx = self.trns.get_by_left(&key)?;
        let res = &self.data[self.uf.find_mut((*idx).into()).0 as usize];
        assert!(res.is_some() || self.trns.get_by_left(&key).is_none());
        res.as_ref()
    }

    pub fn insert(&mut self, key: K, value: T) {
        if self.trns.contains_left(&key) {
            return;
        }
        let idx = self.data.len();
        self.data.push(Some(value));
        let id = self.uf.make_set();
        assert_eq!(id.0, idx as u32);
        self.trns.insert(key, idx);
    }

    pub fn union(&mut self, key1: &K, key2: &K) -> Option<(K, K)> {
        let mut idx1 = *self.trns.get_by_left(&key1)?;
        let mut idx2 = *self.trns.get_by_left(&key2)?;
        // I need to union by the keys of trns
        let mut k1 = self.uf.find_mut(idx1.into());
        let mut k2 = self.uf.find_mut(idx2.into());
        if self.trns.get_by_right(&(k1.0 as usize)) > self.trns.get_by_right(&(k2.0 as usize)) {
            std::mem::swap(&mut k1, &mut k2);
            std::mem::swap(&mut idx1, &mut idx2);
        }
        let (root1, root2) = self.uf.union_no_swap(k1, k2);
        if root1 != root2 {
            let old2 = std::mem::take(&mut self.data[root2.0 as usize]).unwrap();
            self.data[root1.0 as usize].as_mut().unwrap().merge(old2);
        }
        let key1 = self.trns.get_by_right(&(root1.0 as usize))?;
        let key2 = self.trns.get_by_right(&(root2.0 as usize))?;
        Some((key1.clone(), key2.clone()))
    }

    pub fn find(&self, key: &K) -> Option<K> {
        let idx = *self.trns.get_by_left(&key)?;
        let root = self.uf.find(idx.into());
        self.trns.get_by_right(&(root.0 as usize)).cloned()
    }

    pub fn find_mut(&mut self, key: &K) -> Option<K> {
        let idx = *self.trns.get_by_left(&key)?;
        let root = self.uf.find_mut(idx.into());
        self.trns.get_by_right(&(root.0 as usize)).cloned()
    }

    /// Remove a node from the union-find. It will not remove the group, but it will remove a single node.
    /// Fails if the node is a leader.
    pub fn remove(&mut self, t: &K, keys_to_check: Option<impl IntoIterator<Item = K>>) -> Option<()> {
        let t_i = *self.trns.get_by_left(t)?;
        let leader = self.find_mut(t)?;
        let leader_i = *self.trns.get_by_left(&leader)?;
        if &leader == t {
            return None;
        }
        if let Some(keys) = keys_to_check {
            for k in keys {
                let k_i = *self.trns.get_by_left(&k).unwrap();
                let k_lead = self.find_mut(&k).unwrap();
                // Should have updated the leader of k
                assert!(k_i == t_i || k_lead == leader);
                assert!(self.trns.get_by_right(&(self.uf.parent(k_i.into()).0 as usize)).unwrap() == &leader);
            }
        } else {
            let keys = self.uf.parents.iter()
                .filter(|k| k.0 as usize == t_i)
                .copied()
                .collect_vec();
            for k in keys {
                // Update them all to the leader
                *self.uf.parent_mut(k) = leader_i.into();
            }
        }
        self.data[t_i] = None;
        self.trns.remove_by_left(t);
        Some(())
    }
}

impl<T:Default + Merge, K: Clone + Ord + Debug> UnionFindWrapper<T, K> {
    pub(crate) fn debug_print_all(&self) {
        for (k, v) in self.trns.iter() {
            println!("{:?}: {:?}", k, v);
        }
        for p in &self.uf.parents {
            println!("{:?}", p);
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::unionfind::SimpleUnionFind;
    use super::*;

    fn ids(us: impl IntoIterator<Item = usize>) -> Vec<Id> {
        us.into_iter().map(|u| u.into()).collect()
    }

    #[test]
    fn union_find() {
        let n = 10;
        let id = Id::from;

        let mut uf = SimpleUnionFind::default();
        for _ in 0..n {
            uf.make_set();
        }

        // test the initial condition of everyone in their own set
        assert_eq!(uf.parents, ids(0..n));

        // build up one set
        uf.union(id(0), id(1));
        uf.union(id(0), id(2));
        uf.union(id(0), id(3));

        // build up another set
        uf.union(id(6), id(7));
        uf.union(id(6), id(8));
        uf.union(id(6), id(9));

        // this should compress all paths
        for i in 0..n {
            uf.find_mut(id(i));
        }

        // indexes:         0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        let expected = vec![0, 0, 0, 0, 4, 5, 6, 6, 6, 6];
        assert_eq!(uf.parents, ids(expected));
    }
}
