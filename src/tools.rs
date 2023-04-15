pub mod tools {
    use std::collections::hash_map::RandomState;
    use std::hash::Hash;

    use itertools::MultiProduct;
    use itertools::Itertools;
    use indexmap::IndexMap;

// fn combinations<'a, T: 'a, I: Iterator<Item = &'a T> + Clone>(mut sets: impl Iterator<Item = I>) -> impl Iterator<Item = Vec<&'a T>> {
//     let first = sets.next();
//     let second = sets.next();
//     if first.is_none() || second.is_none() {
//         return iter::empty();
//     }
//
//     let initial = Itertools::cartesian_product(first.unwrap(), second.unwrap())
//         .map(|p| vec![p.0, p.1]);
//     let res = sets.fold(initial, |res, i| Itertools::cartesian_product(res, i));
//     res.unwrap_or(iter::empty())
// }

    pub fn product<'a, T: 'a + Clone>(vecs: &[&'a Vec<T>]) -> Vec<Vec<&'a T>> {
        if vecs.is_empty() {
            return vec![vec![]];
        }

        if vecs.len() == 1 {
            return vecs[0].iter().map(|t| vec![t]).collect();
        }

        let rec_res = product(&vecs[1..vecs.len()]);
        let initial_set = &vecs[0];
        let mut res = Vec::new();
        for s in initial_set.iter() {
            for r in rec_res.iter() {
                let mut new_r = r.clone();
                new_r.push(s);
                res.push(new_r)
            }
        }

        return res;
    }

    #[allow(dead_code)]
    pub(crate) fn combinations<T: Clone, I: Clone + Iterator<Item=T>>(iters: impl Iterator<Item=I>) -> MultiProduct<I> {
        iters.multi_cartesian_product()
    }

    pub fn choose<K>(vec: &[K], size: usize) -> Vec<Vec<&K>> {
        if size == 1 {
            let mut res = Vec::default();
            vec.iter().for_each(|k| res.push(vec![k.clone()]));
            return res;
        }
        if size == 0 || size > vec.len() {
            return Vec::default();
        }

        let mut res = Vec::default();
        for (i, k) in vec.iter().enumerate() {
            let mut rec_res = choose(&vec[i + 1..], size - 1);
            for v in rec_res.iter_mut() {
                v.push(k);
            }
            res.extend(rec_res);
        }
        res
    }

    pub trait Grouped<T> {
        fn grouped<K: Hash + Eq, F: Fn(&T) -> K>(&mut self, key: F) -> IndexMap<K, Vec<T>>;
    }

    impl<T, I: Iterator<Item=T>> Grouped<T> for I {
        fn grouped<K: Hash + Eq, F: Fn(&T) -> K>(&mut self, key: F) -> IndexMap<K, Vec<T>, RandomState> {
            let mut res = IndexMap::new();
            self.for_each(|t| res.entry(key(&t)).or_insert(Vec::new()).push(t));
            res
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::FromIterator;
    use indexmap::IndexSet;

    use itertools::Itertools;

    use crate::tools::tools::choose;
    use crate::tools::tools::combinations;

    #[test]
    fn check_comb_amount() {
        let v1 = vec![1, 2, 3];
        let v2 = vec![4, 5, 6];
        let combs = combinations(vec![v1.iter(), v2.iter()].into_iter()).collect_vec();
        assert_eq!(combs.len(), 9);
        for v in &combs {
            assert_eq!(v.len(), 2);
        }
        // No doubles
        let as_set: IndexSet<&Vec<&i32>> = IndexSet::from_iter(combs.iter());
        assert_eq!(as_set.len(), 9);
    }

    #[test]
    fn check_choose_amount() {
        let v3 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        for i in 1..9 {
            let chosen = choose(&v3, i);
            for v in &chosen {
                assert_eq!(v.len(), i);
            }
            let as_set: IndexSet<&Vec<&i32>> = IndexSet::from_iter(chosen.iter());
            assert_eq!(chosen.len(), as_set.len());
        }
        assert_eq!(choose(&v3, 2).len(), 36);
    }
}