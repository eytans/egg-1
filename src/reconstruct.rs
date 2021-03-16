// use std::collections::{HashMap, HashSet};
// use std::iter;
// use std::rc::Rc;
//
// use egg::{EGraph, Id, Language, SymbolLang, RecExpr, EClass};
// use itertools::Itertools;
// use crate::{EGraph, SymbolLang, Id, RecExpr, EClass, Language};
//
//
// pub fn reconstruct(graph: &EGraph<SymbolLang, ()>, class: Id, max_depth: usize) -> Option<RecExpr<SymbolLang>> {
//     let mut translations: HashMap<Id, RecExpr<SymbolLang>> = HashMap::new();
//     let classes = graph.classes().into_iter().map(|c| (c.id, c)).collect();
//     reconstruct_inner(&classes, class, max_depth, &mut translations);
//     translations.get(&class).map(|x| x.clone())
// }
//
// pub fn reconstruct_edge(graph: &EGraph<SymbolLang, ()>, class: Id, edge: SymbolLang, max_depth: usize) -> Option<RecExpr<SymbolLang>> {
//     let mut translations: HashMap<Id, RecExpr<SymbolLang>> = HashMap::new();
//     let classes = graph.classes().into_iter().map(|c| (c.id, c)).collect();
//     for child in &edge.children {
//         reconstruct_inner(&classes, *child, max_depth - 1, &mut translations);
//     }
//     build_translation(&mut translations, &edge, class);
//     translations.get(&class).map(|x| x.clone())
// }
//
// fn reconstruct_inner(classes: &HashMap<Id, &EClass<SymbolLang, ()>>, class: Id, max_depth: usize, translations: &mut HashMap<Id, RecExpr<SymbolLang>>) {
//     if max_depth == 0 || translations.contains_key(&class) {
//         return;
//     }
//     let cur_class = classes.get(&class).unwrap();
//     let mut inner_ids = vec![];
//     for edge in &cur_class.nodes {
//         if edge.children.iter().all(|c| translations.contains_key(c)) {
//             build_translation(translations, &edge, class);
//             return;
//         }
//         inner_ids.push(edge);
//     }
//     inner_ids.sort_by_key(|c| c.children.len());
//     for edge in inner_ids {
//         for id in &edge.children {
//             reconstruct_inner(classes, *id, max_depth - 1, translations);
//         }
//         if edge.children.iter().all(|c| translations.contains_key(c)) {
//             build_translation(translations, &edge, class);
//             return;
//         }
//     }
// }
//
// fn build_translation(translations: &mut HashMap<Id, RecExpr<SymbolLang>>, edge: &SymbolLang, id: Id) {
//     let mut res = vec![];
//     let children = edge.children.iter().map(|c| {
//         let cur_len = res.len();
//         res.extend(translations[&c].as_ref().iter().cloned().map(|s| s.map_children(|child| Id::from(usize::from(child) + cur_len))));
//         Id::from(res.len() - 1)
//     }).collect_vec();
//     res.push(SymbolLang::new(edge.op, children));
//     translations.insert(id, RecExpr::from(res));
// }
//
// pub fn reconstruct_all(graph: &EGraph<SymbolLang, ()>, max_depth: usize) -> HashMap<Id, HashSet<Rc<Tree>>> {
//     for c in graph.classes() {
//         let set = c.nodes.iter().collect::<HashSet<&SymbolLang>>();
//         for c1 in graph.classes() {
//             for n in c1.nodes.iter() {
//                 assert!(!set.contains(n) || c.id == c1.id)
//             }
//         }
//     }
//     let edges = graph.classes().filter(|c| graph.find(c.id) == c.id).flat_map(|c| iter::once(c.id).cycle().zip(c.nodes.iter())).collect_vec();
//     // let idToType = inputEdges.filter(_.edgeType.identifier == Language.typeId).map(e => (e.sources(0), Programs.reconstruct(inputEdges, e.sources(1)).head)).toMap
//     let mut known_terms: HashMap<Id, HashSet<Rc<Tree>>> = HashMap::new();
//     let mut last_level: HashSet<(Id, Rc<Tree>)> = edges.iter().filter_map(|(id, e)| {
//         if e.is_leaf() {
//             let tree = Rc::new(Tree::leaf(e.op.to_string()));
//             if !known_terms.contains_key(id) {
//                 known_terms.insert(*id, HashSet::new());
//             }
//             known_terms.get_mut(id).unwrap().insert(tree.clone());
//             Some((*id, tree))
//         } else {
//             None
//         }
//     }).collect();
//     let mut levels = vec![last_level];
//
//     // id to edges that might be available
//     // let edges_by_reqiurments = collection.mutable.HashMultiMap.empty[HyperTermId, (Int, HyperEdge[HyperTermId, HyperTermIdentifier])]
//     let mut edges_by_reqiurments = HashMap::new();
//     for x in edges.iter() {
//         for (i, s) in x.1.children.iter().enumerate() {
//             assert_eq!(graph.find(*s), *s);
//             if !edges_by_reqiurments.contains_key(s) {
//                 edges_by_reqiurments.insert(*s, HashSet::new());
//             }
//             edges_by_reqiurments.get_mut(s).unwrap().insert((i, *x));
//         }
//     };
//
//     // The reconstruct itself.
//     for d in 0..max_depth {
//         info!("depth {}", d);
//         last_level = HashSet::new();
//         for e in levels.last().unwrap() {
//             for v in edges_by_reqiurments.get(&e.0) {
//                 for (i_to_fill, e_to_fill) in v {
//                     let params_to_fill = e_to_fill.1.children.iter().take(*i_to_fill).chain(&e_to_fill.1.children[i_to_fill + 1..]);
//                     let trees = params_to_fill.filter_map(|h_id| known_terms.get(h_id).map_or(None, |x| Some(x.iter().cloned()))).collect_vec();
//                     if trees.len() == e_to_fill.1.children.len() - 1 {
//                         for mut fillers in combinations(trees.into_iter()) {
//                             fillers.insert(*i_to_fill, e.1.clone());
//                             last_level.insert((e.0, Rc::new(Tree::branch(e_to_fill.1.op.to_string(), fillers))));
//                         }
//                     }
//                 }
//             }
//         }
//
//         for e in last_level.clone() {
//             if !known_terms.contains_key(&e.0) {
//                 known_terms.insert(e.0, HashSet::new());
//             }
//             known_terms.get_mut(&e.0).unwrap().insert(e.1);
//         }
//
//         levels.push(last_level);
//     }
//
//     known_terms
//     // known_terms.into_iter().flat_map(|x| iter::once(x.0).cycle().zip(x.1.into_iter())).collect::<HashSet<(Id, Rc<Tree>)>>()
// }
