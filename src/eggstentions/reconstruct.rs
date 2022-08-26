use std::iter;
use std::rc::Rc;

use crate::{EGraph, Id, Language, SymbolLang, RecExpr, EClass, ColorId};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use log::info;

use crate::tools::tools::combinations;
use crate::eggstentions::tree::Tree;

pub fn reconstruct(graph: &EGraph<SymbolLang, ()>, class: Id, max_depth: usize)
    -> Option<RecExpr<SymbolLang>> {
    reconstruct_colored(graph, None, class, max_depth)
}

pub fn reconstruct_colored(graph: &EGraph<SymbolLang, ()>, color: Option<ColorId>, class: Id, max_depth: usize) -> Option<RecExpr<SymbolLang>> {
    let mut translations: IndexMap<Id, RecExpr<SymbolLang>> = IndexMap::new();
    let class = graph.find(class);
    reconstruct_inner(&graph, class, max_depth, color, &mut translations);
    translations.get(&class).map(|x| x.clone())
}

pub fn reconstruct_edge(graph: &EGraph<SymbolLang, ()>, class: Id, edge: SymbolLang, max_depth: usize) -> Option<RecExpr<SymbolLang>> {
    let mut translations: IndexMap<Id, RecExpr<SymbolLang>> = IndexMap::new();
    for child in &edge.children {
        reconstruct_inner(&graph, *child, max_depth - 1, None, &mut translations);
    }
    build_translation(graph, None, &mut translations, &edge, class);
    translations.get(&class).map(|x| x.clone())
}

fn reconstruct_inner(graph: &EGraph<SymbolLang, ()>, class: Id, max_depth: usize,
                     color: Option<ColorId>, translations: &mut IndexMap<Id, RecExpr<SymbolLang>>) {
    if max_depth == 0 || translations.contains_key(&class) {
        return;
    }
    let cur_class = &graph[class];
    let mut inner_ids = vec![];
    check_class(graph, color, class, translations, &mut inner_ids, &cur_class);
    color.map(|c| {
        for x in graph.get_color(c) {
            let ids = x.black_ids(class);
            if let Some(ids) = ids {
                for id in ids {
                    let colorded_class = &graph[*id];
                    check_class(graph, color, *id, translations, &mut inner_ids, &colorded_class)
                }
            }
        }
    });
    inner_ids.sort_by_key(|c| c.children.len());
    for edge in inner_ids {
        for id in &edge.children {
            reconstruct_inner(graph, *id, max_depth - 1, color, translations);

        }
        if edge.children.iter().all(|c| translations.contains_key(c) ||
            color.map_or(false, |c_id| graph.get_color(c_id).map_or(false, |x|
                x.black_ids(class).map_or(false, |ids|
                    ids.iter().find(|id| translations.contains_key(*id)).is_some())))) {
            build_translation(graph, color, translations, &edge, class);
            return;
        }
    }
}

fn check_class<'a>(graph: &EGraph<SymbolLang, ()>, color: Option<ColorId>, class: Id, translations: &mut IndexMap<Id, RecExpr<SymbolLang>>, mut inner_ids: &mut Vec<&'a SymbolLang>, colorded_class: &'a EClass<SymbolLang, ()>) {
    for edge in &colorded_class.nodes {
        if edge.children.iter().all(|c| translations.contains_key(c)) {
            build_translation(graph, color, translations, &edge, class);
            return;
        }
        inner_ids.push(&edge);
    }
}

fn build_translation(graph: &EGraph<SymbolLang, ()>, color: Option<ColorId>, translations: &mut IndexMap<Id, RecExpr<SymbolLang>>, edge: &SymbolLang, id: Id) {
    let mut res = vec![];
    let mut children = vec![];
    for c in edge.children.iter() {
        let cur_len = res.len();
        let translation = translations.get(c).or_else(||
            color.map(|c_id|
                graph.get_color(c_id).map(|x|
                    x.black_ids(*c).map(|ids|
                        // Build translation is only called when a translation exists
                        ids.iter().find_map(|id| translations.get(id)))))
                .flatten().flatten().flatten()
        );
        if translation.is_none() { return; }
        res.extend(translation.unwrap().as_ref().iter().cloned().map(|s| s.map_children(|child| Id::from(usize::from(child) + cur_len))));
        children.push(Id::from(res.len() - 1));
    };
    res.push(SymbolLang::new(edge.op, children));
    translations.insert(id, RecExpr::from(res));
}

pub fn reconstruct_all(graph: &EGraph<SymbolLang, ()>, max_depth: usize) -> IndexMap<Id, IndexSet<Rc<Tree>>> {
    for c in graph.classes() {
        let set = c.nodes.iter().collect::<IndexSet<&SymbolLang>>();
        for c1 in graph.classes() {
            for n in c1.nodes.iter() {
                assert!(!set.contains(&n) || c.id == c1.id)
            }
        }
    }
    let edges = graph.classes().filter(|c| graph.find(c.id) == c.id).flat_map(|c| iter::once(c.id).cycle().zip(c.nodes.iter())).collect_vec();
    // let idToType = inputEdges.filter(_.edgeType.identifier == Language.typeId).map(e => (e.sources(0), Programs.reconstruct(inputEdges, e.sources(1)).head)).toMap
    let mut known_terms: IndexMap<Id, IndexSet<Rc<Tree>>> = IndexMap::new();
    let mut last_level: IndexSet<(Id, Rc<Tree>)> = edges.iter().filter_map(|(id, e)| {
        if e.is_leaf() {
            let tree = Rc::new(Tree::leaf(e.op.to_string()));
            if !known_terms.contains_key(id) {
                known_terms.insert(*id, IndexSet::new());
            }
            known_terms.get_mut(id).unwrap().insert(tree.clone());
            Some((*id, tree))
        } else {
            None
        }
    }).collect();
    let mut levels = vec![last_level];

    // id to edges that might be available
    // let edges_by_reqiurments = collection.mutable.HashMultiMap.empty[HyperTermId, (Int, HyperEdge[HyperTermId, HyperTermIdentifier])]
    let mut edges_by_reqiurments = IndexMap::new();
    for x in edges.iter() {
        for (i, s) in x.1.children.iter().enumerate() {
            assert_eq!(graph.find(*s), *s);
            if !edges_by_reqiurments.contains_key(s) {
                edges_by_reqiurments.insert(*s, IndexSet::new());
            }
            edges_by_reqiurments.get_mut(s).unwrap().insert((i, *x));
        }
    };

    // The reconstruct itself.
    for d in 0..max_depth {
        info!("depth {}", d);
        last_level = IndexSet::new();
        for e in levels.last().unwrap() {
            for v in edges_by_reqiurments.get(&e.0) {
                for (i_to_fill, e_to_fill) in v {
                    let params_to_fill = e_to_fill.1.children.iter().take(*i_to_fill).chain(&e_to_fill.1.children[i_to_fill + 1..]);
                    let trees = params_to_fill.filter_map(|h_id| known_terms.get(h_id).map_or(None, |x| Some(x.iter().cloned()))).collect_vec();
                    if trees.len() == e_to_fill.1.children.len() - 1 {
                        for mut fillers in combinations(trees.into_iter()) {
                            fillers.insert(*i_to_fill, e.1.clone());
                            last_level.insert((e.0, Rc::new(Tree::branch(e_to_fill.1.op.to_string(), fillers))));
                        }
                    }
                }
            }
        }

        for e in last_level.clone() {
            if !known_terms.contains_key(&e.0) {
                known_terms.insert(e.0, IndexSet::new());
            }
            known_terms.get_mut(&e.0).unwrap().insert(e.1);
        }

        levels.push(last_level);
    }

    known_terms
    // known_terms.into_iter().flat_map(|x| iter::once(x.0).cycle().zip(x.1.into_iter())).collect::<IndexSet<(Id, Rc<Tree>)>>()
}
