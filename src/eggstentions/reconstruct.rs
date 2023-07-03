use std::collections::{HashMap, HashSet};

use crate::{EGraph, Id, Language, SymbolLang, RecExpr, EClass, ColorId};
use indexmap::IndexMap;
use itertools::Itertools;
use crate::eggstentions::tree::Tree;

/// Reconstructs a RecExpr from an eclass.
pub fn reconstruct(graph: &EGraph<SymbolLang, ()>, class: Id, max_depth: usize)
    -> Option<RecExpr<SymbolLang>> {
    reconstruct_colored(graph, None, class, max_depth)
}

/// Reconstructs a RecExpr from an eclass under a specific colored assumption.
pub fn reconstruct_colored(graph: &EGraph<SymbolLang, ()>, color: Option<ColorId>, class: Id, max_depth: usize) -> Option<RecExpr<SymbolLang>> {
    let mut translations: IndexMap<Id, RecExpr<SymbolLang>> = IndexMap::new();
    let class = graph.find(class);
    reconstruct_inner(&graph, class, max_depth, color, &mut translations);
    translations.get(&class).map(|x| x.clone())
}

/// Reconstructs a RecExpr from an eclass, but filtering to start with `edge`.
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
        if let Some(x) = graph.get_color(c) {
            let ids = x.black_ids(graph, class);
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
                x.black_ids(graph, class).map_or(false, |ids|
                    ids.iter().find(|id| translations.contains_key(*id)).is_some())))) {
            build_translation(graph, color, translations, &edge, class);
            return;
        }
    }
}

fn check_class<'a>(graph: &EGraph<SymbolLang, ()>, color: Option<ColorId>, class: Id, translations: &mut IndexMap<Id, RecExpr<SymbolLang>>, inner_ids: &mut Vec<&'a SymbolLang>, colorded_class: &'a EClass<SymbolLang, ()>) {
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
                    x.black_ids(graph, *c).map(|ids|
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

/// Reconstructs a RecExpr for each EClass in the graph.
pub fn reconstruct_all(graph: &EGraph<SymbolLang, ()>, color: Option<ColorId>, max_depth: usize)
    -> IndexMap<Id, Tree> {
    let mut translations: IndexMap<Id, SymbolLang> = IndexMap::default();
    let mut edge_in_need: HashMap<Id, Vec<(Id, SymbolLang)>> = HashMap::default();

    let mut todo = HashSet::new();

    let mut layers = vec![vec![]];
    // Initialize data structures (translations, and which edges might be "free" next)
    for c in graph.classes()
        .filter(|c| c.color().is_none() || c.color() == color) {
        let fixed_id = graph.opt_colored_find(color, c.id);
        for n in &c.nodes {
            let fixed_n = if color.is_some() {
                graph.colored_canonize(*color.as_ref().unwrap(), n)
            } else {
                n.clone()
            };
            if n.children().is_empty() {
                todo.insert(fixed_id);
                translations.insert(fixed_id, fixed_n);
                layers.last_mut().unwrap().push(fixed_id);
            } else {
                for ch in fixed_n.children() {
                    let fixed_child = graph.opt_colored_find(color, *ch);
                    // this might be a bit expensive to do for each edge
                    edge_in_need.entry(fixed_child).or_default().push((fixed_id, fixed_n.clone()));
                }
            }
        }
    }
    let mut res = IndexMap::new();
    for (id, n) in translations.iter() {
        res.insert(*id, Tree::leaf(n.op.to_string()));
    }

    let empty = vec![];
    // Build layers
    for _ in 0..max_depth {
        layers.push(vec![]);
        let doing = std::mem::take(&mut todo);
        for c in doing {
            for (trg, n) in edge_in_need.get(&c).unwrap_or(&empty) {
                if (!translations.contains_key(trg)) &&
                    n.children().iter().all(|ch| translations.contains_key(ch)) {
                    translations.insert(*trg, n.clone());
                    todo.insert(*trg);
                    layers.last_mut().unwrap().push(*trg);
                }
            }
        }
    }

    // Build translations
    for l in layers.iter().dropping(1) {
        for id in l {
            let n = &translations[id];
            let new_tree = Tree::branch(n.op.to_string(), n.children().iter().map(|ch| res[ch].clone()).collect());
            res.insert(*id, new_tree);
        }
    }
    res
}
