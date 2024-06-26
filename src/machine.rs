use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;
use crate::*;
use indexmap::{IndexMap, IndexSet};
use invariants::dassert;
use itertools::{Either, Itertools};
use itertools::Either::{Right, Left};
use log::trace;
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};
use crate::expression_ops::{IntoTree, RecExpSlice, Tree};

type EarlyStopFn<'a, L, N> = Rc<RefCell<dyn FnMut(&Machine<'a, L, N>) -> bool>>;

/// An iterator for match results
struct Machine<'a, L: Language, N: Analysis<L>> {
    reg: Vec<Id>,
    // a buffer to re-use for lookups
    lookup: Vec<Id>,
    #[cfg(feature = "colored")]
    color: Option<ColorId>,
    egraph: &'a EGraph<L, N>,
    instructions: &'a [Instruction<L>],
    subst: Subst,
    add_colors: bool,
    stack: Vec<MachineContext>,
    early_stop: Option<EarlyStopFn<'a, L, N>>,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
struct Reg(u32);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Program<L> {
    instructions: Vec<Instruction<L>>,
    subst: Subst,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Instruction<L> {
    Bind { node: L, eclass: Reg, out: Reg },
    Compare { i: Reg, j: Reg },
    Lookup { term: Vec<ENodeOrReg<L>>, i: Reg },
    Scan { out: Reg, top_pat: Either<L, Option<Reg>> },
    ColorJump { orig: Reg, out: Reg },
    Not { sub_prog: Program<L> },
    Nop,
    Or { root: Var, sub_progs: Vec<Program<L>>, out: Reg },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum ENodeOrReg<L> {
    ENode(L),
    Reg(Reg),
}

#[inline(always)]
fn for_each_matching_node<L, D>(
    eclass: &EClass<L, D>,
    node: &L,
    mut f: impl FnMut(&L) -> (),
)
    where
        L: Language,
{
    #[allow(enum_intrinsics_non_enums)]
    if eclass.nodes.len() < 50 {
        eclass
            .nodes
            .iter()
            .filter(|n| node.matches(n))
            .for_each(f)
    } else {
        debug_assert!(node.children().iter().all(|id| *id == Id::from(0)));
        debug_assert!(eclass.nodes.windows(2).all(|w| w[0] < w[1]));
        let start = eclass.nodes.binary_search(node).unwrap_or_else(|i| i);
        let matching = eclass.nodes[..start].iter().rev()
            .take_while(|x| node.matches(x))
            .chain(eclass.nodes[start..].iter().take_while(|x| node.matches(x)));
        debug_assert_eq!(
            matching.clone().count(),
            eclass.nodes.iter().filter(|n| node.matches(n)).count(),
            "matching node {:?}\nstart={}\n{:?} != {:?}\nnodes: {:?}",
            node,
            start,
            matching.clone().collect::<IndexSet<_>>(),
            eclass
                .nodes
                .iter()
                .filter(|n| node.matches(n))
                .collect::<IndexSet<_>>(),
            eclass.nodes
        );
        matching.for_each(&mut f);
    }
}

struct MachineContext {
    instruction_index: usize,
    color: Option<ColorId>,
    truncate: usize,
    to_push: SmallVec<[Id; 4]>,
}

impl MachineContext {
    fn new(instruction_index: usize, color: Option<ColorId>, truncate: usize, to_push: SmallVec<[Id; 4]>) -> Self {
        Self {
            instruction_index,
            color,
            truncate,
            to_push
        }
    }
}

impl<'a, L: Language, A: Analysis<L>> Machine<'a, L, A> {
    pub(crate) fn new(
        color: Option<ColorId>,
        egraph: &'a EGraph<L, A>,
        instructions: &'a [Instruction<L>],
        subst: Subst,
        add_colors: bool,
    ) -> Self {
        let mut stack: Vec<MachineContext> = Vec::with_capacity(4096);
        stack.push(MachineContext::new(0, color, 1, SmallVec::new()));
        Machine {
            reg: vec![],
            lookup: vec![],
            color,
            egraph,
            instructions,
            subst,
            add_colors,
            stack,
            early_stop: None,
        }
    }

    #[inline(always)]
    fn reg(&self, reg: Reg) -> Id {
        self.reg[reg.0 as usize]
    }
}

impl<'a, L: Language, A: Analysis<L>> Iterator for Machine<'a, L, A> {
    type Item = Subst;

    fn next(&mut self) -> Option<Self::Item> {
        let egraph = self.egraph;
        let instructions = self.instructions;
        let add_colors = self.add_colors;
        while !self.stack.is_empty() {
            let mut current_state = self.stack.pop().unwrap();
            trace!("Instruction index {} - Popped state with color {:?}", current_state.instruction_index, current_state.color);
            self.reg.truncate(current_state.truncate);
            self.reg.extend(current_state.to_push.drain(..));
            self.color = current_state.color;
            let early = std::mem::take(&mut self.early_stop);
            if let Some(f) = early.as_ref() {
                let mut r = f.borrow_mut();
                if r(self) {
                    continue;
                }
            }
            self.early_stop = early;
            let mut index = current_state.instruction_index;
            while index < instructions.len() {
                let instruction = &instructions[index];
                match instruction {
                    Instruction::Bind { eclass, out, node } => {
                        let class_color = egraph[self.reg(*eclass)].color();
                        trace!("Instruction index {} - Binding (cur color: {:?}) node {} @ {} (color: {:?}) for out reg {}", index, self.color, node.display_op(), self.reg(*eclass), class_color, out.0);
                        dassert!(class_color.is_none() || class_color == self.color
                            || (self.color.is_some() && egraph.get_colors_parents(self.color.unwrap()).contains(&class_color.unwrap())));
                        for_each_matching_node(&egraph[self.reg(*eclass)], node, |matched| {
                            let out = *out;
                            trace!("Pusing to stack color: {:?}, truncate to {} and push {}", self.color, out.0, matched.children().iter().join(", "));
                            let to_push = SmallVec::from_slice(matched.children());
                            self.stack.push(MachineContext::new(index + 1, self.color, out.0 as usize, to_push));
                        });
                        break;
                    }
                    Instruction::Scan { out, top_pat } => {
                        let run = |machine: &mut Machine<L, A>, id| {
                            let cur_color = machine.color;
                            let cls_color = egraph[id].color();
                            let cls_is_parent = cls_color.is_none() || (cur_color.is_some() &&
                                egraph.get_colors_parents(cur_color.unwrap()).contains(&cls_color.unwrap()));
                            // Not in lineage or no adding colors and is decendant
                            if cls_color.is_some() && cls_color != cur_color && (!cls_is_parent) &&
                                ((!add_colors) || (cur_color.is_some() && !machine.egraph.get_colors_parents(cls_color.unwrap()).contains(&cur_color.unwrap()))) {
                                return;
                            }
                            let new_color = if cls_is_parent {
                                machine.color
                            } else {
                                cls_color
                            };
                            let out = *out;
                            trace!("Pushing to stack color: {:?}, truncate to {} and push {}", new_color, out.0, id);
                            machine.stack.push(MachineContext::new(index + 1, new_color, out.0 as usize, smallvec![id]));
                        };

                        match top_pat {
                            Left(node) => {
                                trace!("Instruction index {} - Scanning for {} with color {:?}", node.display_op(), index, self.color);
                                if let Some(ids) = egraph.classes_by_op_id().get(&node.op_id()) {
                                    for class in ids {
                                        run(self, *class);
                                    }
                                }
                            }
                            Right(opt_reg_var) => {
                                if let Some(reg_var) = opt_reg_var {
                                    trace!("Instruction index {} - Scanning for reg {}={} with color {:?}", reg_var.0, self.reg(*reg_var), index, self.color);
                                    run(self, self.reg(*reg_var));
                                } else {
                                    trace!("Instruction index {} - Scanning for any with color {:?}", index, self.color);
                                    for class in egraph.classes() {
                                        run(self, class.id);
                                    }
                                }
                            }
                        }
                        break;
                    }
                    Instruction::Compare { i, j } => {
                        let fixed_i = egraph.opt_colored_find(self.color, self.reg(*i));
                        let fixed_j = egraph.opt_colored_find(self.color, self.reg(*j));
                        trace!("Instruction index {} - Comparing (color: {:?}) reg {} and reg {} (found to be {} and {})", index, self.color, i.0, j.0, fixed_i, fixed_j);
                        if fixed_i != fixed_j {
                            if let Some(eqs) = egraph.get_base_equalities(self.color, fixed_i) {
                                if eqs.into_iter().any(|id| id == fixed_j) {
                                    trace!("Found base match for compare");
                                    continue;
                                }
                            }
                            if add_colors {
                                if let Some(eqs) = egraph.get_decendent_colored_equalities(self.color, fixed_i) {
                                    for (cid, _id) in eqs.into_iter().filter(|(_cid, id)| *id == fixed_j) {
                                        trace!("Pushing to stack with new color {:?}", cid);
                                        self.stack.push(MachineContext::new(index + 1, Some(cid), self.reg.len(), smallvec![]));
                                    }
                                }
                            }
                            break;
                        }
                    }
                    Instruction::Lookup { term, i } => {
                        assert!(self.color.is_none(), "Lookup instruction is an optimization for non colored search");
                        assert!(!add_colors, "Lookup instruction is an optimization for non colored search");
                        trace!("Instruction index {} - Looking up {:?} in reg {}", index, term, i.0);
                        self.lookup.clear();
                        for node in term {
                            match node {
                                ENodeOrReg::ENode(node) => {
                                    let look = |i| self.lookup[usize::from(i)];
                                    match egraph.lookup(node.clone().map_children(look)) {
                                        Some(id) => self.lookup.push(id),
                                        None => break,
                                    }
                                }
                                ENodeOrReg::Reg(r) => {
                                    self.lookup.push(egraph.find(self.reg(*r)));
                                }
                            }
                        }

                        let id = egraph.find(self.reg(*i));
                        if self.lookup.last().copied() != Some(id) {
                            break;
                        }
                    }
                    Instruction::ColorJump { orig, out } => {
                        let id = egraph.find(self.reg(*orig));
                        let eq_it: Option<Box<dyn Iterator<Item=(ColorId, Id)>>> = if !add_colors {
                            if let Some(c) = self.color {
                                let c = c;
                                if let Some(b) = egraph.get_base_equalities(self.color, id) {
                                    Some(Box::new(b.map(move |id| (c, id))))
                                } else {
                                    None
                                }
                            } else {
                                Some(Box::new(std::iter::empty()))
                            }
                        } else {
                            egraph.get_lineage_equalities(self.color, id)
                        };
                        trace!("Instruction index {} - Color jumping from reg {}={} (color: {:?})", index, orig.0, self.reg(*orig), self.color);
                        if let Some(eqs) = eq_it {
                            for (c_id, jumped_id) in eqs {
                                let jumped_id = egraph.find(jumped_id);
                                if jumped_id == id {
                                    continue;
                                }
                                trace!("Pushing to stack with new id (new color {:?}) reg {}={}", c_id, orig.0, jumped_id);
                                let color = if self.color.is_none() || egraph.get_colors_parents(c_id).len() > egraph.get_colors_parents(self.color.unwrap()).len() {
                                    Some(c_id)
                                } else {
                                    self.color
                                };
                                self.stack.push(MachineContext::new(index + 1, color, out.0 as usize, smallvec![jumped_id]));
                            }
                        }
                        trace!("Done color jump continuing run with color {:?} and reg {}={}", self.color, orig.0, self.reg(*orig));
                        dassert!(self.reg.len() == out.0 as usize);
                        self.reg.push(self.reg(*orig));
                        // Not breaking so will continue to next instruction with current setup.
                    }
                    Instruction::Not { sub_prog } => {
                        if sub_prog.inner_run_from(egraph, self, false).next().is_some() {
                            break;
                        }
                    }
                    Instruction::Nop => {
                        // do nothing
                    }
                    Instruction::Or { root, out, sub_progs } => {
                        let done: Rc<RefCell<BTreeSet<(Option<ColorId>, Id)>>> = Default::default();
                        let cloned = done.clone();
                        let out = *out;
                        let early_stop = Rc::new(RefCell::new(move |machine: &Machine<'a, L, A>| {
                            if machine.reg.len() <= out.0 as usize {
                                return false;
                            }
                            let id = machine.reg(out);
                            if done.borrow().contains(&(machine.color.clone(), id)) || done.borrow().contains(&(None, id)) {
                                return true;
                            }
                            if let Some(c) = machine.color {
                                if let Some(mut eqs) = machine.egraph.get_base_equalities(Some(c), id) {
                                    let parents = machine.egraph.get_colors_parents(c);
                                    if eqs.any(|id| done.borrow().contains(&(None, id)) ||
                                        done.borrow().contains(&(Some(c), id)) ||
                                        parents.iter().any(|p| done.borrow().contains(&(Some(*p), id)))) {
                                        return true;
                                    }
                                }
                            }
                            false
                        }));
                        let done = cloned;
                        sub_progs.iter().for_each(|p| {
                            p.run_for_or(egraph, self, add_colors, early_stop.clone())
                                .for_each(|s| {
                                    done.borrow_mut().insert((s.color.clone(), *s.get(*root).unwrap()));
                                })
                        });
                        for (color, id) in done.borrow().iter() {
                            if let Some(c) = color {
                                if let Some(eqs) = egraph.get_base_equalities(Some(*c), *id) {
                                    for eq in eqs {
                                        if done.borrow().contains(&(None, eq)) ||
                                            egraph.get_colors_parents(*c)
                                                .into_iter()
                                                .any(|p| done.borrow().contains(&(Some(*p), eq))) {
                                            continue;
                                        }
                                    }
                                }
                            }
                            self.stack.push(MachineContext::new(
                                index + 1, *color, out.0 as usize, smallvec![*id],
                            ));
                        }
                        break;
                    }
                };
                index += 1;
            }
            if index == instructions.len() {
                let subst_vec = self.subst
                    .vec
                    .iter()
                    // HACK we are reusing Ids here, this is bad
                    .map(|(v, reg_id)| (*v, self.reg(Reg(usize::from(*reg_id) as u32))))
                    .collect();
                return Some(Subst { vec: subst_vec, color: self.color.clone() });
            }
        }
        None
    }
}

struct Compiler<L> {
    v2r: IndexMap<Var, Reg>,
    free_vars: Vec<IndexSet<Var>>,
    subtree_size: Vec<usize>,
    todo_nodes: IndexMap<(Id, Reg), L>,
    instructions: Vec<Instruction<L>>,
    next_reg: Reg,
}

impl<L: Language> Compiler<L> {
    fn new() -> Self {
        Self {
            free_vars: Default::default(),
            subtree_size: Default::default(),
            v2r: Default::default(),
            todo_nodes: Default::default(),
            instructions: Default::default(),
            next_reg: Reg(0),
        }
    }

    fn add_todo(&mut self, pattern: &PatternAst<L>, id: Id, reg: Reg) {
        match &pattern.as_ref()[id.0 as usize] {
            ENodeOrVar::Var(v) => {
                if let Some(&j) = self.v2r.get(v) {
                    self.instructions.push(Instruction::Compare { i: reg, j })
                } else {
                    self.v2r.insert(*v, reg);
                }
            }
            ENodeOrVar::ENode(pat, _name) => {
                self.todo_nodes.insert((id, reg), pat.clone());
            }
        }
    }

    fn load_pattern(&mut self, pattern: &PatternAst<L>) {
        let len = pattern.as_ref().len();
        self.free_vars = Vec::with_capacity(len);
        self.subtree_size = Vec::with_capacity(len);

        for node in pattern.as_ref() {
            let mut free = IndexSet::default();
            let mut size = 0;
            match node {
                ENodeOrVar::ENode(n, _) => {
                    size = 1;
                    for &child in n.children() {
                        free.extend(&self.free_vars[usize::from(child)]);
                        size += self.subtree_size[usize::from(child)];
                    }
                }
                ENodeOrVar::Var(v) => {
                    free.insert(*v);
                }
            }
            self.free_vars.push(free);
            self.subtree_size.push(size);
        }
    }

    fn next(&mut self) -> Option<((Id, Reg), L)> {
        // we take the max todo according to this key
        // - prefer grounded
        // - prefer more free variables
        // - prefer smaller term
        let key = |(id, _): &&(Id, Reg)| {
            let i = usize::from(*id);
            let n_bound = self.free_vars[i]
                .iter()
                .filter(|v| self.v2r.contains_key(*v))
                .count();
            let n_free = self.free_vars[i].len() - n_bound;
            let size = self.subtree_size[i] as isize;
            (n_free == 0, n_free, -size)
        };

        self.todo_nodes
            .keys()
            .max_by_key(key)
            .copied()
            .map(|k| (k, self.todo_nodes.remove(&k).unwrap()))
    }

    /// check to see if this e-node corresponds to a term that is grounded by
    /// the variables bound at this point
    fn is_ground_now(&self, id: Id) -> bool {
        self.free_vars[usize::from(id)]
            .iter()
            .all(|v| self.v2r.contains_key(v))
    }

    fn compile(&mut self, patternbinder: Option<Var>, pattern: &PatternAst<L>) {
        self.load_pattern(pattern);
        let last_i = pattern.as_ref().len() - 1;

        let mut next_out = self.next_reg;

        // Check if patternbinder already bound in v2r
        // Behavior common to creating a new pattern
        let add_new_pattern = |comp: &mut Compiler<L>| {
            if !comp.instructions.is_empty() {
                // After first pattern needs scan
                let last = pattern.as_ref().last().unwrap();
                let top_pat = match last {
                    ENodeOrVar::ENode(node, _) => { Left(node.clone()) }
                    ENodeOrVar::Var(v) => { Right(comp.v2r.get(v).copied()) }
                };
                comp.instructions.push(Instruction::Scan { out: comp.next_reg, top_pat });
            }
            comp.add_todo(pattern, Id::from(last_i), comp.next_reg);
        };

        if let Some(v) = patternbinder {
            if let Some(&i) = self.v2r.get(&v) {
                // patternbinder already bound
                if matches!(pattern.as_ref()[last_i], ENodeOrVar::ENode(_, _)) {
                    self.introduce_color_jump(next_out, i);
                    self.v2r[&v] = next_out;
                    next_out.0 += 1;
                    self.next_reg.0 += 1;
                }
                self.add_todo(pattern, Id::from(last_i), self.v2r[&v]);
            } else {
                // patternbinder is new variable
                next_out.0 += 1;
                add_new_pattern(self);
                self.v2r.insert(v, self.next_reg); //add to known variables.
            }
        } else {
            // No pattern binder
            next_out.0 += 1;
            add_new_pattern(self);
        }

        let mut first_bind = true;
        while let Some(((id, mut reg), node)) = self.next() {
            if self.is_ground_now(id) && (!node.is_leaf()) && !cfg!(feature = "colored") {
                let extracted = pattern.extract(id);
                self.instructions.push(Instruction::Lookup {
                    i: reg,
                    term: extracted
                        .as_ref()
                        .iter()
                        .map(|n| match n {
                            ENodeOrVar::ENode(n, _name) => ENodeOrReg::ENode(n.clone()),
                            ENodeOrVar::Var(v) => ENodeOrReg::Reg(self.v2r[v]),
                        })
                        .collect(),
                });
            } else {
                if !first_bind {
                    self.introduce_color_jump(next_out, reg);
                    reg = next_out;
                    next_out.0 += 1;
                } else {
                    first_bind = false;
                }
                let out = next_out;
                next_out.0 += node.len() as u32;

                // zero out the children so Bind can use it to sort
                let op = node.clone().map_children(|_| Id::from(0));
                self.instructions.push(Instruction::Bind {
                    eclass: reg,
                    node: op,
                    out,
                });

                for (i, &child) in node.children().iter().enumerate() {
                    self.add_todo(pattern, child, Reg(out.0 + i as u32));
                }
            }
        }
        self.next_reg = next_out;
    }

    fn introduce_color_jump(&mut self, next_out: Reg, orig: Reg) {
        self.instructions.push(Instruction::ColorJump { orig, out: next_out });
        self.todo_nodes = std::mem::take(&mut self.todo_nodes)
            .into_iter()
            .map(|((id, reg), node)| if reg == orig {
                ((id, next_out), node)
            } else { ((id, reg), node) })
            .collect();
    }

    fn compile_sub_program(&mut self, patternbinder: Option<Var>, pattern: &PatternAst<L>) -> Program<L> {
        let mut compiler = Compiler::new();
        compiler.next_reg = self.next_reg;
        compiler.v2r = self.v2r.clone();
        // Adding a nop so it will be treated as a "not first" pattern
        compiler.instructions.push(Instruction::Nop);
        compiler.compile(patternbinder, pattern);
        compiler.extract()
    }

    fn extract(self) -> Program<L> {
        let mut subst = Subst::default();
        for (v, r) in self.v2r {
            subst.insert(v, Id::from(r.0 as usize));
        }
        Program {
            instructions: self.instructions,
            subst,
        }
    }
}

impl<L: Language> Program<L> {
    pub(crate) fn compile_from_pat(pattern: &PatternAst<L>) -> Self {
        let mut compiler = Compiler::new();
        compiler.compile(None, pattern);
        let program = compiler.extract();
        log::debug!("Compiled {:?} to {:?}", pattern.as_ref(), program);
        program
    }

    pub(crate) fn compile_from_multi_pat(patterns: &[(Var, PatternAst<L>)], or_patterns: &Vec<(Var, Vec<PatternAst<L>>)>, not_patterns: &[(Var, PatternAst<L>)]) -> Self {
        assert!(!patterns.is_empty());
        let mut compiler = Compiler::new();
        let mut all_normal_vars = BTreeSet::new();
        for (v, p) in patterns {
            all_normal_vars.insert(*v);
            p.as_ref().iter().for_each(|node_or_var| {
                if let ENodeOrVar::Var(v) = node_or_var {
                    all_normal_vars.insert(*v);
                }
            });
        }
        let or_patterns = or_patterns.into_iter().map(|(or_v, or_ps)| {
            let mut all_vars: BTreeSet<Var> = or_ps.iter().flat_map(|p| p.as_ref().iter().filter_map(|node_or_var| {
                if let ENodeOrVar::Var(v) = node_or_var {
                    Some(*v)
                } else {
                    None
                }
            })).collect();
            all_vars.insert(*or_v);
            let intersection = all_vars.intersection(&all_normal_vars).copied().collect_vec();
            (false, *or_v, or_ps, intersection)
        }).collect_vec();
        all_normal_vars.extend(or_patterns.iter().map(|(_, v, _, _)| *v));
        let mut not_patterns = not_patterns.into_iter().map(|(not_v, not_p)| {
            let mut all_vars: BTreeSet<Var> = not_p.as_ref().iter().filter_map(|node_or_var| {
                if let ENodeOrVar::Var(v) = node_or_var {
                    Some(*v)
                } else {
                    None
                }
            }).collect();
            all_vars.insert(*not_v);
            let intersection = all_vars.intersection(&all_normal_vars).copied().collect_vec();
            (false, *not_v, not_p, intersection)
        }).collect_vec();
        let mut or_patterns = or_patterns.into_iter().filter(|(_, v, ps, intersection)| {
            ps.iter().all(|p| {
                !patterns.iter().any(|(var, pat)|
                    *var == *v && Self::patterns_agree(p.into_tree(), pat.into_tree(), intersection))
            })
        }).collect_vec();
        for (var, pattern) in patterns {
            compiler.compile(Some(*var), pattern);
            for (done, or_v, or_ps, intersection) in or_patterns.iter_mut() {
                if !*done && intersection.iter().all(|v| compiler.v2r.contains_key(v)) {
                    *done = true;
                    // Or patterns keep running to match root for all "base" hierarchy colors. That is if black matches
                    // continue to next eclass root.
                    // Otherwise try all colors.
                    // Another possible optimization is skipping "already matched" colors.
                    let compiled = or_ps.iter().map(|p| compiler.compile_sub_program(Some(*or_v), p)).collect_vec();
                    let out = compiler.next_reg;
                    compiler.next_reg.0 += 1;
                    compiler.instructions.push(Instruction::Or {
                        root: *or_v,
                        sub_progs: compiled,
                        out,
                    });
                    if let Some(r) = compiler.v2r.get(or_v) {
                        compiler.instructions.push(Instruction::Compare {
                            i: out,
                            j: *r,
                        });
                    } else {
                        compiler.v2r[&*or_v] = out;
                    }
                }
            }
            for (done, not_v, not_p, intersection) in not_patterns.iter_mut() {
                if !*done && intersection.iter().all(|v| compiler.v2r.contains_key(v)) {
                    *done = true;
                    let res = compiler.compile_sub_program(Some(*not_v), *not_p);
                    compiler.instructions.push(Instruction::Not {
                        sub_prog: res,
                    });
                }
            }
        }
        assert!(not_patterns.iter().all(|(done, _, _, _)| *done), "Not patterns not done");
        compiler.extract()
    }

    pub fn run_with_limit<A>(
        &self,
        egraph: &EGraph<L, A>,
        eclass: Id,
        limit: usize,
    ) -> Vec<Subst>
    where
        A: Analysis<L>,
    {
        self.inner_run(egraph, eclass, None, false, limit)
    }

    #[allow(dead_code)]
    pub fn run<A>(
        &self,
        egraph: &EGraph<L, A>,
        eclass: Id,
    ) -> Vec<Subst>
        where
            A: Analysis<L>,
    {
        self.run_with_limit(egraph, eclass, usize::MAX)
    }

    pub fn colored_run_with_limit<A>(&self,
                          egraph: &EGraph<L, A>,
                          eclass: Id,
                          color: Option<ColorId>,
                          limit: usize,
    ) -> Vec<Subst>
        where
            A: Analysis<L>,
    {
        self.inner_run(egraph, eclass, color, true, limit)
    }

    fn inner_run<A>(
        &self,
        egraph: &EGraph<L, A>,
        eclass: Id,
        opt_color: Option<ColorId>,
        run_color: bool,
        limit: usize,
    ) -> Vec<Subst>
        where
            A: Analysis<L>,
    {
        assert!(egraph.is_clean(), "Tried to search a dirty e-graph!");
        let class_color = egraph[eclass].color();
        dassert!(opt_color.is_none() || class_color.is_none() || opt_color == class_color,
                 "Tried to run a colored program on an eclass with a different color: {:?} vs {:?}",
                 opt_color, class_color);
        let opt_color = class_color;
        let mut machine = Machine::new(opt_color, egraph, &self.instructions, self.subst.clone(), run_color);
        assert_eq!(machine.reg.len(), 0);
        machine.reg.push(eclass);

        let matches = machine.take(limit).collect_vec();
        log::trace!("Ran program, found {:?}", matches);
        matches
    }

    fn inner_run_from<'a, A>(
        &'a self,
        egraph: &'a EGraph<L, A>,
        old_machine: &Machine<'a, L, A>,
        add_colors: bool,
    ) -> impl Iterator<Item = Subst> + 'a
        where
            A: Analysis<L>,
    {
        let mut machine = Machine::new(old_machine.color, egraph, &self.instructions, self.subst.clone(), add_colors);
        machine.reg = old_machine.reg.clone();
        machine.stack[0].truncate = machine.reg.len();
        machine
    }

    fn run_for_or<'a, A>(
        &'a self,
        egraph: &'a EGraph<L, A>,
        old_machine: &Machine<'a, L, A>,
        run_color: bool,
        early_stop: EarlyStopFn<'a, L, A>,
    ) -> impl Iterator<Item = Subst> + 'a
        where
            A: Analysis<L>,
    {
        let mut machine = Machine::new(old_machine.color, egraph, &self.instructions, self.subst.clone(), run_color);
        machine.reg = old_machine.reg.clone();
        machine.stack[0].truncate = machine.reg.len();
        machine.early_stop = Some(early_stop);
        machine
    }

    fn patterns_agree(orig: RecExpSlice<ENodeOrVar<L>>, compared: RecExpSlice<ENodeOrVar<L>>, common_vars: &Vec<Var>) -> bool {
        match orig.root() {
            ENodeOrVar::ENode(enode, _) => {
                if let ENodeOrVar::ENode(c_enode, _) = compared.root() {
                    if enode.op_id() != c_enode.op_id() {
                        return false;
                    }
                    for (orig_child, compared_child) in orig.children().into_iter().zip(compared.children()) {
                        if !Self::patterns_agree(orig_child, compared_child, common_vars) {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            }
            ENodeOrVar::Var(v) => {
                (!common_vars.contains(v)) || compared.root() == orig.root()
            }
        }
    }
}