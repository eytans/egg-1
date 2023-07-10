use crate::*;
use std::result;
use indexmap::{IndexMap, IndexSet};
use invariants::dassert;
use itertools::Either;
use itertools::Either::{Right, Left};

type Result = result::Result<(), ()>;

#[derive(Default)]
struct Machine {
    reg: Vec<Id>,
    // a buffer to re-use for lookups
    lookup: Vec<Id>,
    #[cfg(feature = "colored")]
    color: Option<ColorId>,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Reg(u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program<L> {
    instructions: Vec<Instruction<L>>,
    subst: Subst,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Instruction<L> {
    Bind { node: L, eclass: Reg, out: Reg },
    Compare { i: Reg, j: Reg },
    Lookup { term: Vec<ENodeOrReg<L>>, i: Reg },
    Scan { out: Reg, top_pat: Either<L, Option<Reg>> },
    ColorJump { orig: Reg },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ENodeOrReg<L> {
    ENode(L),
    Reg(Reg),
}

#[inline(always)]
fn for_each_matching_node<L, D>(
    eclass: &EClass<L, D>,
    node: &L,
    mut f: impl FnMut(&L) -> Result,
) -> Result
    where
        L: Language,
{
    #[allow(enum_intrinsics_non_enums)]
    if eclass.nodes.len() < 50 {
        eclass
            .nodes
            .iter()
            .filter(|n| node.matches(n))
            .try_for_each(f)
    } else {
        debug_assert!(node.children().iter().all(|id| *id == Id::from(0)));
        debug_assert!(eclass.nodes.windows(2).all(|w| w[0] < w[1]));
        let start = eclass.nodes.binary_search(node).unwrap_or_else(|i| i);
        let mut matching = eclass.nodes[..start].iter().rev()
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
        matching.try_for_each(&mut f)
    }
}

impl Machine {
    #[inline(always)]
    fn reg(&self, reg: Reg) -> Id {
        self.reg[reg.0 as usize]
    }

    fn run<L, N>(&mut self,
                 egraph: &EGraph<L, N>,
                 instructions: &[Instruction<L>],
                 subst: &Subst,
                 yield_fn: &mut impl FnMut(&Self, &Subst) -> Result,
    ) -> Result
        where
            L: Language,
            N: Analysis<L>,
    {
        self.inner_run(egraph, instructions, subst, false, yield_fn)
    }

    fn colored_run<L, N>(&mut self,
                 egraph: &EGraph<L, N>,
                 instructions: &[Instruction<L>],
                 subst: &Subst,
                 yield_fn: &mut impl FnMut(&Self, &Subst) -> Result,
    ) -> Result
        where
            L: Language,
            N: Analysis<L>,
    {
        self.inner_run(egraph, instructions, subst, true, yield_fn)
    }

    fn inner_run<L, N>(
        &mut self,
        egraph: &EGraph<L, N>,
        instructions: &[Instruction<L>],
        subst: &Subst,
        colored_jumps: bool,
        yield_fn: &mut impl FnMut(&Self, &Subst) -> Result,
    ) -> Result
        where
            L: Language,
            N: Analysis<L>,
    {
        let mut instructions = instructions.iter();
        while let Some(instruction) = instructions.next() {
            match instruction {
                Instruction::Bind { eclass, out, node } => {
                    let class_color = egraph[self.reg(*eclass)].color();
                    dassert!(class_color.is_none() || class_color == self.color);
                    let remaining_instructions = instructions.as_slice();
                    return for_each_matching_node(&egraph[self.reg(*eclass)], node, |matched| {
                        self.reg.truncate(out.0 as usize);
                        matched.for_each(|id| self.reg.push(id));
                        self.inner_run(egraph, remaining_instructions, subst, colored_jumps, yield_fn)
                    });
                }
                Instruction::Scan { out, top_pat } => {
                    let remaining_instructions = instructions.as_slice();
                    let mut run = |machine: &mut Machine, id| {
                        let cur_color = machine.color.clone();
                        let class_color = egraph[id].color();
                        if cur_color.is_some() && cur_color != class_color {
                            return Ok(());
                        }
                        machine.color = class_color;
                        machine.reg.truncate(out.0 as usize);
                        machine.reg.push(id);
                        machine.inner_run(egraph, remaining_instructions, subst, colored_jumps, yield_fn)?;
                        machine.color = cur_color;
                        Ok(())
                    };

                    match top_pat {
                        Either::Left(node) => {
                            if let Some(ids) = egraph.classes_by_op_id().get(&node.op_id()) {
                                for class in ids {
                                    run(self, *class)?;
                                }
                            }
                        }
                        Either::Right(opt_reg_var) => {
                            if let Some(reg_var) = opt_reg_var {
                                run(self, self.reg(*reg_var))?;
                            } else {
                                for class in egraph.classes() {
                                    run(self, class.id)?;
                                }
                            }
                        }
                    }
                    return Ok(());
                }
                Instruction::Compare { i, j } => {
                    let fixed_i = egraph.opt_colored_find(self.color.clone(), self.reg(*i));
                    let fixed_j = egraph.opt_colored_find(self.color.clone(), self.reg(*j));
                    if fixed_i != fixed_j {
                        if colored_jumps && self.color.is_none() {
                            if let Some(eqs) = egraph.get_colored_equalities(fixed_i) {
                                for (cid, _id) in eqs.into_iter().filter(|(_cid, id)| *id == fixed_j) {
                                    self.color = Some(cid);
                                    self.inner_run(egraph, instructions.as_slice(), subst, colored_jumps, yield_fn)?;
                                    self.color = None;
                                }
                            }
                        }
                        return Ok(());
                    }
                }
                Instruction::Lookup { term, i } => {
                    assert!(self.color.is_none(), "Lookup instruction is an optimization for non colored search");
                    assert!(!colored_jumps, "Lookup instruction is an optimization for non colored search");
                    self.lookup.clear();
                    for node in term {
                        match node {
                            ENodeOrReg::ENode(node) => {
                                let look = |i| self.lookup[usize::from(i)];
                                match egraph.lookup(node.clone().map_children(look)) {
                                    Some(id) => self.lookup.push(id),
                                    None => return Ok(()),
                                }
                            }
                            ENodeOrReg::Reg(r) => {
                                self.lookup.push(egraph.find(self.reg(*r)));
                            }
                        }
                    }

                    let id = egraph.find(self.reg(*i));
                    if self.lookup.last().copied() != Some(id) {
                        return Ok(());
                    }
                }
                Instruction::ColorJump { orig } => {
                    let remaining_instructions = instructions.as_slice();
                    if !colored_jumps {
                        return self.run(egraph, remaining_instructions, subst, yield_fn);
                    }
                    // Doing this now to later close yield_fn in a closure.
                    self.colored_run(egraph, remaining_instructions, subst, yield_fn)?;

                    let mut run_jump = |machine: &mut Machine, jump_id| {
                        machine.reg[orig.0 as usize] = jump_id;
                        machine.colored_run(egraph, remaining_instructions, subst, yield_fn)
                    };
                    let id = egraph.find(self.reg(*orig));
                    return if let Some(color) = self.color.as_ref() {
                        // Will also run id as it is part of black_ids
                        if let Some(eqs) = egraph.get_color(*color).unwrap().black_ids(egraph, id) {
                            for jump_id in eqs {
                                if *jump_id == id {
                                    continue;
                                }
                                run_jump(self, *jump_id)?;
                            }
                        }
                        Ok(())
                    } else {
                        if let Some(eqs) = egraph.get_colored_equalities(id) {
                            for (c_id, jumped_id) in eqs {
                                let jumped_id = egraph.find(jumped_id);
                                if jumped_id == id {
                                    continue;
                                }
                                self.color = Some(c_id);
                                run_jump(self, jumped_id)?;
                                self.color = None;
                            }
                        }
                        Ok(())
                    }
                }
            }
        }

        yield_fn(self, subst)
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
                self.add_todo(pattern, Id::from(last_i), i);
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
        while let Some(((id, reg), node)) = self.next() {
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
                let out = next_out;
                next_out.0 += node.len() as u32;

                // zero out the children so Bind can use it to sort
                let op = node.clone().map_children(|_| Id::from(0));
                if !first_bind {
                    self.instructions.push(Instruction::ColorJump {
                        orig: reg,
                    });
                } else {
                    first_bind = false;
                }
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

    pub(crate) fn compile_from_multi_pat(patterns: &[(Var, PatternAst<L>)]) -> Self {
        let mut compiler = Compiler::new();
        for (var, pattern) in patterns {
            compiler.compile(Some(*var), pattern);
        }
        compiler.extract()
    }

    pub fn run<A>(
        &self,
        egraph: &EGraph<L, A>,
        eclass: Id,
    ) -> Vec<Subst>
        where
            A: Analysis<L>,
    {
        self.inner_run(egraph, eclass, None, false)
    }

    pub fn colored_run<A>(&self,
                       egraph: &EGraph<L, A>,
                       eclass: Id,
                       color: Option<ColorId>,
    ) -> Vec<Subst>
        where
            A: Analysis<L>,
    {
        self.inner_run(egraph, eclass, color, true)
    }

    fn inner_run<A>(
        &self,
        egraph: &EGraph<L, A>,
        eclass: Id,
        opt_color: Option<ColorId>,
        run_color: bool,
    ) -> Vec<Subst>
        where
            A: Analysis<L>,
    {
        assert!(egraph.is_clean(), "Tried to search a dirty e-graph!");

        let mut machine = Machine::default();
        assert_eq!(machine.reg.len(), 0);
        machine.reg.push(eclass);

        let class_color = egraph[eclass].color();
        dassert!(opt_color.is_none() || class_color.is_none() || opt_color == class_color,
                 "Tried to run a colored program on an eclass with a different color: {:?} vs {:?}",
                 opt_color, class_color);
        machine.color = class_color;

        let mut matches = Vec::new();
        let mut yield_fn = |machine: &Machine, subst: &Subst| {
            let subst_vec = subst
                .vec
                .iter()
                // HACK we are reusing Ids here, this is bad
                .map(|(v, reg_id)| (*v, machine.reg(Reg(usize::from(*reg_id) as u32))))
                .collect();
            matches.push(Subst { vec: subst_vec, color: machine.color.clone() });
            Ok(())
        };

        if run_color {
            machine.colored_run(
                egraph,
                &self.instructions,
                &self.subst,
                &mut yield_fn,
            ).unwrap_or_default();
        } else {
            assert!(opt_color.is_none());
            machine.run(
                egraph,
                &self.instructions,
                &self.subst,
                &mut yield_fn,
            ).unwrap_or_default();
        }

        log::trace!("Ran program, found {:?}", matches);
        matches
    }
}