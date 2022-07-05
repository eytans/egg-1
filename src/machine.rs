use crate::{Analysis, EClass, EGraph, ENodeOrVar, Id, Language, PatternAst, Subst, Var};
use std::cmp::Ordering;
use indexmap::IndexSet;
use log::warn;
use smallvec::SmallVec;
use crate::ColorId;
use crate::colors::Color;
use itertools::Itertools;

lazy_static!{
    static ref EMPTY_SET: IndexSet<(ColorId, Id)> = IndexSet::new();
}

struct Machine {
    reg: Vec<Id>,
    #[cfg(feature = "colored")]
    color: Option<ColorId>,
}

impl Default for Machine {
    fn default() -> Self {
        Self { reg: vec![], color: Default::default() }
    }
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
    Bind { node: L, i: Reg, out: Reg },
    Compare { i: Reg, j: Reg },
}

#[inline(always)]
fn for_each_matching_node<L, D>(eclass: &EClass<L, D>,
                                node: &L,
                                mut f: impl FnMut(&L))
    where
        L: Language,
{
    let filterer = |n: &L| { node.matches(n) };
    if eclass.nodes.len() < 50 {
        eclass.nodes.iter().filter(|x| filterer(x))
            .for_each(f)
    } else {
        debug_assert!(node.children().iter().all(|&id| id == Id::from(0)));
        debug_assert!(eclass.nodes.windows(2).all(|w| w[0] < w[1]));
        let mut start = eclass.nodes.binary_search(node).unwrap_or_else(|i| i);
        let matching = eclass.nodes[..start].iter().rev()
            .take_while(|x| filterer(x))
            .chain(eclass.nodes[start..].iter().take_while(|x| filterer(x)));
        debug_assert_eq!(
            matching.clone().count(),
            eclass.nodes.iter().filter(|x| filterer(x)).count(),
            "matching node {:?}\nstart={}\n{:?} != {:?}\nnodes: {:?}",
            node,
            start,
            matching.clone().collect::<indexmap::IndexSet<_>>(),
            eclass
                .nodes
                .iter()
                .filter(|x| filterer(x))
                .collect::<indexmap::IndexSet<_>>(),
            eclass.nodes
        );
        matching.for_each(f);
    }
}

impl Machine {
    #[inline(always)]
    fn reg(&self, reg: Reg) -> Id {
        self.reg[reg.0 as usize]
    }

    fn run<L, N>(
        &mut self,
        egraph: &EGraph<L, N>,
        instructions: &[Instruction<L>],
        subst: &Subst,
        yield_fn: &mut impl FnMut(&Self, &Subst),
    ) where
        L: Language,
        N: Analysis<L>,
    {
        let mut instructions = instructions.iter();
        while let Some(instruction) = instructions.next() {
            match instruction {
                Instruction::Bind { i, out, node } => {
                    let remaining_instructions = instructions.as_slice();
                    for_each_matching_node(&egraph[self.reg(*i)], node,
                                           |matched| {
                        self.reg.truncate(out.0 as usize);
                        self.reg.extend_from_slice(matched.children());
                        self.run(egraph, remaining_instructions, subst, yield_fn)
                    });
                    return;
                }
                Instruction::Compare { i, j } => {
                    if egraph.find(self.reg(*i)) != egraph.find(self.reg(*j)) {
                        return;
                    }
                }
            }
        }

        yield_fn(self, subst)
    }

    fn run_colored<L, N>(
        &mut self,
        egraph: &EGraph<L, N>,
        instructions: &[Instruction<L>],
        subst: &Subst,
        yield_fn: &mut impl FnMut(&Self, &Subst),
    ) where
        L: Language,
        N: Analysis<L>,
    {
        let mut instructions = instructions.iter();
        while let Some(instruction) = instructions.next() {
            match instruction {
                Instruction::Bind { i, out, node } => {
                    let remaining_instructions = instructions.as_slice();
                    let mut run_matches = |machine: &mut Machine,
                                           eclass: &EClass<L, N::Data>| {
                        for_each_matching_node(eclass, node, |matched| {
                            machine.reg.truncate(out.0 as usize);
                            machine.reg.extend_from_slice(matched.children());
                            machine.run_colored(egraph, remaining_instructions, subst, yield_fn);
                        });
                    };

                    // Collect colors that should be run on next instruction. Foreach color
                    // keep which eclasses it needs to run on (all except current reg).
                    // If we are already colored we need all ids.
                    run_matches(self, &egraph[self.reg(*i)]);
                    let old_reg = self.reg(*i);
                    if self.color.is_some() {
                        let c = &egraph.get_color(self.color.unwrap()).unwrap();
                        self.run_colored_branches(&egraph, i, &mut run_matches, c, old_reg);
                    } else {
                        for (c, id) in egraph.colored_equivalences.get(&self.reg(*i)).unwrap_or(&EMPTY_SET) {
                            self.reg[i.0 as usize] = *id;
                            self.color = Some(*c);
                            run_matches(self, &egraph[*id]);
                        }
                        self.color = None;
                    }
                    self.reg[i.0 as usize] = old_reg;
                    return;
                }
                Instruction::Compare { i, j } => {
                    if egraph.find(self.reg(*i)) != egraph.find(self.reg(*j)) {
                        if let Some(c) = self.color {
                            if egraph.colored_find(c, self.reg(*i)) != egraph.colored_find(c, self.reg(*j)) {
                                return;
                            }
                        } else {
                            for (c, id) in egraph.colored_equivalences.get(&self.reg(*i)).unwrap_or(&EMPTY_SET) {
                                if *id == self.reg(*j) {
                                    self.color = Some(*c);
                                    self.run_colored(egraph, instructions.as_slice(), subst, yield_fn);
                                }
                            }
                            self.color = None;
                            return;
                        }
                    }
                }
            }
        }

        yield_fn(self, subst)
    }

    fn run_colored_branches<L, N, F>(&mut self, egraph: &EGraph<L, N>, i: &Reg, mut run_matches: &mut F, c: &Color, old_reg: Id)
    where L: Language, N: Analysis<L>, F: FnMut(&mut Machine, &EClass<L, <N as Analysis<L>>::Data>){
        let ids = c.black_ids(old_reg);
        ids.iter().for_each(|&b_ids| { b_ids.iter().filter(|i| *i != &old_reg).for_each(|id| {
            self.reg[i.0 as usize] = *id;
            run_matches(self, &egraph[*id]);
        })});
        self.reg[i.0 as usize] = old_reg;
    }
}

type VarToReg = indexmap::IndexMap<Var, Reg>;
type TodoList<L> = std::collections::BinaryHeap<Todo<L>>;

#[derive(PartialEq, Eq)]
struct Todo<L> {
    reg: Reg,
    pat: ENodeOrVar<L>,
}

impl<L: Language> PartialOrd for Todo<L> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<L: Language> Ord for Todo<L> {
    // TodoList is a max-heap, so we greater is higher priority
    fn cmp(&self, other: &Self) -> Ordering {
        use ENodeOrVar::*;
        match (&self.pat, &other.pat) {
            // fewer children means higher priority
            (ENode(e1, _), ENode(e2, _)) => e2.len().cmp(&e1.len()),
            // Var is higher prio than enode
            (ENode(_, _), Var(_)) => Ordering::Less,
            (Var(_), ENode(_, _)) => Ordering::Greater,
            (Var(_), Var(_)) => Ordering::Equal,
        }
    }
}

struct Compiler<'a, L> {
    pattern: &'a [ENodeOrVar<L>],
    v2r: VarToReg,
    todo: TodoList<L>,
    out: Reg,
}

impl<'a, L: Language> Compiler<'a, L> {
    fn compile(pattern: &'a [ENodeOrVar<L>]) -> Program<L> {
        let last = pattern.last().unwrap();
        let mut compiler = Self {
            pattern,
            v2r: Default::default(),
            todo: Default::default(),
            out: Reg(1),
        };
        compiler.todo.push(Todo {
            reg: Reg(0),
            pat: last.clone(),
        });
        compiler.go()
    }

    fn go(&mut self) -> Program<L> {
        let mut instructions = vec![];
        while let Some(Todo { reg: i, pat }) = self.todo.pop() {
            match pat {
                ENodeOrVar::Var(v) => {
                    if let Some(&j) = self.v2r.get(&v) {
                        instructions.push(Instruction::Compare { i, j })
                    } else {
                        self.v2r.insert(v, i);
                    }
                }
                ENodeOrVar::ENode(node, name) => {
                    let out = self.out;
                    self.out.0 += node.len() as u32;

                    for (id, &child) in node.children().iter().enumerate() {
                        let r = Reg(out.0 + id as u32);
                        self.todo.push(Todo {
                            reg: r,
                            pat: self.pattern[usize::from(child)].clone(),
                        });
                    }

                    // zero out the children so Bind can use it to sort
                    let node = node.map_children(|_| Id::from(0));
                    if let Some(name) = name {
                        self.v2r.insert(name.parse().unwrap(), out);
                    }
                    instructions.push(Instruction::Bind { i, node, out })
                }
            }
        }

        let mut subst = Subst::default();
        for (v, r) in &self.v2r {
            subst.insert(*v, Id::from(r.0 as usize));
        }
        Program {
            instructions,
            subst,
        }
    }
}

impl<L: Language> Program<L> {
    pub(crate) fn compile_from_pat(pattern: &PatternAst<L>) -> Self {
        let program = Compiler::compile(pattern.as_ref());
        log::debug!("Compiled {:?} to {:?}", pattern.as_ref(), program);
        program
    }

    pub fn run<A>(&self, egraph: &EGraph<L, A>, eclass: Id) -> Vec<Subst>
        where
            A: Analysis<L>,
    {
        let mut machine = Machine::default();

        assert_eq!(machine.reg.len(), 0);
        machine.reg.push(eclass);

        let mut substs = Vec::new();
        machine.run(
            egraph,
            &self.instructions,
            &self.subst,
            &mut |machine, subst| {
                let subst_vec = subst.vec.iter()
                    // HACK we are reusing Ids here, this is bad
                    .map(|(v, reg_id)| (*v, machine.reg(Reg(usize::from(*reg_id) as u32))))
                    .collect();
                substs.push(Subst { vec: subst_vec, color: machine.color.clone() })
            },
        );

        log::trace!("Ran program, found {:?}", substs);
        substs
    }

    pub fn colored_run<A>(&self, egraph: &EGraph<L, A>, eclass: Id, color: Option<ColorId>) -> Vec<Subst>
        where
            A: Analysis<L>,
    {
        let mut machine = Machine::default();

        assert_eq!(machine.reg.len(), 0);
        machine.reg.push(eclass);
        if let Some(c) = egraph[eclass].color {
            machine.color = Some(c);
            color.as_ref().map(|c1| assert_eq!(c1, &c));
        }
        if let Some(c) = color {
            machine.color = Some(c);
        }

        let mut substs = Vec::new();
        machine.run_colored(
            egraph,
            &self.instructions,
            &self.subst,
            &mut |machine, subst| {
                let subst_vec = subst.vec.iter()
                    // HACK we are reusing Ids here, this is bad
                    .map(|(v, reg_id)| (*v, machine.reg(Reg(usize::from(*reg_id) as u32))))
                    .collect();
                substs.push(Subst { vec: subst_vec, color: machine.color.clone() })
            },
        );

        log::trace!("Ran program, found {:?}", substs);
        substs
    }
}
