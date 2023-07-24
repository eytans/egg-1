use crate::*;

struct Machine<'a, L: Language, N: Analysis<L>> {
    reg: Vec<Id>,
    // a buffer to re-use for lookups
    lookup: Vec<Id>,
    stack: Vec<MachineContext>,
    instructions: &'a [Instruction<L>],
    egraph: &'a EGraph<L, N>,
    subst: Subst,
}

impl<'a, L: Language, N: Analysis<L>> Machine<'a, L, N> {
    #[inline(always)]
    fn reg(&self, reg: Reg) -> Id {
        self.reg[reg.0 as usize]
    }

    fn new(instructions: &'a [Instruction<L>], egraph: &'a EGraph<L, N>, subst: Subst) -> Self {
        Machine {
            reg: Default::default(),
            lookup: Default::default(),
            stack: Default::default(),
            instructions,
            egraph,
            subst,
        }
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
    Bind { node: L, eclass: Reg, out: Reg },
    Compare { i: Reg, j: Reg },
    Lookup { term: Vec<ENodeOrReg<L>>, i: Reg },
    Scan { out: Reg },
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
        debug_assert!(node.all(|id| id == Id::from(0)));
        debug_assert!(eclass.nodes.windows(2).all(|w| w[0] < w[1]));
        let mut start = eclass.nodes.binary_search(node).unwrap_or_else(|i| i);
        let discrim = std::mem::discriminant(node);
        while start > 0 {
            if std::mem::discriminant(&eclass.nodes[start - 1]) == discrim {
                start -= 1;
            } else {
                break;
            }
        }
        let matching = eclass.nodes[start..]
            .iter()
            .take_while(|&n| std::mem::discriminant(n) == discrim)
            .filter(|n| node.matches(n));
        debug_assert_eq!(
            matching.clone().count(),
            eclass.nodes.iter().filter(|n| node.matches(n)).count(),
            "matching node {:?}\nstart={}\n{:?} != {:?}\nnodes: {:?}",
            node,
            start,
            matching.clone().collect::<HashSet<_>>(),
            eclass
                .nodes
                .iter()
                .filter(|n| node.matches(n))
                .collect::<HashSet<_>>(),
            eclass.nodes
        );
        matching.for_each(&mut f)
    }
}

struct MachineContext {
    instruction_index: usize,
    truncate: usize,
    to_push: Vec<Id>,
}

impl MachineContext {
    fn new(instruction_index: usize, truncate: usize, push: Vec<Id>) -> Self {
        Self {
            instruction_index,
            truncate,
            to_push: push
        }
    }
}

impl<'a, L: Language, N: Analysis<L>> Iterator for Machine<'a, L, N> {
    type Item = Subst;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let current_state = self.stack.pop().unwrap();
            self.reg.truncate(current_state.truncate);
            for id in current_state.to_push {
                self.reg.push(id);
            }
            let mut index = current_state.instruction_index;
            'instr: while index < self.instructions.len() {
                let instruction = &self.instructions[index];
                match instruction {
                    Instruction::Bind { eclass, out, node } => {
                        for_each_matching_node(&self.egraph[self.reg(*eclass)], node, |matched| {
                            let truncate = out.0 as usize;
                            let to_push = matched.children().iter().copied().collect();
                            self.stack.push(MachineContext::new(index + 1, truncate, to_push));
                        });
                        break;
                    }
                    Instruction::Scan { out } => {
                        for class in self.egraph.classes() {
                            let truncate = out.0 as usize;
                            let to_push = vec![class.id];
                            self.stack.push(MachineContext::new(index + 1, truncate, to_push));
                        }
                        break;
                    }
                    Instruction::Compare { i, j } => {
                        if self.egraph.find(self.reg(*i)) != self.egraph.find(self.reg(*j)) {
                            break;
                        }
                    }
                    Instruction::Lookup { term, i } => {
                        self.lookup.clear();
                        for node in term {
                            match node {
                                ENodeOrReg::ENode(node) => {
                                    let look = |i| self.lookup[usize::from(i)];
                                    match self.egraph.lookup(node.clone().map_children(look)) {
                                        Some(id) => self.lookup.push(id),
                                        None => break 'instr,
                                    }
                                }
                                ENodeOrReg::Reg(r) => {
                                    self.lookup.push(self.egraph.find(self.reg(*r)));
                                }
                            }
                        }

                        let id = self.egraph.find(self.reg(*i));
                        if self.lookup.last().copied() != Some(id) {
                            break 'instr;
                        }
                    }
                };
                index += 1;
            }
            if index == self.instructions.len() {
                let subst_vec = self.subst
                    .vec
                    .iter()
                    // HACK we are reusing Ids here, this is bad
                    .map(|(v, reg_id)| (*v, self.reg(Reg(usize::from(*reg_id) as u32))))
                    .collect();
                return Some(Subst { vec: subst_vec });
            }
        }
        return None;
    }
}

struct Compiler<L> {
    v2r: IndexMap<Var, Reg>,
    free_vars: Vec<HashSet<Var>>,
    subtree_size: Vec<usize>,
    todo_nodes: HashMap<(Id, Reg), L>,
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
        match &pattern[id] {
            ENodeOrVar::Var(v) => {
                if let Some(&j) = self.v2r.get(v) {
                    self.instructions.push(Instruction::Compare { i: reg, j })
                } else {
                    self.v2r.insert(*v, reg);
                }
            }
            ENodeOrVar::ENode(pat) => {
                self.todo_nodes.insert((id, reg), pat.clone());
            }
        }
    }

    fn load_pattern(&mut self, pattern: &PatternAst<L>) {
        let len = pattern.as_ref().len();
        self.free_vars = Vec::with_capacity(len);
        self.subtree_size = Vec::with_capacity(len);

        for node in pattern.as_ref() {
            let mut free = HashSet::default();
            let mut size = 0;
            match node {
                ENodeOrVar::ENode(n) => {
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
                comp.instructions
                    .push(Instruction::Scan { out: comp.next_reg });
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

        while let Some(((id, reg), node)) = self.next() {
            if self.is_ground_now(id) && !node.is_leaf() {
                let extracted = pattern.extract(id);
                self.instructions.push(Instruction::Lookup {
                    i: reg,
                    term: extracted
                        .as_ref()
                        .iter()
                        .map(|n| match n {
                            ENodeOrVar::ENode(n) => ENodeOrReg::ENode(n.clone()),
                            ENodeOrVar::Var(v) => ENodeOrReg::Reg(self.v2r[v]),
                        })
                        .collect(),
                });
            } else {
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

    pub fn run_with_limit<A>(
        &self,
        egraph: &EGraph<L, A>,
        eclass: Id,
        limit: usize,
    ) -> Vec<Subst>
    where
        A: Analysis<L>,
    {
        assert!(egraph.clean, "Tried to search a dirty e-graph!");

        if limit == 0 {
            return vec![];
        }

        let mut machine = Machine::new(&self.instructions, egraph, self.subst.clone());
        assert_eq!(machine.reg.len(), 0);
        machine.reg.push(eclass);
        machine.stack.push(MachineContext::new(0,  1, vec![]));

        let matches = machine.into_iter().take(limit).collect();

        log::trace!("Ran program, found {:?}", matches);
        matches
    }
}
