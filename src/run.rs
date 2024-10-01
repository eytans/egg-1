use maplit::{btreemap, btreeset};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use derive_new::new;
use std::fmt;
use std::fmt::Formatter;
use indexmap::IndexMap;
use instant::{Duration, Instant};
use itertools::{iproduct, Itertools};
use log::*;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::iter::IndexedParallelIterator;
use crate::{language, Analysis, Applier, EGraph, Id, Language, MultiPattern, RecExpr, Rewrite, SearchMatches, Searcher, Symbol};
use derive_more::Display;

/** Facilitates running rewrites over an [`EGraph`].

One use for [`EGraph`]s is as the basis of a rewriting system.
Since an egraph never "forgets" state when applying a [`Rewrite`], you
can apply many rewrites many times quite efficiently.
After the egraph is "full" (the rewrites can no longer find new
equalities) or some other condition, the egraph compactly represents
many, many equivalent expressions.
At this point, the egraph is ready for extraction (see [`Extractor`])
which can pick the represented expression that's best according to
some cost function.

This technique is called
[equality saturation](https://www.cs.cornell.edu/~ross/publications/eqsat/)
in general.
However, there can be many challenges in implementing this "outer
loop" of applying rewrites, mostly revolving around which rules to run
and when to stop.

[`Runner`] is `egg`'s provided equality saturation engine that has
reasonable defaults and implements many useful things like saturation
checking, egraph size limits, and customizable rule
[scheduling](trait.RewriteScheduler.html).
Consider using [`Runner`] before rolling your own outer loop.

Here are some of the things [`Runner`] does for you:

- Saturation checking

  [`Runner`] checks to see if any of the rules added anything
  new to the [`EGraph`]. If none did, then it stops, returning
  [`StopReason::Saturated`](enum.StopReason.html#variant.Saturated).

- Iteration limits

  You can set a upper limit of iterations to do in case the search
  doesn't stop for some other reason. If this limit is hit, it stops with
  [`StopReason::IterationLimit`](enum.StopReason.html#variant.IterationLimit).

- [`EGraph`] size limit

  You can set a upper limit on the number of enodes in the egraph.
  If this limit is hit, it stops with
  [`StopReason::NodeLimit`](enum.StopReason.html#variant.NodeLimit).

- Time limit

  You can set a time limit on the runner.
  If this limit is hit, it stops with
  [`StopReason::TimeLimit`](enum.StopReason.html#variant.TimeLimit).

- Rule scheduling

  Some rules enable themselves, blowing up the [`EGraph`] and
  preventing other rewrites from running as many times.
  To prevent this, you can provide your own [`RewriteScheduler`] to
  govern when to run which rules.

  [`BackoffScheduler`] is the default scheduler.

[`Runner`] generates [`Iteration`]s that record some data about
each iteration.
You can add your own data to this by implementing the
[`IterationData`] trait.
[`Runner`] is generic over the [`IterationData`] that it will be in the
[`Iteration`]s, but by default it uses `()`.

[`Runner`]: struct.Runner.html
[`RewriteScheduler`]: trait.RewriteScheduler.html
[`Extractor`]: struct.Extractor.html
[`Rewrite`]: struct.Rewrite.html
[`BackoffScheduler`]: struct.BackoffScheduler.html
[`EGraph`]: struct.EGraph.html
[`Iteration`]: struct.Iteration.html
[`IterationData`]: trait.IterationData.html

# Example
```
use easter_egg::{*, rewrite as rw};

define_language! {
    enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

let rules: &[Rewrite<SimpleLanguage, ()>] = &[
    rw!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
    rw!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),

    rw!("add-0"; "(+ ?a 0)" => "?a"),
    rw!("mul-0"; "(* ?a 0)" => "0"),
    rw!("mul-1"; "(* ?a 1)" => "?a"),
];

pub struct MyIterData {
    smallest_so_far: usize,
}

type MyRunner = Runner<SimpleLanguage, (), MyIterData>;

impl IterationData<SimpleLanguage, ()> for MyIterData {
    fn make(runner: &MyRunner) -> Self {
        let root = runner.roots[0];
        let mut extractor = Extractor::new(&runner.egraph, AstSize);
        MyIterData {
            smallest_so_far: extractor.find_best(root).0,
        }
    }
}

let start = "(+ 0 (* 1 foo))".parse().unwrap();
// Runner is customizable in the builder pattern style.
let runner = MyRunner::new(Default::default())
    .with_iter_limit(10)
    .with_node_limit(10_000)
    .with_expr(&start)
    .with_scheduler(SimpleScheduler)
    .run(rules);


println!(
    "Stopped after {} iterations, reason: {:?}",
    runner.iterations.len(),
    runner.stop_reason
);

```

*/
pub struct Runner<L: Language, N: Analysis<L>, IterData = ()> {
    /// The [`EGraph`](struct.EGraph.html) used.
    pub egraph: EGraph<L, N>,
    /// Data accumulated over each [`Iteration`](struct.Iteration.html).
    pub iterations: Vec<Iteration<IterData>>,
    /// The roots of expressions added by the
    /// [`with_expr`](#method.with_expr()) method, in insertion order.
    pub roots: Vec<Id>,
    /// Why the `Runner` stopped. This will be `None` if it hasn't
    /// stopped yet.
    pub stop_reason: Option<StopReason>,

    /// The hooks added by the
    /// [`with_hook`](#method.with_hook) method, in insertion order.
    #[allow(clippy::type_complexity)]
    pub hooks: Vec<Box<dyn FnMut(&mut Self) -> Result<(), String>>>,

    // limits
    iter_limit: usize,
    node_limit: usize,
    time_limit: Duration,

    start_time: Option<Instant>,
    scheduler: Box<dyn RewriteScheduler<L, N>>,
}

impl<L, N> Default for Runner<L, N, ()>
where
    L: Language + 'static,
    N: Analysis<L> + Default + 'static,
{
    fn default() -> Self {
        Runner::new(N::default())
    }
}

/// Error returned by [`Runner`] when it stops.
///
/// [`Runner`]: struct.Runner.html
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum StopReason {
    /// The egraph saturated, i.e., there was an iteration where we
    /// didn't learn anything new from applying the rules.
    Saturated,
    /// The iteration limit was hit. The data is the iteration limit.
    IterationLimit(usize),
    /// The enode limit was hit. The data is the enode limit.
    NodeLimit(usize),
    /// The time limit was hit. The data is the time limit in seconds.
    TimeLimit(f64),
    /// Some other reason to stop.
    Other(String),
}

/// Struct for search metadata which is how many matches and binds each rule did
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, new, Default, Display)]
#[display("{} matches {} binds", matches, binds)]
struct SearchMetadata {
    matches: usize,
    binds: usize,
}

/// Data generated by running a [`Runner`] one iteration.
///
/// If the `serde-1` feature is enabled, this implements
/// [`serde::Serialize`][ser], which is useful if you want to output
/// this as a JSON or some other format.
///
/// [`Runner`]: struct.Runner.html
/// [ser]: https://docs.rs/serde/latest/serde/trait.Serialize.html
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Iteration<IterData> {
    /// The number of enodes in the egraph at the start of this
    /// iteration.
    pub egraph_nodes: usize,
    /// The number of eclasses in the egraph at the start of this
    /// iteration.
    pub egraph_classes: usize,
    /// A map from rule name to number of times it was _newly_ applied
    /// in this iteration.
    pub applied: IndexMap<Symbol, usize>,
    /// A map from rule name to number of binds it did (as a measure of time)
    pub searched: IndexMap<Symbol, SearchMetadata>,
    /// Seconds spent running hooks.
    pub hook_time: f64,
    /// Seconds spent searching in this iteration.
    pub search_time: f64,
    /// Seconds spent applying rules in this iteration.
    pub apply_time: f64,
    /// Seconds spent [`rebuild`](struct.EGraph.html#method.rebuild)ing
    /// the egraph in this iteration.
    pub rebuild_time: f64,
    /// Total time spent in this iteration, including data generation time.
    pub total_time: f64,
    /// The user provided annotation for this iteration
    pub data: IterData,
    /// If the runner stopped on this iterations, this is the reason
    pub stop_reason: Option<StopReason>,
}

impl<IterData> Display for Iteration<IterData> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Iteration report")?;
        writeln!(f, "================")?;
        writeln!(f, "  Egraph size: {} nodes, {} classes", self.egraph_nodes, self.egraph_classes)?;
        writeln!(f, "  Time: {:.2} total, {:.2} search, {:.2} apply, {:.2} rebuild", self.total_time, self.search_time, self.apply_time, self.rebuild_time)?;
        writeln!(f, "  Stop reason: {:?}", self.stop_reason)?;
        // Print top ten worst rules first
        let worst = self.searched.iter().sorted_by_key(|(_, d)| d.binds).rev().take(10);
        writeln!(f, "  Top ten slowest rules:")?;
        for (name, binds) in worst {
            writeln!(f, "    {}: number of binds {}", name, binds)?;
        }
        // writeln!(f, "  Rule times:")?;
        // for (name, count) in &self.searched {
        //     writeln!(f, "    {}: {}", name, count.as_secs_f64())?;
        // }
        // writeln!(f, "  Applied rules:")?;
        // for (name, count) in &self.applied {
        //     writeln!(f, "    {}: {}", name, count)?;
        // }
        Ok(())
    }
}

type RunnerResult<T> = std::result::Result<T, StopReason>;

impl<L, N, IterData> Runner<L, N, IterData>
where
    L: Language + 'static,
    N: Analysis<L> + 'static,
    IterData: IterationData<L, N>,
{
    /// Create a new `Runner` with the given analysis and default parameters.
    pub fn new(analysis: N) -> Self {
        Self {
            iter_limit: 30,
            node_limit: 100_000,
            time_limit: Duration::from_secs(500),

            egraph: EGraph::new(analysis),
            roots: vec![],
            iterations: vec![],
            stop_reason: None,
            hooks: vec![],

            start_time: None,
            scheduler: Box::new(SimpleScheduler::default()),
        }
    }

    /// Sets the iteration limit. Default: 30
    pub fn with_iter_limit(self, iter_limit: usize) -> Self {
        Self { iter_limit, ..self }
    }

    /// Sets the egraph size limit (in enodes). Default: 10,000
    pub fn with_node_limit(self, node_limit: usize) -> Self {
        Self { node_limit, ..self }
    }

    /// Sets the runner time limit. Default: 5 seconds
    pub fn with_time_limit(self, time_limit: Duration) -> Self {
        Self { time_limit, ..self }
    }

    /// Add a hook to instrument or modify the behavior of a [`Runner`].
    /// Each hook will run at the beginning of each iteration, i.e. before
    /// all the rewrites.
    ///
    /// If your hook modifies the e-graph, make sure to call
    /// [`rebuild`](struct.EGraph.html#method.rebuild).
    ///
    /// # Example
    /// ```
    /// # use easter_egg::*;
    /// let rules: &[Rewrite<SymbolLang, ()>] = &[
    ///     rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
    ///     // probably some others ...
    /// ];
    ///
    /// Runner::<SymbolLang, ()>::default()
    ///     .with_expr(&"(+ 5 2)".parse().unwrap())
    ///     .with_hook(|runner| {
    ///          println!("Egraph is this big: {}", runner.egraph.total_size());
    ///          Ok(())
    ///     })
    ///     .run(rules);
    /// ```
    /// [`Runner`]: struct.Runner.html
    pub fn with_hook<F>(mut self, hook: F) -> Self
    where
        F: FnMut(&mut Self) -> Result<(), String> + 'static,
    {
        self.hooks.push(Box::new(hook));
        self
    }

    /// Change out the [`RewriteScheduler`] used by this [`Runner`].
    /// The default one is [`BackoffScheduler`].
    ///
    /// [`RewriteScheduler`]: trait.RewriteScheduler.html
    /// [`BackoffScheduler`]: struct.BackoffScheduler.html
    /// [`Runner`]: struct.Runner.html
    pub fn with_scheduler(self, scheduler: impl RewriteScheduler<L, N> + 'static) -> Self {
        let scheduler = Box::new(scheduler);
        Self { scheduler, ..self }
    }

    #[allow(dead_code)]
    fn with_boxed_scheduler(self, scheduler: Box<dyn RewriteScheduler<L, N> + 'static>) -> Self {
        Self { scheduler, ..self }
    }

    /// Add an expression to the egraph to be run.
    ///
    /// The eclass id of this addition will be recorded in the
    /// [`roots`](struct.Runner.html#structfield.roots) field, ordered by
    /// insertion order.
    pub fn with_expr(mut self, expr: &RecExpr<L>) -> Self {
        let id = self.egraph.add_expr(expr);
        self.egraph.rebuild();
        self.roots.push(id);
        self
    }

    /// Replace the [`EGraph`](struct.EGraph.html) of this `Runner`.
    pub fn with_egraph(self, egraph: EGraph<L, N>) -> Self {
        Self { egraph, ..self }
    }

    /// Run this `Runner` until it stops.
    /// After this, the field
    /// [`stop_reason`](#structfield.stop_reason) is guaranteed to be
    /// set.
    pub fn run<'a, R>(mut self, rules: R) -> Self
    where
        R: IntoIterator<Item=&'a Rewrite<L, N>>,
        L: 'a,
        N: 'a,
    {
        let rules = rules.into_iter().collect::<Vec<_>>();
        #[cfg(feature = "keep_splits")]
        {
            let mut sched = std::mem::replace(&mut self.scheduler, Box::new(SimpleScheduler::default()));
            for g in self.egraph.all_splits.iter_mut() {
                let mut runner: Runner<L, N> = Runner::new(g.analysis.clone())
                    .with_iter_limit(self.iter_limit)
                    .with_node_limit(self.node_limit)
                    .with_time_limit(self.time_limit)
                    .with_egraph(std::mem::replace(g, EGraph::new(g.analysis.clone())))
                    .with_boxed_scheduler(sched);
                sched = std::mem::replace(&mut runner.scheduler, Box::new(SimpleScheduler::default()));
                let runner = runner.run(rules.iter().cloned());
                *g = runner.egraph;
            }
        }
        check_rules(&rules);
        self.egraph.rebuild();
        loop {
            let mut result = self.run_one(&rules);
            if result.is_ok() {
                result = self.check_limits()
            }
            // we need to check_limits after the iteration is complete to check for iter_limit
            if let Err(stop_reason) = result {
                info!("Stopping: {:?}", stop_reason);
                self.stop_reason = Some(stop_reason);
                // push on a final iteration to mark the end state
                self.iterations.push(Iteration {
                    stop_reason: self.stop_reason.clone(),
                    egraph_nodes: self.egraph.total_number_of_nodes(),
                    egraph_classes: self.egraph.number_of_classes(),
                    data: IterData::make(&self),
                    applied: Default::default(),
                    search_time: Default::default(),
                    hook_time: Default::default(),
                    apply_time: Default::default(),
                    rebuild_time: Default::default(),
                    total_time: Default::default(),
                    searched: Default::default(),
                });
                break;
            } else {
                if let Some(i) = self.iterations.last() {
                    warn!("Iteration data:\n {}", i);
                }
            }
        }

        self
    }

    #[rustfmt::skip]
    /// Prints some information about a runners run.
    pub fn print_report(&self) {
        let search_time: f64 = self.iterations.iter().map(|i| i.search_time).sum();
        let apply_time: f64 = self.iterations.iter().map(|i| i.apply_time).sum();
        let rebuild_time: f64 = self.iterations.iter().map(|i| i.rebuild_time).sum();
        let total_time: f64 = self.iterations.iter().map(|i| i.total_time).sum();

        let iters = self.iterations.len();

        let eg = &self.egraph;
        println!("Runner report");
        println!("=============");
        println!("  Stop reason: {:?}", self.stop_reason.as_ref().unwrap());
        println!("  Iterations: {}", iters);
        println!("  Egraph size: {} nodes, {} classes, {} memo", eg.total_number_of_nodes(), eg.number_of_classes(), eg.total_size());
        println!("  Total time: {}", total_time);
        println!("    Search:  ({:.2}) {}", search_time / total_time, search_time);
        println!("    Apply:   ({:.2}) {}", apply_time / total_time, apply_time);
        println!("    Rebuild: ({:.2}) {}", rebuild_time / total_time, rebuild_time);
    }

    fn run_one(&mut self, rules: &[&Rewrite<L, N>]) -> RunnerResult<()> {
        assert!(self.stop_reason.is_none());

        info!("\nIteration {}", self.iterations.len());

        self.try_start();
        self.check_limits()?;

        let egraph_nodes = self.egraph.total_size();
        let egraph_classes = self.egraph.number_of_classes();

        let hook_time = Instant::now();
        let mut hooks = std::mem::take(&mut self.hooks);
        let mut error = None;
        for hook in &mut hooks {
            if let Err(r) = hook(self).map_err(StopReason::Other) {
                error = Some(r);
                break;
            }
        }
        self.hooks = hooks;
        if let Some(e) = error {
            return Err(e);
        }
        let hook_time = hook_time.elapsed().as_secs_f64();

        let egraph_nodes_after_hooks = self.egraph.total_size();
        let egraph_classes_after_hooks = self.egraph.number_of_classes();

        let i = self.iterations.len();
        trace!("EGraph {:?}", self.egraph.dump());

        let start_time = Instant::now();

        let mut matches = Vec::new();
        // From now on scheduler may not access the runner and the other way around
        // Note this limits check_limits
        let mut sched = std::mem::replace(&mut self.scheduler, Box::new(SimpleScheduler {}));
        let result: RunnerResult<()> =
            sched.search_rewrites(i, &self.egraph, rules, &mut matches, self.start_time.unwrap(), self.time_limit);
        let search_binds = matches.iter().enumerate().map(|(ri, m)| {
            let name: Symbol = rules[ri].name().into();
            let binds = m.as_ref().map(|x| x.binds_done);
            (name, SearchMetadata::new(m.as_ref().map(|x| x.total_substs()).unwrap_or(0), binds.unwrap_or(0)))
        }).collect::<IndexMap<_, _>>();
        self.scheduler = sched;
        // Done with sched limitation
        self.check_limits()?;

        let search_time = start_time.elapsed().as_secs_f64();
        info!("Search time: {}", search_time);

        let apply_time = Instant::now();

        let mut applied = IndexMap::new();
        result.and_then(|_| {
            let temp = Box::new(SimpleScheduler {});
            let mut sched = std::mem::replace(&mut self.scheduler, temp);
            sched.apply_rewrites(&mut self.egraph, rules, i, matches, &mut applied);
            self.scheduler = sched;
            self.check_limits()
        })?;

        let apply_time = apply_time.elapsed().as_secs_f64();
        info!("Apply time: {}", apply_time);

        let rebuild_time = Instant::now();
        self.egraph.rebuild();

        let elapsed = rebuild_time.elapsed();
        let rebuild_time = elapsed.as_secs_f64();
        unsafe {
            REBUILD_TIME += elapsed.as_millis();
        }
        info!("Rebuild time: {}", rebuild_time);
        info!(
            "Size: n={}, e={}",
            self.egraph.total_size(),
            self.egraph.number_of_classes()
        );

        let saturated = applied.is_empty()
            && self.scheduler.can_stop(i)
            && (egraph_nodes == egraph_nodes_after_hooks)
            && (egraph_classes == egraph_classes_after_hooks);

        self.iterations.push(Iteration {
            applied,
            searched: search_binds,
            egraph_nodes,
            egraph_classes,
            hook_time,
            search_time,
            apply_time,
            rebuild_time,
            data: IterData::make(self),
            total_time: start_time.elapsed().as_secs_f64(),
            stop_reason: None,
        });

        if saturated {
            Err(StopReason::Saturated)
        } else {
            Ok(())
        }
    }

    fn try_start(&mut self) {
        self.start_time.get_or_insert_with(Instant::now);
    }

    fn check_limits(&self) -> RunnerResult<()> {
        let elapsed = self.start_time.unwrap().elapsed();
        if elapsed > self.time_limit {
            return Err(StopReason::TimeLimit(elapsed.as_secs_f64()));
        }

        let size = self.egraph.total_size();
        if size > self.node_limit {
            return Err(StopReason::NodeLimit(size));
        }

        if self.iterations.len() >= self.iter_limit {
            return Err(StopReason::IterationLimit(self.iterations.len()));
        }

        Ok(())
    }
}

fn check_rules<L: Language, N: Analysis<L>>(rules: &[&Rewrite<L, N>]) {
    let mut name_counts = IndexMap::new();
    for rw in rules {
        *name_counts.entry(rw.name()).or_default() += 1
    }

    name_counts.retain(|_, count: &mut usize| *count > 1);
    if !name_counts.is_empty() {
        eprintln!("WARNING: Duplicated rule names may affect rule reporting and scheduling.");
        log::warn!("Duplicated rule names may affect rule reporting and scheduling.");
        for (name, &count) in name_counts.iter() {
            assert!(count > 1);
            eprintln!("Rule '{}' appears {} times", name, count);
            log::warn!("Rule '{}' appears {} times", name, count);
        }
    }
}

/** A way to customize how a [`Runner`] runs [`Rewrite`]s.

This gives you a way to prevent certain [`Rewrite`]s from exploding
the [`EGraph`] and dominating how much time is spent while running the
[`Runner`].

[`EGraph`]: struct.EGraph.html
[`Runner`]: struct.Runner.html
[`Rewrite`]: struct.Rewrite.html
*/
#[allow(unused_variables)]
pub trait RewriteScheduler<L, N>
where
    L: Language + 'static,
    N: Analysis<L> + 'static,
{
    /// Whether or not the [`Runner`](struct.Runner.html) is allowed
    /// to say it has saturated.
    ///
    /// This is only called when the runner is otherwise saturated.
    /// Default implementation just returns `true`.
    fn can_stop(&mut self, iteration: usize) -> bool {
        true
    }

    /// A hook allowing you to customize rewrite searching behavior.
    /// Useful to implement rule management.
    ///
    /// Default implementation just calls
    /// [`Rewrite::search`](struct.Rewrite.html#method.search).
    fn search_rewrite(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrite: &Rewrite<L, N>,
    ) -> Option<SearchMatches> {
        rewrite.search(egraph)
    }

    /// A Hook allowing you to customize the rewrite searching behaviour for all rewrites at once.
    /// This is similar to [`search_rewrite`](RewriteScheduler::search_rewrite()), but
    /// requires that the schedualer handle limitation checks of the runner
    fn search_rewrites<'a, 'b>(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrites: &[&'a Rewrite<L, N>],
        matches: &mut Vec<Option<SearchMatches>>,
        start_time: Instant,
        time_limit: Duration,
    ) -> RunnerResult<()> {
        rewrites.iter().try_for_each(|rw| {
            let start_time = Instant::now();
            let ms = self.search_rewrite(iteration, egraph, rw);
            let elapsed = start_time.elapsed();
            unsafe {
                SEARCH_MATCHES += ms.as_ref().map(|x| x.total_substs()).unwrap_or(0);
                SEARCH_TIME += elapsed.as_millis();
            }
            matches.push(ms);
            let elapsed = start_time.elapsed();
            if elapsed > time_limit {
                Err(StopReason::TimeLimit(elapsed.as_secs_f64()))
            } else {
                Ok(())
            }
        })
    }

    /// A hook allowing you to customize rewrite application behavior.
    /// Useful to implement rule management.
    ///
    /// Default implementation just calls
    /// [`Rewrite::apply`](struct.Rewrite.html#method.apply)
    /// and returns number of new applications.
    fn apply_rewrite(
        &mut self,
        iteration: usize,
        egraph: &mut EGraph<L, N>,
        rewrite: &Rewrite<L, N>,
        matches: &Option<SearchMatches>,
    ) -> usize {
        rewrite.apply(egraph, matches).len()
    }


    /// A hook allowing you to customize the rewrite application behaviour for all rewrites at once.
    /// This is similar to [`apply_rewrite`](RewriteScheduler::apply_rewrite()), but
    /// requires that the schedualer handle limitation checks of the runner
    /// and that the scheduler can apply multiple rewrites at once.
    fn apply_rewrites(&mut self, egraph: &mut EGraph<L, N>, rules: &[&Rewrite<L, N>], i: usize, matches: Vec<Option<SearchMatches>>, applied: &mut IndexMap<Symbol, usize>) {
        rules.iter().zip(matches).for_each(|(rw, ms)| {
            if let Some(ms) = ms {
                let total_matches: usize = ms.total_substs();
                debug!("Applying {} {} times", rw.name, total_matches);

                let actually_matched = self.apply_rewrite(i, egraph, rw, &Some(ms));
                if actually_matched > 0 {
                    let symbol: Symbol = rw.name().into();
                    if let Some(count) = applied.get_mut(&symbol) {
                        *count += actually_matched;
                    } else {
                        applied.insert(rw.name().into(), actually_matched);
                    }
                    debug!("Applied {} {} times", rw.name, actually_matched);
                }
            }
        })
    }
}

/// A very simple [`RewriteScheduler`] that runs every rewrite every
/// time.
///
/// Using this is basically turning off rule scheduling.
/// It uses the default implementation for all [`RewriteScheduler`]
/// methods.
///
/// This is not the default scheduler; choose it with the
/// [`with_scheduler`](struct.Runner.html#method.with_scheduler)
/// method.
///
/// [`RewriteScheduler`]: trait.RewriteScheduler.html
#[derive(Default, Clone)]
pub struct SimpleScheduler;

impl<L, N> RewriteScheduler<L, N> for SimpleScheduler
where
    L: Language + 'static,
    N: Analysis<L> + 'static,
{}

/// A [`RewriteScheduler`] that implements exponentional rule backoff.
///
/// For each rewrite, there exists a configurable initial match limit.
/// If a rewrite search yield more than this limit, then we ban this
/// rule for number of iterations, double its limit, and double the time
/// it will be banned next time.
///
/// This seems effective at preventing explosive rules like
/// associativity from taking an unfair amount of resources.
///
/// [`BackoffScheduler`] is configurable in the builder-pattern style.
///
/// [`RewriteScheduler`]: trait.RewriteScheduler.html
/// [`BackoffScheduler`]: struct.BackoffScheduler.html
#[derive(Debug)]
pub struct BackoffScheduler {
    default_match_limit: usize,
    default_ban_length: usize,
    stats: IndexMap<Symbol, RuleStats>,
}

/// Statistics on rule usage
#[derive(Debug, Clone, Copy)]
pub struct RuleStats {
    times_applied: usize,
    banned_until: usize,
    times_banned: usize,
    match_limit: usize,
    ban_length: usize,
}

impl BackoffScheduler {
    /// Set the initial match limit after which a rule will be banned.
    /// Default: 1,000
    pub fn with_initial_match_limit(mut self, limit: usize) -> Self {
        self.default_match_limit = limit;
        self
    }

    /// Set the initial ban length.
    /// Default: 5 iterations
    pub fn with_ban_length(mut self, ban_length: usize) -> Self {
        self.default_ban_length = ban_length;
        self
    }

    fn rule_stats(&mut self, name: Symbol) -> &mut RuleStats {
        if self.stats.contains_key(&name) {
            &mut self.stats[&name]
        } else {
            self.stats.entry(name.to_owned()).or_insert(RuleStats {
                times_applied: 0,
                banned_until: 0,
                times_banned: 0,
                match_limit: self.default_match_limit,
                ban_length: self.default_ban_length,
            })
        }
    }

    /// Never ban a particular rule.
    pub fn do_not_ban(mut self, name: &str) -> Self {
        self.rule_stats(name.into()).match_limit = usize::MAX;
        self
    }

    /// Set the initial match limit for a rule.
    pub fn rule_match_limit(mut self, name: &str, limit: usize) -> Self {
        self.rule_stats(name.into()).match_limit = limit;
        self
    }

    /// Set the initial ban length for a rule.
    pub fn rule_ban_length(mut self, name: &str, length: usize) -> Self {
        self.rule_stats(name.into()).ban_length = length;
        self
    }

    /// Search with a rewrite given a limit and stats. Useful for parallel search as normal API is
    /// insufficient.
    pub fn search_with_stats<'a, L: Language, N: Analysis<L> + 'static>(
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrite: &'a Rewrite<L, N>,
        stats: &mut RuleStats,
    ) -> Option<SearchMatches> {
        if iteration < stats.banned_until {
            debug!(
                "Skipping {} ({}-{}), banned until {}...",
                rewrite.name, stats.times_applied, stats.times_banned, stats.banned_until,
            );
            return None;
        }

        let threshold = stats
            .match_limit
            .checked_shl(stats.times_banned as u32)
            .unwrap();
        let matches = rewrite.search_with_limit(egraph, threshold.saturating_add(1));
        let total_len: usize = matches.iter().map(|m| m.total_substs()).sum();
        if total_len > threshold {
            let ban_length = stats.ban_length << stats.times_banned;
            stats.times_banned += 1;
            stats.banned_until = iteration + ban_length;
            info!(
                "Banning {} ({}-{}) for {} iters: {} < {}",
                rewrite.name,
                stats.times_applied,
                stats.times_banned,
                ban_length,
                threshold,
                total_len,
            );
            None
        } else {
            stats.times_applied += 1;
            matches
        }
    }
}

impl Default for BackoffScheduler {
    fn default() -> Self {
        Self {
            stats: Default::default(),
            default_match_limit: 1_000,
            default_ban_length: 5,
        }
    }
}

impl<L, N> RewriteScheduler<L, N> for BackoffScheduler
where
    L: Language + 'static,
    N: Analysis<L> + 'static,
{
    fn can_stop(&mut self, iteration: usize) -> bool {
        let n_stats = self.stats.len();

        let mut banned: Vec<_> = self
            .stats
            .iter_mut()
            .filter(|(_, s)| s.banned_until > iteration)
            .collect();

        if banned.is_empty() {
            true
        } else {
            let min_ban = banned
                .iter()
                .map(|(_, s)| s.banned_until)
                .min()
                .expect("banned cannot be empty here");

            assert!(min_ban >= iteration);
            let delta = min_ban - iteration;

            let mut unbanned = vec![];
            for (name, s) in &mut banned {
                s.banned_until -= delta;
                if s.banned_until == iteration {
                    unbanned.push(name.as_str());
                }
            }

            assert!(!unbanned.is_empty());
            info!(
                "Banned {}/{}, fast-forwarded by {} to unban {}",
                banned.len(),
                n_stats,
                delta,
                unbanned.join(", "),
            );

            false
        }
    }

    fn search_rewrite<'a>(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrite: &'a Rewrite<L, N>,
    ) -> Option<SearchMatches> {
        let stats = self.rule_stats(rewrite.name().into());
        Self::search_with_stats(iteration, egraph, rewrite, stats)
    }
}

/// A wrapper for a ['RewriteScheduler'] that runs rewrite_search in parallel.
/// This requires that the underlying scheduler is thread safe, and that the language implements
/// [Send] and [Sync].
#[derive(Debug, Clone, Default, Copy)]
pub struct ParallelScheduler {}

#[cfg(feature = "parallel")]
impl<L, N> RewriteScheduler<L, N> for ParallelScheduler
where
    L: Language + Send + Sync,
    N: Analysis<L> + Sync + 'static,
    <N as language::Analysis<L>>::Data: Sync,
{
    fn search_rewrites<'a, 'b>(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrites: &[&'a Rewrite<L, N>],
        matches: &mut Vec<Option<SearchMatches>>,
        start_time: Instant,
        time_limit: Duration,
    ) -> RunnerResult<()> {
        debug!("Searching rewrites in parallel. Creating channel with size {}", rewrites.len());
        let channel = crossbeam::channel::bounded(rewrites.len());
        let res = rewrites.par_iter().enumerate().try_for_each(|(i, rw)| {
            debug!("Searching rw {}", rw.name);
            let rule_start = Instant::now();
            let results = rw.search(egraph);
            let elapsed = rule_start.elapsed();
            if let Some(results) = results {
                channel.0.send((i, results, elapsed)).expect("Channel should be big enough for all messages");
            }
            let elapsed = start_time.elapsed();
            if elapsed > time_limit {
                Err(StopReason::TimeLimit(elapsed.as_secs_f64()))
            } else {
                Ok(())
            }
        });
        drop(channel.0);
        debug!("Finished searching rewrites in parallel. Collecting results");
        matches.resize_with(rewrites.len(), || Default::default());
        let mut res_m: IndexMap<Symbol, Duration> = Default::default();
        for (i, ms, duration) in channel.1 {
            res_m.insert(rewrites[i].name().into(), duration);
            matches[i] = Some(ms);
        }
        Ok(())
    }
}
/// A wrapper for a ['RewriteScheduler'] that runs rewrite_search in parallel.
#[cfg(feature = "parallel")]
pub struct ParallelBackoffScheduler {
    scheduler: BackoffScheduler,
    thread_limit: usize,
}

impl Default for ParallelBackoffScheduler {
    fn default() -> Self {
        ParallelBackoffScheduler { scheduler: Default::default(), thread_limit: num_cpus::get() }
    }
}


impl ParallelBackoffScheduler {
    fn init_rule_stats(&mut self, names: impl Iterator<Item = Symbol>) {
        for name in names {
            self.scheduler.rule_stats(name);
        }
    }

    /// Set the amount of threads to use during search
    pub fn with_thread_limit(mut self, thread_limit: usize) -> Self {
        self.thread_limit = thread_limit;
        self
    }
}

impl<L, N> RewriteScheduler<L, N> for ParallelBackoffScheduler
where
    L: Language + Send + Sync,
    N: Analysis<L> + Sync + 'static,
    <N as language::Analysis<L>>::Data: Sync,
{
    fn can_stop(&mut self, iteration: usize) -> bool {
        <BackoffScheduler as RewriteScheduler<L, N>>::can_stop(&mut self.scheduler, iteration)
    }

    fn search_rewrite<'a>(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrite: &'a Rewrite<L, N>,
    ) -> Option<SearchMatches> {
        self.scheduler.search_rewrite(iteration, egraph, rewrite)
    }

    fn search_rewrites<'a, 'b>(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrites: &[&'a Rewrite<L, N>],
        matches: &mut Vec<Option<SearchMatches>>,
        start_time: Instant,
        time_limit: Duration,
    ) -> RunnerResult<()> {
        debug!("Searching rewrites in parallel. Creating channel with size {}", rewrites.len());
        self.init_rule_stats(rewrites.iter().map(|rw| rw.name().into()));
        let channel = crossbeam::channel::bounded(rewrites.len());
        let mut with_stats = rewrites.iter().map(|rw| {
            // Assuming no dup names
            let symbol: Symbol = rw.name().into();
            let stats = self.scheduler.stats.remove(&symbol).unwrap();
            (*rw, stats)
        }).collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.thread_limit).build().unwrap();
        let res = pool.install(|| {
            with_stats.par_iter_mut().enumerate().try_for_each(|(i, (rw, stats))| {
                debug!("Searching rw {}", rw.name());
                let rule_start = Instant::now();
                let results = BackoffScheduler::search_with_stats(iteration, egraph, rw, stats);
                let elapsed = rule_start.elapsed();
                if let Some(results) = results {
                    channel.0.send((i, results, elapsed)).expect("Channel should be big enough for all messages");
                }
                let elapsed = start_time.elapsed();
                if elapsed > time_limit {
                    Err(StopReason::TimeLimit(elapsed.as_secs_f64()))
                } else {
                    Ok(())
                }
            })
        });
        drop(channel.0);
        debug!("Finished searching rewrites in parallel. Collecting results");
        matches.resize_with(rewrites.len(), || Default::default());
        for (i, ms, time) in channel.1 {
            matches[i] = Some(ms);
        }
        // Return stats
        for (rw, stats) in with_stats {
            self.scheduler.stats.insert(rw.name().into(), stats);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, new)]
pub struct Split<L, N>
where
    L: Language,
    N: Analysis<L> + Clone,
    <N as language::Analysis<L>>::Data: Clone,
{
    // Rule name and searcher
    pub(crate) reason: String,
    pub(crate) egraphs: Vec<EGraph<L, N>>,
}

#[derive(Debug, Clone, new)]
pub struct Spliter<L>
where
    L: Language,
{
    // Rule name and searcher
    rule_name: Symbol,
    searcher: MultiPattern<L>,
    /// Ids present from goal egraph. Do not match on anything else
    goal_ids: Vec<Id>,
    // Appliers
    pub(crate) appliers: Vec<MultiPattern<L>>,
}

impl<L> Spliter<L>
where
    L: Language,
{
    /// Will duplicate e-graph as necessary and apply the split appliers
    pub fn case_split<N>(&self, egraph: &EGraph<L, N>) -> Vec<Split<L, N>>
    where
        N: Analysis<L> + Clone + 'static,
        <N as language::Analysis<L>>::Data: Clone,
    {
        let res = self.searcher.search(egraph);
        if res.is_none() {
            return vec![];
        }
        let matches = res.unwrap().matches.into_iter()
            .flat_map(|(id, subs)| {
                subs.into_iter()
                    .filter(|s| s.vec.iter().all(|(_, id)| self.goal_ids.contains(id)))
                    .map(move |sub| (id, sub))
            }).collect_vec();
        warn!("Doing case split creating {}*{} new e-graphs for {}", matches.len(), self.appliers.len(), self.rule_name);
        // Eclass should be meaningless here, let's verify
        assert_eq!(matches.iter().map(|(_, s)| s).collect::<HashSet<_>>().len(), matches.len());
        if matches.len() > 5 {
            warn!("Too many matches for rule {}. Skipping", self.rule_name);
            return vec![];
        }

        matches.into_iter().map(|(eclass, subst)| {
            let reason = format!("{}_{}", self.rule_name, subst.vec.iter().map(|(_, x)| x).join("_"));
            let egraphs = self.appliers.iter().map(|a| {
                let mut new_egg = (*egraph).clone();
                // Create fake SearchMatches
                let matches = SearchMatches {
                    matches: btreemap! {eclass => btreeset![subst.clone()]},
                    binds_done: 1,
                };
                a.apply_matches(&mut new_egg, &Some(matches));
                new_egg.rebuild();
                new_egg
            }).collect_vec();
            Split::new(reason, egraphs)
        }).collect_vec()
    }
}

/// A wrapper for a ['RewriteScheduler'] that runs rewrite_search in parallel
/// but also manages multiple egraphs at once such that conclusions are collected.
#[cfg(feature = "parallel")]
#[derive(Debug)]
pub struct ParallelBackoffSchedulerWithCases<L, N>
where
    L: Language,
    N: Analysis<L> + Clone,
    <N as language::Analysis<L>>::Data: Clone,
{
    scheduler: BackoffScheduler,
    thread_limit: usize,
    // I am gonna say fuck it right now and just hold additional cases here
    when: usize,
    splitters: Vec<Spliter<L>>,
    cases: BTreeMap<String, Split<L, N>>,
    original_ids: BTreeSet<Id>,
    case_matches: BTreeMap<String, Vec<Vec<Option<SearchMatches>>>>,
}

impl<L, N> Default for ParallelBackoffSchedulerWithCases<L, N>
where
    L: Language,
    N: Analysis<L> + Clone + 'static,
    <N as language::Analysis<L>>::Data: Clone,
{
    fn default() -> Self {
        ParallelBackoffSchedulerWithCases::new(usize::MAX, Default::default())
    }
}

impl<L, N> crate::ParallelBackoffSchedulerWithCases<L, N>
where
    L: Language,
    N: Analysis<L> + Clone + 'static,
    <N as language::Analysis<L>>::Data: Clone,
{
    pub fn new(when: usize, cases: Vec<Spliter<L>>) -> Self {
        crate::ParallelBackoffSchedulerWithCases {
            scheduler: Default::default(),
            thread_limit: num_cpus::get(),
            splitters: cases,
            cases: Default::default(),
            original_ids: Default::default(),
            case_matches: Default::default(),
            when,
        }
    }

    fn do_case_split(&mut self, egraph: &EGraph<L, N>) {
        // Take all cases to do, and duplicate the egraph for each match and applier
        // push the results to cases and save original ids
        self.original_ids = egraph.classes().map(|c| c.id).collect();
        for s in &self.splitters {
            let splits = s.case_split(egraph);
            for split in splits {
                self.cases.insert(split.reason.clone(), split);
            }
        }
    }

    fn intersect_conclusions(&self) -> Vec<(String, Vec<Id>)> {
        let mut concs = vec![];
        for (reason, split) in &self.cases {
            // Each time I only need to look at the original id
            let mut groups: BTreeMap<Vec<Id>, Vec<Id>> = Default::default();
            // fake start
            groups.insert(vec![0.into()], self.original_ids.iter().copied().collect_vec());

            for c in &split.egraphs {
                let mut new_groups: BTreeMap<Vec<Id>, Vec<Id>> = Default::default();
                for (k, group) in groups.into_iter() {
                    if group.len() <= 1 {
                        continue;
                    }
                    for id in group {
                        let node = c.find(id);
                        let mut key = k.clone();
                        key.push(node);
                        new_groups.entry(key).or_insert_with(Vec::new).push(id);
                    }
                }
                groups = new_groups
            }
            for (_, v) in groups {
                if v.len() >= 1 {
                    concs.push((reason.clone(), v));
                }
            }
        }
        concs
    }

    fn init_rule_stats(&mut self, names: &[Symbol]) {
        for name in names {
            self.scheduler.rule_stats(*name);
        }
    }

    /// Set the amount of threads to use during search
    pub fn with_thread_limit(mut self, thread_limit: usize) -> Self {
        self.thread_limit = thread_limit;
        self
    }
}

impl<L, N> RewriteScheduler<L, N> for ParallelBackoffSchedulerWithCases<L, N>
where
    L: Language + Send + Sync,
    N: Analysis<L> + Sync + Clone + 'static,
    <N as language::Analysis<L>>::Data: Clone,
    <N as language::Analysis<L>>::Data: Sync,
{
    fn can_stop(&mut self, iteration: usize) -> bool {
        <BackoffScheduler as RewriteScheduler<L, N>>::can_stop(&mut self.scheduler, iteration)
    }

    fn search_rewrite<'a>(
        &mut self,
        iteration: usize,
        egraph: &EGraph<L, N>,
        rewrite: &'a Rewrite<L, N>,
    ) -> Option<SearchMatches> {
        self.scheduler.search_rewrite(iteration, egraph, rewrite)
    }

    fn search_rewrites<'a, 'b>(&mut self,
                               iteration: usize,
                               egraph: &EGraph<L, N>,
                               rewrites: &[&'a Rewrite<L, N>],
                               matches: &mut Vec<Option<SearchMatches>>,
                               start_time: Instant,
                               time_limit: Duration)
        -> RunnerResult<()> {
        assert!(self.case_matches.is_empty());
        // Initialize a matches vector for each case and rewrite
        for (_, cms) in &mut self.case_matches {
            cms.resize_with(self.cases.len(), || vec![Default::default(); rewrites.len()]);
        }
        debug!("Searching rewrites in parallel. Creating channel with size {}", rewrites.len());
        self.init_rule_stats(&rewrites.iter().map(|rw| rw.name().into()).collect::<Vec<_>>());
        let default_reason = "default".to_string();
        let channel = crossbeam::channel::bounded(rewrites.len());
        let mut egraphs = vec![];
        if self.when == iteration {
            self.do_case_split(egraph);
        }
        if self.when <= iteration {
            for (r, es) in &self.cases {
                for (i, e) in es.egraphs.iter().enumerate() {
                    egraphs.push((r, e, Some(i)))
                }
            }
        }
        egraphs.push((&default_reason, egraph, None));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.thread_limit).build().unwrap();
        pool.install(|| {
            iproduct!(0..rewrites.len(), 0..egraphs.len()).collect_vec().into_par_iter()
                .try_for_each(|(i, e_idx)| {
                    debug!("Searching rw {}", rewrites[i].name);
                    let (r, egraph, idx) = egraphs[e_idx];
                    let symbol: Symbol = rewrites[i].name().into();
                    let mut stats = *self.scheduler.stats.get(&symbol).unwrap();
                    let results = BackoffScheduler::search_with_stats(iteration, egraph, rewrites[i], &mut stats);
                    if let Some(results) = results {
                        channel.0.send((i, idx, r, results, stats)).expect("Channel should be big enough for all messages");
                    }
                    let elapsed = start_time.elapsed();
                    if elapsed > time_limit {
                        Err(StopReason::TimeLimit(elapsed.as_secs_f64()))
                    } else {
                        Ok(())
                    }
                })
        });
        drop(channel.0);
        debug!("Finished searching rewrites in parallel. Collecting results");
        matches.resize_with(rewrites.len(), || Default::default());
        for (i, idx, r, ms, stats) in channel.1 {
            if let Some(idx) = idx {
                self.case_matches.get_mut(r).unwrap()[idx][i] = Some(ms);
            } else {
                matches[i] = Some(ms);
                self.scheduler.stats.insert(rewrites[i].name().into(), stats);
            }
        }
        Ok(())
    }

    fn apply_rewrites(&mut self, egraph: &mut EGraph<L, N>, rules: &[&Rewrite<L, N>], iteration: usize, matches: Vec<Option<SearchMatches>>, mut applied: &mut IndexMap<Symbol, usize>) {
        let mut s = SimpleScheduler {};
        s.apply_rewrites(egraph, rules, iteration, matches, applied);
        if self.when <= iteration {
            let mut cases = std::mem::take(&mut self.cases);
            let mut case_matches = std::mem::take(&mut self.case_matches);
            for (r, cases) in &mut cases {
                for case in cases.egraphs.iter_mut() {
                    let mut applied = IndexMap::default();
                    let matches = case_matches.get_mut(r).unwrap().remove(0);
                    s.apply_rewrites(case, rules, iteration, matches, &mut applied);
                    case.rebuild();
                }
            }
            self.cases = cases;
            self.case_matches = case_matches;
            // now collect all conclusions from cases and put in orig
            let conclusions = self.intersect_conclusions();
            for (reason, mut ids) in conclusions {
                let symbol: Symbol = reason.into();
                let node = ids.pop().unwrap();
                for id in ids {
                    // egraph.union_trusted(id, node, symbol);
                    egraph.union(id, node);
                }
            }
        }
    }
}


/// Custom data to inject into the [`Iteration`]s recorded by a [`Runner`]
///
/// This trait allows you to add custom data to the [`Iteration`]s
/// recorded as a [`Runner`] applies rules.
///
/// See the [`Runner`] docs for an example.
///
/// [`Runner`] is generic over the [`IterationData`] that it will be in the
/// [`Iteration`]s, but by default it uses `()`.
///
/// [`Runner`]: struct.Runner.html
/// [`Iteration`]: struct.Iteration.html
/// [`IterationData`]: trait.IterationData.html
pub trait IterationData<L, N>: Sized
where
    L: Language,
    N: Analysis<L>,
{
    /// Given the current [`Runner`](struct.Runner.html), make the
    /// data to be put in this [`Iteration`](struct.Iteration.html).
    fn make(runner: &Runner<L, N, Self>) -> Self;
}

impl<L, N> IterationData<L, N> for ()
where
    L: Language,
    N: Analysis<L>,
{
    fn make(_: &Runner<L, N, Self>) -> Self {}
}

// Some global static statistics:
// - number of search matches
pub static mut SEARCH_MATCHES: usize = 0;
// - time spent searching
pub static mut SEARCH_TIME: u128 = 0;
// - number of applications
pub static mut APPLICATIONS: usize = 0;
// - time spent applying
pub static mut APPLY_TIME: u128 = 0;
// - time spent rebuilding
pub static mut REBUILD_TIME: u128 = 0;
