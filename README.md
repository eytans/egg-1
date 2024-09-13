# <img src="doc/egg.svg" alt="egg logo" height="40" align="left"> Easter Egg: Colored E-Graphs

[![Build Status](https://github.com/eytans/egg/workflows/Build%20and%20Test/badge.svg?branch=features/color_splits)](https://github.com/eytans/egg/actions)
[![Crates.io](https://img.shields.io/crates/v/egg.svg)](https://crates.io/crates/egg)
[![Released Docs.rs](https://docs.rs/egg/badge.svg)](https://docs.rs/egg/)
[![Master docs](https://img.shields.io/badge/docs-master-blue)](https://egraphs-good.github.io/egg/egg/)

Easter Egg is an extension of the egg library that implements colored e-graphs, enabling efficient representation of multiple congruence relations in a single e-graph structure.

## Features

- Support for multiple congruence relations (colors) in a single e-graph
- Memory-efficient representation of coarsened equality relations
- Optimized algorithms for colored e-graph operations
- Compatible with existing egg functionality
- Has a parallel backoff schedular to run search of rules in parallel.
- Machine has early stop as it is no longer recursive, but stack based machine (I think this could be parallel if we wished)

## Using Easter Egg

Add `easter_egg` to your `Cargo.toml` like this:
```toml
[dependencies]
easter_egg = { git = "https://github.com/eytans/egg.git" }
```

## Developing
Easter Egg is written in Rust.
Typically, you install Rust using rustup.
Run cargo doc --open to build and open the documentation in a browser.
Before committing/pushing, make sure to run cargo test, which runs all the tests.
You should also run cargo fmt to format your code and a linter.

## Tests
You will need graphviz to run the tests.
Running cargo test will run the tests.
Some tests may time out; try cargo test --release if that happens.
There are several interesting tests in the tests directory:

egraph.rs implements basic functionality tests for colored e-graphs.

## Key Concepts

Colored E-Graphs: An extension of e-graphs that efficiently represents multiple congruence relations.
Colored E-Classes: E-classes associated with specific colors, representing coarsened equality relations.
Colored Operations: Modified e-graph operations (merge, rebuild, e-matching) that work with colored e-graphs.

## Performance
Colored e-graphs offer significant memory savings compared to maintaining separate e-graphs for different assumptions, with a slight trade-off in performance for some operations.

## Contributing
Contributions to Easter Egg are welcome! Please feel free to submit issues, pull requests, or reach out with any questions or suggestions.


## Differences from egg

Easter Egg extends the functionality of egg while maintaining a similar API. However, there are some key differences to be aware of:

### API and Stability

- Easter Egg provides a similar API to egg, but it is currently less stable as it's a newer extension.
- A new "colored" API has been added, allowing users to create colors and perform searches with colors.

### Multi-Pattern Handling

Easter Egg handles multi-patterns slightly differently from egg:

- The "=" operator is not allowed inside patterns as an operation.
- Instead, we introduce "|=" and "!=" as replacements for conditions.
- These new operators represent new machine operations used during search.

### Example Usage

```rust
use egg::{EGraph, Rewrite, Runner, Symbol};

// Create a new colored e-graph
let mut egraph = EGraph::new(());

// Create a new color
let blue_color = egraph.add_color();

// Add a term to the e-graph
let x = egraph.add(Symbol::new("x"));
let y = egraph.add(Symbol::new("y"));

// Perform a colored merge
egraph.colored_union(blue_color, x, y);
let runner = ....
```