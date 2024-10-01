// use easter_egg::{rewrite as rw, *};
// use ordered_float::NotNan;
// use serde::{Deserialize, Serialize};
//
// pub type EGraph = easter_egg::EGraph<Math, ConstantFold>;
// pub type Rewrite = easter_egg::Rewrite<Math, ConstantFold>;
//
// pub type Constant = NotNan<f64>;
//
// define_language! {
//     pub enum Math {
//         "d" = Diff([Id; 2]),
//         "i" = Integral([Id; 2]),
//
//         "+" = Add([Id; 2]),
//         "-" = Sub([Id; 2]),
//         "*" = Mul([Id; 2]),
//         "/" = Div([Id; 2]),
//         "pow" = Pow([Id; 2]),
//         "ln" = Ln(Id),
//         "sqrt" = Sqrt(Id),
//
//         "sin" = Sin(Id),
//         "cos" = Cos(Id),
//
//         Constant(Constant),
//         Symbol(Symbol),
//     }
// }
//
// // You could use easter_egg::AstSize, but this is useful for debugging, since
// // it will really try to get rid of the Diff operator
// pub struct MathCostFn;
// impl easter_egg::CostFunction<Math> for MathCostFn {
//     type Cost = usize;
//     fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
//     where
//         C: FnMut(Id) -> Self::Cost,
//     {
//         let op_cost = match enode {
//             Math::Diff(..) => 100,
//             Math::Integral(..) => 100,
//             _ => 1,
//         };
//         enode.fold(op_cost, |sum, i| sum + costs(i))
//     }
// }
//
// #[derive(Default, Serialize, Deserialize, Clone)]
// pub struct ConstantFold;
// impl Analysis<Math> for ConstantFold {
//     type Data = Option<Constant>;
//
//     fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
//         if let (Some(c1), Some(c2)) = (to.as_ref(), from.as_ref()) {
//             assert_eq!(c1, c2);
//         }
//         merge_if_different(to, to.or(from))
//     }
//
//     fn make(egraph: &EGraph, enode: &Math) -> Self::Data {
//         let x = |i: &Id| egraph[*i].data;
//         Some(match enode {
//             Math::Constant(c) => *c,
//             Math::Add([a, b]) => x(a)? + x(b)?,
//             Math::Sub([a, b]) => x(a)? - x(b)?,
//             Math::Mul([a, b]) => x(a)? * x(b)?,
//             Math::Div([a, b]) if x(b) != Some(0.0.into()) => x(a)? / x(b)?,
//             _ => return None,
//         })
//     }
//
//     fn modify(egraph: &mut EGraph, id: Id) {
//         let class = &mut egraph[id];
//         if let Some(c) = class.data {
//             let added = egraph.add(Math::Constant(c));
//             let (id, _did_something) = egraph.union(id, added);
//             // to not prune, comment this out
//             egraph[id].nodes.retain(|n| n.is_leaf());
//
//             assert!(
//                 !egraph[id].nodes.is_empty(),
//                 "empty eclass! {:#?}",
//                 egraph[id]
//             );
//             #[cfg(debug_assertions)]
//             egraph[id].assert_unique_leaves();
//         }
//     }
// }
//
// struct IsConstOrDistinctCondition {
//     v: Var,
//     w: Var,
// }
//
// impl Condition<Math, ConstantFold> for IsConstOrDistinctCondition {
//     fn check(&self, egraph: &mut easter_egg::EGraph<Math, ConstantFold>, _eclass: Id, subst: &Subst) -> bool {
//         egraph.find(subst[self.v]) != egraph.find(subst[self.w])
//             && egraph[subst[self.v]]
//             .nodes
//             .iter()
//             .any(|n| matches!(n, Math::Constant(..) | Math::Symbol(..)))
//     }
//
//     fn check_colored(&self, egraph: &mut easter_egg::EGraph<Math, ConstantFold>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
//         self.check(egraph, eclass, subst).then(|| vec![])
//     }
//
//     fn describe(&self) -> String {
//         "is_const_or_distinct".to_string()
//     }
// }
//
// fn is_const_or_distinct_var(v: &str, w: &str) -> impl Condition<Math, ConstantFold> {
//     let v = v.parse().unwrap();
//     let w = w.parse().unwrap();
//     IsConstOrDistinctCondition { v, w }
// }
//
// struct IsConstCondition {
//     v: Var,
// }
//
// impl Condition<Math, ConstantFold> for IsConstCondition {
//     fn check(&self, egraph: &mut easter_egg::EGraph<Math, ConstantFold>, _eclass: Id, subst: &Subst) -> bool {
//         egraph[subst[self.v]]
//             .nodes
//             .iter()
//             .any(|n| matches!(n, Math::Constant(..)))
//     }
//
//     fn check_colored(&self, egraph: &mut easter_egg::EGraph<Math, ConstantFold>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
//         self.check(egraph, eclass, subst).then(|| vec![])
//     }
//
//     fn describe(&self) -> String {
//         "is_const".to_string()
//     }
// }
//
// fn is_const(var: &str) -> impl Condition<Math, ConstantFold> {
//     let var = var.parse().unwrap();
//     IsConstCondition { v: var }
// }
//
// struct IsSymCondition {
//     v: Var,
// }
//
// impl Condition<Math, ConstantFold> for IsSymCondition {
//     fn check(&self, egraph: &mut easter_egg::EGraph<Math, ConstantFold>, _eclass: Id, subst: &Subst) -> bool {
//         egraph[subst[self.v]]
//             .nodes
//             .iter()
//             .any(|n| matches!(n, Math::Symbol(..)))
//     }
//
//     fn check_colored(&self, egraph: &mut easter_egg::EGraph<Math, ConstantFold>, eclass: Id, subst: &Subst) -> Option<Vec<ColorId>> {
//         self.check(egraph, eclass, subst).then(|| vec![])
//     }
//
//     fn describe(&self) -> String {
//         "is_sym".to_string()
//     }
// }
//
// fn is_sym(var: &str) -> impl Condition<Math, ConstantFold> {
//     let var = var.parse().unwrap();
//     IsSymCondition { v: var }
// }
//
// struct IsNotZeroCondition {
//     v: Var,
// }
//
// #[rustfmt::skip]
// pub fn rules() -> Vec<Rewrite> { vec![
//     rw!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
//     rw!("comm-mul";  "(* ?a ?b)"        => "(* ?b ?a)"),
//     rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
//     rw!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
//
//     rw!("sub-canon"; "(- ?a ?b)" => "(+ ?a (* -1 ?b))"),
//     multi_rewrite!("div-canon"; "?root = (/ ?a ?b), ?b != 0" => "?root = (* ?a (pow ?b -1))"),
//     // rw!("canon-sub"; "(+ ?a (* -1 ?b))"   => "(- ?a ?b)"),
//     // rw!("canon-div"; "(* ?a (pow ?b -1))" => "(/ ?a ?b)" if is_not_zero("?b")),
//
//     rw!("zero-add"; "(+ ?a 0)" => "?a"),
//     rw!("zero-mul"; "(* ?a 0)" => "0"),
//     rw!("one-mul";  "(* ?a 1)" => "?a"),
//
//     rw!("add-zero"; "?a" => "(+ ?a 0)"),
//     rw!("mul-one";  "?a" => "(* ?a 1)"),
//
//     rw!("cancel-sub"; "(- ?a ?a)" => "0"),
//     multi_rewrite!("cancel-div"; "?root = (/ ?a ?a), ?a != 0" => "1"),
//
//     rw!("distribute"; "(* ?a (+ ?b ?c))"        => "(+ (* ?a ?b) (* ?a ?c))"),
//     rw!("factor"    ; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
//
//     rw!("pow-mul"; "(* (pow ?a ?b) (pow ?a ?c))" => "(pow ?a (+ ?b ?c))"),
//     multi_rewrite!("pow0"; "?root = (pow ?x 0), ?x != 0" => "?root = 1"),
//     rw!("pow1"; "(pow ?x 1)" => "?x"),
//     rw!("pow2"; "(pow ?x 2)" => "(* ?x ?x)"),
//     multi_rewrite!("pow-recip"; "?root = (pow ?x -1), ?x != 0" => "?root = (/ 1 ?x)"),
//     rw!("recip-mul-div"; "?root = (* ?x (/ 1 ?x)), ?x != 0" => "?root = 1"),
//
//     rw!("d-variable"; "(d ?x ?x)" => "1" if is_sym("?x")),
//     rw!("d-constant"; "(d ?x ?c)" => "0" if {conditions::MutAndCondition::new(vec![Box::new(is_sym("?x")), Box::new(is_const_or_distinct_var("?c", "?x"))])}),
//
//     rw!("d-add"; "(d ?x (+ ?a ?b))" => "(+ (d ?x ?a) (d ?x ?b))"),
//     rw!("d-mul"; "(d ?x (* ?a ?b))" => "(+ (* ?a (d ?x ?b)) (* ?b (d ?x ?a)))"),
//
//     rw!("d-sin"; "(d ?x (sin ?x))" => "(cos ?x)"),
//     rw!("d-cos"; "(d ?x (cos ?x))" => "(* -1 (sin ?x))"),
//
//     rw!("d-ln"; "(d ?x (ln ?x))" => "(/ 1 ?x)" if is_not_zero("?x")),
//
//     multi_rewrite!("d-power";
//         "?root = (d ?x (pow ?f ?g)), ?f != 0, ?g != 0" =>
//         "?root = (* (pow ?f ?g)
//             (+ (* (d ?x ?f)
//                   (/ ?g ?f))
//                (* (d ?x ?g)
//                   (ln ?f))))"
//     ),
//
//     rw!("i-one"; "(i 1 ?x)" => "?x"),
//     rw!("i-power-const"; "(i (pow ?x ?c) ?x)" =>
//         "(/ (pow ?x (+ ?c 1)) (+ ?c 1))" if is_const("?c")),
//     rw!("i-cos"; "(i (cos ?x) ?x)" => "(sin ?x)"),
//     rw!("i-sin"; "(i (sin ?x) ?x)" => "(* -1 (cos ?x))"),
//     rw!("i-sum"; "(i (+ ?f ?g) ?x)" => "(+ (i ?f ?x) (i ?g ?x))"),
//     rw!("i-dif"; "(i (- ?f ?g) ?x)" => "(- (i ?f ?x) (i ?g ?x))"),
//     rw!("i-parts"; "(i (* ?a ?b) ?x)" =>
//         "(- (* ?a (i ?b ?x)) (i (* (d ?x ?a) (i ?b ?x)) ?x))"),
// ]}
//
// easter_egg::test_fn! {
//     math_associate_adds, [
//         rw!("comm-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
//         rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
//     ],
//     runner = Runner::default()
//         .with_iter_limit(7)
//         .with_scheduler(SimpleScheduler),
//     "(+ 1 (+ 2 (+ 3 (+ 4 (+ 5 (+ 6 7))))))"
//     =>
//     "(+ 7 (+ 6 (+ 5 (+ 4 (+ 3 (+ 2 1))))))"
//     @check |r: Runner<Math, ()>| assert_eq!(r.egraph.number_of_classes(), 127)
// }
//
// easter_egg::test_fn! {
//     #[should_panic(expected = "Could not prove goal 0")]
//     math_fail, rules(),
//     "(+ x y)" => "(/ x y)"
// }
//
// easter_egg::test_fn! {math_simplify_add, rules(), "(+ x (+ x (+ x x)))" => "(* 4 x)" }
// easter_egg::test_fn! {math_powers, rules(), "(* (pow 2 x) (pow 2 y))" => "(pow 2 (+ x y))"}
//
// easter_egg::test_fn! {
//     math_simplify_const, rules(),
//     "(+ 1 (- a (* (- 2 1) a)))" => "1"
// }
//
// easter_egg::test_fn! {
//     math_simplify_root, rules(),
//     runner = Runner::default().with_node_limit(75_000),
//     r#"
//     (/ 1
//        (- (/ (+ 1 (sqrt five))
//              2)
//           (/ (- 1 (sqrt five))
//              2)))"#
//     =>
//     "(/ 1 (sqrt five))"
// }
//
// easter_egg::test_fn! {
//     math_simplify_factor, rules(),
//     "(* (+ x 3) (+ x 1))"
//     =>
//     "(+ (+ (* x x) (* 4 x)) 3)"
// }
//
// easter_egg::test_fn! {math_diff_same,      rules(), "(d x x)" => "1"}
// easter_egg::test_fn! {math_diff_different, rules(), "(d x y)" => "0"}
// easter_egg::test_fn! {math_diff_simple1,   rules(), "(d x (+ 1 (* 2 x)))" => "2"}
// easter_egg::test_fn! {math_diff_simple2,   rules(), "(d x (+ 1 (* y x)))" => "y"}
// easter_egg::test_fn! {math_diff_ln,        rules(), "(d x (ln x))" => "(/ 1 x)"}
//
// easter_egg::test_fn! {
//     diff_power_simple, rules(),
//     "(d x (pow x 3))" => "(* 3 (pow x 2))"
// }
//
// easter_egg::test_fn! {
//     diff_power_harder, rules(),
//     runner = Runner::default()
//         .with_time_limit(std::time::Duration::from_secs(10))
//         .with_iter_limit(60)
//         .with_node_limit(100_000)
//         // HACK this needs to "see" the end expression
//         .with_expr(&"(* x (- (* 3 x) 14))".parse().unwrap()),
//     "(d x (- (pow x 3) (* 7 (pow x 2))))"
//     =>
//     "(* x (- (* 3 x) 14))"
// }
//
// easter_egg::test_fn! {
//     integ_one, rules(), "(i 1 x)" => "x"
// }
//
// easter_egg::test_fn! {
//     integ_sin, rules(), "(i (cos x) x)" => "(sin x)"
// }
//
// easter_egg::test_fn! {
//     integ_x, rules(), "(i (pow x 1) x)" => "(/ (pow x 2) 2)"
// }
//
// easter_egg::test_fn! {
//     integ_part1, rules(), "(i (* x (cos x)) x)" => "(+ (* x (sin x)) (cos x))"
// }
//
// easter_egg::test_fn! {
//     integ_part2, rules(),
//     "(i (* (cos x) x) x)" => "(+ (* x (sin x)) (cos x))"
// }
//
// easter_egg::test_fn! {
//     integ_part3, rules(), "(i (ln x) x)" => "(- (* x (ln x)) x)"
// }
