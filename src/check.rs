//! This is the actual type checker based on the paper.

pub mod builtins;
pub mod error;

mod assumption;
mod id;
mod kinds;
mod pattern;
mod predicate;
mod qualified;
mod scheme;
mod substitution;
mod syntax;
mod trait_env;
mod traits;
mod type_inference;
mod types;

use error::Result;

pub use syntax::*;

pub use assumption::Assumption;
pub use id::Id;
pub use kinds::{HasKind, Kind};
pub use pattern::Pattern;
pub use predicate::Predicate;
pub use qualified::Qualified;
pub use scheme::Scheme;
pub use substitution::Substitution;
pub use trait_env::TraitEnvironment;
pub use traits::{Instance, Trait};
pub use type_inference::{Infer, TypeInference};
pub use types::{Instantiate, Type, TypeConstructor, TypeVariable, Types};

use crate::util::{append, intersection, minus, partition, union, zip_with, zip_with_try};

// #[cfg(test)]
// mod tests {

//     use super::*;

//     #[test]
//     fn test_empty() {
//         // the empty program
//         let program = vec![];
//         let mut ti = TypeInference::default();
//         let ce = TraitEnvironment::default();
//         let assumptions = ti.program(&ce, &[], program).expect("is well typed");
//         assert!(assumptions.is_empty());
//     }

//     #[test]
//     fn test_simple() {
//         // the program `x = 'c'`
//         let program: Program = vec![(
//             // binding groups
//             vec![], // explicit type signatures
//             vec![vec![(
//                 // implicit types
//                 "x".into(), // id of thing being bound
//                 vec![(
//                     vec![],                        // patterns to the left of the `=`
//                     Expr::Lit(Literal::Char('a')), // expression on the right of the `=`
//                 )],
//             )]],
//         )];

//         let mut ti = TypeInference::default();
//         let ce = TraitEnvironment::default();
//         let assumptions = ti.program(&ce, &[], program).expect("is well typed");
//         assert_eq!(assumptions.len(), 1);
//         // i.e. x :: [Char]
//         assert_eq!(
//             format!("{:?}", assumptions),
//             r#"[Assump("x", ForAll([], Then([], Con(Tycon("Char", Star)))))]"#
//         )
//     }

//     #[test]
//     fn test_hello() {
//         // the program `x = "Hello world!"`
//         let program: Program = vec![(
//             // binding groups
//             vec![], // explicit type signatures
//             vec![vec![(
//                 // implicit types
//                 "x".into(), // id of thing being bound
//                 vec![(
//                     vec![],                                          // patterns to the left of the `=`
//                     Expr::Lit(Literal::Str("Hello, world!".into())), // expression on the right of the `=`
//                 )],
//             )]],
//         )];

//         let mut ti = TypeInference::default();
//         let ce = TraitEnvironment::default();
//         let assumptions = ti.program(&ce, &[], program).expect("is well typed");
//         assert_eq!(assumptions.len(), 1);
//         // i.e. x :: [Char]
//         assert_eq!(
//             format!("{:?}", assumptions),
//             r#"[Assump("x", ForAll([], Then([], Ap(Con(Tycon("[]", Fun(Star, Star))), Con(Tycon("Char", Star))))))]"#
//         )
//     }

//     #[test]
//     fn test_defaults() {
//         // the program `x = 1`
//         let program: Program = Program {
//             binding_groups: vec![(
//                 // binding groups
//                 vec![], // explicit type signatures
//                 vec![vec![(
//                     // implicit types
//                     "x".into(), // id of thing being bound
//                     vec![(
//                         vec![],                     // patterns to the left of the `=`
//                         Expr::Lit(Literal::Int(1)), // expression on the right of the `=`
//                     )],
//                 )]],
//             )],
//         };

//         let mut ti = TypeInference::default();

//         let assumptions = ti.program(&ce, &[], program).expect("is well typed");
//         assert_eq!(assumptions.len(), 1);
//         // i.e. x :: [Char]
//         assert_eq!(
//             format!("{:?}", assumptions),
//             // How do we know what Gen(0) is limited to?
//             r#"[Assump(\"x\", ForAll([Star], Then(["Integral"], Gen(0))))]"#
//         )
//     }
// }
