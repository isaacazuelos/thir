//! An attempt at writing a Haskell 98 like type checker based on the paper
//! [_Typing Haskell in Haskell_][thih].
//!
//! The goal here is just to think through the algorithm. I'm _not_ worried
//! about it being a faithful representation of the evaluation given strictness,
//! or an efficient implementation. The core of the paper is in the
//! [`check`] module.
//!
//! [thih]:https://web.cecs.pdx.edu/~mpj/thih/thih.pdf

#![allow(dead_code)]

mod check;
mod util;

use check::{Expr, Infer, Literal, Program, TypeInference};

fn main() {
    // the program `x = "Hello world!"`
    let program = Program {
        binding_groups: vec![(
            // binding groups
            vec![], // explicit type signatures
            vec![vec![(
                // implicit types
                "x".into(), // id of thing being bound
                vec![(
                    vec![],                                          // patterns to the left of the `=`
                    Expr::Lit(Literal::Str("Hello, world!".into())), // expression on the right of the `=`
                )],
            )]],
        )],
    };

    let mut type_context = TypeInference::default();

    // program.infer(&mut type_context)?;

    println!(
        "{:#?}\nare the type constraints of the program:\n{:#?}",
        type_context.assumptions(),
        program
    );
}
