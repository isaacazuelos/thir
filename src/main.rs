//! An attempt at writing a Haskell 98 like type checker based &on the paper
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

use check::{Expr, Literal, Program, TypeInference};

use crate::check::{BindingGroup, Equation, ImplicitBinding};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // the program `x = "Hello world!"`
    let program = Program::new(vec![BindingGroup::new(
        vec![],
        vec![vec![ImplicitBinding::new(
            // implicit types
            "x".into(), // id of thing being bound
            vec![Equation::new(
                vec![],
                Expr::Literal(Literal::String("Hello, world!".into())),
            )],
        )]],
    )]);

    let mut type_context = TypeInference::default();

    type_context.infer_types(&program)?;

    println!("{program}");
    println!("Produces these types");
    println!("{:#?}", type_context);

    Ok(())
}
