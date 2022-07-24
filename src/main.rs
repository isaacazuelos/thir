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

fn main() {
    println!("Hello, world!");
}
