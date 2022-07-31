# Typing Haskell in Rust

An attempt at writing a Haskell 98 like type checker based on the paper
[_Typing Haskell in Haskell_][thih], in Rust.

[thih]:<https://web.cecs.pdx.edu/~mpj/thih/thih.pdf>

The goal here is just to think through the algorithm. I'm _not_ worried about it
being a faithful representation of the evaluation given strictness, or an
efficient implementation in Rust's semantics. The core of the paper is in the
[`check`] module.

The paper's blindly implemented. But there are a few parts I'm not convinced I
fully understand.

There are bugs as well. And writing Haskell in Rust is not the easiest to debug.
Right now I'm working through cleaning things up a bit so I can start to write
tests, and understand where they're actually failing.

No promises this ever gets anywhere further, the initial goal is met -- I've
been forced to read the paper pretty closely.
