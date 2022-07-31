//! Expression syntax

use super::*;

pub type Ambiguity = (TypeVariable, Vec<Predicate>);

pub type ExplicitBinding = (Id, Scheme, Vec<Equation>);

pub type ImplicitBinding = (Id, Vec<Equation>);

pub fn restricted(bs: &[ImplicitBinding]) -> bool {
    bs.iter()
        .any(|(_i, alts)| alts.iter().any(|alt| alt.0.is_empty()))
}

// ### 11.4 Alternatives

// An alternative is what this is calling an 'equation'
//
// i.e. it's each line that's pattern matched in code like this:
//
//     null []    = true
//     null (_:_) = false
//
// The Pattern is each parameter, and the Expr is the right hand side.
pub type Equation = (Vec<Pattern>, Expr);

// TODO: Make struct
pub type BindingGroup = (Vec<ExplicitBinding>, Vec<Vec<ImplicitBinding>>);

#[derive(Debug)]
pub struct Program {
    pub(crate) binding_groups: Vec<BindingGroup>,
}

// TODO: Rename variants.
// TODO: Why does `Expression::Const` contain `Assumption`?

#[derive(Debug, Clone)]
pub enum Expr {
    Var(Id),
    Lit(Literal),
    Const(Assumption),
    Ap(Box<Expr>, Box<Expr>),
    Let(BindingGroup, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Char(char),
    Rat(f64), // I know, but close enough.
    Str(String),
}
