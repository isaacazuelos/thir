//! Patterns

use std::fmt;

use super::{type_inference::Infer, *};

#[derive(Debug, Clone)]
pub enum Pattern {
    Var(Id),                       // `a`
    Wildcard,                      // `_`
    As(Id, Box<Pattern>),          // `id@pat`
    Lit(Literal),                  // `1`
    Npk(Id, usize),                // `n + k` patterns, which are a sin
    Con(Assumption, Vec<Pattern>), // `Constructor p1 p2 ... pn`. What Assumption is this?
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Pattern::Var(v) => write!(f, "{v}"),
            Pattern::Wildcard => write!(f, "_"),
            Pattern::As(n, p) => write!(f, "{n}@({p})"),
            Pattern::Lit(l) => write!(f, "{l}"),
            Pattern::Npk(n, k) => write!(f, "({n} + {k})"),
            Pattern::Con(a, ps) => {
                write!(f, "( {}", a.id())?;

                for p in ps {
                    write!(f, "{p}")?;
                }

                write!(f, ")")
            }
        }
    }
}

impl Infer for Pattern {
    type Output = Type;

    fn infer(&self, _context: &mut TypeInference) -> Result<Self::Output> {
        todo!()
    }
}
