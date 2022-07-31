//! Patterns

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

impl Infer for Pattern {
    type Output = Type;

    fn infer(&self, _context: &mut TypeInference) -> Result<Self::Output> {
        todo!()
    }
}
