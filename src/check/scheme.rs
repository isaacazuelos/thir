//! Type Scheme

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Scheme {
    // These are the generic types. In Rust, the stuff in <> when introducing
    // type variables.
    ForAll(Vec<Kind>, Qualified<Type>),
}

impl Types for Scheme {
    fn apply(&self, s: &[Substitution]) -> Self {
        let Scheme::ForAll(ks, qt) = self.clone();
        Scheme::ForAll(ks, qt.apply(s))
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        let Scheme::ForAll(ks, qt) = self.clone();
        qt.type_variables()
    }
}

impl From<Type> for Scheme {
    fn from(t: Type) -> Self {
        Scheme::ForAll(vec![], Qualified::Then(vec![], t))
    }
}
