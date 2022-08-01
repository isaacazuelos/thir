//! Type Schemes

use std::fmt;

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Scheme {
    kinds: Vec<Kind>,
    qualified_type: Qualified<Type>,
}

impl Scheme {
    pub fn new(kinds: Vec<Kind>, qualified_type: Qualified<Type>) -> Scheme {
        Scheme {
            kinds,
            qualified_type,
        }
    }

    pub fn kinds(&self) -> &[Kind] {
        &self.kinds
    }

    pub fn qualified_type(&self) -> &Qualified<Type> {
        &self.qualified_type
    }
}

impl Types for Scheme {
    fn apply(&self, s: &[Substitution]) -> Self {
        Scheme::new(self.kinds.clone(), self.qualified_type.apply(s))
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        self.qualified_type.type_variables()
    }
}

impl From<Type> for Scheme {
    fn from(t: Type) -> Self {
        Scheme::new(vec![], Qualified::new(vec![], t))
    }
}

impl fmt::Display for Scheme {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(forall ")?;
        f.debug_list().entries(self.kinds()).finish()?;
        write!(f, "{})", self.qualified_type)
    }
}
