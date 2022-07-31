// ## 9. Assumptions

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Assumption {
    id: Id,
    scheme: Scheme,
}

impl Assumption {
    pub fn new(id: Id, scheme: Scheme) -> Self {
        Assumption { id, scheme }
    }

    // TODO: do this with Iterator::find at the call site?
    pub fn find(wanted: &Id, assumptions: &[Assumption]) -> Result<Scheme> {
        for Assumption { id, scheme } in assumptions {
            if id == wanted {
                return Ok(scheme.clone());
            }
        }

        Err(format!("unbound identifier: {wanted}"))
    }
}

impl Types for Assumption {
    fn apply(&self, s: &[Substitution]) -> Self {
        Assumption {
            id: self.id.clone(),
            scheme: self.scheme.apply(s),
        }
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        self.scheme.type_variables()
    }
}
