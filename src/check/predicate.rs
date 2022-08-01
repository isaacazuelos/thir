//! Predicates

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Predicate {
    id: Id,
    t: Type,
}

impl Predicate {
    /// A convenience function for the IsIn constructor.
    pub fn new(id: Id, t: Type) -> Predicate {
        Predicate { id, t }
    }

    pub fn id(&self) -> Id {
        self.id.clone()
    }

    pub fn type_(&self) -> &Type {
        &self.t
    }

    pub fn in_hfn(&self) -> bool {
        // This is pulled out into a function since we call it recursively.
        //
        // TODO: make iterative?
        fn hnf(t: &Type) -> bool {
            match t {
                Type::Variable(_) => true,
                Type::Constructor(_) => false,
                Type::Applied(t, _) => hnf(t),
                Type::Gen(_) => todo!(),
            }
        }

        hnf(self.type_())
    }

    pub fn overlap(&self, q: &Predicate) -> bool {
        self.most_general_unifier_predicate(q).is_ok()
    }

    pub fn most_general_unifier_predicate(&self, b: &Predicate) -> Result<Vec<Substitution>> {
        lift(Type::most_general_unifier, self, b)
    }

    pub fn match_predicate(&self, b: &Predicate) -> Result<Vec<Substitution>> {
        lift(Type::matches, self, b)
    }
}

impl Types for Predicate {
    fn apply(&self, s: &[Substitution]) -> Self {
        Predicate::new(self.id.clone(), self.t.apply(s))
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        self.t.type_variables()
    }
}

impl Instantiate for Predicate {
    fn inst(&self, ts: &[Type]) -> Predicate {
        Predicate::new(self.id.clone(), self.t.inst(ts))
    }
}

/// Just a helper function.
fn lift<M>(m: M, a: &Predicate, b: &Predicate) -> Result<Vec<Substitution>>
where
    M: Fn(&Type, &Type) -> Result<Vec<Substitution>>,
{
    if a.id() == b.id() {
        m(a.type_(), b.type_())
    } else {
        Err("classes differ".into())
    }
}
