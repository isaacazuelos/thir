//! Predicates

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    IsIn(Id, Type),
}

impl Predicate {
    /// A convenience function for the IsIn constructor.
    pub fn is_in(id: impl Into<Id>, t: Type) -> Predicate {
        Predicate::IsIn(id.into(), t)
    }

    pub fn in_hfn(&self) -> bool {
        let Predicate::IsIn(c, t) = self;

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

        hnf(t)
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
        let Predicate::IsIn(i, t) = self;
        Predicate::IsIn(i.clone(), t.apply(s))
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        let Predicate::IsIn(i, t) = self;
        t.type_variables()
    }
}

impl Instantiate for Predicate {
    fn inst(&self, ts: &[Type]) -> Predicate {
        let Predicate::IsIn(c, t) = self;
        Predicate::IsIn(c.clone(), t.inst(ts))
    }
}

/// Just a helper function.
fn lift<M>(m: M, a: &Predicate, b: &Predicate) -> Result<Vec<Substitution>>
where
    M: Fn(&Type, &Type) -> Result<Vec<Substitution>>,
{
    let Predicate::IsIn(i, t) = a;
    let Predicate::IsIn(i_, t_) = b;

    if i == i_ {
        m(t, t_)
    } else {
        Err("classes differ".into())
    }
}
