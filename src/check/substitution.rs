//! Substitutions
//!
//! This [`Subst`] should be based on Iterators, with adapters, to better emulate
//! the Haskell lists. We could also use the same IDs here to make things
//! cheaper.

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Substitution {
    pub(crate) from: TypeVariable,
    pub(crate) to: Type,
}

impl Substitution {
    pub fn new(from: TypeVariable, to: Type) -> Substitution {
        Substitution { from, to }
    }

    pub fn at_at(s1: &[Substitution], s2: &[Substitution]) -> Vec<Substitution> {
        let mut substitutions = Vec::new();

        // [(u, apply s1 t) | (u, t) <- s2]
        for s in s2.iter() {
            substitutions.push(Substitution {
                from: s.from.clone(),
                to: s.to.apply(s1),
            });
        }

        // ++ s1
        for s in s1.iter() {
            substitutions.push(s.clone());
        }

        substitutions
    }

    // We can't quite translate this over any `Monad m`.
    //
    // I'm assuming this is going to be over `Result` for now. We might need to make
    // a few versions of this manually, if we do use it over different `m`.

    // Here `merge` is `@@`, but it checks that the order of the arguments won't
    // matter.
    pub fn merge(s1: &[Substitution], s2: &[Substitution]) -> Result<Vec<Substitution>> {
        let s1_vars: Vec<_> = s1.iter().map(|s| s.from.clone()).collect();
        let s2_vars: Vec<_> = s2.iter().map(|s| s.from.clone()).collect();

        for v in intersection(&s1_vars, &s2_vars) {
            if Type::Variable(v.clone()).apply(s1) != Type::Variable(v).apply(s2) {
                return Err("merge fails".into());
            }
        }

        Ok(union(s1, s2))
    }
}
