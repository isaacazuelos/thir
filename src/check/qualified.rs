use std::fmt;

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Qualified<T> {
    conditions: Vec<Predicate>,
    consequence: T,
}

impl<T> Qualified<T> {
    pub fn new(conditions: Vec<Predicate>, consequence: T) -> Qualified<T> {
        Qualified {
            conditions,
            consequence,
        }
    }

    pub fn conditions(&self) -> &[Predicate] {
        &self.conditions
    }

    pub fn consequence(&self) -> &T {
        &self.consequence
    }
}

impl Qualified<Type> {
    // Is this maybe supposed to be 'qualify' instead of 'quantify', lol?
    pub fn quantify(&self, vs: &[TypeVariable]) -> Scheme {
        let vs_: Vec<TypeVariable> = self
            .type_variables()
            .iter()
            .filter(|v| vs.contains(v))
            .cloned()
            .collect();
        let ks = vs_.iter().map(|v| v.kind().clone()).collect();
        let s: Vec<Substitution> = vs_
            .iter()
            .enumerate()
            .map(|(i, v)| Substitution {
                from: v.clone(),
                to: Type::Gen(i),
            })
            .collect();

        Scheme::new(ks, self.apply(&s))
    }
}

impl<T: fmt::Display> fmt::Display for Qualified<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(")?;
        f.debug_list().entries(self.conditions()).finish()?;
        write!(f, " => ")?;
        write!(f, "{} )", self.consequence())
    }
}

impl<T: Instantiate> Instantiate for Qualified<T> {
    fn inst(&self, ts: &[Type]) -> Qualified<T> {
        Qualified::new(self.conditions.inst(ts), self.consequence.inst(ts))
    }
}

impl<T> Types for Qualified<T>
where
    T: Types,
{
    fn apply(&self, s: &[Substitution]) -> Self {
        Qualified {
            conditions: self.conditions.clone().apply(s),
            consequence: self.consequence.apply(s),
        }
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        union(
            &self.conditions.type_variables(),
            &self.consequence.type_variables(),
        )
    }
}
