use super::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Qualified<T> {
    // This is the `:=>` constructor in the paper.
    //
    // In the final assumptions produced, it's the trait constraints. The stuff
    // in the `where` clauses for Rust.
    Then(Vec<Predicate>, T),
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

        Scheme::ForAll(ks, self.apply(&s))
    }
}

impl<T: Instantiate> Instantiate for Qualified<T> {
    fn inst(&self, ts: &[Type]) -> Qualified<T> {
        let Qualified::Then(ps, t) = self;
        Qualified::Then(ps.inst(ts), t.inst(ts))
    }
}

impl<T: Clone> Qualified<T> {
    pub fn then(pred: &[Predicate], t: T) -> Qualified<T> {
        Qualified::Then(pred.into(), t)
    }

    pub fn consequence(&self) -> &T {
        let Qualified::Then(_, q) = self;
        q
    }
}

impl<T> Types for Qualified<T>
where
    T: Types,
{
    fn apply(&self, s: &[Substitution]) -> Self {
        let Qualified::Then(ps, t) = self;
        Qualified::Then(ps.apply(s), t.apply(s))
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        let Qualified::Then(ps, t) = self;
        union(&ps.type_variables(), &t.type_variables())
    }
}
