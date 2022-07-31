//! Type inference code

use super::{error::*, qualified::Qualified, *};

pub trait Infer {
    type Output;
    fn infer(&self, context: &mut TypeInference) -> Result<Self::Output>;
}

impl<T: Infer> Infer for &[T] {
    type Output = Vec<<T as Infer>::Output>;

    fn infer(&self, context: &mut TypeInference) -> Result<Self::Output> {
        let mut buf = Vec::with_capacity(self.len());

        for item in self.iter() {
            let out = item.infer(context)?;
            buf.push(out);
        }

        Ok(buf)
    }
}

#[derive(Debug, Default)]
pub struct TypeInference {
    substitutions: Vec<Substitution>,
    assumptions: Vec<Assumption>,

    next_var: usize,
    type_class_env: TraitEnvironment,
}

impl TypeInference {
    pub fn substitutions(&self) -> &[Substitution] {
        &self.substitutions
    }

    pub fn assumptions(&self) -> &[Assumption] {
        &self.assumptions
    }

    pub fn new_type_var(&mut self, k: Kind) -> Type {
        let i = self.next_var;
        self.next_var += 1;
        Type::Variable(TypeVariable::new(i, k))
    }

    pub fn assume(&mut self, a: Assumption) {
        self.assumptions.push(a);
    }
}

// These are the older methods from the paper
impl TypeInference {
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<()> {
        let s = self.substitutions();
        let u = &t1.apply(s).most_general_unifier(&t2.apply(s))?;
        self.ext_subst(u);
        Ok(())
    }

    pub fn ext_subst(&mut self, s: &[Substitution]) {
        // We should _definitely_ look at the definition of `at_at` and unpack
        // things a bit here.
        self.substitutions = Substitution::at_at(&self.substitutions, s);
    }

    // TODO: better name?
    pub fn fresh_inst(&mut self, s: &Scheme) -> Qualified<Type> {
        let Scheme::ForAll(ks, qt) = s;

        let ts: Vec<_> = ks.iter().map(|k| self.new_type_var(k.clone())).collect();

        qt.inst(&ts)
    }
}
