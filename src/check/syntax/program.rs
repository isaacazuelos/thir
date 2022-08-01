use crate::check::*;
use std::fmt;

#[derive(Debug)]
pub struct Program {
    binding_groups: Vec<BindingGroup>,
}

impl Program {
    pub fn new(binding_groups: Vec<BindingGroup>) -> Program {
        Program { binding_groups }
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for binding in &self.binding_groups {
            write!(f, "{}", binding)?;
        }

        Ok(())
    }
}

impl Infer for Program {
    type Output = ();

    fn infer(&self, _context: &mut TypeInference) -> Result<Self::Output> {
        //     pub fn program(
        //         &mut self,
        //         ce: &TraitEnvironment,
        //         as_: &[Assumption],
        //         bgs: Program,
        //     ) -> Result<Vec<Assumption>> {
        //         let (ps, as__) = {
        //             let mut ps = vec![];
        //             let mut all_as = vec![];

        //             for bg in &bgs.binding_groups {
        //                 let (p, a) = self.bind_group(ce, as_, &bg)?;
        //                 all_as.extend(a);
        //                 ps.extend(p);
        //             }

        //             (ps, all_as)
        //         };

        //         let s = self.substitutions();

        //         let rs = ce.reduce(&ps.apply(s))?;
        //         let s_ = ce.default_substitutions(vec![], &rs)?;

        //         Ok(as__.apply(&Substitution::at_at(&s_, &s)))
        //     }
        todo!()
    }
}
