use super::*;
use crate::check::*;
use std::fmt;

#[derive(Debug, Clone)]
pub struct BindingGroup {
    explicit: Vec<ExplicitBinding>,
    implicit: Vec<Vec<ImplicitBinding>>,
}

impl BindingGroup {
    pub fn new(
        explicit: Vec<ExplicitBinding>,
        implicit: Vec<Vec<ImplicitBinding>>,
    ) -> BindingGroup {
        BindingGroup { explicit, implicit }
    }
}

impl fmt::Display for BindingGroup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // These aren't necessarily in any order. :/

        for binding in &self.explicit {
            write!(f, "{}", binding)?;
        }

        for bindings in &self.implicit {
            for binding in bindings {
                write!(f, "{}", binding)?;
            }
        }

        Ok(())
    }
}

impl Infer for BindingGroup {
    type Output = Vec<Predicate>;
    fn infer(&self, _context: &mut TypeInference) -> Result<Self::Output> {
        //     fn bind_group(&mut self, (es, iss): &BindingGroup) -> Result<Vec<Predicate>> {
        //         let as__: Vec<Assumption> = es
        //             .iter()
        //             .map(|(v, sc, alts)| Assumption::new(v.clone(), sc.clone()))
        //             .collect();

        //         let (ps, all_as): (Vec<Predicate>, Vec<Assumption>) = {
        //             // inlining tiSeq in the paper because its' easier.
        //             let mut ps = vec![];
        //             let mut all_as = vec![];

        //             for is in iss {
        //                 let (p, a) = self.implicit(ce, &all_as, is)?;
        //                 ps.extend(p);
        //                 all_as.extend(a);
        //             }

        //             (ps, all_as)
        //         };

        //         let qss = {
        //             // had to work with the Results in iterators.
        //             let mut buf = vec![];

        //             for e in es {
        //                 let x = self.explicit(ce, &all_as, e)?;
        //                 buf.extend(x);
        //             }

        //             buf
        //         };

        //         Ok((append(ps, qss), all_as))
        //     }
        todo!()
    }
}
