use super::*;
use crate::check::*;
use std::fmt;

#[derive(Debug, Clone)]
pub struct ImplicitBinding {
    id: Id,
    equations: Vec<Equation>,
}

impl ImplicitBinding {
    pub fn new(id: Id, equations: Vec<Equation>) -> ImplicitBinding {
        ImplicitBinding { id, equations }
    }

    pub fn restricted(bindings: &[ImplicitBinding]) -> bool {
        bindings.iter().any(|binding| {
            binding
                .equations
                .iter()
                .any(|equation| equation.parameters().is_empty())
        })
    }
}

impl fmt::Display for ImplicitBinding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for e in &self.equations {
            write!(f, "{}", self.id)?;

            for p in e.parameters() {
                write!(f, "{} ", p)?;
            }

            write!(f, " = {}", e.body())?;
        }

        Ok(())
    }
}

impl Infer for ImplicitBinding {
    type Output = ();

    fn infer(&self, _context: &mut TypeInference) -> Result<Self::Output> {
        //     fn implicit(&mut self, bs: &[ImplicitBinding]) -> Result<(Vec<Predicate>, Vec<Assumption>)> {
        //         let ts = bs
        //             .iter()
        //             .map(|_| self.new_type_var(Kind::Star))
        //             .collect::<Vec<_>>();

        //         let is: Vec<Id> = bs.iter().map(|b| b.0.clone()).collect();
        //         let scs: Vec<Scheme> = ts.iter().cloned().map(Scheme::from).collect();
        //         let as__ = append(
        //             zip_with(Assumption::new, is.clone(), scs),
        //             self.assumptions.to_vec(),
        //         );
        //         let altss = bs.iter().map(|b| b.1.clone()).collect();

        //         let pss = zip_with_try(|alts, t| self.equations(&alts, t), altss, ts.clone())?;
        //         let s = self.substitutions();

        //         let ps_: Vec<Predicate> = pss.iter().flatten().map(|p| p.apply(s)).collect();
        //         let ts_: Vec<Type> = ts.apply(s);
        //         let fs: Vec<TypeVariable> = self.assumptions().to_vec().apply(s).type_variables();
        //         let vss: Vec<Vec<TypeVariable>> = ts_.iter().map(Types::type_variables).collect();
        //         let gs = minus(
        //             vss.iter()
        //                 .cloned()
        //                 .reduce(|l, r| intersection(&l, &r))
        //                 .unwrap(),
        //             &fs,
        //         );

        //         let (ds, rs) = ce.split(
        //             &fs,
        //             &vss.into_iter().reduce(|a, b| intersection(&a, &b)).unwrap(),
        //             &ps_,
        //         )?;

        //         if restricted(bs) {
        //             let gs_ = minus(gs, &rs.type_variables());
        //             let scs_ = ts_
        //                 .iter()
        //                 .map(|t| Qualified::Then(rs.clone(), t.clone()).quantify(&gs_))
        //                 .collect();
        //             Ok((append(ds, rs), zip_with(Assumption::new, is, scs_)))
        //         } else {
        //             let scs_ = ts_
        //                 .iter()
        //                 .map(|t| Qualified::Then(rs.clone(), t.clone()).quantify(&gs))
        //                 .collect();
        //             Ok((ds, zip_with(Assumption::new, is, scs_)))
        //         }
        //     }
        todo!()
    }
}
