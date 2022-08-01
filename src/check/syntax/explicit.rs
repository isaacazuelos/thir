use super::*;
use crate::check::*;
use std::fmt;

#[derive(Debug, Clone)]
pub struct ExplicitBinding {
    id: Id,
    signature: Scheme,
    equations: Vec<Equation>,
}

impl fmt::Display for ExplicitBinding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{} :: {}", self.id, self.signature)?;

        for e in &self.equations {
            write!(f, "{}", self.id)?;

            for p in e.parameters() {
                write!(f, "{} ", p)?;
            }

            write!(f, "= {}", e.body())?;
        }

        Ok(())
    }
}

impl ExplicitBinding {
    fn new(id: Id, signature: Scheme, equations: Vec<Equation>) -> ExplicitBinding {
        ExplicitBinding {
            id,
            signature,
            equations,
        }
    }
}

impl Infer for ExplicitBinding {
    type Output = ();

    fn infer(&self, _context: &mut TypeInference) -> Result<Self::Output> {
        //     fn explicit(
        //         &mut self,
        //         ce: &TraitEnvironment,
        //         as_: &[Assumption],
        //         (i, sc, alts): &ExplicitBinding,
        //     ) -> Result<Vec<Predicate>> {
        //         let Qualified::Then(qs, t) = self.fresh_inst(sc);
        //         let ps = self.equations(alts, t.clone())?;
        //         let s = self.substitutions();

        //         let qs_ = qs.apply(s);
        //         let t_ = t.apply(s);
        //         let fs = as_.to_vec().apply(s).type_variables();
        //         let gs = minus(t_.type_variables(), &fs);
        //         let sc_ = Qualified::Then(qs_.clone(), t_).quantify(&gs);
        //         let ps_ = ps
        //             .apply(s)
        //             .iter()
        //             .filter(|p| !ce.entails(&qs_, p))
        //             .cloned()
        //             .collect::<Vec<Predicate>>();

        //         let (ds, rs) = ce.split(&fs, &gs, &ps_)?;

        //         if sc != &sc_ {
        //             Err("signature to general".into())
        //         } else if !rs.is_empty() {
        //             Err("context too weak".into())
        //         } else {
        //             Ok(ds)
        //         }
        //     }
        // }
        todo!()
    }
}
