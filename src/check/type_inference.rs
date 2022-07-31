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
    fn unify(&mut self, t1: &Type, t2: &Type) -> Result<()> {
        let s = self.substitutions();
        let u = &t1.apply(s).most_general_unifier(&t2.apply(s))?;
        self.ext_subst(&u);
        Ok(())
    }

    fn ext_subst(&mut self, s: &[Substitution]) {
        // We should _definitely_ look at the definition of `at_at` and unpack
        // things a bit here.
        self.substitutions = Substitution::at_at(&self.substitutions, s);
    }

    // TODO: better name?
    fn fresh_inst(&mut self, s: &Scheme) -> Qualified<Type> {
        let Scheme::ForAll(ks, qt) = s;

        let ts: Vec<_> = ks.iter().map(|k| self.new_type_var(k.clone())).collect();

        qt.inst(&ts)
    }

    fn lit(&mut self, l: &Literal) -> (Vec<Predicate>, Type) {
        match l {
            Literal::Char(_) => (vec![], builtins::character()),
            Literal::Int(_) => {
                let v = self.new_type_var(Kind::Star);
                (vec![Predicate::IsIn("Num".into(), v.clone())], v)
            }
            Literal::Str(_) => (vec![], builtins::string()),
            Literal::Rat(_) => {
                let v = self.new_type_var(Kind::Star);
                (vec![Predicate::IsIn("Fractional".into(), v.clone())], v)
            }
        }
    }
}

// Okay so from the old definitions at the bottom, we need to define an Infer
// instance for the following:

// impl Infer for Expr {}
// impl Infer for Equation {}
// impl Infer for Equations {}
// impl Infer for Program {}
// impl Infer for BindingGroup {}
// impl Infer for Implicit {}
// impl Infer for Explitic {}

// // Old Definitions
// impl TypeInference {
//     fn expr(&mut self, e: &Expr) -> Result<(Vec<Predicate>, Type)> {
//         match e {
//             Expr::Var(i) => {
//                 let scheme = Assumption::find(i, &self.assumptions)?;
//                 let Qualified::Then(ps, t) = self.fresh_inst(&sc);
//                 Ok((ps, t))
//             }
//             Expr::Const(Assumption { scheme, .. }) => {
//                 let Qualified::Then(ps, t) = self.fresh_inst(scheme);
//                 Ok((ps, t))
//             }
//             Expr::Lit(l) => {
//                 let (ps, t) = self.lit(l);
//                 Ok((ps, t))
//             }
//             Expr::Ap(e, f) => {
//                 let (ps, te) = self.expr(e)?;
//                 let (qs, tf) = self.expr(f)?;
//                 let t = self.new_type_var(Kind::Star);
//                 self.unify(&builtins::make_function(tf, t), &te)?;
//                 Ok((append(ps, qs), te))
//             }
//             Expr::Let(bg, e) => {
//                 let ps = self.bind_group(bg)?;
//                 let (qs, t) = self.expr(e)?;
//                 Ok((append(ps, qs), t))
//             }
//         }
//     }

//     fn equation(&mut self, (pats, e): &Equation) -> Result<(Vec<Predicate>, Type)> {
//         let (ps, ts) = self.patterns(pats);
//         let (qs, t) = self.expr(e)?;

//         let folded = ts.iter().cloned().fold(t, builtins::make_function);
//         Ok((append(ps, qs), folded))
//     }

//     fn equations(&mut self, alts: &[Equation], t: Type) -> Result<Vec<Predicate>> {
//         let psts = alts
//             .iter()
//             .map(|a| self.equation(a))
//             .collect::<Result<Vec<_>>>()?;

//         for t2 in psts.iter().map(|t| &t.1) {
//             self.unify(&t, t2)?
//         }

//         Ok(psts.into_iter().flat_map(|(p, _)| p).collect())
//     }

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
