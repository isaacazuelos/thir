//! Expression syntax

use super::*;

#[derive(Debug)]
pub struct Program {
    binding_groups: Vec<BindingGroup>,
}

impl Program {
    pub fn new(binding_groups: Vec<BindingGroup>) -> Program {
        Program { binding_groups }
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

#[derive(Debug, Clone)]
pub struct ImplicitBinding {
    id: Id,
    equations: Vec<Equation>,
}

impl ImplicitBinding {
    pub fn new(id: Id, equations: Vec<Equation>) -> ImplicitBinding {
        ImplicitBinding { id, equations }
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

#[derive(Debug, Clone)]
pub struct ExplicitBinding {
    id: Id,
    signature: Scheme,
    equations: Vec<Equation>,
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

// An alternative is what this is calling an 'equation'
//
// i.e. it's each line that's pattern matched in code like this:
//
//     null []    = true
//     null (_:_) = false
//
// The Pattern is each parameter, and the Expr is the right hand side.
#[derive(Debug, Clone)]
pub struct Equation {
    parameters: Vec<Pattern>,
    body: Expr,
}

impl Equation {
    pub fn new(parameters: Vec<Pattern>, body: Expr) -> Equation {
        Equation { parameters, body }
    }
}

impl Infer for Equation {
    type Output = Type;

    fn infer(&self, context: &mut TypeInference) -> Result<Self::Output> {
        let parameter_types = self.parameters.as_slice().infer(context)?;

        let mut t = self.body.infer(context)?;

        for parameter in parameter_types {
            t = builtins::make_function(parameter, t);
        }

        Ok(t)
    }
}

// TODO: Rename variants.
// TODO: Why does `Expression::Const` contain `Assumption`?

#[derive(Debug, Clone)]
pub enum Expr {
    Application(Box<Expr>, Box<Expr>),
    Constructor(Assumption),
    Let(BindingGroup, Box<Expr>),
    Literal(Literal),
    Variable(Id),
}

impl Infer for Expr {
    type Output = Type;

    fn infer(&self, context: &mut TypeInference) -> Result<Self::Output> {
        match self {
            Expr::Application(e, f) => {
                let te = e.infer(context)?;
                let tf = f.infer(context)?;
                let t = context.new_type_var(Kind::Star);
                context.unify(&builtins::make_function(tf, t.clone()), &te)?;
                Ok(t)
            }

            Expr::Constructor(assumption) => {
                let Qualified::Then(_ps, t) = context.fresh_inst(assumption.scheme());
                Ok(t)
            }

            Expr::Let(bg, e) => {
                let _ps = bg.infer(context)?;
                let t = e.infer(context)?;
                Ok(t)
            }

            Expr::Literal(l) => l.infer(context),

            Expr::Variable(i) => {
                let scheme = Assumption::find(i, context.assumptions())?;
                let Qualified::Then(_ps, t) = context.fresh_inst(&scheme);
                Ok(t)
            }
        }
    }
}

// Note that these are the literals, their exact type at runtime isn't know.
//
// For example, we use an `i64` for Literal::Int, but it's actual type is
// generic, it's `Num a => a`.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Char(char),
    Rational(f64),
    Integral(i64),
    String(String),
}

impl Infer for Literal {
    type Output = Type;

    fn infer(&self, _context: &mut TypeInference) -> Result<Self::Output> {
        match self {
            Literal::Char(_) => Ok(builtins::character()),

            Literal::Rational(_) => {
                // the type is some new type T, where the context knows that
                // T = ForAll a . Fractional a => a
                todo!()
            }

            Literal::Integral(_) => {
                todo!()
            }

            Literal::String(_) => Ok(builtins::string()),
        }
    }
}

pub type Ambiguity = (TypeVariable, Vec<Predicate>);

pub fn restricted(bindings: &[ImplicitBinding]) -> bool {
    bindings.iter().any(|binding| {
        binding
            .equations
            .iter()
            .any(|equation| equation.parameters.is_empty())
    })
}
