use super::*;
use crate::check::*;
use std::fmt;

#[derive(Debug, Clone)]
pub enum Expr {
    Application(Box<Expr>, Box<Expr>),
    Constructor(Assumption),
    Let(BindingGroup, Box<Expr>),
    Literal(Literal),
    Variable(Id),
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Application(lhs, rhs) => write!(f, "({lhs} {rhs})"),
            Expr::Constructor(a) => write!(f, "{:?}", a),
            Expr::Let(e, b) => write!(f, "(let {} in {})", e, b),
            Expr::Literal(l) => write!(f, "{}", l),
            Expr::Variable(v) => write!(f, "{}", v),
        }
    }
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

            Expr::Constructor(assumption) => Ok(context
                .fresh_inst(assumption.scheme())
                .consequence()
                .clone()),

            Expr::Let(bg, e) => {
                let _ps = bg.infer(context)?;
                let t = e.infer(context)?;
                Ok(t)
            }

            Expr::Literal(l) => l.infer(context),

            Expr::Variable(i) => {
                let scheme = Assumption::find(i, context.assumptions())?;
                let qualified = context.fresh_inst(&scheme);
                Ok(qualified.consequence().clone())
            }
        }
    }
}
