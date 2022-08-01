use crate::check::*;
use std::fmt;

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

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Literal::Char(c) => write!(f, "{}", c.escape_default()),
            Literal::Rational(r) => write!(f, "{}", r),
            Literal::Integral(i) => write!(f, "{}", i),
            Literal::String(s) => write!(f, "\"{}\"", s.escape_default()),
        }
    }
}
impl Infer for Literal {
    type Output = Type;

    fn infer(&self, context: &mut TypeInference) -> Result<Self::Output> {
        match self {
            Literal::Char(_) => Ok(builtins::character()),

            Literal::Rational(_) => {
                // the type is some new type T, where the context knows that
                // T = ForAll a . Fractional a => a

                let v = context.new_type_var(Kind::Star);

                context.add_pred(Predicate::new("Num".into(), v.clone()));

                Ok(v)
            }

            Literal::Integral(_) => {
                todo!()
            }

            Literal::String(_) => Ok(builtins::string()),
        }
    }
}
