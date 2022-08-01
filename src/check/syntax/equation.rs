use super::*;
use crate::check::*;

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

    pub fn parameters(&self) -> &[Pattern] {
        &self.parameters
    }

    pub fn body(&self) -> &Expr {
        &self.body
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
