//! Type representation
//!
//! The module is named plural since `type` is a reserved word.

use super::*;

pub trait Types {
    fn apply(&self, s: &[Substitution]) -> Self;
    fn type_variables(&self) -> Vec<TypeVariable>;
}

impl<T> Types for Vec<T>
where
    T: Types,
{
    fn apply(&self, s: &[Substitution]) -> Self {
        self.iter().map(|t| t.apply(s)).collect()
    }
    fn type_variables(&self) -> Vec<TypeVariable> {
        let mut vars: Vec<TypeVariable> = self.iter().flat_map(|t| t.type_variables()).collect();
        vars.dedup();
        vars
    }
}

pub trait Instantiate {
    fn inst(&self, ts: &[Type]) -> Self;
}

impl<A> Instantiate for Vec<A>
where
    A: Instantiate,
{
    fn inst(&self, ts: &[Type]) -> Vec<A> {
        self.iter().map(|a| a.inst(ts)).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Variable(TypeVariable),
    Constructor(TypeConstructor),
    Applied(Box<Type>, Box<Type>),
    Gen(usize), // for Type Schemes, not used until section 8
}

impl Type {
    // The same as the [`Type::Applied`] constructor, but it does the boxing for
    // us. I'll do similar things with other types to clone.
    pub fn apply_to(&self, b: Type) -> Type {
        Type::Applied(Box::new(self.clone()), Box::new(b))
    }

    pub fn most_general_unifier(&self, t2: &Type) -> Result<Vec<Substitution>> {
        match (self, t2) {
            (Type::Applied(l, r), Type::Applied(l_, r_)) => {
                // Cool to see `?` work as a monad here -- it doesn't always!
                let s1 = l.most_general_unifier(l_)?;

                let s2 = r.apply(&s1).most_general_unifier(&r_.apply(&s1))?;

                Ok(Substitution::at_at(&s2, &s1))
            }
            (Type::Variable(u), t) | (t, Type::Variable(u)) => u.var_bind(t),
            (Type::Constructor(t1), Type::Constructor(t2)) if t1 == t2 => Ok(Vec::default()),
            _ => Err(format!("types do not unify: {:?}, {:?}", self, t2)),
        }
    }

    pub fn matches(&self, t2: &Type) -> Result<Vec<Substitution>> {
        match (self, t2) {
            (Type::Applied(l, r), Type::Applied(l_, r_)) => {
                let sl = l.matches(l_)?;
                let sr = r.matches(r_)?;

                Substitution::merge(&sl, &sr)
            }
            (Type::Variable(u), t) if u.kind() == t.kind() => Ok(u.maps_to(t)),
            (Type::Constructor(tc1), Type::Constructor(tc2)) if tc1 == tc2 => Ok(Vec::default()),
            (t1, t2) => Err(format!("types do not match: {:?}, {:?}", t1, t2)),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Variable(v) => write!(f, "{}", v),
            Type::Constructor(c) => write!(f, "{}", c),
            Type::Applied(lhs, rhs) => write!(f, "({} {})", lhs, rhs),
            Type::Gen(n) => write!(f, "<Gen({})>", n),
        }
    }
}

impl Instantiate for Type {
    fn inst(&self, ts: &[Type]) -> Type {
        match self {
            Type::Applied(l, r) => l.inst(ts).apply_to(r.inst(ts)),
            Type::Gen(n) => ts[*n].clone(),
            t => t.clone(),
        }
    }
}

// Since we don't have fancy user-defined operators to use in Rust, I'll have to
// use regular functions with names for instead when operators are defined.
//
// None of the operators in [`std::ops`] really looks like `+->`. Probably `>>`
// is the closes option, but that seems unwise.

impl Types for Type {
    fn apply(&self, s: &[Substitution]) -> Self {
        match self {
            Type::Variable(u) => match u.find_type_in(s) {
                Some(t) => t,
                None => self.clone(),
            },
            Type::Applied(l, r) => l.apply(s).apply_to(r.apply(s)),
            _ => self.clone(),
        }
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        match self {
            Type::Variable(u) => vec![u.clone()],
            Type::Applied(l, r) => union(&l.type_variables(), &r.type_variables()),
            Type::Constructor(_) | Type::Gen(_) => vec![],
        }
    }
}

// Going with tuple types here since Haskell _strongly_ prefers thinking of
// constructor arguments positionally.
//
// Again, this is expensive since it's a big boxy tree structure, but it could
// be a pair of `Copy` indexes into some context if we wanted to be efficient.

#[derive(Debug, PartialEq, Clone)]
pub struct TypeVariable {
    id: Id,
    kind: Kind,
}

impl TypeVariable {
    pub fn new(id: impl Into<Id>, kind: Kind) -> Self {
        TypeVariable {
            id: id.into(),
            kind,
        }
    }

    pub fn maps_to(&self, t: &Type) -> Vec<Substitution> {
        vec![Substitution {
            from: self.clone(),
            to: t.clone(),
        }]
    }

    pub fn find_type_in(&self, substitutions: &[Substitution]) -> Option<Type> {
        for s in substitutions {
            if &s.from == self {
                return Some(s.to.clone());
            }
        }

        None
    }

    fn var_bind(&self, t: &Type) -> Result<Vec<Substitution>> {
        if matches!(t, Type::Variable(t_) if t_ == self) {
            Ok(Vec::default())
        } else if t.type_variables().contains(self) {
            Err("occurs check fails".into())
        } else if self.kind() != t.kind() {
            Err("kinds do not match".into())
        } else {
            Ok(self.maps_to(t))
        }
    }
}

impl std::fmt::Display for TypeVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl HasKind for Type {
    fn kind(&self) -> &Kind {
        match self {
            Type::Gen(_) => unimplemented!(),
            Type::Variable(v) => v.kind(),
            Type::Constructor(c) => c.kind(),
            Type::Applied(t, _) => match t.kind() {
                Kind::Function(_, k) => k,

                // If the type is well-formed, we can't apply something that's
                // not of kind `(* -> *)`.
                _ => panic!("Type is not well formed, is Ap but Kind is not (->)"),
            },
        }
    }
}

impl HasKind for TypeVariable {
    fn kind(&self) -> &Kind {
        &self.kind
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypeConstructor {
    id: Id,
    kind: Kind,
}

impl TypeConstructor {
    pub fn new(id: impl Into<Id>, kind: Kind) -> Self {
        TypeConstructor {
            id: id.into(),
            kind,
        }
    }
}

impl std::fmt::Display for TypeConstructor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl HasKind for TypeConstructor {
    fn kind(&self) -> &Kind {
        &self.kind
    }
}
