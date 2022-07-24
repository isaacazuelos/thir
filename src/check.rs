//! This is the core of the paper, written in mostly the same order as things
//! are presented there.

// ## Preliminaries

use crate::util::{intersection, lookup, union};

// For simplicity, we're not worrying too much about reducing shared references.

type Error = &'static str;
type Id = String;

// ## 3. Kinds

// Our `Kind` and `Type` types aren't exactly cheap because of the boxing.
//
// I think we could put these all in one context and work on IDs, in a
// struct-of-array, data-oriented way.

#[derive(Debug, PartialEq, Clone)]
pub enum Kind {
    Star,
    Fun(Box<Kind>, Box<Kind>),
}

impl Kind {
    // Just for convenience, same as the [`Kind::Fun`] constructor, but it does
    // the boxing for us.
    fn fun(lhs: Kind, rhs: Kind) -> Kind {
        Kind::Fun(Box::new(lhs), Box::new(rhs))
    }
}

// ## 4. Types

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Var(Tyvar),
    Con(Tycon),
    Ap(Box<Type>, Box<Type>),
    TGen(usize), // for Type Schemes, not used until section 8
}

impl Type {
    // The same as the [`Type::Ap`] constructor, but it does the boxing for us.
    fn app(&self, b: Type) -> Type {
        Type::Ap(Box::new(self.clone()), Box::new(b))
    }
}

// Going with tuple types here since Haskell _strongly_ prefers thinking of
// constructor arguments positionally.
//
// Again, this is expensive since it's a big boxy tree structure, but it could
// be a pair of `Copy` indexes into some context if we wanted to be efficient.

#[derive(Debug, PartialEq, Clone)]
pub struct Tyvar(Id, Kind);

#[derive(Debug, PartialEq, Clone)]
pub struct Tycon(Id, Kind);

// Putting these in a module to namespace them instead of using the prefix
// naming scheme used in the paper. A name like `prim::unit` is `tUnit` in the
// paper.
//
// These are functions with no arguments since I can't make the `String` for the
// IDs in `const` contexts.
//
// This is something we'd want to do as part of creating the context in that
// data-oriented approach.

mod prim {
    use super::*;

    pub fn unit() -> Type {
        Type::Con(Tycon(String::from("()"), Kind::Star))
    }

    pub fn character() -> Type {
        Type::Con(Tycon(String::from("Char"), Kind::Star))
    }

    pub fn int() -> Type {
        Type::Con(Tycon(String::from("Int"), Kind::Star))
    }

    pub fn integer() -> Type {
        Type::Con(Tycon(String::from("Integer"), Kind::Star))
    }

    pub fn float() -> Type {
        Type::Con(Tycon(String::from("Float"), Kind::Star))
    }

    pub fn double() -> Type {
        Type::Con(Tycon(String::from("Double"), Kind::Star))
    }

    pub fn list() -> Type {
        Type::Con(Tycon(String::from("[]"), Kind::fun(Kind::Star, Kind::Star)))
    }

    pub fn arrow() -> Type {
        Type::Con(Tycon(
            String::from("(->)"),
            Kind::fun(Kind::Star, Kind::fun(Kind::Star, Kind::Star)),
        ))
    }

    pub fn tuple2() -> Type {
        Type::Con(Tycon(
            String::from("(,)"),
            Kind::fun(Kind::Star, Kind::fun(Kind::Star, Kind::Star)),
        ))
    }

    pub fn string() -> Type {
        super::list(character())
    }
}

// This is the function the paper calls `fn`. We could use `r#fn` but this is
// nicer, imho. We can't exactly make it infix in Rust either.

fn fun(a: Type, b: Type) -> Type {
    prim::arrow().app(a).app(b)
}

fn list(t: Type) -> Type {
    prim::list().app(t)
}

fn pair(a: Type, b: Type) -> Type {
    prim::tuple2().app(a).app(b)
}

trait HasKind {
    fn kind(&self) -> Kind;
}

impl HasKind for Tyvar {
    fn kind(&self) -> Kind {
        self.1.clone()
    }
}

impl HasKind for Tycon {
    fn kind(&self) -> Kind {
        self.1.clone()
    }
}

impl HasKind for Type {
    fn kind(&self) -> Kind {
        match self {
            Type::TGen(_) => todo!(),
            Type::Var(v) => v.kind(),
            Type::Con(c) => c.kind(),
            Type::Ap(t, _) => match t.kind() {
                Kind::Fun(_, k) => (*k).clone(),

                // If the type is well-formed, we can't apply something that's
                // not of kind `(* -> *)`.
                _ => todo!(),
            },
        }
    }
}

// 5. Substitutions
//
// This [`Subst`] should be based on Iterators, with adapters, to better emulate
// the Haskell lists. We could also use the same IDs here to make things
// cheaper.

type Subst = Vec<(Tyvar, Type)>;

fn null_subst() -> Subst {
    Vec::default()
}

// Since we don't have fancy user-defined operators to use in Rust, I'll have to
// use regular functions with names for instead when operators are defined.
//
// None of the operators in [`std::ops`] really looks like `+->`. Probably `>>`
// is the closes option, but that seems unwise.

fn maps_to(u: Tyvar, t: Type) -> Subst {
    vec![(u, t)]
}

/// I had to swap the argument orders here for `self` to work.
trait Types {
    fn apply(&self, s: &Subst) -> Self;
    fn tv(&self) -> Vec<Tyvar>;
}

impl Types for Type {
    fn apply(&self, s: &Subst) -> Self {
        match self {
            Type::Var(u) => match lookup(u, &s) {
                Some(t) => t.clone(),
                None => self.clone(),
            },
            Type::Ap(l, r) => l.apply(s).app(r.apply(s)),
            _ => self.clone(),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        match self {
            Type::Var(u) => vec![u.clone()],
            Type::Ap(l, r) => union(&l.tv(), &r.tv()),
            Type::Con(_) | Type::TGen(_) => todo!(),
        }
    }
}

// Starting to really appreciate Haskell's syntax here.

impl<T> Types for Vec<T>
where
    T: Types,
{
    fn apply(&self, s: &Subst) -> Self {
        self.iter().map(|t| t.apply(s)).collect()
    }
    fn tv(&self) -> Vec<Tyvar> {
        let mut vars: Vec<Tyvar> = self.iter().flat_map(|t| t.tv()).collect();
        vars.dedup();
        vars
    }
}

// It's unfortunate they don't give a more pronounceable name to the `@@`
// function. This is somewhere using iterators would probably pay off a lot.

// The paper includes a properties which would make this a great candidate for
// property-based testing. I'll try to write these out as I go now.
//
// > apply (s1@@s2) = apply s1 . apply s2

fn at_at(s1: &Subst, s2: &Subst) -> Subst {
    let mut buf = Vec::new();

    // [(u, apply s1 t) | (u, t) <- s2]
    for (u, t) in s2 {
        buf.push((u.clone(), t.apply(s1)));
    }

    // ++ s1
    for s in s1 {
        buf.push(s.clone());
    }

    buf
}

// We can't quite translate this over any `Monad m`.
//
// I'm assuming this is going to be over `Result` for now. We might need to make
// a few versions of this manually, if we do use it over different `m`.

// Here `merge` is `@@`, but it checks that the order of the arguments won't
// matter.
fn merge(s1: &Subst, s2: &Subst) -> Result<Subst, Error> {
    let s1_vars = s1.iter().map(|s| s.0.clone()).collect();
    let s2_vars = s2.iter().map(|s| s.0.clone()).collect();

    for v in intersection(&s1_vars, &s2_vars) {
        if Type::Var(v.clone()).apply(s1) != Type::Var(v).apply(s2) {
            return Err("merge fails");
        }
    }

    Ok(union(s1, s2))
}

// ## 6. Unification and Matching
