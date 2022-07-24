//! This is the core of the paper, written in mostly the same order as things
//! are presented there.

#![allow(unused_variables)]

// ## Preliminaries

use crate::util::{intersection, lookup, union};

// For simplicity, we're not worrying too much about reducing shared references.

type Error = &'static str;
type Id = String;

type Result<T> = std::result::Result<T, Error>;

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
            Type::TGen(_) => unimplemented!(),
            Type::Var(v) => v.kind(),
            Type::Con(c) => c.kind(),
            Type::Ap(t, _) => match t.kind() {
                Kind::Fun(_, k) => (*k).clone(),

                // If the type is well-formed, we can't apply something that's
                // not of kind `(* -> *)`.
                _ => panic!("Type is not well formed, is Ap but Kind is not (->)"),
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

fn maps_to(u: &Tyvar, t: &Type) -> Subst {
    vec![(u.clone(), t.clone())]
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
            Type::Con(_) | Type::TGen(_) => unimplemented!(),
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
fn merge(s1: &Subst, s2: &Subst) -> Result<Subst> {
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

// Again, we have `Monad m =>` constraints which aren't worth trying to emulate
// in Rust, at least not here, yet. I'm going to write this with `Result` for
// now, but leave the implementation until I have a better idea of what specific
// `Monad m`s we need. Hopefully it's just one or two...

// mgu is for 'most general unifier'.
fn mgu(t1: &Type, t2: &Type) -> Result<Subst> {
    match (t1, t2) {
        (Type::Ap(l, r), Type::Ap(l_, r_)) => {
            // Cool to see `?` work as a monad here -- it doesn't always!
            let s1 = mgu(l, l_)?;
            let s2 = mgu(&r.apply(&s1), &r_.apply(&s1))?;
            Ok(at_at(&s2, &s1))
        }
        (Type::Var(u), t) | (t, Type::Var(u)) => var_bind(u, t),
        (Type::Con(t1), Type::Con(t2)) if t1 == t2 => Ok(null_subst()),
        _ => Err("types do not unify"),
    }
}

fn var_bind(u: &Tyvar, t: &Type) -> Result<Subst> {
    if matches!(t, Type::Var(t_) if t_ == u) {
        Ok(null_subst())
    } else if t.tv().contains(u) {
        Err("occurs check fails")
    } else if u.kind() != t.kind() {
        Err("kinds do not match")
    } else {
        Ok(maps_to(u, t))
    }
}

fn match_(t1: &Type, t2: &Type) -> Result<Subst> {
    match (t1, t2) {
        (Type::Ap(l, r), Type::Ap(l_, r_)) => {
            let sl = match_(l, l_)?;
            let sr = match_(r, r_)?;
            merge(&sl, &sr)
        }
        (Type::Var(u), t) if u.kind() == t.kind() => Ok(maps_to(u, t)),
        (Type::Con(tc1), Type::Con(tc2)) if tc1 == tc2 => Ok(null_subst()),
        (t1, t2) => Err("types do not match"),
    }
}

// ## 7. Type Classes, Predicates and Qualified Types

// No code in this part! I should check out that Wadler and Blott (1989) paper.

// ### 7.1. Basic Definition

// To better match Haskell's `data Type = Constructor _`, I'm using enums here
// with one variant like `enum Type { Constructor(_) }`. We could just use tuple
// structs, but this lets us name the constructors to better match the paper.

#[derive(Debug, Clone, PartialEq)]
enum Qual<T> {
    // This is the `:=>` constructor in
    Then(Vec<Pred>, T),
}

#[derive(Debug, Clone, PartialEq)]
enum Pred {
    IsIn(Id, Type),
}

impl<T> Types for Qual<T>
where
    T: Types,
{
    fn apply(&self, s: &Subst) -> Self {
        let Qual::Then(ps, t) = self;
        Qual::Then(ps.apply(s), t.apply(s))
    }

    fn tv(&self) -> Vec<Tyvar> {
        let Qual::Then(ps, t) = self;
        union(&ps.tv(), &t.tv())
    }
}

impl Types for Pred {
    fn apply(&self, s: &Subst) -> Self {
        let Pred::IsIn(i, t) = self;
        Pred::IsIn(i.clone(), t.apply(s))
    }

    fn tv(&self) -> Vec<Tyvar> {
        let Pred::IsIn(i, t) = self;
        t.tv()
    }
}

fn mgu_pred(a: &Pred, b: &Pred) -> Result<Subst> {
    lift(mgu, a, b)
}

fn match_pred(a: &Pred, b: &Pred) -> Result<Subst> {
    lift(match_, a, b)
}

// Here come the primes. I'll use `_` in place of `'` where I can.

fn lift<M>(m: M, a: &Pred, b: &Pred) -> Result<Subst>
where
    M: Fn(&Type, &Type) -> Result<Subst>,
{
    let Pred::IsIn(i, t) = a;
    let Pred::IsIn(i_, t_) = b;

    if i == i_ {
        m(t, t_)
    } else {
        Err("classes differ")
    }
}

type Class = (Vec<Id>, Vec<Inst>);
type Inst = Qual<Pred>;

fn ord_example() -> Class {
    (
        // This part tells us that Eq is a 'superclass' of Ord,
        // it's the `class Eq => Ord` part.
        vec![String::from("Eq")],
        // These are instances of the class
        vec![
            // This is the `instance Ord _ where` part for unit, char, int.
            // Notice this isn't the implementation, just the type level stuff.
            Qual::Then(vec![], Pred::IsIn(String::from("Ord"), prim::unit())),
            Qual::Then(vec![], Pred::IsIn(String::from("Ord"), prim::character())),
            Qual::Then(vec![], Pred::IsIn(String::from("Ord"), prim::int())),
            // This one is `Ord a, Ord b => Ord (a, b)`
            Qual::Then(
                vec![
                    // Ord a constraint
                    Pred::IsIn(
                        String::from("Ord"),
                        Type::Var(Tyvar(String::from("a"), Kind::Star)),
                    ),
                    // Ord b constraint
                    Pred::IsIn(
                        String::from("Ord"),
                        Type::Var(Tyvar(String::from("b"), Kind::Star)),
                    ),
                ],
                // => Ord (a, b)
                Pred::IsIn(
                    String::from("Ord"),
                    pair(
                        Type::Var(Tyvar(String::from("a"), Kind::Star)),
                        Type::Var(Tyvar(String::from("b"), Kind::Star)),
                    ),
                ),
            ),
        ],
    )
}

// This is another place where I'm appreciating Haskell's brevity.

// ### 7.2 Class Environments
