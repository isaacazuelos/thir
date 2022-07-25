//! This is the core of the paper, written in mostly the same order as things
//! are presented there.

#![allow(unused_variables)]

// ## Preliminaries

use std::collections::HashMap;

use crate::util::{intersection, lookup, union};

// For simplicity, we're not worrying too much about reducing shared references.

type Error = &'static str;

// kind of ties our hands at defining new ones, but it's nicer to avoid all the
// `.into` calls for now.
type Id = &'static str;

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
    // I'll do similar things with other types to clone.
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
// The types with arrow kinds are functions since I can't make the boxes for the
// arrow kinds in while `const`.
//
// This is something we'd want to do as part of creating the context in that
// data-oriented approach.
mod prim {
    use super::*;

    pub const UNIT: Type = Type::Con(Tycon("()", Kind::Star));
    pub const CHARACTER: Type = Type::Con(Tycon("Char", Kind::Star));
    pub const INT: Type = Type::Con(Tycon("Int", Kind::Star));
    pub const INTEGER: Type = Type::Con(Tycon("Integer", Kind::Star));
    pub const FLOAT: Type = Type::Con(Tycon("Float", Kind::Star));
    pub const DOUBLE: Type = Type::Con(Tycon("Double", Kind::Star));

    pub fn list() -> Type {
        Type::Con(Tycon("[]", Kind::fun(Kind::Star, Kind::Star)))
    }

    pub fn arrow() -> Type {
        Type::Con(Tycon(
            "(->)",
            Kind::fun(Kind::Star, Kind::fun(Kind::Star, Kind::Star)),
        ))
    }

    pub fn tuple_2() -> Type {
        Type::Con(Tycon(
            "(,)",
            Kind::fun(Kind::Star, Kind::fun(Kind::Star, Kind::Star)),
        ))
    }

    pub fn string() -> Type {
        super::list(CHARACTER)
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
    prim::tuple_2().app(a).app(b)
}

trait HasKind {
    fn kind(&self) -> &Kind;
}

impl HasKind for Tyvar {
    fn kind(&self) -> &Kind {
        &self.1
    }
}

impl HasKind for Tycon {
    fn kind(&self) -> &Kind {
        &self.1
    }
}

impl HasKind for Type {
    fn kind(&self) -> &Kind {
        match self {
            Type::TGen(_) => unimplemented!(),
            Type::Var(v) => v.kind(),
            Type::Con(c) => c.kind(),
            Type::Ap(t, _) => match t.kind() {
                Kind::Fun(_, k) => k,

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
            Type::Var(u) => match lookup(u, s) {
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
    let s1_vars: Vec<_> = s1.iter().map(|s| s.0.clone()).collect();
    let s2_vars: Vec<_> = s2.iter().map(|s| s.0.clone()).collect();

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

impl<T: Clone> Qual<T> {
    fn then(pred: &[Pred], t: &T) -> Qual<T> {
        Qual::Then(pred.into(), t.clone())
    }

    fn consequence(&self) -> &T {
        let Qual::Then(_, q) = self;
        q
    }
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
        Pred::IsIn(i, t.apply(s))
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

// I couldn't keep things straight as a tuple, so I'm naming things.
#[derive(Debug, Clone, PartialEq)]
struct Class {
    supers: Vec<Id>,
    instances: Vec<Inst>,
}

impl Class {
    // Keeps the same arg order as the tuple, so we can use it as a constructor
    fn new(supers: &[Id], instances: &[Inst]) -> Self {
        Class {
            supers: supers.into(),
            instances: instances.as_ref().to_vec(),
        }
    }
}

type Inst = Qual<Pred>;

fn ord_example() -> Class {
    Class::new(
        // This part tells us that Eq is a 'superclass' of Ord,
        // it's the `class Eq => Ord` part.
        &["Eq"],
        // These are instances of the class
        &[
            // This is the `instance Ord _ where` part for unit, char, int.
            // Notice this isn't the implementation, just the type level stuff.
            Qual::then(&[], &Pred::IsIn("Ord", prim::UNIT)),
            Qual::then(&[], &Pred::IsIn("Ord", prim::CHARACTER)),
            Qual::then(&[], &Pred::IsIn("Ord", prim::INT)),
            // This one is `Ord a, Ord b => Ord (a, b)`
            Qual::then(
                &[
                    // Ord a constraint
                    Pred::IsIn("Ord", Type::Var(Tyvar("a", Kind::Star))),
                    // Ord b constraint
                    Pred::IsIn("Ord", Type::Var(Tyvar("b", Kind::Star))),
                ],
                // => Ord (a, b)
                &Pred::IsIn(
                    "Ord",
                    pair(
                        Type::Var(Tyvar("a", Kind::Star)),
                        Type::Var(Tyvar("b", Kind::Star)),
                    ),
                ),
            ),
        ],
    )
}

// This is another place where I'm appreciating Haskell's brevity.

// ### 7.2 Class Environments

// The paper uses a function with type `Id -> Maybe Class` here for the field
// `classes`. I'm betting that function is a lookup, so a hashmap is a different
// way to do that.
//
// The way the paper does it is by nesting it's `classes` function. This
// effectively forms a linked list in memory -- I'm not really sure I see this
// and the `modify` definition as better than just using `[Class]` and `lookup`
// but whatever.
//
// Using a hashmap and cloning here is going to be wildly inefficient. I'd need
// to understand the use pattern better to really know how to better translate
// it. Alternatively just `Rc` it.

// The `EnvTransformer` type isn't a natural (or really viable) way to work with
// things in Rust, so I mostly don't use it in favour of fully applying things
// to ClassEnv. Hopefully this works out. Otherwise we might need to get more
// clever/likely-wrong here than I'd like. I'll also use Result so we can have
// some context.

type EnvTransformer = fn(&ClassEnv) -> Result<ClassEnv>;

#[derive(Clone)]
struct ClassEnv {
    classes: HashMap<Id, Class>,
    defaults: Vec<Type>,
}

impl ClassEnv {
    // so we can call the hashmap the way you'd expect if we used a function
    // like the paper.
    fn classes(&self, id: Id) -> Option<&Class> {
        self.classes.get(id)
    }

    fn super_(&self, id: Id) -> &[Id] {
        &self
            .classes(id)
            .expect("super is partial in the paper")
            .supers
    }

    fn insts(&self, id: Id) -> &[Inst] {
        &self
            .classes(id)
            .expect("super is partial in the paper")
            .instances
    }

    fn modify(&self, i: Id, c: Class) -> ClassEnv {
        let mut new = self.clone();
        new.classes.insert(i, c);
        new
    }

    fn initial() -> Self {
        ClassEnv {
            classes: HashMap::default(),
            defaults: Vec::default(),
        }
    }

    // This is our name for `<:>` since I have no idea what to call that.
    //
    // Here it's fully applied, so we'll probably need to be careful translating
    // code that uses this.

    fn compose(&self, f: EnvTransformer, g: EnvTransformer) -> Result<ClassEnv> {
        g(&f(self)?)
    }

    fn add_class(&self, i: Id, is: &[Id]) -> Result<ClassEnv> {
        let mut new = self.clone();
        new.add_class_mut(i, is)?;
        Ok(new)
    }

    fn add_inst(&self, ps: &[Pred], p: &Pred) -> Result<ClassEnv> {
        let mut new = self.clone();
        new.add_inst_mut(ps, p)?;
        Ok(new)
    }

    fn add_prelude_classes(&self) -> Result<ClassEnv> {
        let mut new = self.clone();

        new.add_core_classes_mut()?;
        new.add_num_classes_mut()?;

        Ok(new)
    }

    fn example_insts(&self) -> Result<ClassEnv> {
        let mut new = self.add_prelude_classes()?;

        new.add_inst_mut(&[], &Pred::IsIn("Ord", prim::UNIT))?;
        new.add_inst_mut(&[], &Pred::IsIn("Ord", prim::CHARACTER))?;
        new.add_inst_mut(&[], &Pred::IsIn("Ord", prim::INT))?;
        new.add_inst_mut(
            &[
                Pred::IsIn("Ord", Type::Var(Tyvar("a", Kind::Star))),
                Pred::IsIn("Ord", Type::Var(Tyvar("b", Kind::Star))),
            ],
            &Pred::IsIn(
                "Ord",
                pair(
                    Type::Var(Tyvar("a", Kind::Star)),
                    Type::Var(Tyvar("b", Kind::Star)),
                ),
            ),
        )?;

        Ok(new)
    }

    // Cheating, don't tell Mr. Haskell!

    fn add_core_classes_mut(&mut self) -> Result<()> {
        self.add_class_mut("Eq", &[])?;
        self.add_class_mut("Ord", &["Eq"])?;
        self.add_class_mut("Show", &[])?;
        self.add_class_mut("Read", &[])?;
        self.add_class_mut("Bounded", &[])?;
        self.add_class_mut("Enum", &[])?;
        self.add_class_mut("Functor", &[])?;
        self.add_class_mut("Monad", &[])
    }

    fn add_num_classes_mut(&mut self) -> Result<()> {
        self.add_class_mut("Num", &["Eq", "Show"])?;
        self.add_class_mut("Real", &["Num", "Ord"])?;
        self.add_class_mut("Fractional", &["Num"])?;
        self.add_class_mut("Integral", &["Real", "Enum"])?;
        self.add_class_mut("RealFrac", &["Real", "Fractional"])?;
        self.add_class_mut("Floating", &["Fractional"])?;
        self.add_class_mut("RealFloat", &["ReadFrac, Floating"])
    }

    fn add_class_mut(&mut self, id: Id, supers: &[Id]) -> Result<()> {
        if let Some(class) = self.classes(id) {
            Err("class already defined")
        } else if supers.iter().any(|super_| self.classes(super_).is_none()) {
            Err("superclass not defined")
        } else {
            let class = Class::new(supers, &[]);
            self.classes.insert(id, class);
            Ok(())
        }
    }

    fn add_inst_mut(&mut self, ps: &[Pred], p: &Pred) -> Result<()> {
        let Pred::IsIn(id, _) = p;

        // we could skip this check if `insts` return a result
        if self.classes(id).is_none() {
            return Err("no class instance");
        }

        if self
            .insts(id)
            .iter()
            .map(Qual::consequence)
            .any(|q| overlap(p, q))
        {
            return Err("overlapping instance");
        }

        // There's some important other stuff to check listed at the bottom of
        // page 16, but the paper doesn't do it so I won't here.

        // We mutate the existing class definition to add our new instance.
        //
        // This adds it to the end instead of head, but since overlapping
        // instances aren't legal, we know it's unique so it doesn't matter.
        self.classes
            .get_mut(id)
            .expect("checked above")
            .instances
            .push(Qual::then(ps, p));

        Ok(())
    }
}

fn overlap(p: &Pred, q: &Pred) -> bool {
    mgu_pred(p, q).is_ok()
}

// Not sure I agree with footnote about `isJust` here. I didn't use it in favour
// of `is_none` over `not . defined`.
fn defined<T>(option: Option<T>) -> bool {
    option.is_some()
}

// ### 7.3 Entailment

impl ClassEnv {
    fn by_super(&self, p: &Pred) -> Vec<Pred> {
        let Pred::IsIn(i, t) = p;

        let mut buf: Vec<Pred> = self
            .super_(i)
            .iter()
            .flat_map(|i_| self.by_super(&Pred::IsIn(i_, t.clone())))
            .collect();

        buf.push(p.clone());

        buf
    }

    fn by_inst(&self, p: &Pred) -> Result<Vec<Pred>> {
        let Pred::IsIn(i, t) = p;

        let mut buf = Vec::new();

        for Qual::Then(ps, h) in self.insts(i) {
            let u = match_pred(h, p)?;

            for p in ps {
                buf.push(p.apply(&u));
            }
        }

        Ok(buf)
    }

    fn entail(&self, ps: &[Pred], p: &Pred) -> bool {
        // This `|| match` is in the Haskell too, and I really don't like it
        // there either. Weird choice.
        ps.iter()
            .map(|p| self.by_super(p))
            .any(|supers| supers.contains(p))
            || match self.by_inst(p) {
                Err(_) => false,
                Ok(qs) => qs.iter().all(|q| self.entail(ps, q)),
            }
    }
}

// ## 7.4 Context Reduction
