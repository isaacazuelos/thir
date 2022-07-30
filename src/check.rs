//! This is the core of the paper, written in mostly the same order as things
//! are presented there.
//!
//! As will be immediately clear, this is _not_ idiomatic Rust at all. The point
//! here is to get the algorithm in the paper working. I can worry about making
//! it idiomatic later.

#![allow(unused_variables)]

// ## Preliminaries

use std::collections::HashMap;

use crate::util::{append, intersection, lookup, minus, partition, union, zip_with, zip_with_try};

// For simplicity, we're not worrying too much about reducing shared references.

type Error = &'static str;

// kind of ties our hands at defining new ones, but it's nicer to avoid all the
// `.into` calls for now.
type Id = String;

fn enum_id(i: usize) -> Id {
    format!("v{i}")
}

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
    Gen(usize), // for Type Schemes, not used until section 8
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

    pub fn unit() -> Type {
        Type::Con(Tycon("()".into(), Kind::Star))
    }

    pub fn character() -> Type {
        Type::Con(Tycon("Char".into(), Kind::Star))
    }

    pub fn int() -> Type {
        Type::Con(Tycon("Int".into(), Kind::Star))
    }

    pub fn integer() -> Type {
        Type::Con(Tycon("Integer".into(), Kind::Star))
    }

    pub fn float() -> Type {
        Type::Con(Tycon("Float".into(), Kind::Star))
    }

    pub fn double() -> Type {
        Type::Con(Tycon("Double".into(), Kind::Star))
    }

    pub fn list() -> Type {
        Type::Con(Tycon("[]".into(), Kind::fun(Kind::Star, Kind::Star)))
    }

    pub fn arrow() -> Type {
        Type::Con(Tycon(
            "(->)".into(),
            Kind::fun(Kind::Star, Kind::fun(Kind::Star, Kind::Star)),
        ))
    }

    pub fn tuple_2() -> Type {
        Type::Con(Tycon(
            "(,)".into(),
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
            Type::Gen(_) => unimplemented!(),
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
            Type::Con(_) | Type::Gen(_) => unimplemented!(),
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
    fn then(pred: &[Pred], t: T) -> Qual<T> {
        Qual::Then(pred.into(), t)
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

impl Pred {
    fn is_in(id: impl Into<String>, t: Type) -> Pred {
        Pred::IsIn(id.into(), t)
    }
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
        Pred::IsIn(i.into(), t.apply(s))
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
        &["Eq".into()],
        // These are instances of the class
        &[
            // This is the `instance Ord _ where` part for unit, char, int.
            // Notice this isn't the implementation, just the type level stuff.
            Qual::then(&[], Pred::is_in("Ord", prim::unit())),
            Qual::then(&[], Pred::is_in("Ord", prim::character())),
            Qual::then(&[], Pred::is_in("Ord", prim::int())),
            // This one is `Ord a, Ord b => Ord (a, b)`
            Qual::then(
                &[
                    // Ord a constraint
                    Pred::is_in("Ord", Type::Var(Tyvar("a".into(), Kind::Star))),
                    // Ord b constraint
                    Pred::is_in("Ord", Type::Var(Tyvar("b".into(), Kind::Star))),
                ],
                // => Ord (a, b)
                Pred::IsIn(
                    "Ord".into(),
                    pair(
                        Type::Var(Tyvar("a".into(), Kind::Star)),
                        Type::Var(Tyvar("b".into(), Kind::Star)),
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
    fn classes(&self, id: &Id) -> Option<&Class> {
        self.classes.get(id)
    }

    fn super_(&self, id: &Id) -> &[Id] {
        &self
            .classes(id)
            .expect("super is partial in the paper")
            .supers
    }

    fn insts(&self, id: &Id) -> &[Inst] {
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

        let is: Vec<&str> = is.iter().map(AsRef::as_ref).collect(); // buh
        new.add_class_mut(i, &is)?;
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

        new.add_inst_mut(&[], &Pred::is_in("Ord", prim::unit()))?;
        new.add_inst_mut(&[], &Pred::is_in("Ord", prim::character()))?;
        new.add_inst_mut(&[], &Pred::is_in("Ord", prim::int()))?;
        new.add_inst_mut(
            &[
                Pred::is_in("Ord", Type::Var(Tyvar("a".into(), Kind::Star))),
                Pred::is_in("Ord", Type::Var(Tyvar("b".into(), Kind::Star))),
            ],
            &Pred::IsIn(
                "Ord".into(),
                pair(
                    Type::Var(Tyvar("a".into(), Kind::Star)),
                    Type::Var(Tyvar("b".into(), Kind::Star)),
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

    fn add_class_mut(&mut self, id: impl Into<Id>, supers: &[&str]) -> Result<()> {
        let id = id.into();
        let supers: Vec<Id> = supers.iter().map(|s| String::from(*s)).collect();
        if let Some(class) = self.classes(&id) {
            Err("class already defined")
        } else if supers.iter().any(|super_| self.classes(super_).is_none()) {
            Err("superclass not defined")
        } else {
            let supers: Vec<_> = supers.iter().map(|s| s.into()).collect();
            let class = Class::new(&supers, &[]);
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
            .push(Qual::then(ps, p.clone()));

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
            .flat_map(|i_| self.by_super(&Pred::IsIn(i_.clone(), t.clone())))
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

// ### 7.4 Context Reduction

impl Pred {
    fn in_hfn(&self) -> bool {
        let Pred::IsIn(c, t) = self;

        fn hnf(t: &Type) -> bool {
            match t {
                Type::Var(_) => true,
                Type::Con(_) => false,
                Type::Ap(t, _) => hnf(t),
                Type::Gen(_) => todo!(),
            }
        }

        hnf(t)
    }
}

impl ClassEnv {
    fn to_hnfs(&self, ps: &[Pred]) -> Result<Vec<Pred>> {
        let mut buf = Vec::new();

        for pss in ps.iter().map(|p| self.to_hnf(p)) {
            buf.extend(pss?)
        }

        Ok(buf)
    }

    fn to_hnf(&self, p: &Pred) -> Result<Vec<Pred>> {
        if p.in_hfn() {
            Ok(vec![p.clone()])
        } else {
            match self.by_inst(p) {
                Err(_) => Err("context reduction"),
                Ok(ps) => self.to_hnfs(&ps),
            }
        }
    }

    fn simplify(&self, ps: &[Pred]) -> Vec<Pred> {
        // Here's one of the few places where the iterative solution is clearer
        // to me, it's how the text of the paper describes things too.

        let mut rs = Vec::new();

        for p in ps {
            if !self.entail(&union(&rs, ps), p) {
                rs.push(p.clone());
            }
        }

        rs
    }

    fn reduce(&self, ps: &[Pred]) -> Result<Vec<Pred>> {
        // I love `?` so much. It works so well for `do` code in error-handling
        // monads, since it's _almost_ the same thing.
        let qs = self.to_hnfs(ps)?;
        Ok(self.simplify(&qs))
    }
}

// I think if we use impl Iterator everywhere instead of `&[]` or `Vec`, this
// could be pretty OK. So many clones and Vec allocations. But Haskell's lists
// are lazy, and behave a lot like iterators -- WHNF of `:` is `next` returning
// `Some`, and `[]` means `None`. The iterators mutate, so we don't have a rhs
// to `:`. I wrote a blog draft about this once -- it's actually super cool how
// it works out if you do the WHNF eval by hand.

// ## 8. Type Schemes

#[derive(Debug, Clone, PartialEq)]
enum Scheme {
    ForAll(Vec<Kind>, Qual<Type>),
}

// Okay, I actually need to stop, it's getting late.

impl Types for Scheme {
    fn apply(&self, s: &Subst) -> Self {
        let Scheme::ForAll(ks, qt) = self.clone();
        Scheme::ForAll(ks, qt.apply(s))
    }

    fn tv(&self) -> Vec<Tyvar> {
        let Scheme::ForAll(ks, qt) = self.clone();
        qt.tv()
    }
}

fn quantify(vs: &[Tyvar], qt: Qual<Type>) -> Scheme {
    let vs_: Vec<Tyvar> = qt.tv().iter().filter(|v| vs.contains(v)).cloned().collect();
    let ks = vs_.iter().map(|v| v.kind().clone()).collect();
    let s = vs_
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), Type::Gen(i)))
        .collect();

    Scheme::ForAll(ks, qt.apply(&s))
}

impl From<Type> for Scheme {
    fn from(t: Type) -> Self {
        Scheme::ForAll(vec![], Qual::Then(vec![], t))
    }
}

// ## 9. Assumptions

// I'm not going to try and come up with a name for `:>:`

#[derive(Debug, Clone)]
struct Assump(Id, Scheme);

impl Types for Assump {
    fn apply(&self, s: &Subst) -> Self {
        let Assump(i, sc) = self;
        Assump(i.clone(), sc.apply(s))
    }

    fn tv(&self) -> Vec<Tyvar> {
        let Assump(i, sc) = self;
        sc.tv()
    }
}

fn find(i: &Id, assumptions: &[Assump]) -> Result<Scheme> {
    for Assump(i_, sc) in assumptions {
        if i == i_ {
            return Ok(sc.clone());
        }
    }
    Err("unbound identifier")
}

// ## 10. A Type Inference Monad
//
// Well, not really.

struct TI {
    substitutions: Subst,
    next_var: usize,
}

impl TI {
    fn get_subst(&self) -> &Subst {
        &self.substitutions
    }

    fn unify(&mut self, t1: &Type, t2: &Type) -> Result<()> {
        let s = self.get_subst();
        let u = mgu(&t1.apply(s), &t2.apply(s))?;
        self.ext_subst(&u);
        Ok(())
    }

    fn ext_subst(&mut self, s: &Subst) {
        // We should _definitely_ look at the definition of `at_at` and unpack
        // things a bit here.
        self.substitutions = at_at(&self.substitutions, s);
    }

    fn new_type_var(&mut self, k: Kind) -> Type {
        let i = self.next_var;
        self.next_var += 1;
        Type::Var(Tyvar(enum_id(i), k))
    }

    fn fresh_inst(&mut self, s: &Scheme) -> Qual<Type> {
        let Scheme::ForAll(ks, qt) = s;

        let ts: Vec<_> = ks.iter().map(|k| self.new_type_var(k.clone())).collect();

        qt.inst(&ts)
    }
}

trait Instantiate {
    fn inst(&self, ts: &[Type]) -> Self;
}

impl Instantiate for Type {
    fn inst(&self, ts: &[Type]) -> Type {
        match self {
            Type::Ap(l, r) => l.inst(ts).app(r.inst(ts)),
            Type::Gen(n) => ts[*n].clone(),
            t => t.clone(),
        }
    }
}

impl Instantiate for Vec<Type> {
    fn inst(&self, ts: &[Type]) -> Vec<Type> {
        self.iter().map(|t| t.inst(ts)).collect()
    }
}

impl<T: Instantiate> Instantiate for Qual<T> {
    fn inst(&self, ts: &[Type]) -> Qual<T> {
        todo!()
    }
}

impl Instantiate for Pred {
    fn inst(&self, ts: &[Type]) -> Pred {
        todo!()
    }
}

// ## 11. Type Inference
//
// Here we go!

// I will not be using this much, outside of as a reference against the text.
//
// I _suspect_ we can move these other args into our TI state, but we'd need to
// confirm that changes to them don't 'backtrack' -- which might happen as you
// traverse scopes. I'm pretty sure you can't have a scoped instance, so the
// `ClassEnv` should be fine. I'm not sure I understand exactly how Assump is
// used, especially around `TI::pat`, so I'll stick to args for now.
type Infer<E, T> = dyn Fn(&mut TI, &ClassEnv, &[Assump], E) -> Result<(Vec<Pred>, T)>;

#[derive(Debug, Clone, PartialEq)]
enum Literal {
    Int(i64),
    Char(char),
    Rat(f64), // I know, but close enough.
    Str(String),
}

impl TI {
    fn lit(&mut self, l: &Literal) -> (Vec<Pred>, Type) {
        match l {
            Literal::Char(_) => (vec![], prim::character()),
            Literal::Int(_) => {
                let v = self.new_type_var(Kind::Star);
                (vec![Pred::IsIn("Num".into(), v.clone())], v)
            }
            Literal::Str(_) => (vec![], prim::string()),
            Literal::Rat(_) => {
                let v = self.new_type_var(Kind::Star);
                (vec![Pred::IsIn("Fractional".into(), v.clone())], v)
            }
        }
    }
}

// ### 11.2 Patterns

#[derive(Debug, Clone)]
enum Pat {
    Var(Id),               // `a`
    Wildcard,              // `_`
    As(Id, Box<Pat>),      // `id@pat`
    Lit(Literal),          // `1`
    Npk(Id, usize),        // `n + k` patterns, which are a sin
    Con(Assump, Vec<Pat>), // `Constructor(**pats)`, not sure what Assump is
}

impl TI {
    // This might need to be changed to return a Result, it's weird it not.
    fn pat(&mut self, p: &Pat) -> (Vec<Pred>, Vec<Assump>, Type) {
        match p {
            Pat::Var(i) => {
                let v = self.new_type_var(Kind::Star);
                (vec![], vec![Assump(i.clone(), v.clone().into())], v)
            }
            Pat::Wildcard => {
                let v = self.new_type_var(Kind::Star);
                (vec![], vec![], v)
            }
            Pat::As(i, pat) => {
                let (ps, mut as_, t) = self.pat(pat);
                as_.push(Assump(i.clone(), t.clone().into()));
                (ps, as_, t)
            }
            Pat::Lit(l) => {
                let (ps, t) = self.lit(l);
                (ps, vec![], t)
            }
            Pat::Npk(i, k) => {
                let t = self.new_type_var(Kind::Star);
                (
                    vec![Pred::IsIn("Integeral".into(), t.clone())],
                    vec![Assump(i.clone(), t.clone().into())],
                    t,
                )
            }
            Pat::Con(a, pats) => {
                let Assump(i, sc) = a;
                let (ps, as_, ts) = self.pats(pats);
                let t_ = self.new_type_var(Kind::Star);
                let Qual::Then(qs, t) = self.fresh_inst(sc);

                let folded = ts.iter().cloned().fold(t_.clone(), fun);
                self.unify(&t, &folded)
                    // We almost certainly need Result here...
                    .expect("Monad Transformer? I hardly know 'er");
                (append(ps, qs), as_, t_)
            }
        }
    }

    fn pats(&mut self, pats: &[Pat]) -> (Vec<Pred>, Vec<Assump>, Vec<Type>) {
        let mut ps = vec![];
        let mut as_ = vec![];
        let mut ts = vec![];

        for (p, a, t) in pats.iter().map(|pat| self.pat(pat)) {
            ps.extend(p);
            as_.extend(a);
            ts.push(t);
        }

        (ps, as_, ts)
    }
}

// ### 11.3 Expressions

#[derive(Debug, Clone)]
enum Expr {
    Var(Id),
    Lit(Literal),
    Const(Assump),
    Ap(Box<Expr>, Box<Expr>),
    Let(BindGroup, Box<Expr>),
}

impl TI {
    fn expr(&mut self, ce: &ClassEnv, as_: &[Assump], e: &Expr) -> Result<(Vec<Pred>, Type)> {
        match e {
            Expr::Var(i) => {
                let sc = find(i, as_)?;
                let Qual::Then(ps, t) = self.fresh_inst(&sc);
                Ok((ps, t))
            }
            Expr::Const(Assump(i, sc)) => {
                let Qual::Then(ps, t) = self.fresh_inst(sc);
                Ok((ps, t))
            }
            Expr::Lit(l) => {
                let (ps, t) = self.lit(l);
                Ok((ps, t))
            }
            Expr::Ap(e, f) => {
                let (ps, te) = self.expr(ce, as_, e)?;
                let (qs, tf) = self.expr(ce, as_, f)?;
                let t = self.new_type_var(Kind::Star);
                self.unify(&fun(tf, t), &te)?;
                Ok((append(ps, qs), te))
            }
            Expr::Let(bg, e) => {
                let (ps, as__) = self.bind_group(ce, as_, bg)?;
                let (qs, t) = self.expr(ce, &append(as__, as_.into()), e)?;
                Ok((append(ps, qs), t))
            }
        }
    }
}

// ### 11.4 Alternatives

type Alt = (Vec<Pat>, Expr);

impl TI {
    fn alt(&mut self, ce: &ClassEnv, as_: &[Assump], (pats, e): &Alt) -> Result<(Vec<Pred>, Type)> {
        let (ps, as__, ts) = self.pats(pats);
        let (qs, t) = self.expr(ce, &append(as__, as_.into()), e)?;

        let folded = ts.iter().cloned().fold(t, fun);
        Ok((append(ps, qs), folded))
    }

    // Now that's a signature!
    fn alts(&mut self, ce: &ClassEnv, as_: &[Assump], alts: &[Alt], t: Type) -> Result<Vec<Pred>> {
        let psts = alts
            .iter()
            .map(|a| self.alt(ce, as_, a))
            .collect::<Result<Vec<_>>>()?;

        for t2 in psts.iter().map(|t| &t.1) {
            self.unify(&t, t2)?
        }

        Ok(psts.into_iter().flat_map(|(p, _)| p).collect())
    }
}

// ### 11.5

fn split(ce: &ClassEnv, fs: &[Tyvar], gs: &[Tyvar], ps: &[Pred]) -> Result<(Vec<Pred>, Vec<Pred>)> {
    let ps_ = ce.reduce(ps)?;
    let (ds, rs) = partition(|p| p.tv().iter().all(|t| fs.contains(t)), ps_);
    let rs_ = ce.defaulted_preds(append(fs.to_vec(), gs.to_vec()), &rs)?;
    Ok((ds, minus(rs, &rs_)))
}

// #### 11.5.1

type Ambiguity = (Tyvar, Vec<Pred>);

const NUM_CLASSES: &[&str] = &[
    "Num",
    "Integral",
    "Floating",
    "Fractional",
    "Real",
    "RealFloat",
    "RealFrac",
];

fn num_classes() -> Vec<Id> {
    NUM_CLASSES.iter().map(|s| (*s).into()).collect()
}

const STD_CLASSES: &[&str] = &[
    "Eq",
    "Ord",
    "Show",
    "Read",
    "Bounded",
    "Enum",
    "Ix",
    "Functor",
    "Monad",
    "MonadPlus",
];

fn std_classes() -> Vec<Id> {
    [STD_CLASSES, NUM_CLASSES]
        .iter()
        .flat_map(|i| i.iter())
        .map(|s| s.to_string())
        .collect()
}

impl ClassEnv {
    fn candidates(&self, (v, qs): &Ambiguity) -> Vec<Type> {
        let mut is = vec![];
        let mut ts = vec![];
        for Pred::IsIn(i, t) in qs.iter() {
            is.push(i);
            ts.push(t);
        }

        if !ts.iter().all(|t| t == &&Type::Var(v.clone())) {
            return vec![];
        }

        if !is.iter().any(|i| num_classes().contains(i)) {
            return vec![];
        }

        if !is.iter().all(|i| std_classes().contains(i)) {
            return vec![];
        }

        let mut t_s = vec![];

        for t_ in &self.defaults {
            if !is
                .iter()
                .map(|i| Pred::IsIn(i.to_string(), t_.clone()))
                .all(|q| self.entail(&[], &q))
            {
                return vec![];
            } else {
                t_s.push(t_.clone());
            }
        }

        // there's no way I got this right

        t_s
    }

    fn with_defaults<F, T>(&self, f: F, vs: &[Tyvar], ps: &[Pred]) -> Result<T>
    where
        F: Fn(&[Ambiguity], &[Type]) -> T,
    {
        todo!()
    }
}

impl ClassEnv {
    fn defaulted_preds(&self, vs: Vec<Tyvar>, ps: &[Pred]) -> Result<Vec<Pred>> {
        self.with_defaults(
            |vps, ts| vps.iter().flat_map(|a| a.1.clone()).collect(),
            &vs,
            ps,
        )
    }

    fn default_subst(&self, vs: Vec<Tyvar>, ps: &[Pred]) -> Result<Subst> {
        self.with_defaults(
            |vps, ts| {
                vps.iter()
                    .map(|(fst, _)| fst.clone())
                    .zip(ts.iter().cloned())
                    .collect()
            },
            &vs,
            ps,
        )
    }
}
// ### 11.6 Bind Groups

type Expl = (Id, Scheme, Vec<Alt>);

impl TI {
    fn expl(&mut self, ce: &ClassEnv, as_: &[Assump], (i, sc, alts): &Expl) -> Result<Vec<Pred>> {
        let Qual::Then(qs, t) = self.fresh_inst(sc);
        let ps = self.alts(ce, as_, alts, t.clone())?;
        let s = self.get_subst();

        let qs_ = qs.apply(s);
        let t_ = t.apply(s);
        let fs = as_.to_vec().apply(s).tv();
        let gs = minus(t_.tv(), &fs);
        let sc_ = quantify(&gs, Qual::Then(qs_.clone(), t_));
        let ps_ = ps
            .apply(s)
            .iter()
            .filter(|p| !ce.entail(&qs_, p))
            .cloned()
            .collect::<Vec<Pred>>();

        let (ds, rs) = split(ce, &fs, &gs, &ps_)?;

        if sc != &sc_ {
            Err("signature to general")
        } else if !rs.is_empty() {
            Err("context too weak")
        } else {
            Ok(ds)
        }
    }
}

type Impl = (Id, Vec<Alt>);

fn restricted(bs: &[Impl]) -> bool {
    bs.iter()
        .any(|(i, alts)| alts.iter().any(|alt| alt.0.is_empty()))
}

impl TI {
    fn impls(
        &mut self,
        ce: &ClassEnv,
        as_: &[Assump],
        bs: &[Impl],
    ) -> Result<(Vec<Pred>, Vec<Assump>)> {
        let ts = bs
            .iter()
            .map(|_| self.new_type_var(Kind::Star))
            .collect::<Vec<_>>();

        let is: Vec<Id> = bs.iter().map(|b| b.0.clone()).collect();
        let scs: Vec<Scheme> = ts.iter().cloned().map(Scheme::from).collect();
        let as__ = append(zip_with(Assump, is.clone(), scs), as_.to_vec());
        let altss = bs.iter().map(|b| b.1.clone()).collect();

        let pss = zip_with_try(|alts, t| self.alts(ce, as_, &alts, t), altss, ts.clone())?;
        let s = self.get_subst();

        let ps_: Vec<Pred> = pss.iter().flatten().map(|p| p.apply(s)).collect();
        let ts_: Vec<Type> = ts.apply(s);
        let fs: Vec<Tyvar> = as_.to_vec().apply(s).tv();
        let vss: Vec<Vec<Tyvar>> = ts_.iter().map(Types::tv).collect();
        let gs = minus(
            vss.iter()
                .cloned()
                .reduce(|l, r| intersection(&l, &r))
                .unwrap(),
            &fs,
        );

        let (ds, rs) = split(
            ce,
            &fs,
            &vss.into_iter().reduce(|a, b| intersection(&a, &b)).unwrap(),
            &ps_,
        )?;

        if restricted(bs) {
            let gs_ = minus(gs, &rs.tv());
            let scs_ = ts_
                .iter()
                .map(|t| quantify(&gs_, Qual::Then(rs.clone(), t.clone())))
                .collect();
            Ok((append(ds, rs), zip_with(Assump, is, scs_)))
        } else {
            let scs_ = ts_
                .iter()
                .map(|t| quantify(&gs, Qual::Then(rs.clone(), t.clone())))
                .collect();
            Ok((ds, zip_with(Assump, is, scs_)))
        }
    }
}

type BindGroup = (Vec<Expl>, Vec<Vec<Impl>>);
type Program = Vec<BindGroup>;

impl TI {
    fn bind_group(
        &mut self,
        ce: &ClassEnv,
        as_: &[Assump],
        (es, iss): &BindGroup,
    ) -> Result<(Vec<Pred>, Vec<Assump>)> {
        let as__: Vec<Assump> = es
            .iter()
            .map(|(v, sc, alts)| Assump(v.clone(), sc.clone()))
            .collect();

        let (ps, all_as): (Vec<Pred>, Vec<Assump>) = {
            // inlining tiSeq in the paper because its' easier.
            let mut ps = vec![];
            let mut all_as = vec![];

            for is in iss {
                let (p, a) = self.impls(ce, &all_as, is)?;
                ps.extend(p);
                all_as.extend(a);
            }

            (ps, all_as)
        };

        let qss = {
            // had to work with the Results in iterators.
            let mut buf = vec![];

            for e in es {
                let x = self.expl(ce, &all_as, e)?;
                buf.extend(x);
            }

            buf
        };

        Ok((append(ps, qss), all_as))
    }

    fn program(&mut self, ce: &ClassEnv, as_: &[Assump], bgs: Program) -> Result<Vec<Assump>> {
        let (ps, as__) = {
            let mut ps = vec![];
            let mut all_as = vec![];

            for bg in bgs {
                let (p, a) = self.bind_group(ce, as_, &bg)?;
                all_as.extend(a);
                ps.extend(p);
            }

            (ps, all_as)
        };

        let s = self.get_subst();

        let rs = ce.reduce(&ps.apply(s))?;
        let s_ = ce.default_subst(vec![], &rs)?;

        Ok(as__.apply(&at_at(&s_, s)))
    }
}
