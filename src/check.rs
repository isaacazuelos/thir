//! This is the core of the paper, written in mostly the same order as things
//! are presented there.
//!
//! As will be immediately clear, this is _not_ idiomatic Rust at all. The point
//! here is to get the algorithm in the paper working. I can worry about making
//! it idiomatic later.

// There are a lot.
#![allow(unused_variables)]

// ## Preliminaries

use std::{collections::HashMap, rc::Rc};

use crate::util::{append, intersection, minus, partition, union, zip_with, zip_with_try};

type Error = String;

// These are still not really interned, so we'll probably end up with duplicates
// of things like "Eq". Still better than `String` everywhere though.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Id {
    name: Rc<String>,
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl From<&str> for Id {
    fn from(s: &str) -> Self {
        Id {
            name: Rc::new(String::from(s)),
        }
    }
}

impl From<usize> for Id {
    fn from(i: usize) -> Self {
        Id {
            name: Rc::new(format!("v{i}")),
        }
    }
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
    Function(Box<Kind>, Box<Kind>),
}

impl Kind {
    // Just for convenience, same as the [`Kind::Fun`] constructor, but it does
    // the boxing for us.
    fn function(lhs: Kind, rhs: Kind) -> Kind {
        Kind::Function(Box::new(lhs), Box::new(rhs))
    }
}

// ## 4. Types

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
    fn apply_to(&self, b: Type) -> Type {
        Type::Applied(Box::new(self.clone()), Box::new(b))
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
    fn new(id: impl Into<Id>, kind: Kind) -> Self {
        TypeVariable {
            id: id.into(),
            kind,
        }
    }

    fn maps_to(&self, t: &Type) -> Vec<Substitution> {
        vec![Substitution {
            from: self.clone(),
            to: t.clone(),
        }]
    }

    fn find_type_in(&self, substitutions: &[Substitution]) -> Option<Type> {
        for s in substitutions {
            if &s.from == self {
                return Some(s.to.clone());
            }
        }

        None
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypeConstructor {
    id: Id,
    kind: Kind,
}

impl TypeConstructor {
    fn new(id: impl Into<Id>, kind: Kind) -> Self {
        TypeConstructor {
            id: id.into(),
            kind,
        }
    }
}

// Putting these in a module to namespace them instead of using the prefix
// naming scheme used in the paper. A name like `prim::unit` is `tUnit` in the
// paper.
//
// The types with arrow kinds are functions since I can't make the boxes for the
// arrow kinds in while `const`.
//
// TODO: lazy_static?
mod prim {
    use super::*;

    pub fn unit() -> Type {
        Type::Constructor(TypeConstructor::new("()", Kind::Star))
    }

    pub fn character() -> Type {
        Type::Constructor(TypeConstructor::new("Char", Kind::Star))
    }

    pub fn int() -> Type {
        Type::Constructor(TypeConstructor::new("Int", Kind::Star))
    }

    pub fn integer() -> Type {
        Type::Constructor(TypeConstructor::new("Integer", Kind::Star))
    }

    pub fn float() -> Type {
        Type::Constructor(TypeConstructor::new("Float", Kind::Star))
    }

    pub fn double() -> Type {
        Type::Constructor(TypeConstructor::new("Double", Kind::Star))
    }

    pub fn list() -> Type {
        Type::Constructor(TypeConstructor::new(
            "[]",
            Kind::function(Kind::Star, Kind::Star),
        ))
    }

    pub fn arrow() -> Type {
        Type::Constructor(TypeConstructor::new(
            "(->)",
            Kind::function(Kind::Star, Kind::function(Kind::Star, Kind::Star)),
        ))
    }

    pub fn tuple_2() -> Type {
        Type::Constructor(TypeConstructor::new(
            "(,)",
            Kind::function(Kind::Star, Kind::function(Kind::Star, Kind::Star)),
        ))
    }

    pub fn string() -> Type {
        make_list(character())
    }

    // This is the function the paper calls `fn`. We could use `r#fn` but this is
    // nicer, imho. We can't exactly make it infix in Rust either.

    pub fn make_function(a: Type, b: Type) -> Type {
        prim::arrow().apply_to(a).apply_to(b)
    }

    pub fn make_list(t: Type) -> Type {
        prim::list().apply_to(t)
    }

    pub fn make_pair(a: Type, b: Type) -> Type {
        prim::tuple_2().apply_to(a).apply_to(b)
    }
}

trait HasKind {
    fn kind(&self) -> &Kind;
}

impl HasKind for TypeVariable {
    fn kind(&self) -> &Kind {
        &self.kind
    }
}

impl HasKind for TypeConstructor {
    fn kind(&self) -> &Kind {
        &self.kind
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

// 5. Substitutions
//
// This [`Subst`] should be based on Iterators, with adapters, to better emulate
// the Haskell lists. We could also use the same IDs here to make things
// cheaper.

#[derive(Debug, Clone, PartialEq)]
struct Substitution {
    from: TypeVariable,
    to: Type,
}

impl Substitution {
    fn new(from: TypeVariable, to: Type) -> Substitution {
        Substitution { from, to }
    }
}

fn at_at(s1: &[Substitution], s2: &[Substitution]) -> Vec<Substitution> {
    let mut substitutions = Vec::new();

    // [(u, apply s1 t) | (u, t) <- s2]
    for s in s2.iter() {
        substitutions.push(Substitution {
            from: s.from.clone(),
            to: s.to.apply(s1),
        });
    }

    // ++ s1
    for s in s1.iter() {
        substitutions.push(s.clone());
    }

    substitutions
}

// Since we don't have fancy user-defined operators to use in Rust, I'll have to
// use regular functions with names for instead when operators are defined.
//
// None of the operators in [`std::ops`] really looks like `+->`. Probably `>>`
// is the closes option, but that seems unwise.

/// I had to swap the argument orders here for `self` to work.
trait Types {
    fn apply(&self, s: &[Substitution]) -> Self;
    fn type_variables(&self) -> Vec<TypeVariable>;
}

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

// Starting to really appreciate Haskell's syntax here.

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

// We can't quite translate this over any `Monad m`.
//
// I'm assuming this is going to be over `Result` for now. We might need to make
// a few versions of this manually, if we do use it over different `m`.

// Here `merge` is `@@`, but it checks that the order of the arguments won't
// matter.
fn merge(s1: &[Substitution], s2: &[Substitution]) -> Result<Vec<Substitution>> {
    let s1_vars: Vec<_> = s1.iter().map(|s| s.from.clone()).collect();
    let s2_vars: Vec<_> = s2.iter().map(|s| s.from.clone()).collect();

    for v in intersection(&s1_vars, &s2_vars) {
        if Type::Variable(v.clone()).apply(s1) != Type::Variable(v).apply(s2) {
            return Err("merge fails".into());
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
fn most_general_unifier(t1: &Type, t2: &Type) -> Result<Vec<Substitution>> {
    match (t1, t2) {
        (Type::Applied(l, r), Type::Applied(l_, r_)) => {
            // Cool to see `?` work as a monad here -- it doesn't always!
            let s1 = most_general_unifier(l, l_)?;
            let s2 = most_general_unifier(&r.apply(&s1), &r_.apply(&s1))?;
            Ok(at_at(&s2, &s1))
        }
        (Type::Variable(u), t) | (t, Type::Variable(u)) => var_bind(u, t),
        (Type::Constructor(t1), Type::Constructor(t2)) if t1 == t2 => Ok(Vec::default()),
        _ => Err(format!("types do not unify: {:?}, {:?}", t1, t2)),
    }
}

fn var_bind(u: &TypeVariable, t: &Type) -> Result<Vec<Substitution>> {
    if matches!(t, Type::Variable(t_) if t_ == u) {
        Ok(Vec::default())
    } else if t.type_variables().contains(u) {
        Err("occurs check fails".into())
    } else if u.kind() != t.kind() {
        Err("kinds do not match".into())
    } else {
        Ok(u.maps_to(t))
    }
}

fn match_(t1: &Type, t2: &Type) -> Result<Vec<Substitution>> {
    match (t1, t2) {
        (Type::Applied(l, r), Type::Applied(l_, r_)) => {
            let sl = match_(l, l_)?;
            let sr = match_(r, r_)?;
            merge(&sl, &sr)
        }
        (Type::Variable(u), t) if u.kind() == t.kind() => Ok(u.maps_to(t)),
        (Type::Constructor(tc1), Type::Constructor(tc2)) if tc1 == tc2 => Ok(Vec::default()),
        (t1, t2) => Err(format!("types do not match: {:?}, {:?}", t1, t2)),
    }
}

// ## 7. Type Classes, Predicates and Qualified Types

// No code in this part! I should check out that Wadler and Blott (1989) paper.

// ### 7.1. Basic Definition

// To better match Haskell's `data Type = Constructor _`, I'm using enums here
// with one variant like `enum Type { Constructor(_) }`. We could just use tuple
// structs, but this lets us name the constructors to better match the paper.

#[derive(Debug, Clone, PartialEq)]
pub enum Qualified<T> {
    // This is the `:=>` constructor in the paper.
    //
    // In the final assumptions produced, it's the trait constraints. The stuff
    // in the `where` clauses for Rust.
    Then(Vec<Predicate>, T),
}

impl<T: Clone> Qualified<T> {
    fn then(pred: &[Predicate], t: T) -> Qualified<T> {
        Qualified::Then(pred.into(), t)
    }

    fn consequence(&self) -> &T {
        let Qualified::Then(_, q) = self;
        q
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    IsIn(Id, Type),
}

impl Predicate {
    fn is_in(id: impl Into<Id>, t: Type) -> Predicate {
        Predicate::IsIn(id.into(), t)
    }
}

impl<T> Types for Qualified<T>
where
    T: Types,
{
    fn apply(&self, s: &[Substitution]) -> Self {
        let Qualified::Then(ps, t) = self;
        Qualified::Then(ps.apply(s), t.apply(s))
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        let Qualified::Then(ps, t) = self;
        union(&ps.type_variables(), &t.type_variables())
    }
}

impl Types for Predicate {
    fn apply(&self, s: &[Substitution]) -> Self {
        let Predicate::IsIn(i, t) = self;
        Predicate::IsIn(i.clone(), t.apply(s))
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        let Predicate::IsIn(i, t) = self;
        t.type_variables()
    }
}

fn most_general_unifier_predicate(a: &Predicate, b: &Predicate) -> Result<Vec<Substitution>> {
    lift(most_general_unifier, a, b)
}

fn match_predicate(a: &Predicate, b: &Predicate) -> Result<Vec<Substitution>> {
    lift(match_, a, b)
}

// Here come the primes. I'll use `_` in place of `'` where I can.

fn lift<M>(m: M, a: &Predicate, b: &Predicate) -> Result<Vec<Substitution>>
where
    M: Fn(&Type, &Type) -> Result<Vec<Substitution>>,
{
    let Predicate::IsIn(i, t) = a;
    let Predicate::IsIn(i_, t_) = b;

    if i == i_ {
        m(t, t_)
    } else {
        Err("classes differ".into())
    }
}

// I couldn't keep things straight as a tuple, so I'm naming things.
#[derive(Debug, Clone, PartialEq)]
struct TypeClass {
    super_classes: Vec<Id>,
    instances: Vec<Instance>,
}

impl TypeClass {
    // Keeps the same arg order as the tuple, so we can use it as a constructor
    fn new(supers: &[Id], instances: &[Instance]) -> Self {
        TypeClass {
            super_classes: supers.into(),
            instances: instances.as_ref().to_vec(),
        }
    }
}

type Instance = Qualified<Predicate>;

fn ord_example() -> TypeClass {
    TypeClass::new(
        // This part tells us that Eq is a 'superclass' of Ord,
        // it's the `class Eq => Ord` part.
        &["Eq".into()],
        // These are instances of the class
        &[
            // This is the `instance Ord _ where` part for unit, char, int.
            // Notice this isn't the implementation, just the type level stuff.
            Qualified::then(&[], Predicate::is_in("Ord", prim::unit())),
            Qualified::then(&[], Predicate::is_in("Ord", prim::character())),
            Qualified::then(&[], Predicate::is_in("Ord", prim::int())),
            // This one is `Ord a, Ord b => Ord (a, b)`
            Qualified::then(
                &[
                    // Ord a constraint
                    Predicate::is_in("Ord", Type::Variable(TypeVariable::new("a", Kind::Star))),
                    // Ord b constraint
                    Predicate::is_in("Ord", Type::Variable(TypeVariable::new("b", Kind::Star))),
                ],
                // => Ord (a, b)
                Predicate::IsIn(
                    "Ord".into(),
                    prim::make_pair(
                        Type::Variable(TypeVariable::new("a", Kind::Star)),
                        Type::Variable(TypeVariable::new("b", Kind::Star)),
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

type EnvironmentTransformer = fn(&TypeClassEnvironment) -> Result<TypeClassEnvironment>;

#[derive(Clone, Default, Debug)]
pub struct TypeClassEnvironment {
    classes: HashMap<Id, TypeClass>,
    defaults: Vec<Type>,
}

impl TypeClassEnvironment {
    /// Get a type class by [`Id`].
    fn get(&self, id: &Id) -> Option<&TypeClass> {
        self.classes.get(id)
    }

    /// Get all the super classes of a type class by [`Id`].
    fn super_classes_of(&self, id: &Id) -> &[Id] {
        &self
            .get(id)
            .expect("super is partial in the paper")
            .super_classes
    }

    /// Get the [`Instance`]s of a type class by [`Id`].
    fn instances_of(&self, id: &Id) -> &[Instance] {
        &self
            .get(id)
            .expect("insts is partial in the paper")
            .instances
    }

    fn add_type_class(&mut self, id: impl Into<Id>, supers: &[Id]) -> Result<()> {
        let id = id.into();

        if let Some(class) = self.get(&id) {
            return Err(format!("class already defined: {id}"));
        }

        for superclass in supers {
            if !self.classes.contains_key(superclass) {
                return Err(format!("superclass {superclass} not defined for {id}"));
            }
        }

        let class = TypeClass::new(&supers, &[]);
        self.classes.insert(id, class);
        Ok(())
    }

    fn add_instance_mut(&mut self, ps: &[Predicate], p: &Predicate) -> Result<()> {
        let Predicate::IsIn(id, _) = p;

        // we could skip this check if `insts` return a result
        if self.get(id).is_none() {
            return Err(format!("no class instance for {id}"));
        }

        if self
            .instances_of(id)
            .iter()
            .map(Qualified::consequence)
            .any(|q| overlap(p, q))
        {
            return Err(format!("overlapping instances"));
        }

        // There's some important other stuff to check listed at the bottom of
        // page 16, but the paper doesn't do it so I won't here.

        self.classes
            .get_mut(id)
            .unwrap()
            .instances
            .push(Qualified::then(ps, p.clone()));

        Ok(())
    }
}

// These methods are used to create a loaded environment, which we can use for testing.
impl TypeClassEnvironment {
    fn example_instances() -> TypeClassEnvironment {
        let mut new = TypeClassEnvironment::default();

        new.add_prelude_classes();

        new.add_instance_mut(&[], &Predicate::is_in("Ord", prim::unit()))
            .unwrap();
        new.add_instance_mut(&[], &Predicate::is_in("Ord", prim::character()))
            .unwrap();
        new.add_instance_mut(&[], &Predicate::is_in("Ord", prim::int()))
            .unwrap();
        new.add_instance_mut(
            &[
                Predicate::is_in("Ord", Type::Variable(TypeVariable::new("a", Kind::Star))),
                Predicate::is_in("Ord", Type::Variable(TypeVariable::new("b", Kind::Star))),
            ],
            &Predicate::IsIn(
                "Ord".into(),
                prim::make_pair(
                    Type::Variable(TypeVariable::new("a", Kind::Star)),
                    Type::Variable(TypeVariable::new("b", Kind::Star)),
                ),
            ),
        )
        .unwrap();

        new
    }

    fn add_prelude_classes(&mut self) {
        self.add_core_type_classes();
        self.add_num_type_classes();
    }

    fn add_core_type_classes(&mut self) {
        self.add_type_class("Eq", &[]).unwrap();
        self.add_type_class("Ord", &["Eq".into()]).unwrap();
        self.add_type_class("Show", &[]).unwrap();
        self.add_type_class("Read", &[]).unwrap();
        self.add_type_class("Bounded", &[]).unwrap();
        self.add_type_class("Enum", &[]).unwrap();
        self.add_type_class("Functor", &[]).unwrap();
        self.add_type_class("Monad", &[]).unwrap();
    }

    fn add_num_type_classes(&mut self) {
        self.add_type_class("Num", &["Eq".into(), "Show".into()])
            .unwrap();
        self.add_type_class("Real", &["Num".into(), "Ord".into()])
            .unwrap();
        self.add_type_class("Fractional", &["Num".into()]).unwrap();
        self.add_type_class("Integral", &["Real".into(), "Enum".into()])
            .unwrap();
        self.add_type_class("RealFrac", &["Real".into(), "Fractional".into()])
            .unwrap();
        self.add_type_class("Floating", &["Fractional".into()])
            .unwrap();
        self.add_type_class("RealFloat", &["RealFrac".into(), "Floating".into()])
            .unwrap();
    }
}

fn overlap(p: &Predicate, q: &Predicate) -> bool {
    most_general_unifier_predicate(p, q).is_ok()
}

// ### 7.3 Entailment

impl TypeClassEnvironment {
    fn by_super_class(&self, p: &Predicate) -> Vec<Predicate> {
        let Predicate::IsIn(i, t) = p;

        let mut buf: Vec<Predicate> = self
            .super_classes_of(i)
            .iter()
            .flat_map(|i_| self.by_super_class(&Predicate::IsIn(i_.clone(), t.clone())))
            .collect();

        buf.push(p.clone());

        buf
    }

    fn by_instance(&self, p: &Predicate) -> Result<Vec<Predicate>> {
        let Predicate::IsIn(i, t) = p;

        let mut buf = Vec::new();

        for Qualified::Then(ps, h) in self.instances_of(i) {
            let u = match_predicate(h, p)?;

            for p in ps {
                buf.push(p.apply(&u));
            }
        }

        Ok(buf)
    }

    fn entails(&self, ps: &[Predicate], p: &Predicate) -> bool {
        // This `|| match` is in the Haskell too, and I really don't like it
        // there either. Weird choice.
        ps.iter()
            .map(|p| self.by_super_class(p))
            .any(|supers| supers.contains(p))
            || match self.by_instance(p) {
                Err(_) => false,
                Ok(qs) => qs.iter().all(|q| self.entails(ps, q)),
            }
    }
}

// ### 7.4 Context Reduction

impl Predicate {
    fn in_hfn(&self) -> bool {
        let Predicate::IsIn(c, t) = self;

        fn hnf(t: &Type) -> bool {
            match t {
                Type::Variable(_) => true,
                Type::Constructor(_) => false,
                Type::Applied(t, _) => hnf(t),
                Type::Gen(_) => todo!(),
            }
        }

        hnf(t)
    }
}

impl TypeClassEnvironment {
    fn to_hnfs(&self, ps: &[Predicate]) -> Result<Vec<Predicate>> {
        let mut buf = Vec::new();

        for pss in ps.iter().map(|p| self.to_hnf(p)) {
            buf.extend(pss?)
        }

        Ok(buf)
    }

    fn to_hnf(&self, p: &Predicate) -> Result<Vec<Predicate>> {
        if p.in_hfn() {
            Ok(vec![p.clone()])
        } else {
            match self.by_instance(p) {
                Err(_) => Err("context reduction".into()),
                Ok(ps) => self.to_hnfs(&ps),
            }
        }
    }

    fn simplify(&self, ps: &[Predicate]) -> Vec<Predicate> {
        // Here's one of the few places where the iterative solution is clearer
        // to me, it's how the text of the paper describes things too.

        let mut rs = Vec::new();

        for p in ps {
            if !self.entails(&union(&rs, ps), p) {
                rs.push(p.clone());
            }
        }

        rs
    }

    fn reduce(&self, ps: &[Predicate]) -> Result<Vec<Predicate>> {
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
pub enum Scheme {
    // These are the generic types. In Rust, the stuff in <> when introducing
    // type variables.
    ForAll(Vec<Kind>, Qualified<Type>),
}

impl Types for Scheme {
    fn apply(&self, s: &[Substitution]) -> Self {
        let Scheme::ForAll(ks, qt) = self.clone();
        Scheme::ForAll(ks, qt.apply(s))
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        let Scheme::ForAll(ks, qt) = self.clone();
        qt.type_variables()
    }
}

fn quantify(vs: &[TypeVariable], qt: Qualified<Type>) -> Scheme {
    let vs_: Vec<TypeVariable> = qt
        .type_variables()
        .iter()
        .filter(|v| vs.contains(v))
        .cloned()
        .collect();
    let ks = vs_.iter().map(|v| v.kind().clone()).collect();
    let s: Vec<Substitution> = vs_
        .iter()
        .enumerate()
        .map(|(i, v)| Substitution {
            from: v.clone(),
            to: Type::Gen(i),
        })
        .collect();

    Scheme::ForAll(ks, qt.apply(&s))
}

impl From<Type> for Scheme {
    fn from(t: Type) -> Self {
        Scheme::ForAll(vec![], Qualified::Then(vec![], t))
    }
}

// ## 9. Assumptions

// I'm not going to try and come up with a name for `:>:`

#[derive(Debug, Clone, PartialEq)]
pub struct Assumption(Id, Scheme);

impl Types for Assumption {
    fn apply(&self, s: &[Substitution]) -> Self {
        let Assumption(i, sc) = self;
        Assumption(i.clone(), sc.apply(s))
    }

    fn type_variables(&self) -> Vec<TypeVariable> {
        let Assumption(i, sc) = self;
        sc.type_variables()
    }
}

fn find(i: &Id, assumptions: &[Assumption]) -> Result<Scheme> {
    for Assumption(i_, sc) in assumptions {
        if i == i_ {
            return Ok(sc.clone());
        }
    }
    Err(format!("unbound identifier: {i}"))
}

// ## 10. A Type Inference Monad
//
// Well, not really.

#[derive(Debug, Default)]
pub struct TypeInference {
    substitutions: Vec<Substitution>,
    next_var: usize,
}

impl TypeInference {
    fn get_subst(&self) -> &[Substitution] {
        &self.substitutions
    }

    fn unify(&mut self, t1: &Type, t2: &Type) -> Result<()> {
        let s = self.get_subst();
        let u = most_general_unifier(&t1.apply(s), &t2.apply(s))?;
        self.ext_subst(&u);
        Ok(())
    }

    fn ext_subst(&mut self, s: &[Substitution]) {
        // We should _definitely_ look at the definition of `at_at` and unpack
        // things a bit here.
        self.substitutions = at_at(&self.substitutions, s);
    }

    fn new_type_var(&mut self, k: Kind) -> Type {
        let i = self.next_var;
        self.next_var += 1;
        Type::Variable(TypeVariable::new(i, k))
    }

    fn fresh_inst(&mut self, s: &Scheme) -> Qualified<Type> {
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
            Type::Applied(l, r) => l.inst(ts).apply_to(r.inst(ts)),
            Type::Gen(n) => ts[*n].clone(),
            t => t.clone(),
        }
    }
}

impl<A> Instantiate for Vec<A>
where
    A: Instantiate,
{
    fn inst(&self, ts: &[Type]) -> Vec<A> {
        self.iter().map(|a| a.inst(ts)).collect()
    }
}

impl<T: Instantiate> Instantiate for Qualified<T> {
    fn inst(&self, ts: &[Type]) -> Qualified<T> {
        let Qualified::Then(ps, t) = self;
        Qualified::Then(ps.inst(ts), t.inst(ts))
    }
}

impl Instantiate for Predicate {
    fn inst(&self, ts: &[Type]) -> Predicate {
        let Predicate::IsIn(c, t) = self;
        Predicate::IsIn(c.clone(), t.inst(ts))
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
type Infer<E, T> = dyn Fn(
    &mut TypeInference,
    &TypeClassEnvironment,
    &[Assumption],
    E,
) -> Result<(Vec<Predicate>, T)>;

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Char(char),
    Rat(f64), // I know, but close enough.
    Str(String),
}

impl TypeInference {
    fn lit(&mut self, l: &Literal) -> (Vec<Predicate>, Type) {
        match l {
            Literal::Char(_) => (vec![], prim::character()),
            Literal::Int(_) => {
                let v = self.new_type_var(Kind::Star);
                (vec![Predicate::IsIn("Num".into(), v.clone())], v)
            }
            Literal::Str(_) => (vec![], prim::string()),
            Literal::Rat(_) => {
                let v = self.new_type_var(Kind::Star);
                (vec![Predicate::IsIn("Fractional".into(), v.clone())], v)
            }
        }
    }
}

// ### 11.2 Patterns

#[derive(Debug, Clone)]
pub enum Pat {
    Var(Id),                   // `a`
    Wildcard,                  // `_`
    As(Id, Box<Pat>),          // `id@pat`
    Lit(Literal),              // `1`
    Npk(Id, usize),            // `n + k` patterns, which are a sin
    Con(Assumption, Vec<Pat>), // `Constructor(**pats)`, not sure what Assump is
}

impl TypeInference {
    // This might need to be changed to return a Result, it's weird it not.
    fn pat(&mut self, p: &Pat) -> (Vec<Predicate>, Vec<Assumption>, Type) {
        match p {
            Pat::Var(i) => {
                let v = self.new_type_var(Kind::Star);
                (vec![], vec![Assumption(i.clone(), v.clone().into())], v)
            }
            Pat::Wildcard => {
                let v = self.new_type_var(Kind::Star);
                (vec![], vec![], v)
            }
            Pat::As(i, pat) => {
                let (ps, mut as_, t) = self.pat(pat);
                as_.push(Assumption(i.clone(), t.clone().into()));
                (ps, as_, t)
            }
            Pat::Lit(l) => {
                let (ps, t) = self.lit(l);
                (ps, vec![], t)
            }
            Pat::Npk(i, k) => {
                let t = self.new_type_var(Kind::Star);
                (
                    vec![Predicate::IsIn("Integeral".into(), t.clone())],
                    vec![Assumption(i.clone(), t.clone().into())],
                    t,
                )
            }
            Pat::Con(a, pats) => {
                let Assumption(i, sc) = a;
                let (ps, as_, ts) = self.pats(pats);
                let t_ = self.new_type_var(Kind::Star);
                let Qualified::Then(qs, t) = self.fresh_inst(sc);

                let folded = ts.iter().cloned().fold(t_.clone(), prim::make_function);
                self.unify(&t, &folded)
                    // We almost certainly need Result here...
                    .expect("Monad Transformer? I hardly know 'er");
                (append(ps, qs), as_, t_)
            }
        }
    }

    fn pats(&mut self, pats: &[Pat]) -> (Vec<Predicate>, Vec<Assumption>, Vec<Type>) {
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
pub enum Expr {
    Var(Id),
    Lit(Literal),
    Const(Assumption),
    Ap(Box<Expr>, Box<Expr>),
    Let(BindingGroup, Box<Expr>),
}

impl TypeInference {
    fn expr(
        &mut self,
        ce: &TypeClassEnvironment,
        as_: &[Assumption],
        e: &Expr,
    ) -> Result<(Vec<Predicate>, Type)> {
        match e {
            Expr::Var(i) => {
                let sc = find(i, as_)?;
                let Qualified::Then(ps, t) = self.fresh_inst(&sc);
                Ok((ps, t))
            }
            Expr::Const(Assumption(i, sc)) => {
                let Qualified::Then(ps, t) = self.fresh_inst(sc);
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
                self.unify(&prim::make_function(tf, t), &te)?;
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

// An alternative is what this is calling an 'equation'
//
// i.e. it's each line that's pattern matched in code like this:
//
//     null []    = true
//     null (_:_) = false
//
// The Vec is each parameter, and the Expr is the right hand side.
type Equations = (Vec<Pat>, Expr);

impl TypeInference {
    fn alt(
        &mut self,
        ce: &TypeClassEnvironment,
        as_: &[Assumption],
        (pats, e): &Equations,
    ) -> Result<(Vec<Predicate>, Type)> {
        let (ps, as__, ts) = self.pats(pats);
        let (qs, t) = self.expr(ce, &append(as__, as_.into()), e)?;

        let folded = ts.iter().cloned().fold(t, prim::make_function);
        Ok((append(ps, qs), folded))
    }

    fn alts(
        &mut self,
        ce: &TypeClassEnvironment,
        as_: &[Assumption],
        alts: &[Equations],
        t: Type,
    ) -> Result<Vec<Predicate>> {
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

fn split(
    ce: &TypeClassEnvironment,
    fs: &[TypeVariable],
    gs: &[TypeVariable],
    ps: &[Predicate],
) -> Result<(Vec<Predicate>, Vec<Predicate>)> {
    let ps_ = ce.reduce(ps)?;
    let (ds, rs) = partition(|p| p.type_variables().iter().all(|t| fs.contains(t)), ps_);
    let rs_ = ce.defaulted_predicates(append(fs.to_vec(), gs.to_vec()), &rs)?;
    Ok((ds, minus(rs, &rs_)))
}

// #### 11.5.1

type Ambiguity = (TypeVariable, Vec<Predicate>);

const NUM_CLASSES: &[&str] = &[
    "Num",
    "Integral",
    "Floating",
    "Fractional",
    "Real",
    "RealFloat",
    "RealFrac",
];

fn num_type_classes() -> Vec<Id> {
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

fn standard_type_classes() -> Vec<Id> {
    [STD_CLASSES, NUM_CLASSES]
        .iter()
        .flat_map(|i| i.iter())
        .cloned()
        .map(|s| Id::from(s))
        .collect()
}

impl TypeClassEnvironment {
    fn ambiguities(&self, vs: &[TypeVariable], ps: &[Predicate]) -> Vec<Ambiguity> {
        let mut buf = vec![];

        for v in minus(ps.to_vec().type_variables(), vs).iter() {
            let ps = ps
                .iter()
                .filter_map(|p| {
                    if p.type_variables().contains(v) {
                        Some(p.clone())
                    } else {
                        None
                    }
                })
                .collect();

            buf.push((v.clone(), ps));
        }

        buf
    }

    fn candidates(&self, (v, qs): &Ambiguity) -> Vec<Type> {
        let mut is: Vec<Id> = vec![];
        let mut ts = vec![];

        for Predicate::IsIn(i, t) in qs.iter() {
            is.push(i.clone());
            ts.push(t);
        }

        if !ts.iter().all(|t| t == &&Type::Variable(v.clone())) {
            return vec![];
        }

        if !is.iter().any(|i| num_type_classes().contains(i)) {
            return vec![];
        }

        if !is.iter().all(|i| standard_type_classes().contains(i)) {
            return vec![];
        }

        let mut t_s = vec![];

        for t_ in &self.defaults {
            if !is
                .iter()
                .map(|i| Predicate::IsIn(i.clone(), t_.clone()))
                .all(|q| self.entails(&[], &q))
            {
                return vec![];
            } else {
                t_s.push(t_.clone());
            }
        }

        // there's no way I got this right

        t_s
    }

    fn with_defaults<F, T>(&self, f: F, vs: &[TypeVariable], ps: &[Predicate]) -> Result<T>
    where
        F: Fn(&[Ambiguity], &[Type]) -> T,
    {
        let vps = self.ambiguities(vs, ps);
        let tss: Vec<Vec<_>> = vps.iter().map(|vp| self.candidates(vp)).collect();

        if tss.iter().any(|ts| ts.is_empty()) {
            Err("cannot resolve ambiguity".into())
        } else {
            let ts: Vec<Type> = tss.iter().map(|ts| ts.first().unwrap().clone()).collect();

            Ok(f(&vps, &ts))
        }
    }
}

impl TypeClassEnvironment {
    fn defaulted_predicates(
        &self,
        vs: Vec<TypeVariable>,
        ps: &[Predicate],
    ) -> Result<Vec<Predicate>> {
        self.with_defaults(
            |vps, ts| vps.iter().flat_map(|a| a.1.clone()).collect(),
            &vs,
            ps,
        )
    }

    fn default_substitutions(
        &self,
        vs: Vec<TypeVariable>,
        ps: &[Predicate],
    ) -> Result<Vec<Substitution>> {
        self.with_defaults(
            |vps, ts| {
                vps.iter()
                    .map(|(fst, _)| fst.clone())
                    .zip(ts.iter().cloned())
                    .map(|(from, to)| Substitution::new(from, to))
                    .collect()
            },
            &vs,
            ps,
        )
    }
}
// ### 11.6 Bind Groups

type ExplicitBinding = (Id, Scheme, Vec<Equations>);

impl TypeInference {
    fn expl(
        &mut self,
        ce: &TypeClassEnvironment,
        as_: &[Assumption],
        (i, sc, alts): &ExplicitBinding,
    ) -> Result<Vec<Predicate>> {
        let Qualified::Then(qs, t) = self.fresh_inst(sc);
        let ps = self.alts(ce, as_, alts, t.clone())?;
        let s = self.get_subst();

        let qs_ = qs.apply(s);
        let t_ = t.apply(s);
        let fs = as_.to_vec().apply(s).type_variables();
        let gs = minus(t_.type_variables(), &fs);
        let sc_ = quantify(&gs, Qualified::Then(qs_.clone(), t_));
        let ps_ = ps
            .apply(s)
            .iter()
            .filter(|p| !ce.entails(&qs_, p))
            .cloned()
            .collect::<Vec<Predicate>>();

        let (ds, rs) = split(ce, &fs, &gs, &ps_)?;

        if sc != &sc_ {
            Err("signature to general".into())
        } else if !rs.is_empty() {
            Err("context too weak".into())
        } else {
            Ok(ds)
        }
    }
}

type ImplicitBinding = (Id, Vec<Equations>);

fn restricted(bs: &[ImplicitBinding]) -> bool {
    bs.iter()
        .any(|(i, alts)| alts.iter().any(|alt| alt.0.is_empty()))
}

impl TypeInference {
    fn impls(
        &mut self,
        ce: &TypeClassEnvironment,
        as_: &[Assumption],
        bs: &[ImplicitBinding],
    ) -> Result<(Vec<Predicate>, Vec<Assumption>)> {
        let ts = bs
            .iter()
            .map(|_| self.new_type_var(Kind::Star))
            .collect::<Vec<_>>();

        let is: Vec<Id> = bs.iter().map(|b| b.0.clone()).collect();
        let scs: Vec<Scheme> = ts.iter().cloned().map(Scheme::from).collect();
        let as__ = append(zip_with(Assumption, is.clone(), scs), as_.to_vec());
        let altss = bs.iter().map(|b| b.1.clone()).collect();

        let pss = zip_with_try(|alts, t| self.alts(ce, as_, &alts, t), altss, ts.clone())?;
        let s = self.get_subst();

        let ps_: Vec<Predicate> = pss.iter().flatten().map(|p| p.apply(s)).collect();
        let ts_: Vec<Type> = ts.apply(s);
        let fs: Vec<TypeVariable> = as_.to_vec().apply(s).type_variables();
        let vss: Vec<Vec<TypeVariable>> = ts_.iter().map(Types::type_variables).collect();
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
            let gs_ = minus(gs, &rs.type_variables());
            let scs_ = ts_
                .iter()
                .map(|t| quantify(&gs_, Qualified::Then(rs.clone(), t.clone())))
                .collect();
            Ok((append(ds, rs), zip_with(Assumption, is, scs_)))
        } else {
            let scs_ = ts_
                .iter()
                .map(|t| quantify(&gs, Qualified::Then(rs.clone(), t.clone())))
                .collect();
            Ok((ds, zip_with(Assumption, is, scs_)))
        }
    }
}

pub type BindingGroup = (Vec<ExplicitBinding>, Vec<Vec<ImplicitBinding>>);
pub type Program = Vec<BindingGroup>;

impl TypeInference {
    fn bind_group(
        &mut self,
        ce: &TypeClassEnvironment,
        as_: &[Assumption],
        (es, iss): &BindingGroup,
    ) -> Result<(Vec<Predicate>, Vec<Assumption>)> {
        let as__: Vec<Assumption> = es
            .iter()
            .map(|(v, sc, alts)| Assumption(v.clone(), sc.clone()))
            .collect();

        let (ps, all_as): (Vec<Predicate>, Vec<Assumption>) = {
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

    pub fn program(
        &mut self,
        ce: &TypeClassEnvironment,
        as_: &[Assumption],
        bgs: Program,
    ) -> Result<Vec<Assumption>> {
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
        let s_ = ce.default_substitutions(vec![], &rs)?;

        Ok(as__.apply(&at_at(&s_, s)))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_empty() {
        // the empty program
        let program = vec![];
        let mut ti = TypeInference::default();
        let ce = TypeClassEnvironment::default();
        let assumptions = ti.program(&ce, &[], program).expect("is well typed");
        assert!(assumptions.is_empty());
    }

    #[test]
    fn test_simple() {
        // the program `x = 'c'`
        let program: Program = vec![(
            // binding groups
            vec![], // explicit type signatures
            vec![vec![(
                // implicit types
                "x".into(), // id of thing being bound
                vec![(
                    vec![],                        // patterns to the left of the `=`
                    Expr::Lit(Literal::Char('a')), // expression on the right of the `=`
                )],
            )]],
        )];

        let mut ti = TypeInference::default();
        let ce = TypeClassEnvironment::default();
        let assumptions = ti.program(&ce, &[], program).expect("is well typed");
        assert_eq!(assumptions.len(), 1);
        // i.e. x :: [Char]
        assert_eq!(
            format!("{:?}", assumptions),
            r#"[Assump("x", ForAll([], Then([], Con(Tycon("Char", Star)))))]"#
        )
    }

    #[test]
    fn test_hello() {
        // the program `x = "Hello world!"`
        let program: Program = vec![(
            // binding groups
            vec![], // explicit type signatures
            vec![vec![(
                // implicit types
                "x".into(), // id of thing being bound
                vec![(
                    vec![],                                          // patterns to the left of the `=`
                    Expr::Lit(Literal::Str("Hello, world!".into())), // expression on the right of the `=`
                )],
            )]],
        )];

        let mut ti = TypeInference::default();
        let ce = TypeClassEnvironment::default();
        let assumptions = ti.program(&ce, &[], program).expect("is well typed");
        assert_eq!(assumptions.len(), 1);
        // i.e. x :: [Char]
        assert_eq!(
            format!("{:?}", assumptions),
            r#"[Assump("x", ForAll([], Then([], Ap(Con(Tycon("[]", Fun(Star, Star))), Con(Tycon("Char", Star))))))]"#
        )
    }

    #[test]
    fn test_defaults() {
        // the program `x = 1`
        let program: Program = vec![(
            // binding groups
            vec![], // explicit type signatures
            vec![vec![(
                // implicit types
                "x".into(), // id of thing being bound
                vec![(
                    vec![],                     // patterns to the left of the `=`
                    Expr::Lit(Literal::Int(1)), // expression on the right of the `=`
                )],
            )]],
        )];

        let mut ti = TypeInference::default();

        let ce = TypeClassEnvironment::example_instances();

        let assumptions = ti.program(&ce, &[], program).expect("is well typed");
        assert_eq!(assumptions.len(), 1);
        // i.e. x :: [Char]
        assert_eq!(
            format!("{:?}", assumptions),
            // How do we know what Gen(0) is limited to?
            r#"[Assump(\"x\", ForAll([Star], Then(["Integral"], Gen(0))))]"#
        )
    }
}
