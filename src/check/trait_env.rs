//! Trait Environments
//!
//! The paper uses a function with type `Id -> Maybe Class` here for the field
//! `classes`. I'm betting that function is a lookup, so a hashmap is a different
//! way to do that.
//!
//! The way the paper does it is by nesting it's `classes` function. This
//! effectively forms a linked list in memory -- I'm not really sure I see this
//! and the `modify` definition as better than just using `[Class]` and `lookup`
//! but whatever.
//!
//! Using a hashmap and cloning here is going to be wildly inefficient. I'd need
//! to understand the use pattern better to really know how to better translate
//! it. Alternatively just `Rc` it.

use std::collections::HashMap;

use super::*;

#[derive(Clone, Default, Debug)]
// TODO: rethink collection types?
pub struct TraitEnvironment {
    classes: HashMap<Id, Trait>,
    defaults: Vec<Type>,
}

impl TraitEnvironment {
    /// Get a type class by [`Id`].
    fn get(&self, id: &Id) -> Option<&Trait> {
        self.classes.get(id)
    }

    /// Get all the super classes of a type class by [`Id`].
    fn super_traits_of(&self, id: &Id) -> &[Id] {
        &self
            .get(id)
            .expect("super is partial in the paper")
            .super_traits
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

        if self.get(&id).is_some() {
            return Err(format!("class already defined: {id}"));
        }

        for superclass in supers {
            if !self.classes.contains_key(superclass) {
                return Err(format!("superclass {superclass} not defined for {id}"));
            }
        }

        let class = Trait::new(supers, &[]);
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
            .any(|q| p.overlap(q))
        {
            return Err("overlapping instances".into());
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
impl TraitEnvironment {
    fn example_instances() -> TraitEnvironment {
        let mut new = TraitEnvironment::default();

        new.add_prelude_classes();

        new.add_instance_mut(&[], &Predicate::is_in("Ord", builtins::unit()))
            .unwrap();
        new.add_instance_mut(&[], &Predicate::is_in("Ord", builtins::character()))
            .unwrap();
        new.add_instance_mut(&[], &Predicate::is_in("Ord", builtins::int()))
            .unwrap();
        new.add_instance_mut(
            &[
                Predicate::is_in("Ord", Type::Variable(TypeVariable::new("a", Kind::Star))),
                Predicate::is_in("Ord", Type::Variable(TypeVariable::new("b", Kind::Star))),
            ],
            &Predicate::IsIn(
                "Ord".into(),
                builtins::make_pair(
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

impl TraitEnvironment {
    fn by_super_class(&self, p: &Predicate) -> Vec<Predicate> {
        let Predicate::IsIn(i, t) = p;

        let mut buf: Vec<Predicate> = self
            .super_traits_of(i)
            .iter()
            .flat_map(|i_| self.by_super_class(&Predicate::IsIn(i_.clone(), t.clone())))
            .collect();

        buf.push(p.clone());

        buf
    }

    fn by_instance(&self, p: &Predicate) -> Result<Vec<Predicate>> {
        let Predicate::IsIn(i, _t) = p;

        let mut buf = Vec::new();

        for Qualified::Then(ps, h) in self.instances_of(i) {
            let u = h.match_predicate(p)?;

            for p in ps {
                buf.push(p.apply(&u));
            }
        }

        Ok(buf)
    }

    pub fn entails(&self, ps: &[Predicate], p: &Predicate) -> bool {
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

impl TraitEnvironment {
    // TODO: to_hnfs is a terrible name.
    fn to_hnfs(&self, ps: &[Predicate]) -> Result<Vec<Predicate>> {
        let mut buf = Vec::new();

        for pss in ps.iter().map(|p| self.to_hnf(p)) {
            buf.extend(pss?)
        }

        Ok(buf)
    }

    // TODO: HNF is a terrible name.
    pub fn to_hnf(&self, p: &Predicate) -> Result<Vec<Predicate>> {
        if p.in_hfn() {
            Ok(vec![p.clone()])
        } else {
            match self.by_instance(p) {
                Err(_) => Err("context reduction".into()),
                Ok(ps) => self.to_hnfs(&ps),
            }
        }
    }

    pub fn simplify(&self, ps: &[Predicate]) -> Vec<Predicate> {
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

    pub fn reduce(&self, ps: &[Predicate]) -> Result<Vec<Predicate>> {
        // I love `?` so much. It works so well for `do` code in error-handling
        // monads, since it's _almost_ the same thing.
        let qs = self.to_hnfs(ps)?;
        Ok(self.simplify(&qs))
    }
}

impl TraitEnvironment {
    pub fn defaulted_predicates(
        &self,
        vs: Vec<TypeVariable>,
        ps: &[Predicate],
    ) -> Result<Vec<Predicate>> {
        self.with_defaults(
            |vps, _ts| vps.iter().flat_map(|a| a.1.clone()).collect(),
            &vs,
            ps,
        )
    }

    pub fn default_substitutions(
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
        .map(Id::from)
        .collect()
}

impl TraitEnvironment {
    pub fn ambiguities(&self, vs: &[TypeVariable], ps: &[Predicate]) -> Vec<Ambiguity> {
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

    pub fn candidates(&self, (v, qs): &Ambiguity) -> Vec<Type> {
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

    pub fn with_defaults<F, T>(&self, f: F, vs: &[TypeVariable], ps: &[Predicate]) -> Result<T>
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

    pub fn split(
        &self,
        fs: &[TypeVariable],
        gs: &[TypeVariable],
        ps: &[Predicate],
    ) -> Result<(Vec<Predicate>, Vec<Predicate>)> {
        let ps_ = self.reduce(ps)?;
        let (ds, rs) = partition(|p| p.type_variables().iter().all(|t| fs.contains(t)), ps_);
        let rs_ = self.defaulted_predicates(append(fs.to_vec(), gs.to_vec()), &rs)?;
        Ok((ds, minus(rs, &rs_)))
    }
}
