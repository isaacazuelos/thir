//! Traits

use super::*;

pub type Instance = Qualified<Predicate>;

#[derive(Debug, Clone, PartialEq)]
pub struct Trait {
    pub(crate) super_traits: Vec<Id>,
    pub(crate) instances: Vec<Instance>,
}

impl Trait {
    // Keeps the same arg order as the tuple, so we can use it as a constructor
    pub fn new(super_traits: &[Id], instances: &[Instance]) -> Self {
        Trait {
            super_traits: super_traits.into(),
            instances: instances.as_ref().to_vec(),
        }
    }

    pub fn ord_example() -> Trait {
        Trait::new(
            // This part tells us that Eq is a 'superclass' of Ord,
            // it's the `class Eq => Ord` part.
            &["Eq".into()],
            // These are instances of the class
            &[
                // This is the `instance Ord _ where` part for unit, char, int.
                // Notice this isn't the implementation, just the type level stuff.
                Qualified::then(&[], Predicate::is_in("Ord", builtins::unit())),
                Qualified::then(&[], Predicate::is_in("Ord", builtins::character())),
                Qualified::then(&[], Predicate::is_in("Ord", builtins::int())),
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
                        builtins::make_pair(
                            Type::Variable(TypeVariable::new("a", Kind::Star)),
                            Type::Variable(TypeVariable::new("b", Kind::Star)),
                        ),
                    ),
                ),
            ],
        )
    }
}
