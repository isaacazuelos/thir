//! Putting these in a module to namespace them instead of using the prefix
//! naming scheme used in the paper. A name like `prim::unit` is `tUnit` in the
//! paper.
//!
//! The types with arrow kinds are functions since I can't make the boxes for the
//! arrow kinds in while `const`.

// TODO: lazy_static?

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
    builtins::arrow().apply_to(a).apply_to(b)
}

pub fn make_list(t: Type) -> Type {
    builtins::list().apply_to(t)
}

pub fn make_pair(a: Type, b: Type) -> Type {
    builtins::tuple_2().apply_to(a).apply_to(b)
}
