//! Kinds
//!
//! The module is named plural to match `types`

// Our `Kind` and `Type` types aren't exactly cheap because of the boxing.
//
// I think we could put these all in one context and work on IDs, in a
// struct-of-array, data-oriented way.

pub trait HasKind {
    fn kind(&self) -> &Kind;
}

#[derive(Debug, PartialEq, Clone)]
pub enum Kind {
    Star,
    Function(Box<Kind>, Box<Kind>),
}

impl Kind {
    // Just for convenience, same as the [`Kind::Function`] constructor, but it
    // does the boxing for us.
    pub(crate) fn function(lhs: Kind, rhs: Kind) -> Kind {
        Kind::Function(Box::new(lhs), Box::new(rhs))
    }
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kind::Star => write!(f, "*"),
            Kind::Function(from, to) => write!(f, "({} -> {})", from, to),
        }
    }
}
