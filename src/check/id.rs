//! Names of things

use std::rc::Rc;

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
