use std::fmt;
use std::str::FromStr;
use thiserror::Error;

use crate::{Analysis, EGraph, Id, Symbol};
use crate::ColorId;
use std::fmt::Formatter;
use serde::{Deserialize, Serialize};

/// A variable for use in [`Pattern`]s or [`Subst`]s.
///
/// This implements [`FromStr`], and will only parse if it has a
/// leading `?`.
///
/// [`Pattern`]: struct.Pattern.html
/// [`Subst`]: struct.Subst.html
/// [`FromStr`]: https://doc.rust-lang.org/std/str/trait.FromStr.html
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Var(Symbol);

#[derive(Debug, Error)]
pub enum VarParseError {
    #[error("pattern variable {0:?} should have a leading question mark")]
    MissingQuestionMark(String),
}

impl FromStr for Var {
    type Err = VarParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use VarParseError::*;

        if s.starts_with('?') && s.len() > 1 {
            Ok(Var(s.into()))
        } else {
            Err(MissingQuestionMark(s.to_owned()))
        }
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A substitition mapping [`Var`]s to eclass [`Id`]s.
///
/// [`Var`]: struct.Var.html
/// [`Id`]: struct.Id.html
#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Subst {
    pub(crate) vec: smallvec::SmallVec<[(Var, Id); 3]>,
    #[cfg(feature = "colored")]
    pub(crate) color: Option<ColorId>,
}

impl Subst {
    pub(crate) fn fix<L: crate::Language, A: Analysis<L>>(&mut self, egraph: &EGraph<L, A>) {
        let color = self.color;
        for (_var, id) in &mut self.vec {
            *id = egraph.opt_colored_find(color, *id);
        }
    }
}

impl Subst {
    /// Create a `Subst` with the given initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vec: smallvec::SmallVec::with_capacity(capacity),
            color: None,
        }
    }

    #[cfg(feature = "colored")]
    pub fn colored_with_capacity(capacity: usize, color: Option<ColorId>) -> Self {
        Self {
            vec: smallvec::SmallVec::with_capacity(capacity),
            color,
        }
    }

    /// Insert something, returning the old `Id` if present.
    pub fn insert(&mut self, var: Var, id: Id) -> Option<Id> {
        for pair in &mut self.vec {
            if pair.0 == var {
                return Some(std::mem::replace(&mut pair.1, id));
            }
        }
        self.vec.push((var, id));
        None
    }

    /// Retrieve a `Var`, returning `None` if not present.
    #[inline(never)]
    pub fn get(&self, var: Var) -> Option<&Id> {
        self.vec
            .iter()
            .find_map(|(v, id)| if *v == var { Some(id) } else { None })
    }

    #[cfg(feature = "colored")]
    pub fn color(&self) -> Option<ColorId> {
        self.color
    }

    pub fn merge(&self, sub2: Subst) -> Subst {
        assert!(self.color.is_none() || sub2.color.is_none() || self.color == sub2.color);
        let mut new = self.clone();
        if new.color.is_none() && sub2.color.is_some() {
            new.color = sub2.color.clone();
        }
        for (var, id) in sub2.vec {
            if let Some(vid) = self.get(var) {
                assert!(vid == &id);
            } else {
                new.insert(var, id);
            }
        }
        new
    }
}

impl std::ops::Index<Var> for Subst {
    type Output = Id;

    fn index(&self, var: Var) -> &Self::Output {
        match self.get(var) {
            Some(id) => id,
            None => panic!("Var '{}={}' not found in {:?}", var.0, var, self),
        }
    }
}

impl fmt::Display for Subst {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:#?}", self)
    }
}

impl fmt::Debug for Subst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.vec.len();
        write!(f, "{{")?;
        for i in 0..len {
            let (var, id) = &self.vec[i];
            write!(f, "{}: {}", var, id)?;
            if i < len - 1 {
                write!(f, ", ")?;
            }
        }
        if cfg!(feature = "colored") {
            write!(f, " color: {}", self.color.map_or("None".to_string(), |x| x.to_string()))?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_parse() {
        assert_eq!(Var::from_str("?a").unwrap().to_string(), "?a");
        assert_eq!(Var::from_str("?abc 123").unwrap().to_string(), "?abc 123");
        assert!(Var::from_str("a").is_err());
        assert!(Var::from_str("a?").is_err());
        assert!(Var::from_str("?").is_err());
    }
}
