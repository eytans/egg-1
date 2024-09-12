use crate::{Language, Pattern};

/// A trait for pretty printing a pattern.
pub trait PrettyString {
    /// Returns a string representation of the pattern.
    fn pretty_string(&self) -> String;
}

impl<L: Language> PrettyString for Pattern<L> {
    fn pretty_string(&self) -> String {
        self.pretty(500)
    }
}