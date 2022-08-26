use crate::{Language, Pattern};

pub trait PrettyString {
    fn pretty_string(&self) -> String;
}

impl<L: Language> PrettyString for Pattern<L> {
    fn pretty_string(&self) -> String {
        self.pretty(500)
    }
}