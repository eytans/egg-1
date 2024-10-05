use std::fmt;
use std::str::FromStr;
use std::sync::Mutex;

use indexmap::{IndexMap, IndexSet};
use once_cell::sync::Lazy;
use std::fmt::Formatter;
use std::iter::FromIterator;
use itertools::Itertools;
use serde::de::{Error, MapAccess};
use serde::{Deserialize, Deserializer, Serialize};
use serde::ser::SerializeMap;

static STRINGS: Lazy<Mutex<IndexMap<u32, &'static str>>> = Lazy::new(Default::default);
// If in test mode create function to get the strings
pub fn get_strings() -> &'static Mutex<IndexMap<u32, &'static str>> {
    &STRINGS
}

// If in test mode create function to clear the strings
#[cfg(test)]
pub fn clear_strings() {
    STRINGS.lock().unwrap().clear();
}

/// An interned string.
///
/// Internally, `egg` frequently compares [`Var`]s and elements of
/// [`Language`]s. To keep comparisons fast, `egg` provides [`Symbol`] a simple
/// wrapper providing interned strings.
///
/// You may wish to use [`Symbol`] in your own [`Language`]s to increase
/// performance and keep enode sizes down (a [`Symbol`] is only 4 bytes,
/// compared to 24 for a `String`.)
///
/// A [`Symbol`] is simply a wrapper around an integer.
/// When creating a [`Symbol`] from a string, `egg` looks up it up in a global
/// table, returning the index (inserting it if not found).
/// That integer is used to cheaply implement
/// `Copy`, `Clone`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, and `Hash`.
///
/// The internal symbol cache leaks the strings, which should be
/// fine if you only put in things like variable names and identifiers.
///
/// # Example
/// ```rust
/// use easter_egg::Symbol;
///
/// assert_eq!(Symbol::from("foo"), Symbol::from("foo"));
/// assert_eq!(Symbol::from("foo"), "foo".parse().unwrap());
///
/// assert_ne!(Symbol::from("foo"), Symbol::from("bar"));
/// ```
///
/// [`Var`]: struct.Var.html
/// [`Symbol`]: struct.Symbol.html
/// [`Language`]: trait.Language.html
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(pub(crate) u32);

impl Symbol {
    /// Get the string that this symbol represents
    pub fn as_str(self) -> &'static str {
        let i = self.0 as usize;
        let strings = STRINGS
            .lock()
            .unwrap_or_else(|err| panic!("Failed to acquire egg's global string cache: {}", err));
        strings.get(&(i as u32)).unwrap()
    }
}

impl serde::Serialize for Symbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let name = self.as_str().to_string();
        let index = self.0.to_string();
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("name", &name)?;
        map.serialize_entry("index", &index)?;
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for Symbol {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        struct SymbolVisitor;

        impl<'de> serde::de::Visitor<'de> for SymbolVisitor {
            type Value = Symbol;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("A string representing a symbol and it's index")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error> where A: MapAccess<'de> {
                // deserialize name from map
                let mut name: Option<String> = None;
                let mut str_index: Option<String> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "name" => {
                            if name.is_some() {
                                return Err(A::Error::duplicate_field("name"));
                            }
                            name = Some(map.next_value()?);
                        }
                        "index" => {
                            if str_index.is_some() {
                                return Err(A::Error::duplicate_field("index"));
                            }
                            str_index = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(A::Error::unknown_field(&key, &["name", "index"]));
                        }
                    }
                }
                let index: u32 = str_index.unwrap().parse().unwrap();
                let name = Box::leak(name.unwrap().into_boxed_str());
                let mut strings = STRINGS
                    .lock()
                    .unwrap_or_else(|err| panic!("Failed to acquire egg's global string cache: {}", err));
                if let Some(existing) = strings.get(&index) {
                    assert_eq!(*existing, name);
                } else {
                    assert!(strings.values().find(|&&v| v == name).is_none());
                    assert!(!strings.contains_key(&index));
                    strings.insert(index, name);
                }
                Ok(Symbol(index))
            }
        }

        deserializer.deserialize_map(SymbolVisitor)
    }
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn intern(s: &str) -> Symbol {
    let mut strings = STRINGS
        .lock()
        .unwrap_or_else(|err| panic!("Failed to acquire egg's global string cache: {}", err));
    let i = match strings.iter().find(|(_, n)| **n == s) {
        Some((i, _)) => {
            *i
        },
        None => {
            let i = (0..strings.len()+1).find(|i| !strings.contains_key(&(*i as u32))).unwrap();
            let old = strings.insert(i as u32, leak(s));
            assert!(old.is_none());
            i as u32
        },
    };
    Symbol(i)
}

impl<S: AsRef<str>> From<S> for Symbol {
    fn from(s: S) -> Self {
        intern(s.as_ref())
    }
}

impl FromStr for Symbol {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

pub(crate) trait JoinDisp {
    #[allow(dead_code)]
    fn disp_string(self) -> String;
    fn sep_string(self, sep: &str) -> String;
}

impl<I> JoinDisp for I where I: Iterator,
                              I::Item: fmt::Display {
    fn disp_string(self) -> String {
        self.sep_string(", ")
    }

    fn sep_string(self, sep: &str) -> String {
        self.map(|x| format!("{}", x)).join(sep)
    }
}

pub trait Singleton<T> {
    fn singleton(t: T) -> Self;
}

impl<T, FI> Singleton<T> for FI
where FI: FromIterator<T> {
    fn singleton(t: T) -> Self {
        FI::from_iter(std::iter::once(t))
    }
}



/** A data structure to maintain a queue of unique elements.

Notably, insert/pop operations have O(1) expected amortized runtime complexity.
*/
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    set: IndexSet<T>,
    queue: std::collections::VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: IndexSet::default(),
            queue: std::collections::VecDeque::new(),
        }
    }
}

#[allow(dead_code)]
impl<T> UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    pub fn insert(&mut self, t: T) {
        if self.set.insert(t.clone()) {
            self.queue.push_back(t);
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for t in iter.into_iter() {
            self.insert(t);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let res = self.queue.pop_front();
        res.as_ref().map(|t| self.set.remove(t));
        res
    }

    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }
}

impl<T> IntoIterator for UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    type Item = T;

    type IntoIter = <std::collections::VecDeque<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.queue.into_iter()
    }
}

impl<A> FromIterator<A> for UniqueQueue<A>
where
    A: Eq + std::hash::Hash + Clone,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let mut queue = UniqueQueue::default();
        for t in iter {
            queue.insert(t);
        }
        queue
    }
}
