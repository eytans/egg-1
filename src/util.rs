use std::fmt;
use std::str::FromStr;
use std::sync::Mutex;

use indexmap::{IndexMap, IndexSet};
use once_cell::sync::Lazy;
use std::fmt::Formatter;
use std::iter::FromIterator;
use itertools::Itertools;
use serde::de::{Error, MapAccess};
use serde::Deserializer;
use serde::ser::{SerializeMap, SerializeTuple};

static STRINGS: Lazy<Mutex<IndexMap<u32, &'static str>>> = Lazy::new(Default::default);
// If in test mode create function to get the strings
#[cfg(test)]
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
/// use egg::Symbol;
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
                    strings.insert(index, name);
                }
                Ok(Symbol(index))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                where E: serde::de::Error {
                // split value by first #
                let mut split = value.splitn(2, '#').collect_vec();
                assert_eq!(split.len(), 2);
                let index = split[0].parse().unwrap();
                let name = split[1];
                let mut strings = STRINGS
                    .lock()
                    .unwrap_or_else(|err| panic!("Failed to acquire egg's global string cache: {}", err));
                if let Some(existing) = strings.get(&(index as u32)) {
                    assert_eq!(*existing, name);
                } else {
                    assert!(strings.values().find(|&&v| v == name).is_none());
                    strings.insert(index as u32, Box::leak(name.to_string().into_boxed_str()));
                }
                Ok(Symbol(index))
            }
        }

        deserializer.deserialize_str(SymbolVisitor)
    }
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

fn intern(s: &str) -> Symbol {
    let mut strings = STRINGS
        .lock()
        .unwrap_or_else(|err| panic!("Failed to acquire egg's global string cache: {}", err));
    let i = match strings.iter().find_position(|(_, n)| **n == s) {
        Some((i, _)) => i,
        None => {
            let i = strings.len();
            strings.insert(i as u32, leak(s));
            i
        },
    };
    Symbol(i as u32)
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