//! A FFI-safe sorted map backed by a `RVec<Tuple2<K, V>>`.

use abi_stable::{
    StableAbi,
    std_types::{RVec, Tuple2},
};
use std::fmt::{self, Debug, Formatter};

/// A FFI-safe ordered map backed by a sorted `RVec<Tuple2<K, V>>`.
///
/// Keys are kept in ascending `Ord` order. Lookup and insertion are O(n) for
/// rare mutations and O(log n) for reads via binary search.  This is intended
/// as a drop-in replacement for `BTreeMap` where deterministic iteration order
/// matters.
#[repr(C)]
#[derive(StableAbi, Clone, PartialEq, Eq, Default)]
pub struct RSortedMap<K, V>(RVec<Tuple2<K, V>>);

impl<K: Debug, V: Debug> Debug for RSortedMap<K, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.0.iter().map(|Tuple2(k, v)| (k, v))).finish()
    }
}

impl<K: Ord, V> RSortedMap<K, V> {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self(RVec::new())
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns a reference to the value for `key`, or `None` if absent.
    pub fn get(&self, key: &K) -> Option<&V> {
        let pos = self.0.iter().position(|Tuple2(k, _)| k == key)?;
        Some(&self.0[pos].1)
    }

    /// Returns `true` if the map contains `key`.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns a mutable reference to the value for `key`, or `None` if absent.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let pos = self.0.iter().position(|Tuple2(k, _)| k == key)?;
        Some(&mut self.0[pos].1)
    }

    /// Returns a mutable reference to the value for `key`, inserting `default` if absent.
    ///
    /// Equivalent to `BTreeMap::entry(key).or_insert(default)`. O(n).
    pub fn get_or_insert(&mut self, key: K, default: V) -> &mut V {
        let mut v: Vec<Tuple2<K, V>> = std::mem::take(&mut self.0).into();
        let pos = match v.binary_search_by(|Tuple2(k, _)| k.cmp(&key)) {
            Ok(i) => i,
            Err(i) => {
                v.insert(i, Tuple2(key, default));
                i
            }
        };
        self.0 = v.into();
        &mut self.0[pos].1
    }

    /// Removes `key` and returns its value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Inserts `key → value`. Returns the previous value if the key was already present.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut v: Vec<Tuple2<K, V>> = std::mem::take(&mut self.0).into();
        let old = match v.binary_search_by(|Tuple2(k, _)| k.cmp(&key)) {
            Ok(i) => Some(std::mem::replace(&mut v[i].1, value)),
            Err(i) => {
                v.insert(i, Tuple2(key, value));
                None
            }
        };
        self.0 = v.into();
        old
    }

    /// Removes `key` and returns `(key, value)` if it was present.
    pub fn remove_entry(&mut self, key: &K) -> Option<(K, V)> {
        let mut v: Vec<Tuple2<K, V>> = std::mem::take(&mut self.0).into();
        let result = match v.binary_search_by(|Tuple2(k, _)| k.cmp(key)) {
            Ok(i) => {
                let Tuple2(k, val) = v.remove(i);
                Some((k, val))
            }
            Err(_) => None,
        };
        self.0 = v.into();
        result
    }

    /// Iterates over `(&K, &V)` pairs in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.0.iter().map(|Tuple2(k, v)| (k, v))
    }

    /// Iterates over keys in order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.0.iter().map(|Tuple2(k, _)| k)
    }

    /// Iterates over values in key order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.iter().map(|Tuple2(_, v)| v)
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for RSortedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K: Ord, V> IntoIterator for RSortedMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::iter::Map<abi_stable::std_types::vec::IntoIter<Tuple2<K, V>>, fn(Tuple2<K, V>) -> (K, V)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|Tuple2(k, v)| (k, v))
    }
}
