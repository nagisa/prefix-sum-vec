#[doc = include_str!("../README.mkd")]

/// A data structure for space-efficient storage of repeating sequences.
///
/// Much like [`Vec`], this is a growable array-like type. This data structure is capaable of
/// storing a contiguous sequence of repeating values only once, thus “compressing” the
/// representation of the sequence. Conversely index lookups are `O(log n)`, rather than `O(1)` as
/// provided by `Vec`.
///
/// Note, however, that this data structure can become less space-efficient than a plain `Vec` if
/// the data is incompressible.
///
/// [`Vec`]: std::vec::Vec
///
/// # Examples
///
/// ```
/// let mut prefix_sum_vec = prefix_sum_vec::PrefixSumVec::new();
///
/// prefix_sum_vec.try_push_many(12, "strings").expect("could not push");
/// assert_eq!(prefix_sum_vec.max_index(), Some(&11));
///
/// assert_eq!(prefix_sum_vec[0], "strings");
/// assert_eq!(prefix_sum_vec[11], "strings");
/// ```
#[derive(Debug)]
pub struct PrefixSumVec<T, Idx = usize> {
    /// Keys between ((keys[n-1] + 1) or 0) and keys[n] (both included) have value values[n]
    indices: Vec<Idx>,
    values: Vec<T>,
}

impl<T, Idx> PrefixSumVec<T, Idx> {
    /// Create a new, empty [`PrefixSumVec`].
    ///
    /// Does not allocate.
    pub fn new() -> Self {
        Self {
            indices: vec![],
            values: vec![],
        }
    }

    /// Clears the data structure, removing all contained values.
    ///
    /// This method has no effect on the allocated capacity. Due to the compressed representation,
    /// a repeating value in a contiguous sequence is dropped only once.
    pub fn clear(&mut self) {
        // TODO: needs panic safety.
        self.indices.clear();
        self.values.clear();
    }

    /// Get the current maximum index that can be used for indexing this data structure.
    ///
    /// Will return `None` while this data structure contains no elements. In practice this is the
    /// number of elements contained within data structure minus one.
    ///
    /// **Complexity**: `O(1)`
    pub fn max_index(&self) -> Option<&Idx> {
        self.indices.last()
    }

    fn grow_if_necessary(&mut self) -> Result<(), std::collections::TryReserveError> {
        if self.values.len() == self.values.capacity() {
            self.values.try_reserve(1)?;
        }
        if self.indices.len() == self.indices.capacity() {
            self.values.try_reserve(1)?;
        }
        Ok(())
    }
}

impl<T, Idx: Ord> PrefixSumVec<T, Idx> {
    /// Find the value by an index.
    ///
    /// If the value at this index is not present in the data structure, `None` is returned.
    ///
    /// **Complexity**: `O(log n)`
    pub fn get(&self, index: &Idx) -> Option<&T> {
        match self.indices.binary_search(index) {
            // If this index would be inserted at the end of the list, then the
            // index is out of bounds and we return a None.
            //
            // If `Ok` is returned we found the index exactly, or if `Err` is
            // returned the position is the one which is the least index
            // greater than `idx`, which is still the type of `idx` according
            // to our "compressed" representation. In both cases we access the
            // list at index `i`.
            Ok(i) | Err(i) => self.values.get(i),
        }
    }
}

impl<T, Idx: Index> PrefixSumVec<T, Idx> {

    fn new_index(&self, additional_count: Idx) -> Result<Idx, TryPushError> {
        match self.max_index() {
            Some(current_max) => additional_count
                .checked_add(current_max)
                .ok_or(TryPushError::Overflow),
            None => Ok(additional_count.decrement()),
        }
    }

    /// Append `count` copies of a `value` to the back of the collection.
    ///
    /// This method will not inspect the values currently stored in the data structure in search
    /// for further compression opportunities. Instead the provided value will be stored compressed
    /// as-if a sequence of `count` repeating `value`s were provided.
    ///
    /// **Complexity**: `O(1)` amortized.
    pub fn try_push_many(&mut self, count: Idx, value: T) -> Result<(), TryPushError> {
        if count.is_zero() {
            return Ok(());
        }
        let new_index = self.new_index(count)?;
        self.grow_if_necessary().map_err(TryPushError::Reserve)?;
        self.indices.push(new_index);
        self.values.push(value);
        Ok(())
    }
}

impl<T: PartialEq, Idx: Index> PrefixSumVec<T, Idx> {
    /// Append `count` copies of a `value` to the back of the collection, attempting compression.
    ///
    /// This method will not inspect the values currently stored in the data structure in search
    /// for further compression opportunities. Instead the provided value will be stored compressed
    /// as-if a sequence of `count` repeating `value`s were provided.
    ///
    /// **Complexity**: `O(1)` amortized.
    pub fn try_push_more(&mut self, count: Idx, value: T) -> Result<(), TryPushError> {
        if count.is_zero() {
            return Ok(());
        }
        if let Some(lastval) = self.values.last_mut() {
            if PartialEq::eq(&*lastval, &value) {
                // We can "just" increment the index.
                let new_index = self.new_index(count)?;
                let old_index = self.indices.pop();
                self.indices.push(new_index);
                // This drop gives some panic safety. We don’t end up with oddly mismatching
                // index and value array lengths if dropping the `Idx` panics.
                drop(old_index);
                return Ok(());
            }
        }
        self.try_push_many(count, value)
    }
}

/// A type suitable for indexing into [`PrefixSumVec`].
pub trait Index: Sized {
    /// Is the value `zero`?
    fn is_zero(&self) -> bool;
    /// Decrement the index by one.
    ///
    /// This will never be called for values for which `is_zero` returns `true`.
    fn decrement(self) -> Self;
    /// Add the two indices together, checking for overflow.
    ///
    /// In case an overflow occurs, this must return `None`.
    fn checked_add(self, other: &Self) -> Option<Self>;
}

macro_rules! impl_primitive_index {
    ($($ty: ty),*) => {
        $(impl Index for $ty {
            fn is_zero(&self) -> bool { *self == 0 }
            fn decrement(self) -> Self { self - 1 }
            fn checked_add(self, other: &Self) -> Option<Self> { self.checked_add(*other) }
        })*
    }
}

impl_primitive_index!(u8, u16, u32, u64, u128, usize);
impl_primitive_index!(i8, i16, i32, i64, i128, isize);

impl<K, Idx: Ord> std::ops::Index<Idx> for PrefixSumVec<K, Idx> {
    type Output = K;

    fn index(&self, index: Idx) -> &Self::Output {
        self.get(&index).expect("index out of range")
    }
}

impl<K, Idx: Ord> std::ops::Index<&Idx> for PrefixSumVec<K, Idx> {
    type Output = K;

    fn index(&self, index: &Idx) -> &Self::Output {
        self.get(index).expect("index out of range")
    }
}

#[non_exhaustive]
#[derive(Debug, PartialEq, Eq)]
pub enum TryPushError {
    /// Reserving the additional space for the storage has failed.
    Reserve(std::collections::TryReserveError),
    /// The index cannot contain the values required to represent the count of values.
    Overflow,
}

impl std::error::Error for TryPushError {}
impl std::fmt::Display for TryPushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TryPushError::Reserve(_) => "could not reserve additional capacity",
            TryPushError::Overflow => "the index type overflowed",
        })
    }
}

// TODO: should we instead process the element so that it returns (count, value) rather than
// (prefix_sum, value)?
pub struct CompressedIter<'a, T, Idx> {
    inner: std::iter::Zip<std::slice::Iter<'a, Idx>, std::slice::Iter<'a, T>>,
}

impl<'a, T, Idx> Iterator for CompressedIter<'a, T, Idx> {
    type Item = (&'a Idx, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    fn count(self) -> usize {
        self.inner.count()
    }

    fn last(self) -> Option<Self::Item> {
        self.inner.last()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n)
    }

    fn for_each<F: FnMut(Self::Item)>(self, f: F) {
        self.inner.for_each(f)
    }

    fn collect<B: std::iter::FromIterator<Self::Item>>(self) -> B {
        self.inner.collect()
    }

    fn partition<B, F>(self, f: F) -> (B, B)
    where
        B: Default + Extend<Self::Item>,
        F: FnMut(&Self::Item) -> bool,
    {
        self.inner.partition(f)
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.inner.fold(init, f)
    }

    fn reduce<F>(self, f: F) -> Option<Self::Item>
    where
        F: FnMut(Self::Item, Self::Item) -> Self::Item,
    {
        self.inner.reduce(f)
    }

    fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.inner.all(f)
    }

    fn any<F: FnMut(Self::Item) -> bool>(&mut self, f: F) -> bool {
        self.inner.any(f)
    }

    fn find<P: FnMut(&Self::Item) -> bool>(&mut self, predicate: P) -> Option<Self::Item> {
        self.inner.find(predicate)
    }

    fn find_map<B, F: FnMut(Self::Item) -> Option<B>>(&mut self, f: F) -> Option<B> {
        self.inner.find_map(f)
    }

    fn position<P: FnMut(Self::Item) -> bool>(&mut self, predicate: P) -> Option<usize> {
        self.inner.position(predicate)
    }

    fn rposition<P: FnMut(Self::Item) -> bool>(&mut self, predicate: P) -> Option<usize> {
        self.inner.rposition(predicate)
    }

    fn max(self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        self.inner.max()
    }

    fn min(self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        self.inner.min()
    }

    fn max_by_key<B: Ord, F: FnMut(&Self::Item) -> B>(self, f: F) -> Option<Self::Item> {
        self.inner.max_by_key(f)
    }

    fn max_by<F>(self, compare: F) -> Option<Self::Item>
    where
        F: FnMut(&Self::Item, &Self::Item) -> std::cmp::Ordering,
    {
        self.inner.max_by(compare)
    }

    fn min_by_key<B: Ord, F: FnMut(&Self::Item) -> B>(self, f: F) -> Option<Self::Item> {
        self.inner.min_by_key(f)
    }

    fn min_by<F>(self, compare: F) -> Option<Self::Item>
    where
        F: FnMut(&Self::Item, &Self::Item) -> std::cmp::Ordering,
    {
        self.inner.min_by(compare)
    }

    fn sum<S: std::iter::Sum<Self::Item>>(self) -> S {
        self.inner.sum()
    }

    fn product<P: std::iter::Product<Self::Item>>(self) -> P {
        self.inner.product()
    }

    fn cmp<I>(self, other: I) -> std::cmp::Ordering
    where
        I: IntoIterator<Item = Self::Item>,
        Self::Item: Ord,
    {
        self.inner.cmp(other)
    }

    fn partial_cmp<I>(self, other: I) -> Option<std::cmp::Ordering>
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
    {
        self.inner.partial_cmp(other)
    }

    fn eq<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialEq<I::Item>,
    {
        self.inner.eq(other)
    }

    fn ne<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialEq<I::Item>,
    {
        self.inner.ne(other)
    }

    fn lt<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
    {
        self.inner.lt(other)
    }

    fn le<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
        Self: Sized,
    {
        self.inner.le(other)
    }

    fn gt<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
    {
        self.inner.gt(other)
    }

    fn ge<I>(self, other: I) -> bool
    where
        I: IntoIterator,
        Self::Item: PartialOrd<I::Item>,
    {
        self.inner.ge(other)
    }
}

impl<'a, T, Idx> DoubleEndedIterator for CompressedIter<'a, T, Idx> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back()
    }
}

impl<'a, T, Idx> ExactSizeIterator for CompressedIter<'a, T, Idx> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, T, Idx> std::iter::FusedIterator for CompressedIter<'a, T, Idx> {}
impl<'a, T, Idx> CompressedIter<'a, T, Idx> {
    #[allow(dead_code)]
    fn _assert_fused_iterator(mut self) {
        fn is_fused<T: std::iter::FusedIterator>(_: T) {}
        is_fused(&mut self.inner);
    }
}

impl<'a, T, Idx> IntoIterator for &'a PrefixSumVec<T, Idx> {
    type Item = (&'a Idx, &'a T);
    type IntoIter = CompressedIter<'a, T, Idx>;
    fn into_iter(self) -> Self::IntoIter {
        CompressedIter {
            inner: self.indices.iter().zip(self.values.iter()),
        }
    }
}

// TODO: should we implement something like this (an iterator that “decompresses” the
// CompressedIter)
// impl<'a, T, Idx: Clone> CompressedIter<'a, T, Idx> {
//     fn expand(mut self) -> Iter<'a, T, Idx> {
//         Iter {
//             last: self.next().map(|(idx, val)| (idx.clone(), val)),
//             inner: self,
//         }
//     }
// }


#[cfg(test)]
mod tests {
    use super::{PrefixSumVec, TryPushError};

    #[test]
    fn empty_partial_map() {
        let map = PrefixSumVec::<u32, u32>::new();
        assert_eq!(None, map.get(&0));
        assert_eq!(None, map.max_index());
    }

    #[test]
    fn basic_function() {
        let mut map = PrefixSumVec::<u32, u32>::new();
        assert_eq!(None, map.max_index());
        for i in 0..10 {
            map.try_push_many(1, i).unwrap();
            assert_eq!(Some(&i), map.max_index());
        }
        for i in 0..10 {
            assert_eq!(Some(&i), map.get(&i));
        }
        assert_eq!(None, map.get(&10));
        assert_eq!(None, map.get(&0xFFFF_FFFF));
    }

    #[test]
    fn zero_count() {
        let mut map = PrefixSumVec::<u32, u32>::new();
        assert_eq!(Ok(()), map.try_push_many(0, 0));
        assert_eq!(None, map.max_index());
        assert_eq!(Ok(()), map.try_push_many(10, 42));
        assert_eq!(Some(&9), map.max_index());
        assert_eq!(Ok(()), map.try_push_many(0, 43));
        assert_eq!(Some(&9), map.max_index());
    }

    #[test]
    fn close_to_limit() {
        let mut map = PrefixSumVec::<u32, u32>::new();
        assert_eq!(Ok(()), map.try_push_many(0xFFFF_FFFE, 42));
        // we added values 0..=0xFFFF_FFFD
        assert_eq!(Some(&42), map.get(&0xFFFF_FFFD));
        assert_eq!(None, map.get(&0xFFFF_FFFE));

        assert_eq!(Err(TryPushError::Overflow), map.try_push_many(100, 93));
        // overflowing does not change the map
        assert_eq!(Some(&42), map.get(&0xFFFF_FFFD));
        assert_eq!(None, map.get(&0xFFFF_FFFE));

        assert_eq!(Ok(()), map.try_push_many(1, 322));
        // we added value at index 0xFFFF_FFFE (which is the 0xFFFF_FFFFth value)
        assert_eq!(Some(&42), map.get(&0xFFFF_FFFD));
        assert_eq!(Some(&322), map.get(&0xFFFF_FFFE));
        assert_eq!(None, map.get(&0xFFFF_FFFF));

        assert_eq!(Err(TryPushError::Overflow), map.try_push_many(2, 1234));
        // can't add that much more stuff...
        assert_eq!(Some(&42), map.get(&0xFFFF_FFFD));
        assert_eq!(Some(&322), map.get(&0xFFFF_FFFE));
        assert_eq!(Some(&0xFFFF_FFFE), map.max_index());
        assert_eq!(None, map.get(&0xFFFF_FFFF));

        assert_eq!(Ok(()), map.try_push_many(1, 1234));
        // but we can add just one more value still.
        assert_eq!(Some(&42), map.get(&0xFFFF_FFFD));
        assert_eq!(Some(&322), map.get(&0xFFFF_FFFE));
        assert_eq!(Some(&0xFFFF_FFFF), map.max_index());
        assert_eq!(Some(&1234), map.get(&0xFFFF_FFFF));

        assert_eq!(Ok(()), map.try_push_many(0, 12345));
        // no more capacity.
        assert_eq!(Err(TryPushError::Overflow), map.try_push_many(1, 12345));
    }

    #[test]
    fn try_push_more() {
        let mut map = PrefixSumVec::<u32, u32>::new();
        assert_eq!(Ok(()), map.try_push_more(0, 0));
        assert_eq!(None, map.max_index());
        assert_eq!(Ok(()), map.try_push_more(10, 42));
        assert_eq!(Some(&9), map.max_index());
        assert_eq!(Ok(()), map.try_push_more(0, 43));
        assert_eq!(Some(&9), map.max_index());
        assert_eq!(Ok(()), map.try_push_more(10, 42));
        assert_eq!(Some(&19), map.max_index());
        assert_eq!(Ok(()), map.try_push_more(10, 44));
        assert_eq!(Some(&29), map.max_index());
        let representation = map.into_iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>();
        assert_eq!(vec![(19, 42), (29, 44)], representation);
    }
}
