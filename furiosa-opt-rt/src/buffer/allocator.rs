use std::ops::Range;

const ALIGN: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    at: usize,
    len: usize,
}

impl Span {
    const fn end(self) -> usize {
        self.at + self.len
    }
}

/// First-fit allocator over one span of a device memory window, tracking free spans in address
/// order.
pub(crate) struct Allocator {
    free: Vec<Span>,
}

impl Allocator {
    /// Hands out `free`, narrowed to whole aligned units.
    pub(crate) fn new(free: Range<usize>) -> Self {
        let at = free.start.next_multiple_of(ALIGN);
        let end = free.end & !(ALIGN - 1);
        let free = (end > at).then_some(Span { at, len: end - at }).into_iter().collect();
        Self { free }
    }

    pub(crate) fn alloc(&mut self, len: usize) -> Option<usize> {
        let len = Self::rounded(len)?;
        let index = self.free.iter().position(|span| span.len >= len)?;
        let span = self.free[index];
        if span.len == len {
            self.free.remove(index);
        } else {
            self.free[index] = Span {
                at: span.at + len,
                len: span.len - len,
            };
        }
        Some(span.at)
    }

    pub(crate) fn release(&mut self, at: usize, len: usize) {
        let span = Span {
            at,
            len: Self::rounded(len).expect("valid prior allocation length"),
        };
        let index = self.free.partition_point(|other| other.at < at);

        debug_assert!(
            index
                .checked_sub(1)
                .and_then(|previous| self.free.get(previous))
                .is_none_or(|previous| previous.end() <= at)
                && self.free.get(index).is_none_or(|next| span.end() <= next.at),
            "{at:#x}..{:#x} was already free",
            span.end(),
        );
        self.free.insert(index, span);
        self.merge(index);
    }

    const fn rounded(len: usize) -> Option<usize> {
        let len = if len == 0 { 1 } else { len };
        match len.checked_add(ALIGN - 1) {
            Some(len) => Some(len & !(ALIGN - 1)),
            None => None,
        }
    }

    fn merge(&mut self, index: usize) {
        if let Some(next) = self.free.get(index + 1).copied()
            && self.free[index].end() == next.at
        {
            self.free[index].len += next.len;
            self.free.remove(index + 1);
        }
        if let Some(previous) = index.checked_sub(1)
            && self.free[previous].end() == self.free[index].at
        {
            self.free[previous].len += self.free[index].len;
            self.free.remove(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_released_span() {
        let mut memory = Allocator::new(0..4096);
        let at = memory.alloc(256).expect("an allocation");

        memory.release(at, 256);

        assert_eq!(memory.alloc(256), Some(at));
    }

    #[test]
    fn allocates_from_the_start_of_its_span() {
        let mut memory = Allocator::new(1000..2048);

        assert_eq!(memory.alloc(256), Some(1024));
        assert_eq!(memory.alloc(1024), None);
        assert_eq!(memory.alloc(768), Some(1280));
    }
}
