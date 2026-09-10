//! One ring of words with two index registers, as the driver lays every queue out.
//!
//! - One slot stays empty so full and empty differ.
//! - [`Producer`] and [`Consumer`] each remember their own index and read the other side's register
//!   only when they must: across PCIe that read is the expensive step.
//! - A [`Ring`] becomes exactly one side, so it is not `Copy`: two live sides would each remember
//!   an index the other moves. A side made anew re-reads its register.

use core::cell::Cell;
use core::task::Poll;

use crate::reg::Reg;

#[derive(Debug)]
pub struct Ring {
    entries: Reg<u64>,
    producer: Reg<u32>,
    consumer: Reg<u32>,
    capacity: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// An index register held a value at or past the capacity.
    Index,
    /// More words were asked for than the ring holds.
    Words,
}

impl Ring {
    /// Panics below two entries: the empty slot would leave no room to hold one.
    ///
    /// # Safety
    ///
    /// `entries` must reach `capacity` words, and all three must stay mapped while the ring is used.
    pub const unsafe fn new(entries: Reg<u64>, producer: Reg<u32>, consumer: Reg<u32>, capacity: usize) -> Self {
        assert!(capacity >= 2, "a ring keeps one slot empty, so it holds two at least");
        Self {
            entries,
            producer,
            consumer,
            capacity,
        }
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Entries from `consumer` up to `producer`, both below the capacity.
    pub const fn distance(&self, producer: usize, consumer: usize) -> usize {
        (producer + self.capacity - consumer) % self.capacity
    }

    /// Entries a producer at `producer` may still write with a consumer at `consumer`.
    pub const fn space(&self, producer: usize, consumer: usize) -> usize {
        self.capacity - self.distance(producer, consumer) - 1
    }

    pub const fn wrap(&self, index: usize) -> usize {
        index % self.capacity
    }

    fn checked(&self, index: u32) -> Result<usize, Error> {
        let index = index as usize;
        (index < self.capacity).then_some(index).ok_or(Error::Index)
    }

    fn read(&self, index: usize) -> u64 {
        // SAFETY: `new` promised `capacity` entries and `index` is below it.
        unsafe { self.entries.offset(index) }.read()
    }

    fn write(&self, index: usize, word: u64) {
        // SAFETY: as in `read`.
        unsafe { self.entries.offset(index) }.write(word)
    }

    // On the device the rings are Device memory, which the core keeps in order only within one
    // peripheral; the entries and the index registers are not one, so every hand-off has a barrier.

    /// Orders index reads before the entry reads that follow.
    fn acquire() {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: a barrier alone.
        unsafe {
            core::arch::asm!("dmb oshld", options(nostack, preserves_flags))
        };
        #[cfg(not(target_arch = "aarch64"))]
        core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
    }

    /// Orders entry writes before the producer index write that follows.
    fn release() {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: a barrier alone.
        unsafe {
            core::arch::asm!("dmb oshst", options(nostack, preserves_flags))
        };
        #[cfg(not(target_arch = "aarch64"))]
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    }

    /// Orders entry reads before the consumer index write that follows; a store-only barrier
    /// would let the producer overwrite an entry still being read.
    fn consumed() {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: a barrier alone.
        unsafe {
            core::arch::asm!("dmb osh", options(nostack, preserves_flags))
        };
        #[cfg(not(target_arch = "aarch64"))]
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    }
}

/// The writing side of a ring. It owns the producer index and remembers the space it last saw
/// below the consumer, so a write that fits costs no register read.
#[derive(Debug)]
pub struct Producer {
    ring: Ring,
    index: Cell<usize>,
    space: Cell<usize>,
}

impl Producer {
    /// Takes the producer side over, reading both indices once.
    pub fn new(ring: Ring) -> Result<Self, Error> {
        let index = ring.checked(ring.producer.read())?;
        let consumer = ring.checked(ring.consumer.read())?;
        let space = ring.space(index, consumer);
        Ok(Self {
            ring,
            index: Cell::new(index),
            space: Cell::new(space),
        })
    }

    /// Words that fit right now, reading the consumer's register.
    pub fn space(&self) -> Result<usize, Error> {
        let consumer = self.ring.checked(self.ring.consumer.read())?;
        let space = self.ring.space(self.index.get(), consumer);
        self.space.set(space);
        Ok(space)
    }

    /// Whether `words` fit, reading the consumer's register only when the remembered space says
    /// they do not.
    pub fn fits(&self, words: usize) -> Result<bool, Error> {
        Ok(words <= self.space.get() || words <= self.space()?)
    }

    /// Writes every word and moves the index once; pending, with the ring unchanged, while they
    /// do not fit.
    pub fn write_all(&self, words: &[u64]) -> Poll<Result<(), Error>> {
        match self.fits(words.len()) {
            Ok(true) => {}
            Ok(false) => return Poll::Pending,
            Err(error) => return Poll::Ready(Err(error)),
        }
        let mut index = self.index.get();
        for &word in words {
            self.ring.write(index, word);
            index = self.ring.wrap(index + 1);
        }
        if !words.is_empty() {
            Ring::release();
            self.ring.producer.write(index as u32);
            self.index.set(index);
            self.space.set(self.space.get() - words.len());
        }
        Poll::Ready(Ok(()))
    }
}

/// The reading side of a ring. It owns the consumer index; the producer's register is read on
/// every look, since new words arrive through it.
#[derive(Debug)]
pub struct Consumer {
    ring: Ring,
    index: Cell<usize>,
}

impl Consumer {
    /// Takes the consumer side over, reading its index once.
    pub fn new(ring: Ring) -> Result<Self, Error> {
        let index = ring.checked(ring.consumer.read())?;
        Ok(Self {
            ring,
            index: Cell::new(index),
        })
    }

    /// Words published and not yet consumed.
    pub fn available(&self) -> Result<usize, Error> {
        let producer = self.ring.checked(self.ring.producer.read())?;
        Ring::acquire();
        Ok(self.ring.distance(producer, self.index.get()))
    }

    /// Copies published words from `offset` past the index, without moving it.
    pub fn copy(&self, offset: usize, output: &mut [u64]) -> Result<(), Error> {
        let available = self.available()?;
        if offset > available || output.len() > available - offset {
            return Err(Error::Words);
        }
        for (step, word) in output.iter_mut().enumerate() {
            *word = self.ring.read(self.ring.wrap(self.index.get() + offset + step));
        }
        Ok(())
    }

    /// Moves the index past `words` copied words.
    pub fn consume(&self, words: usize) -> Result<(), Error> {
        if words > self.available()? {
            return Err(Error::Words);
        }
        if words != 0 {
            let index = self.ring.wrap(self.index.get() + words);
            Ring::consumed();
            self.ring.consumer.write(index as u32);
            self.index.set(index);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(entries: &mut [u64], producer: &mut u32, consumer: &mut u32) -> Ring {
        // SAFETY: the test owns this memory for the ring's lifetime.
        unsafe {
            Ring::new(
                Reg::new(entries.as_mut_ptr()),
                Reg::new(producer),
                Reg::new(consumer),
                entries.len(),
            )
        }
    }

    #[test]
    fn count_wraps_one_slot_empty() {
        let (mut entries, mut producer, mut consumer) = ([0; 127], 0, 0);
        let ring = ring(&mut entries, &mut producer, &mut consumer);

        assert_eq!(ring.distance(5, 2), 3);
        assert_eq!(ring.distance(2, 125), 4);
        assert_eq!(ring.space(2, 2), 126);
        assert_eq!(ring.space(1, 2), 0);
        assert_eq!(ring.wrap(127), 0);
    }

    #[test]
    fn fill_drain_across_wrap() {
        // Each side is a party of its own, so each gets its own ring over the same registers.
        let (mut entries, mut producer, mut consumer) = ([0; 4], 2, 2);
        let writer = Producer::new(ring(&mut entries, &mut producer, &mut consumer)).unwrap();
        let reader = Consumer::new(ring(&mut entries, &mut producer, &mut consumer)).unwrap();

        assert_eq!(writer.write_all(&[7, 8, 9]), Poll::Ready(Ok(())));
        assert_eq!(writer.write_all(&[10]), Poll::Pending);
        assert_eq!(reader.available(), Ok(3));
        let mut words = [0; 2];
        reader.copy(1, &mut words).unwrap();
        assert_eq!(words, [8, 9]);
        assert_eq!(reader.copy(2, &mut words), Err(Error::Words));
        reader.consume(3).unwrap();
        assert_eq!(reader.available(), Ok(0));
        assert_eq!(writer.write_all(&[10]), Poll::Ready(Ok(())));
        assert_eq!(consumer, 1);
        assert_eq!(producer, 2);
    }

    #[test]
    fn rejects_a_ring_of_one_entry() {
        let (mut entries, mut producer, mut consumer) = ([0; 1], 0, 0);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ring(
                &mut entries,
                &mut producer,
                &mut consumer
            )))
            .is_err()
        );
    }

    #[test]
    fn rejects_an_index_past_the_capacity() {
        let (mut entries, mut producer, mut consumer) = ([0; 4], 4, 0);
        assert_eq!(
            Producer::new(ring(&mut entries, &mut producer, &mut consumer)).err(),
            Some(Error::Index)
        );
        assert_eq!(
            Consumer::new(ring(&mut entries, &mut producer, &mut consumer))
                .unwrap()
                .available(),
            Err(Error::Index)
        );
    }
}
