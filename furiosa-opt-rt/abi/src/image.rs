//! A compiled device function, one struct for the compiler that writes it and the host and device that
//! read it.
//!
//! - A bincode header, then the task code: each chunk on a [`TASK_ALIGN`](crate::image::TASK_ALIGN) line of its own, so a
//!   DMA engine moving lines carries one chunk without its neighbors.
//! - A chunk spans at most [`TASK_CHUNK_LIMIT`](crate::image::TASK_CHUNK_LIMIT): the device holds two at once.
//! - An image is built by [`Image::new`] or [`Image::parse`] and nothing else, so one that exists is
//!   well formed.

use alloc::vec::Vec;

use crate::{ABI_VERSION, CHIP_PES, CLUSTER_PES};

const MAGIC: [u8; 4] = *b"FOPT";
/// Every task chunk starts on this boundary and owns whole lines up to the next one.
pub const TASK_ALIGN: usize = 256;
/// The most bytes one task chunk may span; the compiler splits a function into chunks no larger.
pub const TASK_CHUNK_LIMIT: usize = 1 << 20;

#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    #[error("not a device function image")]
    Magic,
    #[error("image version {0} is unsupported")]
    Version(u32),
    #[error("image is truncated")]
    Truncated,
    #[error("image metadata is malformed")]
    Malformed,
    #[error("a task chunk exceeds {TASK_CHUNK_LIMIT} bytes")]
    Chunk,
    #[error("{chips} chips of {pes} PEs is no topology: a function uses 1 to {CHIP_PES} PEs of each chip")]
    Topology { chips: u8, pes: u8 },
    #[error("{columns} task columns do not serve {chips} chips of {pes} PEs")]
    Columns { columns: usize, chips: u8, pes: u8 },
    #[error("span marker {0} is not an id a profile hit can carry (below {MARKER_IDS})")]
    Marker(u16),
}

/// Which device memory a static lives in.
#[derive(bincode::Encode, bincode::Decode, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    Dram,
    Sram,
}

/// Constant data the function reads, loaded once with it.
#[derive(bincode::Encode, bincode::BorrowDecode, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weight<'a> {
    pub kind: Memory,
    pub bytes: &'a [u8],
}

/// Scratch the function needs, allocated once with it.
#[derive(bincode::Encode, bincode::Decode, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stack {
    pub kind: Memory,
    pub size: u64,
}

/// Marker ids a profile hit can carry: its low fourteen bits.
pub const MARKER_IDS: u16 = 1 << 14;

/// One argument slot a launch binds: what memory the task expects there and the least it may hold.
#[derive(bincode::Encode, bincode::Decode, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub kind: Memory,
    pub size: u64,
}

/// One profiled region of a chunk's code, between two profile markers.
#[derive(bincode::Encode, bincode::BorrowDecode, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span<'a> {
    pub name: &'a str,
    pub begin: u16,
    pub end: u16,
}

/// What a profiled call records: `depth` marker hits per chunk, and the spans those hits bound,
/// per chunk.
#[derive(bincode::Encode, bincode::BorrowDecode, Clone, Debug, PartialEq, Eq, Default)]
pub struct Profile<'a> {
    pub depth: u32,
    pub spans: Vec<Vec<Span<'a>>>,
}

/// What the compiler states about one device function; [`Image::new`] checks it. Arguments are the
/// weight, one buffer per stack, then one per slot; `inputs` and `outputs` index into `slots`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Parts<'a> {
    /// Devices the function runs on, and the PEs of each it uses.
    pub chips: u8,
    pub pes: u8,
    /// Task code per chunk, then per cluster: `chips` times the clusters `pes` touch.
    pub tasks: Vec<Vec<&'a [u8]>>,
    pub weight: Option<Weight<'a>>,
    pub stacks: Vec<Stack>,
    pub slots: Vec<Slot>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub profile: Profile<'a>,
}

/// One compiled device function, checked once: see [`Parts`] for what it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image<'a>(Parts<'a>);

/// Everything but the task bytes, which follow it on [`TASK_ALIGN`] in the order `tasks` lists
/// their lengths.
#[derive(bincode::Encode, bincode::BorrowDecode)]
struct Header<'a> {
    magic: [u8; 4],
    version: u32,
    chips: u8,
    pes: u8,
    tasks: Vec<Vec<u32>>,
    weight: Option<Weight<'a>>,
    stacks: Vec<Stack>,
    slots: Vec<Slot>,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    profile: Profile<'a>,
}

impl<'a> Image<'a> {
    /// The image `parts` describe, once they agree: one task column per cluster of the topology,
    /// every input and output naming a slot, every span's markers ids a profile hit can carry.
    pub fn new(parts: Parts<'a>) -> Result<Self, Error> {
        if parts.chips == 0 || !(1..=CHIP_PES).contains(&parts.pes) {
            return Err(Error::Topology {
                chips: parts.chips,
                pes: parts.pes,
            });
        }
        let columns = parts.tasks.first().map_or(0, Vec::len);
        if columns != usize::from(parts.chips) * usize::from(parts.pes.div_ceil(CLUSTER_PES)) {
            return Err(Error::Columns {
                columns,
                chips: parts.chips,
                pes: parts.pes,
            });
        }
        if parts.tasks.iter().any(|chunk| chunk.len() != columns) || parts.profile.spans.len() != parts.tasks.len() {
            return Err(Error::Malformed);
        }
        if parts.tasks.iter().flatten().any(|task| task.len() > TASK_CHUNK_LIMIT) {
            return Err(Error::Chunk);
        }
        if !parts
            .inputs
            .iter()
            .chain(&parts.outputs)
            .all(|&slot| (slot as usize) < parts.slots.len())
        {
            return Err(Error::Malformed);
        }
        if let Some(marker) = parts
            .profile
            .spans
            .iter()
            .flatten()
            .flat_map(|span| [span.begin, span.end])
            .find(|&marker| marker >= MARKER_IDS)
        {
            return Err(Error::Marker(marker));
        }
        Ok(Self(parts))
    }

    /// Devices the function runs on.
    pub fn chips(&self) -> u8 {
        self.0.chips
    }

    /// PEs of each chip the function uses.
    pub fn pes(&self) -> u8 {
        self.0.pes
    }

    /// Task code per chunk, then per cluster.
    pub fn tasks(&self) -> &[Vec<&'a [u8]>] {
        &self.0.tasks
    }

    pub fn weight(&self) -> Option<Weight<'a>> {
        self.0.weight
    }

    pub fn stacks(&self) -> &[Stack] {
        &self.0.stacks
    }

    pub fn slots(&self) -> &[Slot] {
        &self.0.slots
    }

    /// Slots a launch's inputs bind, by index into [`Self::slots`].
    pub fn inputs(&self) -> &[u32] {
        &self.0.inputs
    }

    /// Slots a launch's outputs bind, by index into [`Self::slots`].
    pub fn outputs(&self) -> &[u32] {
        &self.0.outputs
    }

    pub fn profile(&self) -> &Profile<'a> {
        &self.0.profile
    }

    /// The bytes the compiler writes: everything, the weight's bytes included.
    pub fn encode(&self) -> Result<Vec<u8>, Error> {
        let Parts {
            chips,
            pes,
            tasks,
            weight,
            stacks,
            slots,
            inputs,
            outputs,
            profile,
        } = &self.0;
        let header = Header {
            magic: MAGIC,
            version: ABI_VERSION,
            chips: *chips,
            pes: *pes,
            tasks: tasks
                .iter()
                .map(|chunk| chunk.iter().map(|task| task.len() as u32).collect())
                .collect(),
            weight: *weight,
            stacks: stacks.clone(),
            slots: slots.clone(),
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            profile: profile.clone(),
        };
        let mut out = bincode::encode_to_vec(header, config()).map_err(|_| Error::Malformed)?;
        for task in tasks.iter().flatten() {
            out.resize(out.len().next_multiple_of(TASK_ALIGN), 0);
            out.extend_from_slice(task);
        }
        out.resize(out.len().next_multiple_of(TASK_ALIGN), 0);
        Ok(out)
    }

    /// The bytes a cluster reads: the header and the tasks. The weight goes to a buffer of its own
    /// on the device, so its kind rides here and its bytes do not.
    pub fn staging(&self) -> Result<Vec<u8>, Error> {
        let weight = self.0.weight.map(|weight| Weight { bytes: &[], ..weight });
        Self(Parts {
            weight,
            ..self.0.clone()
        })
        .encode()
    }

    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        if bytes.get(..MAGIC.len()) != Some(&MAGIC) {
            return Err(Error::Magic);
        }
        let (header, used): (Header<'a>, _) =
            bincode::borrow_decode_from_slice(bytes, config()).map_err(|_| Error::Truncated)?;
        if header.version != ABI_VERSION {
            return Err(Error::Version(header.version));
        }
        let mut at = used;
        let mut tasks = Vec::with_capacity(header.tasks.len());
        for chunk in &header.tasks {
            let mut slots = Vec::with_capacity(chunk.len());
            for &len in chunk {
                let len = len as usize;
                if len > TASK_CHUNK_LIMIT {
                    return Err(Error::Chunk);
                }
                at = at.next_multiple_of(TASK_ALIGN);
                slots.push(bytes.get(at..at + len).ok_or(Error::Truncated)?);
                at += len;
            }
            tasks.push(slots);
        }
        Self::new(Parts {
            chips: header.chips,
            pes: header.pes,
            tasks,
            weight: header.weight,
            stacks: header.stacks,
            slots: header.slots,
            inputs: header.inputs,
            outputs: header.outputs,
            profile: header.profile,
        })
    }
}

/// Fixed-width little-endian integers, so the header is the same bytes on every host and device.
const fn config() -> bincode::config::Configuration<bincode::config::LittleEndian, bincode::config::Fixint> {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    /// Two chips of eight PEs: four clusters, two chunks.
    fn parts() -> Parts<'static> {
        Parts {
            chips: 2,
            pes: 8,
            tasks: vec![
                vec![b"one".as_slice(), b"two!", b"three", b"four"],
                vec![b"five", b"six", b"seven", b"eight"],
            ],
            weight: Some(Weight {
                kind: Memory::Dram,
                bytes: b"weights",
            }),
            stacks: vec![Stack {
                kind: Memory::Sram,
                size: 4096,
            }],
            slots: vec![
                Slot {
                    kind: Memory::Dram,
                    size: 64,
                },
                Slot {
                    kind: Memory::Dram,
                    size: 64,
                },
                Slot {
                    kind: Memory::Dram,
                    size: 128,
                },
            ],
            inputs: vec![0, 1],
            outputs: vec![2],
            profile: Profile {
                depth: 8,
                spans: vec![
                    vec![Span {
                        name: "first",
                        begin: 1,
                        end: 2,
                    }],
                    vec![],
                ],
            },
        }
    }

    fn image() -> Image<'static> {
        Image::new(parts()).expect("well formed")
    }

    #[test]
    fn parse_inverts_encode() {
        let image = image();
        let bytes = image.encode().expect("encodes");

        assert_eq!(Image::parse(&bytes).expect("parses"), image);
    }

    #[test]
    fn staging_drops_weight_bytes() {
        let bytes = image().staging().expect("stages");
        let staged = Image::parse(&bytes).expect("parses");

        assert_eq!(staged.weight().map(|weight| weight.bytes), Some(&[][..]));
        assert_eq!(staged.weight().map(|weight| weight.kind), Some(Memory::Dram));
        assert_eq!((staged.chips(), staged.pes(), staged.tasks()), (2, 8, image().tasks()));
    }

    #[test]
    fn places_every_chunk_on_its_own_lines() {
        let bytes = image().encode().expect("encodes");
        let parsed = Image::parse(&bytes).expect("parses");
        let offset = |task: &[u8]| task.as_ptr() as usize - bytes.as_ptr() as usize;

        for task in parsed.tasks().iter().flatten() {
            assert_eq!(offset(task) % TASK_ALIGN, 0);
        }
        assert_eq!(bytes.len() % TASK_ALIGN, 0);
        assert!(
            offset(parsed.tasks()[0][1]) - offset(parsed.tasks()[0][0]) >= TASK_ALIGN,
            "a short chunk still owns whole lines"
        );
    }

    #[test]
    fn rejects_other_versions_and_ragged_chunks() {
        let mut bytes = image().encode().expect("encodes");
        bytes[MAGIC.len()..MAGIC.len() + 4].copy_from_slice(&(ABI_VERSION + 1).to_le_bytes());
        assert_eq!(Image::parse(&bytes), Err(Error::Version(ABI_VERSION + 1)));
        assert_eq!(Image::parse(b"NOPE"), Err(Error::Magic));
        assert_eq!(Image::parse(b"FOPT"), Err(Error::Truncated));

        let mut ragged = parts();
        ragged.tasks[1].pop();
        assert_eq!(Image::new(ragged), Err(Error::Malformed));

        let mut unbound = parts();
        unbound.outputs = vec![3];
        assert_eq!(Image::new(unbound), Err(Error::Malformed));
    }

    #[test]
    fn rejects_a_topology_its_columns_do_not_serve() {
        let mut wide = parts();
        wide.chips = 1;
        assert_eq!(
            Image::new(wide),
            Err(Error::Columns {
                columns: 4,
                chips: 1,
                pes: 8,
            })
        );

        let mut narrow = parts();
        narrow.pes = 4;
        assert_eq!(
            Image::new(narrow),
            Err(Error::Columns {
                columns: 4,
                chips: 2,
                pes: 4,
            })
        );

        let mut none = parts();
        none.chips = 0;
        assert_eq!(Image::new(none), Err(Error::Topology { chips: 0, pes: 8 }));

        let mut many = parts();
        many.pes = CHIP_PES + 1;
        assert_eq!(
            Image::new(many),
            Err(Error::Topology {
                chips: 2,
                pes: CHIP_PES + 1,
            })
        );
    }

    #[test]
    fn rejects_a_marker_no_hit_can_carry() {
        let mut wide = parts();
        wide.profile.spans[0][0].end = MARKER_IDS;
        assert_eq!(Image::new(wide), Err(Error::Marker(MARKER_IDS)));

        let mut bytes = image().encode().expect("encodes");
        let marker = bytes
            .windows(4)
            .position(|window| window == [1, 0, 2, 0])
            .expect("the span's markers are consecutive u16s in the header");
        bytes[marker..marker + 2].copy_from_slice(&MARKER_IDS.to_le_bytes());
        assert_eq!(Image::parse(&bytes), Err(Error::Marker(MARKER_IDS)));
    }
}
