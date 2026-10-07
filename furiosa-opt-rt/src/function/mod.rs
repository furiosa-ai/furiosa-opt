//! A device function, loaded on a runtime from its image; launch it with buffers; wait for the
//! launch.

use std::sync::Arc;

use furiosa_opt_abi::args;
use furiosa_opt_abi::image::{Image, Memory, Slot};
use furiosa_opt_ipc::{ProfileRecord, ProfileRequest, Staged};

use crate::buffer::{Buffer, Reservation};
use crate::{Device, Error, Result};

mod profile;
use profile::Profile;
pub use profile::Span;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum FunctionError {
    #[error("the image is invalid: {0}")]
    Image(#[from] furiosa_opt_abi::image::Error),
    #[error("the binding is invalid: {0}")]
    InvalidBinding(&'static str),
    #[error("a device allocates device memory, not SRAM")]
    UnsupportedStaticMemory,
    #[error("slot {slot} expects a {expects:?} buffer, the argument is a {has:?} one")]
    WrongSlotMemory { slot: u32, expects: Memory, has: Memory },
    #[error("slot {slot} needs {needs} bytes on every device, the buffer has {has}")]
    BufferTooSmall { slot: u32, needs: u64, has: u64 },
    #[error("slot {0} is bound to different buffer bases")]
    ConflictingSlot(u32),
    #[error("slots {0} and {1} overlap and at least one is writable")]
    AliasedSlots(u32, u32),
    #[error("SRAM slot {0} is not aligned to the resident address alignment")]
    UnalignedSlot(u32),
    #[error("a {bytes} byte weight does not divide over {chips} chips")]
    WeightNotDivisible { bytes: usize, chips: usize },
    #[error("the image is for {image_chips} chips of {image_pes} PEs, the device has {device_chips} of {device_pes}")]
    WrongTopology {
        image_chips: u8,
        image_pes: u8,
        device_chips: usize,
        device_pes: u8,
    },
}

/// A device function loaded on one device: its image staged, its statics
/// allocated, launchable until dropped, when all of that goes back.
pub struct Function {
    device: Arc<Device>,
    /// The image in device memory, and which load put it there.
    staging: Buffer,
    token: u64,
    /// The weight, a reserved slot, then one buffer per stack: the arguments before the caller's.
    statics: [Buffer; args::IO_BEGIN],
    signature: Signature,
    profile: Profile,
}

/// A launch's argument bindings and memory requirements.
struct Signature {
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    slots: Vec<Slot>,
    /// One past the highest physical SRAM byte the function's temporaries use.
    sram_end: u64,
}

impl Function {
    /// Buffers one launch may bind, inputs and outputs together; the wire carries them inline.
    pub const MAX_ARGS: usize = furiosa_opt_ipc::MAX_LAUNCH_ARGS;

    /// Stages `image` on the device and allocates its statics.
    pub async fn load(device: &Arc<Device>, image: &Image<'_>) -> Result<Self> {
        if usize::from(image.chips()) != device.chips() || image.pes() != device.pes() {
            return Err(FunctionError::WrongTopology {
                image_chips: image.chips(),
                image_pes: image.pes(),
                device_chips: device.chips(),
                device_pes: device.pes(),
            }
            .into());
        }
        let dram = |kind| match kind {
            Memory::Dram => Ok(()),
            Memory::Sram => Err(Error::Function(FunctionError::UnsupportedStaticMemory)),
        };
        let weight = match image.weight() {
            Some(weight) => {
                dram(weight.kind)?;
                let chips = device.chips();
                if !weight.bytes.len().is_multiple_of(chips) {
                    return Err(FunctionError::WeightNotDivisible {
                        bytes: weight.bytes.len(),
                        chips,
                    }
                    .into());
                }
                let buffer = device.alloc(Memory::Dram, weight.bytes.len() / chips)?;
                device.write([(weight.bytes, buffer.on_all())]).await?;
                buffer
            }
            None => device.alloc(Memory::Dram, 0)?,
        };
        let statics = std::array::try_from_fn(|slot| -> Result<Buffer> {
            match slot {
                args::WEIGHT => Ok(weight.clone()),
                args::RESERVED => device.alloc(Memory::Dram, 0),
                _ => {
                    let stack = &image.stacks()[slot - args::STACK_BEGIN];
                    dram(stack.kind)?;
                    let size = usize::try_from(stack.size)
                        .map_err(|_| FunctionError::InvalidBinding("a stack does not fit this platform"))?;
                    device.alloc(Memory::Dram, size)
                }
            }
        })?;
        let bytes = image.staging().map_err(FunctionError::Image)?;
        let staging = device.alloc(Memory::Dram, bytes.len())?;
        device
            .write(device.ranks().map(|rank| (bytes.as_slice(), staging.on(rank))))
            .await?;
        Ok(Self {
            device: Arc::clone(device),
            token: device.tokens.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            staging,
            statics,
            signature: Signature {
                inputs: image.inputs().to_vec(),
                outputs: image.outputs().to_vec(),
                slots: image.slots().to_vec(),
                sram_end: image.sram_end(),
            },
            profile: Profile::new(image)?,
        })
    }

    /// The device this function is loaded on; arguments must be its buffers.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Submits one launch; the returned `Launch` borrows this function until it is waited. Waits
    /// only for room in the device queue, never for the launch itself. A launch dropped before its
    /// wait finished blocks there until the device answers, since the device still uses its buffers.
    pub fn launch(&self, inputs: &[Buffer], outputs: &[Buffer]) -> Result<Launch<'_>> {
        self.submit(inputs, outputs, None)
    }

    /// This function with its launches profiled at `level`, `Info` or finer: a launch made through it
    /// is waited for the spans the image names, in device cycles.
    pub fn profiled(&self, level: log::Level) -> Result<Profiled<'_>> {
        Ok(Profiled {
            function: self,
            request: self.profile.request(level)?,
        })
    }

    fn submit(&self, inputs: &[Buffer], outputs: &[Buffer], profile: Option<ProfileRequest>) -> Result<Launch<'_>> {
        if !inputs
            .iter()
            .chain(outputs)
            .all(|buffer| buffer.belongs_to(&self.device.allocations))
        {
            return Err(Error::ForeignBuffer);
        }
        let buffers = self.signature.bind(&self.statics, inputs, outputs)?;
        // Admitted before submission: until the device answers, no resident may be placed under
        // the function's temporaries.
        let reservation = self.device.allocations.reserve(self.signature.sram_end as usize)?;
        let ticket = self.device.engine.submit(&crate::device::engine::Launch {
            image: Staged {
                addr: self.staging.addr() as u64,
                len: self.staging.size() as u64,
                token: self.token,
            },
            args: &buffers,
            profile,
        })?;
        Ok(Launch {
            function: self,
            pending: Some(Pending {
                ticket,
                buffers,
                reservation,
            }),
        })
    }
}

/// One submitted launch. `wait` consumes it, so a launch completes once.
#[must_use = "a call holds its answer slot until it is waited"]
pub struct Launch<'function> {
    function: &'function Function,
    pending: Option<Pending>,
}

struct Pending {
    /// Held until the device has answered, through a wait or, for a launch dropped before
    /// that, a drain on drop: the operands go back only then.
    ticket: crate::device::engine::Ticket,
    /// The device addresses these until it answers, so the launch holds them rather than trusting
    /// the caller to keep its own handles alive.
    buffers: Vec<Buffer>,
    reservation: Reservation,
}

impl Launch<'_> {
    pub async fn wait(mut self) -> Result<()> {
        self.collect().await.map(drop)
    }

    /// Waits for the device's answer and lets go of the ticket only then: a wait dropped part-way
    /// leaves the ticket for the drop to drain.
    async fn collect(&mut self) -> Result<Vec<Vec<ProfileRecord>>> {
        let Some(pending) = self.pending.as_mut() else {
            return Ok(Vec::new());
        };
        let answered = self.function.device.engine.wait(&mut pending.ticket).await;
        self.finish();
        answered
    }

    fn finish(&mut self) {
        if let Some(Pending {
            buffers, reservation, ..
        }) = self.pending.take()
        {
            drop(buffers);
            drop(reservation);
        }
    }
}

impl Drop for Launch<'_> {
    fn drop(&mut self) {
        // Dropping unwaited would hand a running launch's operands to the next caller, so the
        // answer is collected here instead.
        if let Some(pending) = self.pending.as_mut() {
            self.function.device.engine.drain(&mut pending.ticket);
        }
        self.finish();
    }
}

/// A function whose launches are profiled; see [`Function::profiled`].
pub struct Profiled<'function> {
    function: &'function Function,
    /// `None` when the image names no spans: the launch runs unprofiled and reports none.
    request: Option<ProfileRequest>,
}

impl<'function> Profiled<'function> {
    /// Submits one launch on the same terms as [`Function::launch`].
    pub fn launch(&self, inputs: &[Buffer], outputs: &[Buffer]) -> Result<Trace<'function>> {
        self.function.submit(inputs, outputs, self.request).map(Trace)
    }
}

/// One submitted profiled launch; `wait` yields its spans.
#[must_use = "a call holds its answer slot until it is waited"]
pub struct Trace<'function>(Launch<'function>);

impl Trace<'_> {
    pub async fn wait(self) -> Result<Vec<Span>> {
        let Trace(mut launch) = self;
        let records = launch.collect().await?;
        launch.function.profile.resolve(&records)
    }
}

impl Signature {
    /// The launch's argument buffers: the statics, then each slot's buffer.
    /// Every slot must be bound, by an input, an output, or an input that doubles as the output
    /// when `outputs` is empty. Retains the buffers until the launch completes.
    fn bind(&self, statics: &[Buffer], inputs: &[Buffer], outputs: &[Buffer]) -> Result<Vec<Buffer>> {
        if inputs.len() != self.inputs.len()
            || (!outputs.is_empty() && outputs.len() != self.outputs.len())
            || (outputs.is_empty() && self.outputs.iter().any(|slot| !self.inputs.contains(slot)))
        {
            return Err(FunctionError::InvalidBinding("the supplied args do not match the image").into());
        }
        let mut bound: Vec<Option<&Buffer>> = vec![None; self.slots.len()];
        for (&index, buffer) in self.inputs.iter().zip(inputs).chain(self.outputs.iter().zip(outputs)) {
            if self.slots[index as usize].kind == Memory::Sram
                && let Some(previous) = bound[index as usize]
                && !previous.same_base(buffer)
            {
                return Err(FunctionError::ConflictingSlot(index).into());
            }
            bound[index as usize] = Some(buffer);
        }
        if statics.len() + self.slots.len() > Function::MAX_ARGS {
            return Err(
                FunctionError::InvalidBinding("the function takes more arguments than a launch carries").into(),
            );
        }
        let mut args = statics.to_vec();
        for (index, slot) in (0..).zip(&self.slots) {
            let buffer = bound[index as usize].ok_or(FunctionError::InvalidBinding("an arg slot is not bound"))?;
            let has = buffer.memory();
            if slot.kind != has {
                return Err(FunctionError::WrongSlotMemory {
                    slot: index,
                    expects: slot.kind,
                    has,
                }
                .into());
            }
            if (buffer.size() as u64) < slot.size {
                return Err(FunctionError::BufferTooSmall {
                    slot: index,
                    needs: slot.size,
                    has: buffer.size() as u64,
                }
                .into());
            }
            if has == Memory::Sram && !(buffer.addr() as u64).is_multiple_of(furiosa_opt_abi::dm::ALIGNMENT) {
                return Err(FunctionError::UnalignedSlot(index).into());
            }
            args.push(buffer.clone());
        }
        for (i, first) in bound.iter().enumerate() {
            for (j, second) in bound.iter().enumerate().skip(i + 1) {
                if self.slots[i].kind == Memory::Sram
                    && (self.outputs.contains(&(i as u32)) || self.outputs.contains(&(j as u32)))
                    && first.unwrap().overlaps(
                        self.slots[i].size as usize,
                        second.unwrap(),
                        self.slots[j].size as usize,
                    )
                {
                    return Err(FunctionError::AliasedSlots(i as u32, j as u32).into());
                }
            }
        }
        Ok(args)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::buffer::Allocations;

    fn dram(size: u64) -> Slot {
        Slot {
            kind: Memory::Dram,
            size,
        }
    }

    fn signature(inputs: Vec<u32>, outputs: Vec<u32>, slots: Vec<Slot>) -> Signature {
        Signature {
            inputs,
            outputs,
            slots,
            sram_end: 0,
        }
    }

    fn buffer(size: usize) -> Buffer {
        let allocations = Arc::new(Allocations::new(0..4096));
        allocations.alloc(Memory::Dram, size).expect("buffer")
    }

    #[tokio::test]
    async fn rejects_other_topology_image() {
        let device = Arc::new(Device::stub(&[0], 8));
        let image = Image::new(furiosa_opt_abi::image::Parts {
            stacks: vec![
                furiosa_opt_abi::image::Stack {
                    kind: furiosa_opt_abi::image::Memory::Dram,
                    size: 0,
                };
                furiosa_opt_abi::args::STACK_COUNT
            ],
            chips: 2,
            pes: 8,
            tasks: vec![vec![&[][..]; 4]],
            profile: furiosa_opt_abi::image::Profile {
                depth: Default::default(),
                spans: vec![vec![]],
            },
            ..Default::default()
        })
        .expect("well formed");

        assert_eq!(
            Function::load(&device, &image).await.err(),
            Some(Error::Function(FunctionError::WrongTopology {
                image_chips: 2,
                image_pes: 8,
                device_chips: 1,
                device_pes: 8,
            }))
        );
    }

    #[test]
    fn rejects_wrong_args_before_launch() {
        assert!(
            signature(vec![0], vec![1], vec![dram(1), dram(1)])
                .bind(&[], &[], &[])
                .is_err()
        );
    }

    #[test]
    fn empty_outputs_alias_input() {
        let buffer = buffer(1);

        let args = signature(vec![0], vec![0], vec![dram(1)])
            .bind(&[], std::slice::from_ref(&buffer), &[])
            .expect("aliased output binds");

        assert_eq!(
            args.iter().map(|buffer| buffer.bind(0)).collect::<Vec<_>>(),
            &[buffer.addr() as u64]
        );
    }

    #[test]
    fn empty_outputs_need_alias() {
        let buffer = buffer(1);

        assert!(
            signature(vec![0], vec![1], vec![dram(1), dram(1)])
                .bind(&[], std::slice::from_ref(&buffer), &[])
                .is_err()
        );
    }

    #[test]
    fn repeated_slot_base() {
        let allocations = Arc::new(Allocations::new(0..4096));
        let buffer = allocations.alloc(Memory::Sram, 512).unwrap();
        let signature = signature(
            vec![0],
            vec![0],
            vec![Slot {
                kind: Memory::Sram,
                size: 256,
            }],
        );
        assert!(
            signature
                .bind(&[], std::slice::from_ref(&buffer), std::slice::from_ref(&buffer))
                .is_ok()
        );
        assert_eq!(
            signature
                .bind(&[], std::slice::from_ref(&buffer), &[buffer.slice(256..512)])
                .err(),
            Some(Error::Function(FunctionError::ConflictingSlot(0)))
        );
    }

    #[test]
    fn preserves_dram_binding() {
        let buffer = buffer(512);
        let shared = [buffer.clone(), buffer.clone()];
        assert!(
            signature(vec![0, 1], vec![1], vec![dram(256); 2])
                .bind(&[], &shared, &[])
                .is_ok()
        );
        let output = buffer.slice(256..512);
        let args = signature(vec![0], vec![0], vec![dram(256)])
            .bind(&[], std::slice::from_ref(&buffer), std::slice::from_ref(&output))
            .unwrap();
        assert_eq!(
            args.iter().map(|buffer| buffer.bind(0)).collect::<Vec<_>>(),
            &[output.addr() as u64]
        );
    }

    #[test]
    fn rejects_writable_overlap() {
        let allocations = Arc::new(Allocations::new(0..4096));
        let buffer = allocations.alloc(Memory::Sram, 768).unwrap();
        let slot = Slot {
            kind: Memory::Sram,
            size: 512,
        };
        let inputs = [buffer.clone(), buffer.slice(256..768)];
        assert!(
            signature(vec![0, 1], vec![], vec![slot; 2])
                .bind(&[], &inputs, &[])
                .is_ok()
        );
        assert_eq!(
            signature(vec![0, 1], vec![1], vec![slot; 2])
                .bind(&[], &inputs, &[])
                .err(),
            Some(Error::Function(FunctionError::AliasedSlots(0, 1)))
        );
        let slot = Slot { size: 256, ..slot };
        assert!(
            signature(vec![0, 1], vec![1], vec![slot; 2])
                .bind(&[], &inputs, &[])
                .is_ok()
        );
    }

    #[test]
    fn rejects_unaligned_slices() {
        let allocations = Arc::new(Allocations::new(0..4096));
        let buffer = allocations.alloc(Memory::Sram, 512).unwrap();
        let slot = Slot {
            kind: Memory::Sram,
            size: 256,
        };
        assert_eq!(
            signature(vec![0], vec![], vec![slot])
                .bind(&[], &[buffer.slice(1..257)], &[])
                .err(),
            Some(Error::Function(FunctionError::UnalignedSlot(0)))
        );
    }

    #[test]
    fn rejects_a_buffer_smaller_than_its_slot() {
        let small = buffer(256);

        assert_eq!(
            signature(vec![0], vec![0], vec![dram(512)])
                .bind(&[], std::slice::from_ref(&small), &[])
                .err(),
            Some(Error::Function(FunctionError::BufferTooSmall {
                slot: 0,
                needs: 512,
                has: 256,
            }))
        );
        assert!(
            signature(vec![0], vec![0], vec![dram(256)])
                .bind(&[], std::slice::from_ref(&small), &[])
                .is_ok()
        );
    }

    #[test]
    fn rejects_wrong_memory() {
        let dram_buffer = buffer(256);
        let allocations = Arc::new(Allocations::new(0..4096));
        let resident = allocations.alloc(Memory::Sram, 256).expect("resident");
        let sram = Slot {
            kind: Memory::Sram,
            size: 256,
        };

        assert_eq!(
            signature(vec![0], vec![0], vec![sram])
                .bind(&[], std::slice::from_ref(&dram_buffer), &[])
                .err(),
            Some(Error::Function(FunctionError::WrongSlotMemory {
                slot: 0,
                expects: Memory::Sram,
                has: Memory::Dram,
            }))
        );
        assert_eq!(
            signature(vec![0], vec![0], vec![dram(256)])
                .bind(&[], std::slice::from_ref(&resident), &[])
                .err(),
            Some(Error::Function(FunctionError::WrongSlotMemory {
                slot: 0,
                expects: Memory::Dram,
                has: Memory::Sram,
            }))
        );
        let args = signature(vec![0], vec![0], vec![sram])
            .bind(&[], std::slice::from_ref(&resident), &[])
            .expect("a resident binds an SRAM slot");
        assert_eq!(args.iter().map(Buffer::memory).collect::<Vec<_>>(), &[Memory::Sram]);
        assert_eq!(
            args.iter().map(|buffer| buffer.bind(0)).collect::<Vec<_>>(),
            &[resident.addr() as u64]
        );
    }

    #[tokio::test]
    async fn releases_completed_launches() {
        let device = Arc::new(Device::stub(&[0], 1));
        let image = Image::new(furiosa_opt_abi::image::Parts {
            stacks: vec![
                furiosa_opt_abi::image::Stack {
                    kind: furiosa_opt_abi::image::Memory::Dram,
                    size: 0,
                };
                furiosa_opt_abi::args::STACK_COUNT
            ],
            chips: 1,
            pes: 1,
            tasks: vec![vec![&[][..]]],
            profile: furiosa_opt_abi::image::Profile {
                spans: vec![vec![]],
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
        let function = Function {
            device: Arc::clone(&device),
            staging: device.alloc(Memory::Dram, 1).unwrap(),
            token: 0,
            statics: std::array::from_fn(|_| device.alloc(Memory::Dram, 0).unwrap()),
            signature: Signature {
                sram_end: crate::dm::RESIDENT_END,
                ..signature(vec![0], vec![], vec![dram(1)])
            },
            profile: Profile::new(&image).unwrap(),
        };
        let filler = device.alloc(Memory::Dram, 2560).unwrap();
        for wait in [false, true] {
            let input = device.alloc(Memory::Dram, 1).unwrap();
            let address = input.addr();
            let launch = function.launch(std::slice::from_ref(&input), &[]).unwrap();
            drop(input);
            assert!(matches!(
                device.alloc(Memory::Dram, 1),
                Err(Error::Allocation(crate::AllocError::OutOfMemory { .. }))
            ));
            assert!(matches!(
                device.alloc(Memory::Sram, 1),
                Err(Error::Allocation(crate::AllocError::Busy { .. }))
            ));
            if wait {
                launch.wait().await.unwrap();
            } else {
                drop(launch);
            }
            assert_eq!(device.alloc(Memory::Dram, 1).unwrap().addr(), address);
            assert!(device.alloc(Memory::Sram, 1).is_ok());
        }
        drop(filler);
    }
}
