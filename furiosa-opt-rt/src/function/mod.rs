//! A device function, loaded on a runtime from its image; launch it with buffers; wait for the
//! launch.

use std::sync::Arc;

use furiosa_opt_abi::image::{Image, Memory, Slot};
use furiosa_opt_ipc::{ProfileRecord, ProfileRequest, Staged};

use crate::buffer::Buffer;
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
    #[error("slot {slot} expects SRAM, which a device's buffers are not")]
    UnsupportedSlotMemory { slot: u32 },
    #[error("slot {slot} needs {needs} bytes on every chip, the buffer has {has}")]
    BufferTooSmall { slot: u32, needs: u64, has: u64 },
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
    statics: Vec<Buffer>,
    signature: Signature,
    profile: Profile,
}

/// Which argument slots a launch binds, by index past the statics, and what each slot takes.
struct Signature {
    inputs: Vec<u32>,
    outputs: Vec<u32>,
    slots: Vec<Slot>,
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
                let buffer = device.alloc(weight.bytes.len() / chips)?;
                device.write([(weight.bytes, buffer.on_all())]).await?;
                buffer
            }
            None => device.alloc(0)?,
        };
        let mut statics = vec![weight, device.alloc(0)?];
        for stack in image.stacks() {
            dram(stack.kind)?;
            let size = usize::try_from(stack.size)
                .map_err(|_| FunctionError::InvalidBinding("a stack does not fit this platform"))?;
            statics.push(device.alloc(size)?);
        }
        let bytes = image.staging().map_err(FunctionError::Image)?;
        let staging = device.alloc(bytes.len())?;
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
            .all(|buffer| buffer.belongs_to(&self.device.allocator))
        {
            return Err(Error::ForeignBuffer);
        }
        let args = self.signature.bind(&self.statics, inputs, outputs)?;
        let ticket = self.device.engine.submit(&crate::device::engine::Launch {
            image: Staged {
                addr: self.staging.addr() as u64,
                len: self.staging.size() as u64,
                token: self.token,
            },
            args: args.words(),
            profile,
        })?;
        Ok(Launch {
            ticket: Some(ticket),
            function: self,
            operands: inputs.iter().chain(outputs).cloned().collect(),
        })
    }
}

/// One submitted launch. `wait` consumes it, so a launch completes once.
#[must_use = "a call holds its answer slot until it is waited"]
pub struct Launch<'function> {
    /// Held until the device has answered, through a wait or, for a launch dropped before
    /// that, a drain on drop: the operands go back only then.
    ticket: Option<crate::device::engine::Ticket>,
    function: &'function Function,
    /// The device addresses these until it answers, so the launch holds them rather than trusting
    /// the caller to keep its own handles alive.
    #[allow(dead_code)]
    operands: Vec<Buffer>,
}

impl Launch<'_> {
    pub async fn wait(mut self) -> Result<()> {
        self.collect().await.map(drop)
    }

    /// Waits for the device's answer and lets go of the ticket only then: a wait dropped part-way
    /// leaves the ticket for the drop to drain.
    async fn collect(&mut self) -> Result<Vec<Vec<ProfileRecord>>> {
        let Some(ticket) = self.ticket.as_mut() else {
            return Ok(Vec::new());
        };
        let answered = self.function.device.engine.wait(ticket).await;
        self.ticket = None;
        answered
    }
}

impl Drop for Launch<'_> {
    fn drop(&mut self) {
        // Dropping unwaited would hand a running launch's operands to the next caller, so the
        // answer is collected here instead.
        if let Some(ticket) = self.ticket.as_mut() {
            self.function.device.engine.drain(ticket);
        }
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
    /// The launch's arguments as offsets into device memory: the statics, then each slot's buffer.
    /// Every slot must be bound, by an input, an output, or an input that doubles as the output
    /// when `outputs` is empty.
    fn bind(&self, statics: &[Buffer], inputs: &[Buffer], outputs: &[Buffer]) -> Result<Args> {
        if inputs.len() != self.inputs.len()
            || (!outputs.is_empty() && outputs.len() != self.outputs.len())
            || (outputs.is_empty() && self.outputs.iter().any(|slot| !self.inputs.contains(slot)))
        {
            return Err(FunctionError::InvalidBinding("the supplied args do not match the image").into());
        }
        let bound = |slot: u32| {
            self.inputs
                .iter()
                .zip(inputs)
                .chain(self.outputs.iter().zip(outputs))
                .rev()
                .find_map(|(&bound, buffer)| (bound == slot).then_some(buffer))
        };
        let mut args = Args::default();
        for buffer in statics {
            args.push(buffer)?;
        }
        for (index, slot) in (0..).zip(&self.slots) {
            let buffer = bound(index).ok_or(FunctionError::InvalidBinding("an arg slot is not bound"))?;
            if slot.kind != Memory::Dram {
                return Err(FunctionError::UnsupportedSlotMemory { slot: index }.into());
            }
            if (buffer.size() as u64) < slot.size {
                return Err(FunctionError::BufferTooSmall {
                    slot: index,
                    needs: slot.size,
                    has: buffer.size() as u64,
                }
                .into());
            }
            args.push(buffer)?;
        }
        Ok(args)
    }
}

/// A launch's arguments as offsets into the runtime's device memory.
struct Args {
    words: [u64; Function::MAX_ARGS],
    len: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            words: [0; Function::MAX_ARGS],
            len: 0,
        }
    }
}

impl Args {
    fn push(&mut self, buffer: &Buffer) -> Result<()> {
        let slot = self.words.get_mut(self.len).ok_or(FunctionError::InvalidBinding(
            "the function takes more arguments than a launch carries",
        ))?;
        *slot = buffer.addr() as u64;
        self.len += 1;
        Ok(())
    }

    fn words(&self) -> &[u64] {
        &self.words[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::buffer::Allocator;

    fn dram(size: u64) -> Slot {
        Slot {
            kind: Memory::Dram,
            size,
        }
    }

    fn signature(inputs: Vec<u32>, outputs: Vec<u32>, slots: Vec<Slot>) -> Signature {
        Signature { inputs, outputs, slots }
    }

    fn buffer(size: usize) -> Buffer {
        let allocator = Arc::new(Mutex::new(Allocator::new(0..4096)));
        Buffer::alloc(&allocator, size).expect("buffer")
    }

    #[tokio::test]
    async fn rejects_other_topology_image() {
        let device = Arc::new(Device::stub(&[0], 8));
        let image = Image::new(furiosa_opt_abi::image::Parts {
            chips: 2,
            pes: 8,
            tasks: vec![vec![&[][..]; 4]],
            profile: furiosa_opt_abi::image::Profile {
                depth: 0,
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

        assert_eq!(args.words(), &[buffer.addr() as u64]);
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
    fn rejects_a_device_buffer_in_an_sram_slot() {
        let buffer = buffer(256);
        let sram = Slot {
            kind: Memory::Sram,
            size: 256,
        };

        assert_eq!(
            signature(vec![0], vec![0], vec![sram])
                .bind(&[], std::slice::from_ref(&buffer), &[])
                .err(),
            Some(Error::Function(FunctionError::UnsupportedSlotMemory { slot: 0 }))
        );
    }
}
