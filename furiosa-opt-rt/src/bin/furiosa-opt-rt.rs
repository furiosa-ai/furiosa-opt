//! Runs one compiled device function on a device: inputs and outputs are files of the exact bytes
//! each argument holds, so the tool is the call contract with no host code around it.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use furiosa_opt_rt::{Device, Function, FunctionError, Image};

#[derive(Debug, Parser)]
#[command(name = "furiosa-opt-rt")]
struct Invocation {
    /// Devices the device may be chosen from, comma-separated; any exposed device unless given.
    #[arg(long, value_delimiter = ',')]
    among: Option<Vec<u8>>,
    /// The compiled function; the device opens with the topology it was compiled for.
    #[arg(value_name = "function.bin")]
    image: PathBuf,
    /// One file per input argument, in declaration order, holding the argument's bytes for every
    /// device in chip order.
    #[arg(long = "input", value_name = "FILE")]
    inputs: Vec<PathBuf>,
    /// One path per output argument, in declaration order; written with the bytes the function
    /// left, for every chip in chip order.
    #[arg(long = "output", value_name = "FILE")]
    outputs: Vec<PathBuf>,
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let invocation = Invocation::parse();
    let bytes = std::fs::read(&invocation.image)?;
    let image = Image::parse(&bytes).map_err(FunctionError::Image)?;
    let mut builder = Device::builder((image.chips(), image.pes()));
    if let Some(among) = invocation.among.clone() {
        builder = builder.among(among);
    }
    let device = Arc::new(builder.open()?);
    let chips = device.chips();
    // Each argument's file holds every chip's bytes in chip order: the slot's size per chip.
    let slot = |index: u32| image.slots()[index as usize].size as usize;
    if invocation.inputs.len() != image.inputs().len() || invocation.outputs.len() != image.outputs().len() {
        return Err(format!(
            "the function takes {} inputs and {} outputs, not {} and {}",
            image.inputs().len(),
            image.outputs().len(),
            invocation.inputs.len(),
            invocation.outputs.len()
        )
        .into());
    }
    let function = Function::load(&device, &image).await?;

    let inputs = invocation
        .inputs
        .iter()
        .zip(image.inputs())
        .map(|(path, &index)| {
            let bytes = std::fs::read(path)?;
            if bytes.len() != slot(index) * chips {
                return Err(format!(
                    "{}: {} bytes, but the argument takes {} on each of {chips} chips",
                    path.display(),
                    bytes.len(),
                    slot(index)
                )
                .into());
            }
            Ok((bytes, device.alloc(slot(index))?))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    device
        .write(inputs.iter().map(|(bytes, buffer)| (bytes.as_slice(), buffer.on_all())))
        .await?;
    let outputs = image
        .outputs()
        .iter()
        .map(|&index| device.alloc(slot(index)))
        .collect::<Result<Vec<_>, _>>()?;

    let inputs = inputs.into_iter().map(|(_, buffer)| buffer).collect::<Vec<_>>();
    function.launch(&inputs, &outputs)?.wait().await?;

    let mut results = outputs
        .iter()
        .map(|buffer| vec![0; buffer.size() * chips])
        .collect::<Vec<_>>();
    device
        .read(
            outputs
                .iter()
                .zip(&mut results)
                .map(|(buffer, bytes)| (buffer.on_all(), bytes.as_mut_slice())),
        )
        .await?;
    invocation
        .outputs
        .iter()
        .zip(results)
        .try_for_each(|(path, bytes)| std::fs::write(path, bytes))?;
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `RUST_LOG=furiosa_opt_firmware=debug` sets what the clusters log; without a logger the level
    // it asks for is always off.
    env_logger::init();
    run().await
}

#[cfg(test)]
mod tests {
    use super::Invocation;
    use clap::Parser;

    #[test]
    fn the_image_is_required() {
        assert!(Invocation::try_parse_from(["furiosa-opt-rt", "--input", "a.bin"]).is_err());
    }

    #[test]
    fn arguments_keep_declaration_order() {
        let invocation = Invocation::try_parse_from([
            "furiosa-opt-rt",
            "--among",
            "2,3",
            "function.bin",
            "--input",
            "a.bin",
            "--input",
            "b.bin",
            "--output",
            "c.bin",
        ])
        .expect("valid invocation");

        assert_eq!(invocation.among, Some(vec![2, 3]));
        assert_eq!(invocation.inputs, ["a.bin", "b.bin"].map(std::path::PathBuf::from));
        assert_eq!(invocation.outputs, ["c.bin"].map(std::path::PathBuf::from));
    }
}
