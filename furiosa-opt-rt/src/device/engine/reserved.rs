use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use crate::device::dma::DmaQueue;
use crate::function::FunctionError;
use crate::{Error, Result};
use furiosa_opt_abi::bootloader;
use furiosa_opt_ipc::{Identity, MAX_GROUP, ProfileRecord, ProfileRequest, Request, Response, Staged};
use nix::errno::Errno;
use nix::fcntl::{Flock, FlockArg, Flockable};

use super::dram::Dram;
use super::link::Link;
use super::mmio::{CLUSTERS_PER_DEVICE, PeRange, node, open_rw};
use super::{Chip, DRAM_BASE};

/// Past the device's own launch limit, so a launch the device timed out still answers before the
/// host gives up on it.
const GRACE: Duration = Duration::from_secs(2);

/// The `log` target whose level the device inherits.
pub const DEVICE_LOG_TARGET: &str = "furiosa_opt_firmware";

/// Clusters one device has at most: every cluster of every chip.
pub(crate) const MAX_CLUSTERS: usize = MAX_GROUP * CLUSTERS_PER_DEVICE as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cluster {
    chip: u8,
    pes: PeRange,
}

impl Cluster {
    pub(crate) const fn new(chip: u8, pes: PeRange) -> Self {
        Self { chip, pes }
    }

    pub(crate) const fn chip(self) -> u8 {
        self.chip
    }

    pub(crate) const fn pes(self) -> PeRange {
        self.pes
    }

    /// The clusters at `placement` on each of `chips`, ascending by chip.
    pub(crate) fn across(chips: &[u8], placement: &[PeRange]) -> Result<Vec<Self>> {
        let unique = chips.iter().copied().collect::<BTreeSet<_>>();
        if unique.is_empty() {
            return Err(Error::Topology("a device requires at least one chip".into()));
        }
        if unique.len() != chips.len() {
            return Err(Error::Topology(format!("a chip is named twice in {chips:?}")));
        }
        Ok(unique
            .into_iter()
            .flat_map(|chip| placement.iter().map(move |&pes| Self::new(chip, pes)))
            .collect())
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct Peer {
    file: u32,
    npu: u32,
    dram: u64,
    size: u64,
}

#[repr(C, packed)]
struct GroupConfig {
    local: u32,
    ats: u16,
    remotes: u16,
    peers: [Peer; MAX_GROUP],
}

nix::ioctl_readwrite!(set_group, b'N', 0x05, GroupConfig);

/// The device's chips as the host holds them: joined in the driver, PEs held through fusion nodes,
/// one memory window per chip with the firmware staged. Launches go through the [`Launcher`].
pub(crate) struct Reserved {
    /// One per chip, in ascending chip order: a chip's dense index is its index here.
    drams: Vec<Dram>,
    _nodes: Vec<Flock<File>>,
}

/// A device's launch path: one link per cluster, and where the device's memory starts inside each
/// cluster's window at [`DRAM_BASE`].
///
/// A rank that fails a launch, or leaves it unanswered, poisons the device and every later launch
/// fails with that first failure: the other ranks may hold a half-run task nothing later can trust.
#[derive(Default)]
pub(crate) struct Launcher {
    links: Vec<Link>,
    windows: Vec<u64>,
    /// How long a launch may go unanswered before it fails.
    timeout: Duration,
    /// The first failure, once a rank has failed a launch.
    poison: Option<Error>,
}

/// One launch as the runtime states it: the staged image and the arguments as offsets into the
/// device's memory, which every cluster translates through its own window.
pub(crate) struct Launch<'a> {
    pub(crate) image: Staged,
    pub(crate) args: &'a [u64],
    pub(crate) profile: Option<ProfileRequest>,
}

/// One submitted launch: the request id every cluster answers it under, and the answers so far.
#[derive(Clone, Debug)]
pub(crate) struct Ticket {
    ids: [u32; MAX_CLUSTERS],
    answered: [bool; MAX_CLUSTERS],
    clusters: usize,
    deadline: Instant,
    /// Each cluster's profile records once it answered; empty when the launch is not profiled.
    records: Vec<Vec<ProfileRecord>>,
}

impl Reserved {
    /// Reserves and joins the chips of `clusters`, stages the firmware image on each through `dma`
    /// and brings every cluster up on it; a cluster fails a launch that runs past `timeout`.
    pub(crate) fn open(
        clusters: Vec<Cluster>,
        images: Images,
        timeout: Duration,
        dma: &mut DmaQueue,
    ) -> Result<(Self, Launcher)> {
        if clusters
            .iter()
            .enumerate()
            .any(|(index, cluster)| clusters[..index].contains(cluster))
        {
            return Err(Error::Topology(
                "a device topology names one link more than once".into(),
            ));
        }
        if clusters.len() > MAX_CLUSTERS {
            return Err(Error::Topology(format!(
                "a device has at most {MAX_CLUSTERS} clusters, not {}",
                clusters.len()
            )));
        }
        let mut members = clusters.iter().map(|cluster| cluster.chip()).collect::<Vec<_>>();
        members.sort_unstable();
        members.dedup();
        let first = *members
            .first()
            .ok_or_else(|| Error::Device("a device of no chips".into()))?;
        Identity::new(members.clone(), first)
            .map_err(|why| Error::Device(format!("{members:?} is no group: {why}")))?;
        let limit = Self::launch_limit(
            timeout,
            members.iter().filter_map(|&chip| Self::driver_limit(chip)).min(),
        )?;

        let nodes = Self::reserve(
            members
                .iter()
                .map(|&chip| open_rw(&node(chip, "dmar")).map(|dmar| (chip, dmar)))
                .collect::<Result<Vec<_>>>()?,
            members.len() > 1,
        )?;
        members
            .iter()
            .zip(&nodes)
            .try_for_each(|(&chip, dmar)| Self::join(chip, &members, dmar))?;

        // Time and pid: unique among the stagings a boot could confuse, no more.
        let token = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as u64)
            ^ u64::from(std::process::id()) << 48;
        // The host waits past the device's own limit, so a launch the device timed out still answers.
        let patience = Duration::from_secs(u64::from(limit)) + GRACE;
        // A chip's window is shared by its clusters, so its share is theirs together.
        let masks = clusters.iter().fold(BTreeMap::<u8, u8>::new(), |mut masks, cluster| {
            *masks.entry(cluster.chip()).or_default() |= cluster.pes().mask();
            masks
        });
        let mut drams = BTreeMap::new();
        let (links, windows) = clusters
            .iter()
            .enumerate()
            .map(|(column, cluster)| {
                let dram = match drams.entry(cluster.chip()) {
                    Entry::Occupied(dram) => dram.into_mut(),
                    Entry::Vacant(slot) => slot.insert(Dram::open(
                        cluster.chip(),
                        masks[&cluster.chip()],
                        images.device,
                        token,
                    )?),
                };
                let identity = Identity::new(members.clone(), cluster.chip())
                    .map_err(|why| Error::Device(format!("npu{} is outside this device: {why}", cluster.chip())))?;
                let mut link = Link::open(&identity, cluster.pes(), images.boot, patience)?;
                Self::boot(&mut link, dram, dma, images)?;
                Self::initialize(&mut link, &identity, column as u8, limit)?;
                Ok((link, dram.arg(0)?))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        Ok((
            Self {
                drams: drams.into_values().collect(),
                _nodes: nodes,
            },
            Launcher {
                links,
                windows,
                timeout: patience,
                poison: None,
            },
        ))
    }

    fn join(chip: u8, members: &[u8], dmar: &Flock<File>) -> Result<()> {
        let mut group = GroupConfig {
            local: u32::from(chip),
            ats: 0,
            remotes: (members.len() - 1) as u16,
            peers: [Peer::default(); MAX_GROUP],
        };
        for (slot, &peer) in group
            .peers
            .iter_mut()
            .zip(members.iter().filter(|&&member| member != chip))
        {
            *slot = Peer {
                file: u32::from(peer),
                npu: u32::from(peer),
                ..Peer::default()
            };
        }

        // SAFETY: the descriptor is this call's own, and `group` is the writable ioctl payload.
        unsafe { set_group(dmar.as_raw_fd(), &mut group) }
            .map_err(|why| Error::Device(format!("npu{chip} joining {members:?}: {why}")))?;
        Ok(())
    }

    /// Has the bootloader on `link` start the firmware image staged in `dram`. A device's first
    /// cluster finds the window, writing the image at each candidate until a boot lands on it.
    fn boot(link: &mut Link, dram: &mut Dram, dma: &mut DmaQueue, images: Images) -> Result<()> {
        let probing = dram.window().is_none();
        for candidate in dram.candidates() {
            if probing {
                dram.write(dma, candidate, images.device)?;
            }
            let placed = dram.placed();
            match link.boot(
                DRAM_BASE + candidate + placed.offset as u64,
                images.device.len() as u64,
                placed.token,
            )? {
                bootloader::Code::Booted => {
                    log::debug!("npu{} booted, its window at {candidate:#x}", dram.chip());
                    dram.located(candidate);
                    return Ok(());
                }
                bootloader::Code::NoImage => {}
                other => {
                    return Err(Error::Firmware(format!(
                        "the bootloader refused the firmware image: {other:?}"
                    )));
                }
            }
        }
        Err(Error::Memory(format!(
            "no candidate window showed the firmware image staged at {:#x}",
            dram.placed().offset
        )))
    }

    /// What the device logs, as the finest level this process's logger lets through for the
    /// `furiosa_opt_firmware` target; the device's lines reach `pe_log` at that level and above.
    fn device_log_level() -> u8 {
        [
            log::Level::Trace,
            log::Level::Debug,
            log::Level::Info,
            log::Level::Warn,
            log::Level::Error,
        ]
        .into_iter()
        .find(|&level| log::log_enabled!(target: DEVICE_LOG_TARGET, level))
        .map_or(0, |level| level as u8)
    }

    /// The firmware's launch limit, `timeout` in whole seconds rounded up: at least one, fitting
    /// the wire, and ending [`GRACE`] before the driver's own deadline `limit` when it states one.
    fn launch_limit(timeout: Duration, limit: Option<Duration>) -> Result<u32> {
        let secs = timeout
            .as_secs()
            .checked_add(u64::from(timeout.subsec_nanos() != 0))
            .and_then(|secs| u32::try_from(secs).ok())
            .ok_or_else(|| {
                Error::Timeout(format!(
                    "{timeout:?} does not fit the launch limit's 32 bits of seconds"
                ))
            })?;
        if secs == 0 {
            return Err(Error::Timeout("a launch limit of zero fails every launch".into()));
        }
        if let Some(limit) = limit
            && Duration::from_secs(u64::from(secs)) + GRACE >= limit
        {
            return Err(Error::Timeout(format!(
                "a {secs}s launch limit plus the {}s grace for its answer reaches the driver's {}ms deadline",
                GRACE.as_secs(),
                limit.as_millis()
            )));
        }
        Ok(secs)
    }

    /// How long the driver leaves a doorbell unanswered before failing it, as `chip` states it;
    /// `None` when it states none, which leaves the launch limit unchecked against it.
    fn driver_limit(chip: u8) -> Option<Duration> {
        let path = format!("/sys/class/rngd_mgmt/rngd!npu{chip}mgmt/pe_ipc_timeout_ms");
        let millis = std::fs::read_to_string(&path)
            .map_err(|why| why.to_string())
            .and_then(|text| text.trim().parse::<u64>().map_err(|why| why.to_string()));
        match millis {
            Ok(millis) => Some(Duration::from_millis(millis)),
            Err(why) => {
                log::debug!("{path}: {why}; the launch limit goes unchecked against the driver's");
                None
            }
        }
    }

    /// Gives the cluster its identity, its task column and its launch limit in seconds, and checks
    /// it took them.
    fn initialize(link: &mut Link, identity: &Identity, column: u8, timeout_secs: u32) -> Result<()> {
        match link.call(&Request::Initialize {
            identity: identity.clone(),
            column,
            timeout_secs,
            log_level: Self::device_log_level(),
        })? {
            Response::Initialized { chip, rank, members }
                if (chip, rank, members) == (identity.chip(), identity.rank(), identity.size()) =>
            {
                Ok(())
            }
            Response::Initialized { chip, rank, members } => Err(Error::Device(format!(
                "npu{} came up as npu{chip} rank {rank} of {members}, not rank {} of {}",
                identity.chip(),
                identity.rank(),
                identity.size(),
            ))),
            other => Err(Error::Device(format!("a bring up was answered with {other:?}"))),
        }
    }

    /// The offsets free on every chip: an allocation is one offset on all of them.
    pub(crate) fn free(&self) -> Result<Range<usize>> {
        self.drams
            .iter()
            .map(Dram::free)
            .reduce(|all, one| all.start.max(one.start)..all.end.min(one.end))
            .ok_or_else(|| Error::Memory("the device has no memory".into()))
    }

    /// Each chip's memory as the PDMA addresses it, in dense chip order.
    pub(crate) fn chips(&self) -> Result<Vec<Chip>> {
        self.drams
            .iter()
            .map(|dram| {
                Ok(Chip {
                    index: dram.chip(),
                    base: DRAM_BASE + dram.arg(0)?,
                })
            })
            .collect()
    }
}

/// The two images a cluster runs: the signed bootloader the driver loads, and the firmware image
/// the bootloader brings up from device memory.
#[derive(Clone, Copy)]
pub(crate) struct Images {
    pub(crate) boot: &'static [u8],
    pub(crate) device: &'static [u8],
}

impl Reserved {
    /// Reserves every chip's node, or none. A single-chip device holds only its PEs' fusion
    /// node; a multi-chip device joins chips no other device may see, so it takes them whole.
    fn reserve<T: Flockable>(nodes: impl IntoIterator<Item = (u8, T)>, whole: bool) -> Result<Vec<Flock<T>>> {
        nodes
            .into_iter()
            .map(|(chip, node)| Self::lock(chip, node, whole))
            .collect()
    }

    fn lock<T: Flockable>(chip: u8, node: T, whole: bool) -> Result<Flock<T>> {
        let arg = if whole {
            FlockArg::LockExclusiveNonblock
        } else {
            FlockArg::LockSharedNonblock
        };
        Flock::lock(node, arg).map_err(|(_, why)| Error::reserving(chip, why))
    }
}

impl Error {
    /// A driver refusal while taking PEs: held by another process is [`Error::Reserved`], which a
    /// topology search passes over; anything else is a fault it must report.
    pub(super) fn reserving(chip: u8, why: Errno) -> Self {
        match why {
            Errno::EBUSY | Errno::EAGAIN => Self::Reserved(chip),
            why => Self::Device(format!("reserving npu{chip}: {why}")),
        }
    }
}

impl Launcher {
    /// Submits `launch` to every cluster. The ticket's deadline starts now: a launch the
    /// clusters leave unanswered past it fails.
    pub(crate) fn submit(&mut self, launch: &Launch<'_>) -> Result<Ticket> {
        if let Some(first) = &self.poison {
            return Err(Error::Poisoned(Box::new(first.clone())));
        }
        if launch.args.len() > furiosa_opt_ipc::MAX_LAUNCH_ARGS {
            return Err(
                FunctionError::InvalidBinding("the function takes more arguments than a launch carries").into(),
            );
        }
        let deadline = Instant::now() + self.timeout;
        let mut args = [0; furiosa_opt_ipc::MAX_LAUNCH_ARGS];
        let mut ticket = Ticket {
            ids: [0; MAX_CLUSTERS],
            answered: [false; MAX_CLUSTERS],
            clusters: self.links.len(),
            deadline,
            records: launch
                .profile
                .map_or_else(Vec::new, |_| vec![Vec::new(); self.links.len()]),
        };
        for (rank, (link, &window)) in self.links.iter_mut().zip(&self.windows).enumerate() {
            for (word, &offset) in args.iter_mut().zip(launch.args) {
                *word = window + offset;
            }
            let submitted = link.submit(
                &Request::Launch {
                    args: furiosa_opt_ipc::Args::from(&args[..launch.args.len()]),
                    function: Staged {
                        addr: DRAM_BASE + window + launch.image.addr,
                        len: launch.image.len,
                        token: launch.image.token,
                    },
                    profile: launch.profile,
                },
                deadline,
            );
            match submitted {
                Ok(id) => ticket.ids[rank] = id,
                Err(why) => {
                    // The ranks before this one are running the launch.
                    ticket.clusters = rank;
                    return Err(self.abandon(&ticket, Error::Device(format!("rank {rank}: {why}"))));
                }
            }
        }
        Ok(ticket)
    }

    /// Ends a launch that failed part-way with `why`: collects or frees every unanswered rank of
    /// `ticket`, so no slot stays held, and poisons the device with the first such failure.
    fn abandon(&mut self, ticket: &Ticket, why: Error) -> Error {
        for rank in 0..ticket.clusters {
            if !ticket.answered[rank] {
                self.links[rank].abandon(ticket.ids[rank], ticket.deadline);
            }
        }
        self.poison.get_or_insert_with(|| why.clone());
        why
    }

    /// Advances `ticket`; `true` once every cluster has answered. A profiled launch's records land
    /// in the ticket as each cluster answers. A rank that fails ends the launch and poisons the
    /// device; see [`Launcher`].
    pub(crate) fn poll(&mut self, ticket: &mut Ticket) -> Result<bool> {
        self.advance(ticket).map_err(|why| self.abandon(ticket, why))
    }

    fn advance(&mut self, ticket: &mut Ticket) -> Result<bool> {
        let timed_out = Instant::now() >= ticket.deadline;
        for rank in 0..ticket.clusters {
            if ticket.answered[rank] {
                continue;
            }
            let id = ticket.ids[rank];
            let answer = self.links[rank]
                .poll(id)
                .map_err(|why| Error::Device(format!("rank {rank}: {why}")))?;
            let Some(answer) = answer else {
                if timed_out {
                    return Err(Error::Device(format!(
                        "rank {rank} left request {id} unanswered for {}s",
                        self.timeout.as_secs()
                    )));
                }
                continue;
            };
            // Answered, well or badly: its slot is free and nothing remains to collect from it.
            ticket.answered[rank] = true;
            match answer {
                Response::Launched {
                    code: 0,
                    records: answered,
                } => {
                    if let Some(slot) = ticket.records.get_mut(rank) {
                        *slot = answered.into();
                    }
                }
                Response::Launched { code, .. } => {
                    return Err(Error::Device(format!(
                        "rank {rank} ended a launch with {code}: {}",
                        furiosa_opt_ipc::status::describe(code)
                    )));
                }
                Response::Failed { code, message } => {
                    return Err(Error::Device(format!(
                        "rank {rank} ended a launch with {code}, {}: {message}",
                        furiosa_opt_ipc::status::describe(code)
                    )));
                }
                other => return Err(Error::Device(format!("rank {rank} answered a launch with {other:?}"))),
            }
        }
        Ok(ticket.answered[..ticket.clusters].iter().all(|&answered| answered))
    }
}

impl Ticket {
    /// Takes the records every cluster answered with, once [`Launcher::poll`] reported them all.
    pub(crate) fn records(&mut self) -> Vec<Vec<ProfileRecord>> {
        std::mem::take(&mut self.records)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use crate::Error;

    use std::time::Duration;

    use super::{Cluster, GRACE, Launch, Launcher, PeRange, Reserved, Staged};

    fn eight_pes() -> Vec<PeRange> {
        PeRange::placements(8).expect("placements").remove(0)
    }

    #[test]
    fn poisoned_group_repeats_first_failure() {
        let first = Error::Device("rank 1 ended a launch with 7".into());
        let mut launcher = Launcher {
            poison: Some(first.clone()),
            ..Launcher::default()
        };
        let launch = Launch {
            image: Staged {
                addr: 0,
                len: 0,
                token: 0,
            },
            args: &[],
            profile: None,
        };

        assert_eq!(launcher.submit(&launch).err(), Some(Error::Poisoned(Box::new(first))));
    }

    #[test]
    fn rejects_a_device_named_twice() {
        assert!(matches!(
            Cluster::across(&[1, 2, 1], &eight_pes()),
            Err(Error::Topology(_))
        ));
    }

    #[test]
    fn a_launch_limit_is_whole_seconds_rounded_up() {
        assert_eq!(Reserved::launch_limit(Duration::from_millis(4500), None), Ok(5));
        assert_eq!(Reserved::launch_limit(Duration::from_secs(5), None), Ok(5));
    }

    #[test]
    fn rejects_out_of_range_limit() {
        assert!(matches!(
            Reserved::launch_limit(Duration::ZERO, None),
            Err(Error::Timeout(_))
        ));
        assert!(matches!(
            Reserved::launch_limit(Duration::from_secs(u64::from(u32::MAX) + 1), None),
            Err(Error::Timeout(_))
        ));
        assert!(matches!(
            Reserved::launch_limit(Duration::MAX, None),
            Err(Error::Timeout(_))
        ));
        assert_eq!(
            Reserved::launch_limit(Duration::from_secs(u64::from(u32::MAX)), None),
            Ok(u32::MAX)
        );
    }

    #[test]
    fn launch_limit_precedes_driver_deadline() {
        let driver = Some(Duration::from_secs(60));
        let last = 60 - GRACE.as_secs() - 1;

        assert_eq!(
            Reserved::launch_limit(Duration::from_secs(last), driver),
            Ok(last as u32)
        );
        assert!(matches!(
            Reserved::launch_limit(Duration::from_secs(last + 1), driver),
            Err(Error::Timeout(_))
        ));
        assert!(matches!(
            Reserved::launch_limit(Duration::from_millis(last * 1000 + 1), driver),
            Err(Error::Timeout(_))
        ));
    }

    #[test]
    fn groups_clusters_by_chip_and_rank() {
        assert_eq!(
            Cluster::across(&[2, 1], &eight_pes()).expect("the clusters of a device"),
            [
                Cluster::new(1, PeRange::new(0, 4).expect("pe0-3")),
                Cluster::new(1, PeRange::new(4, 4).expect("pe4-7")),
                Cluster::new(2, PeRange::new(0, 4).expect("pe0-3")),
                Cluster::new(2, PeRange::new(4, 4).expect("pe4-7")),
            ],
        );
    }

    #[test]
    fn rejects_empty_and_repeated_devices() {
        assert!(Cluster::across(&[], &eight_pes()).is_err());
        assert!(Cluster::across(&[0, 0], &eight_pes()).is_err());
        assert_eq!(
            Cluster::across(&[0], &[PeRange::new(2, 2).expect("pe2-3")])
                .expect("one cluster")
                .len(),
            1
        );
    }

    #[test]
    fn whole_device_excludes_others() {
        let node = NamedTempFile::new().expect("node");
        let whole = Reserved::lock(0, node.reopen().expect("first node"), true).expect("whole device");

        assert!(matches!(
            Reserved::lock(0, node.reopen().expect("second node"), true),
            Err(Error::Reserved(0))
        ));
        assert!(matches!(
            Reserved::lock(0, node.reopen().expect("third node"), false),
            Err(Error::Reserved(0))
        ));

        drop(whole);
    }

    #[test]
    fn pe_groups_share_free_device() {
        let node = NamedTempFile::new().expect("node");
        let first = Reserved::lock(0, node.reopen().expect("first node"), false).expect("first share");

        assert!(Reserved::lock(0, node.reopen().expect("second node"), false).is_ok());
        assert!(matches!(
            Reserved::lock(0, node.reopen().expect("third node"), true),
            Err(Error::Reserved(0))
        ));

        drop(first);
    }

    #[test]
    fn releases_the_node_when_dropped() {
        let node = NamedTempFile::new().expect("node");
        let reservation = Reserved::lock(0, node.reopen().expect("first node"), true).expect("first reservation");

        drop(reservation);

        assert!(Reserved::lock(0, node.reopen().expect("second node"), true).is_ok());
    }

    #[test]
    fn partial_reservation_releases_all() {
        let first = NamedTempFile::new().expect("first node");
        let second = NamedTempFile::new().expect("second node");
        let held = Reserved::lock(1, second.reopen().expect("held node"), true).expect("held reservation");

        assert!(
            Reserved::reserve(
                [
                    (0, first.reopen().expect("first reservation")),
                    (1, second.reopen().expect("second reservation")),
                ],
                true,
            )
            .is_err()
        );
        assert!(Reserved::lock(0, first.reopen().expect("reacquired node"), true).is_ok());

        drop(held);
    }
}
