//! What a profiled call reports: the image names spans by the hits that begin and end them, the
//! device records the hits with their cycle counts, and the two resolve into cycle windows.

use furiosa_opt_abi::image::{Depth, Image};
use furiosa_opt_ipc::{
    PROFILE_CAPACITY, PROFILE_CHUNK_CAPACITY, PROFILE_TOTAL_CAPACITY, ProfileRecord, ProfileRequest,
};

use super::FunctionError;
use crate::{Error, Result};

/// One span of a profiled call in device cycles, named as the image names it, on the cluster
/// ranked `cluster` in the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub name: String,
    pub cluster: usize,
    pub begin: u64,
    pub end: u64,
}

/// Per-level hit bounds and the ring available to each chunk.
pub(crate) struct Profile {
    depth: Depth,
    ring: u32,
    markers: Vec<Marker>,
}

struct Marker {
    chunk: u32,
    name: String,
    begin: u16,
    end: u16,
}

impl Profile {
    pub(crate) fn new(image: &Image<'_>) -> Result<Self> {
        let chunks =
            u32::try_from(image.tasks().len()).map_err(|_| FunctionError::InvalidBinding("too many chunks"))?;
        let ring = if chunks <= PROFILE_CHUNK_CAPACITY {
            PROFILE_TOTAL_CAPACITY
                .checked_div(chunks)
                .unwrap_or(0)
                .min(PROFILE_CAPACITY)
        } else {
            0
        };
        Ok(Self {
            depth: image.profile().depth,
            ring,
            markers: image
                .profile()
                .spans
                .iter()
                .enumerate()
                .flat_map(|(chunk, spans)| {
                    spans.iter().map(move |span| Marker {
                        chunk: chunk as u32,
                        name: span.name.to_owned(),
                        begin: span.begin,
                        end: span.end,
                    })
                })
                .collect(),
        })
    }

    /// The request that profiles one call at `level`, or `None` when the image names nothing to
    /// record. The device records nothing below `Info`.
    pub(crate) fn request(&self, level: log::Level) -> Result<Option<ProfileRequest>> {
        let depth = match level {
            log::Level::Info => self.depth.info,
            log::Level::Debug => self.depth.debug,
            log::Level::Trace => self.depth.trace,
            log::Level::Error | log::Level::Warn => {
                return Err(Error::Profile(format!("the device records nothing at level {level}")));
            }
        };
        if depth == 0 || self.markers.is_empty() {
            return Ok(None);
        }
        // A full ring wraps its write pointer to zero and is indistinguishable from empty.
        if depth >= self.ring {
            return Err(Error::Profile(format!(
                "{level} needs {depth} hits per chunk, but the profile ring holds {} slots",
                self.ring,
            )));
        }
        ProfileRequest::new(level as u8, self.ring)
            .map(Some)
            .map_err(|why| Error::Profile(why.to_string()))
    }

    /// Every completed span, per cluster, in end-hit order: a begin hit opens and the next matching
    /// end closes, so a looped span yields one window per pass and an unmatched hit yields none.
    pub(crate) fn resolve(&self, records: &[Vec<ProfileRecord>]) -> Result<Vec<Span>> {
        if records
            .iter()
            .flatten()
            .any(|record| record.hits.len() >= self.ring as usize)
        {
            return Err(Error::Profile("a chunk filled its profile ring".into()));
        }
        Ok(records
            .iter()
            .enumerate()
            .flat_map(|(cluster, records)| records.iter().flat_map(move |record| self.spans(cluster, record)))
            .collect())
    }

    fn spans(&self, cluster: usize, record: &ProfileRecord) -> Vec<Span> {
        let markers = self
            .markers
            .iter()
            .filter(|marker| marker.chunk == record.chunk)
            .collect::<Vec<_>>();
        // Per marker, the cycle of the begin hit awaiting its end.
        let mut begun = vec![None; markers.len()];
        let mut spans = Vec::new();
        for &raw in &record.hits {
            // A raw hit: its marker id in the low fourteen bits, its cycle count above sixteen.
            let (id, cycle) = ((raw & 0x3fff) as u16, raw >> 16);
            for (marker, begun) in markers.iter().zip(&mut begun) {
                match *begun {
                    Some(begin) if marker.end == id => {
                        spans.push(Span {
                            name: marker.name.clone(),
                            cluster,
                            begin,
                            end: cycle,
                        });
                        *begun = None;
                    }
                    None if marker.begin == id => *begun = Some(cycle),
                    _ => {}
                }
            }
        }
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> Profile {
        Profile {
            depth: Depth {
                info: 4,
                debug: 4,
                trace: 4,
            },
            ring: 8,
            markers: vec![
                Marker {
                    chunk: 0,
                    name: "first".into(),
                    begin: 1,
                    end: 2,
                },
                Marker {
                    chunk: 1,
                    name: "second".into(),
                    begin: 1,
                    end: 3,
                },
            ],
        }
    }

    #[test]
    fn spans_pair_within_chunk() {
        let records = vec![vec![
            ProfileRecord {
                chunk: 0,
                hits: vec![1 | 100 << 16, 2 | 250 << 16],
            },
            ProfileRecord {
                chunk: 1,
                hits: vec![1 | 300 << 16],
            },
        ]];

        assert_eq!(
            profile().resolve(&records).unwrap(),
            [Span {
                name: "first".into(),
                cluster: 0,
                begin: 100,
                end: 250
            }]
        );
    }

    #[test]
    fn a_span_in_a_loop_yields_one_window_per_pass() {
        let records = vec![vec![ProfileRecord {
            chunk: 0,
            hits: vec![
                2 | 50 << 16,
                1 | 100 << 16,
                2 | 250 << 16,
                1 | 300 << 16,
                2 | 450 << 16,
                1 | 500 << 16,
            ],
        }]];

        assert_eq!(
            profile()
                .resolve(&records)
                .unwrap()
                .iter()
                .map(|span| (span.begin, span.end))
                .collect::<Vec<_>>(),
            [(100, 250), (300, 450)]
        );
    }

    #[test]
    fn spans_name_the_cluster_that_recorded_them() {
        let record = || ProfileRecord {
            chunk: 0,
            hits: vec![1 | 100 << 16, 2 | 250 << 16],
        };

        assert_eq!(
            profile()
                .resolve(&[vec![record()], vec![record()]])
                .unwrap()
                .iter()
                .map(|span| span.cluster)
                .collect::<Vec<_>>(),
            [0, 1]
        );
    }

    /// Arming a ring no chunk can fill makes this unreachable, so it reports an invariant the
    /// device broke rather than an outcome a caller can meet.
    #[test]
    fn reports_a_chunk_that_filled_its_ring() {
        let filled = vec![vec![ProfileRecord {
            chunk: 0,
            hits: vec![1 | 100 << 16; 8],
        }]];

        assert!(matches!(profile().resolve(&filled), Err(Error::Profile(_))));
    }

    /// Room past what the image names, so the chunk that reaches every marker cannot wrap.
    #[test]
    fn arms_a_ring_no_chunk_can_fill() {
        let named = 270;
        let image = Image::new(furiosa_opt_abi::image::Parts {
            stacks: vec![
                furiosa_opt_abi::image::Stack {
                    kind: furiosa_opt_abi::image::Memory::Dram,
                    size: 0,
                };
                furiosa_opt_abi::args::STACK_COUNT
            ],
            chips: 1,
            pes: 8,
            tasks: vec![vec![&[][..]; 2]],
            profile: furiosa_opt_abi::image::Profile {
                depth: Depth {
                    info: named,
                    debug: named,
                    trace: named,
                },
                spans: vec![vec![furiosa_opt_abi::image::Span {
                    name: "only",
                    begin: 0,
                    end: 1,
                }]],
            },
            ..Default::default()
        })
        .expect("well formed");

        let profile = Profile::new(&image).expect("a one-chunk image fits");

        assert!(
            profile.ring > named,
            "a ring of {} holds no more than the {named} hits the image names",
            profile.ring
        );
    }

    /// An image whose chunks share the response evenly can name exactly the hits one chunk's ring
    /// holds, and that is the ring that wraps, so it is refused rather than armed.
    #[test]
    fn refuses_a_ring_an_image_could_fill_exactly() {
        let chunks = 2;
        let image = |named| {
            Image::new(furiosa_opt_abi::image::Parts {
                stacks: vec![
                    furiosa_opt_abi::image::Stack {
                        kind: furiosa_opt_abi::image::Memory::Dram,
                        size: 0,
                    };
                    furiosa_opt_abi::args::STACK_COUNT
                ],
                chips: 1,
                pes: 8,
                tasks: vec![vec![&[][..]; 2]; chunks as usize],
                profile: furiosa_opt_abi::image::Profile {
                    depth: Depth {
                        info: named,
                        debug: named,
                        trace: named,
                    },
                    spans: vec![
                        vec![furiosa_opt_abi::image::Span {
                            name: "only",
                            begin: 0,
                            end: 1,
                        }];
                        chunks as usize
                    ],
                },
                ..Default::default()
            })
            .expect("well formed")
        };

        let fills = PROFILE_TOTAL_CAPACITY / chunks;
        assert!(
            Profile::new(&image(fills)).unwrap().request(log::Level::Info).is_err(),
            "{fills} hits fill the ring they are given"
        );
        assert!(
            Profile::new(&image(fills - 1))
                .unwrap()
                .request(log::Level::Info)
                .is_ok()
        );
    }

    #[test]
    fn requests_only_levels_the_device_records() {
        assert!(matches!(profile().request(log::Level::Warn), Err(Error::Profile(_))));
        assert_eq!(
            profile().request(log::Level::Debug).expect("debug request"),
            Some(ProfileRequest::new(4, 8).expect("profile"))
        );
    }

    #[test]
    fn permits_smaller_profile() {
        let image = Image::new(furiosa_opt_abi::image::Parts {
            stacks: vec![
                furiosa_opt_abi::image::Stack {
                    kind: furiosa_opt_abi::image::Memory::Dram,
                    size: 0,
                };
                furiosa_opt_abi::args::STACK_COUNT
            ],
            chips: 1,
            pes: 8,
            tasks: vec![vec![&[][..]; 2]; 5],
            profile: furiosa_opt_abi::image::Profile {
                depth: Depth {
                    info: 2,
                    debug: 100,
                    trace: 8191,
                },
                spans: vec![
                    vec![furiosa_opt_abi::image::Span {
                        name: "Task",
                        begin: 0,
                        end: 1
                    }];
                    5
                ],
            },
            ..Default::default()
        })
        .unwrap();
        let profile = Profile::new(&image).expect("loading does not require profiling capacity");
        assert_eq!(profile.ring, PROFILE_TOTAL_CAPACITY / 5);
        assert!(profile.request(log::Level::Info).unwrap().is_some());
        assert!(profile.request(log::Level::Debug).unwrap().is_some());
        assert!(matches!(profile.request(log::Level::Trace), Err(Error::Profile(_))));
    }

    #[test]
    fn rejects_full_ring() {
        let mut profile = profile();
        profile.depth = Depth {
            info: 7,
            debug: 8,
            trace: 9,
        };
        assert!(profile.request(log::Level::Info).unwrap().is_some());
        assert!(profile.request(log::Level::Debug).is_err());
        assert!(profile.request(log::Level::Trace).is_err());
        profile.ring = 9;
        assert!(profile.request(log::Level::Debug).unwrap().is_some());
        assert!(profile.request(log::Level::Trace).is_err());
        profile.ring = 10;
        assert!(profile.request(log::Level::Trace).unwrap().is_some());
    }

    #[test]
    fn reports_full_ring() {
        let profile = profile();
        let records = vec![vec![ProfileRecord {
            chunk: 0,
            hits: vec![1; profile.ring as usize],
        }]];
        assert!(matches!(profile.resolve(&records), Err(Error::Profile(_))));
    }
}
