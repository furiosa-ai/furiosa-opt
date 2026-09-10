//! What a launch answers with in its `code`: zero when every task exited cleanly, otherwise why
//! the firmware stopped it. The firmware writes these, the host reads them.

/// A task received arguments it could not take.
pub const BAD_ARGS: i32 = 1;
/// A task panicked; `Response::Failed` carries its message.
pub const PANICKED: i32 = 3;
/// A task ran past the launch limit.
pub const TIMED_OUT: i32 = 4;
/// The cluster was never initialized with its identity.
pub const NO_IDENTITY: i32 = 6;
/// The staged image could not be used: outside device memory, unparsable, or too large a chunk.
pub const BAD_IMAGE: i32 = 7;
/// A task made a syscall the firmware could not serve: a peer outside the cluster, or a ring
/// whose index register holds nonsense.
pub const BAD_SYSCALL: i32 = 8;

/// What `code` says, for a message a person reads.
pub fn describe(code: i32) -> &'static str {
    match code {
        0 => "finished",
        BAD_ARGS => "the task refused its arguments",
        PANICKED => "the task panicked",
        TIMED_OUT => "the launch ran past its time limit",
        NO_IDENTITY => "the cluster was never initialized",
        BAD_IMAGE => "the staged image could not be used",
        BAD_SYSCALL => "the task made a syscall the firmware could not serve",
        _ => "an unknown status",
    }
}
