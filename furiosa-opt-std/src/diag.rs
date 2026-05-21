//! Test-time panic hook that reads `build-log.jsonl` and renders kernel-compilation failures in rustc's
//! aesthetic.
//!
//! The schema is owned by the plugin (`npu_opt::protocol::log`); this module defines its own minimal
//! deserialization struct and only reads the fields needed for rendering. Field names must match the
//! producer's schema.

use std::fmt;
use std::path::Path;
use std::sync::Once;

use anstyle::{AnsiColor, Style};
use serde::Deserialize;

const MARKER: &str = "__FURIOSA_KERNEL_LOAD_FAILURE__\n";

/// Returns a panic payload the rendering hook will intercept, or `None` when no matching failure row exists
/// (the caller falls back to a plain IO error).
pub(crate) fn failure_payload(path: &str) -> Option<String> {
    let row = log_row(path)?;
    Some(format!("{MARKER}{row}"))
}

/// Installs the rendering panic hook, at most once per process.
pub fn install_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(rendered) = try_render(info) {
                anstream::eprintln!("\n{rendered}\n");
            } else {
                default(info);
            }
        }));
    });
}

fn try_render(info: &std::panic::PanicHookInfo<'_>) -> Option<String> {
    let payload = info
        .payload()
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| info.payload().downcast_ref::<&str>().copied())?;
    let json = payload.strip_prefix(MARKER)?;
    let row: LogRow = serde_json::from_str(json).ok()?;
    Some(render(&row))
}

fn log_row(path: &str) -> Option<String> {
    let path = Path::new(path);
    let stem = path.file_stem()?.to_str()?;
    let log = std::fs::read_to_string(path.with_file_name("build-log.jsonl")).ok()?;
    log.lines()
        .find(|line| match serde_json::from_str::<LogRow>(line) {
            Ok(row) => row.stage != Stage::Edf && row.fn_name.replace("::", "__") == stem,
            Err(_) => false,
        })
        .map(str::to_string)
}

/// Local view of the schema owned by `npu_opt::protocol::log`. Only the fields this renderer needs are
/// deserialized; extras are ignored so the log format can grow without breaking the runtime.
#[derive(Deserialize)]
struct LogRow {
    #[serde(rename = "fn")]
    fn_name: String,
    stage: Stage,
    file: Option<String>,
    line: Option<u32>,
    col: Option<u32>,
    reason: Option<String>,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Stage {
    Mir,
    Visa,
    Lir,
    Edf,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Mir => "MIR → VISA translation",
            Self::Visa => "VISA → LIR translation",
            Self::Lir => "LIR → kernel code generation",
            Self::Edf => "edf",
        })
    }
}

fn render(row: &LogRow) -> String {
    let err = Style::new().fg_color(Some(AnsiColor::Red.into())).bold();
    let loc = Style::new().fg_color(Some(AnsiColor::Blue.into())).bold();
    let bold = Style::new().bold();

    let mut out = format!(
        "{err}error{err:#}{bold}: kernel `{}` was not compiled ({}){bold:#}",
        row.fn_name, row.stage
    );
    if let (Some(file), Some(line), Some(col)) = (&row.file, row.line, row.col) {
        out.push_str(&format!("\n{loc}  -->{loc:#} {file}:{line}:{col}"));
        append_snippet(&mut out, file, line, col, loc, err);
    }
    if let Some(reason) = &row.reason {
        let mut lines = reason.lines();
        if let Some(first) = lines.next() {
            out.push_str(&format!("\n{loc}   ={loc:#} {bold}note{bold:#}: {first}"));
            for tail in lines {
                out.push_str(&format!("\n           {tail}"));
            }
        }
    }
    out.push_str(&format!(
        "\n{loc}   ={loc:#} {bold}help{bold:#}: run `cargo furiosa-opt compiler build --device-function {}` to reproduce",
        row.fn_name
    ));
    out
}

fn append_snippet(out: &mut String, file: &str, line: u32, col: u32, loc: Style, err: Style) {
    let Ok(source) = std::fs::read_to_string(file) else {
        return;
    };
    let Some(src_line) = source.lines().nth(line.saturating_sub(1) as usize) else {
        return;
    };
    let no = line.to_string();
    let pad = " ".repeat(no.len());
    let caret_indent = " ".repeat(col.saturating_sub(1) as usize);
    out.push_str(&format!(
        "\n{pad} {loc}|{loc:#}\n{loc}{no} |{loc:#} {src_line}\n{pad} {loc}|{loc:#} {caret_indent}{err}^{err:#}\n{pad} {loc}|{loc:#}"
    ));
}
