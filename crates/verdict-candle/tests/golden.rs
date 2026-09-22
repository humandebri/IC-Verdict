//! Checkpoint parity gate for the openJev (GLiClass) port.
//!
//! Ignored by default: it needs the generated canonical INT8 pack, rebuilt with
//! `tools/pack_verdict.py` and deliberately not committed. It is the only test here
//! that compares against evidence this repository did not produce — the checkpoint
//! author's own recorded decisions (`reports/v2/predictions_v2.jsonl` in
//! `Heman10x-NGU/openJev-verdict-2.0`, replayed on `data/real_banking_test.jsonl`).
//!
//!   python3 tools/pack_verdict.py --safetensors models/verdict-151m/model.safetensors \
//!       --config models/verdict-151m/config.json \
//!       --tokenizer models/verdict-151m/tokenizer.json \
//!       --out models/verdict-pack \
//!       --source-revision 70fa19828074e1199e4a793c4af4dba3bfd1d222
//!   cargo build --release -p verdict-candle
//!   cargo test --release -p verdict-candle --test golden -- --ignored --nocapture
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("workspace root")
}

/// Floor for the parity set. The committed `models/verdict-parity/cases.jsonl` holds
/// 1000 cases and `verdict-infer check` has no `--limit` here, so a smaller count means
/// the inputs or the predictions file were truncated -- a gate that passes on a
/// truncated input is not a gate.
const MIN_PARITY_CASES: usize = 1000;

#[test]
#[ignore = "requires models/verdict-pack and models/verdict-parity"]
fn agrees_with_the_authors_recorded_decisions() {
    let root = root();
    let pack = root.join("models/verdict-pack");
    let tokenizer = root.join("models/verdict-151m/tokenizer.json");
    let parity = root.join("models/verdict-parity");
    let binary = root.join("target/release/verdict-infer");
    for (path, what) in [
        (pack.join("model.bin"), "pack (tools/pack_verdict.py)"),
        (tokenizer.clone(), "tokenizer (huggingface.co/heman10x/rlcd-modernbert-151m)"),
        (parity.join("cases.jsonl"), "cases"),
        (parity.join("predictions.jsonl"), "recorded predictions"),
        (binary.clone(), "verdict-infer release binary"),
    ] {
        assert!(path.exists(), "missing {what}: {}", path.display());
    }

    let output = Command::new(&binary)
        .args([
            "check",
            "--pack",
            pack.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--cases",
            parity.join("cases.jsonl").to_str().unwrap(),
            "--predictions",
            parity.join("predictions.jsonl").to_str().unwrap(),
        ])
        .output()
        .expect("run verdict-infer");
    let text = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    println!("{text}");
    assert!(output.status.success(), "verdict-infer check failed");

    let summary = text.lines().find(|line| line.starts_with("cases=")).expect("summary line");
    let field = |key: &str| -> Option<usize> {
        summary
            .split_whitespace()
            .filter_map(|token| token.split_once('='))
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value.split('/').next().unwrap_or(value).parse().expect("count"))
    };
    let cases = field("cases").expect("cases= in the summary");
    let matched = field("argmax_match").expect("argmax_match= in the summary");
    // `verdict-infer check` skips a case whose id has no recorded prediction. Without
    // this assertion a predictions file holding a single matching line passed the gate,
    // which is why the recorded run's `skipped_no_truth=0` was not actually enforced.
    let skipped = field("skipped_no_truth").expect("skipped_no_truth= in the summary");
    assert!(cases >= MIN_PARITY_CASES, "the parity set shrank to {cases} cases: {summary}");
    assert_eq!(skipped, 0, "cases were skipped for want of a recorded truth: {summary}");
    assert!(matched as f64 / cases as f64 >= 0.995, "argmax ratio below 99.5%: {summary}");
    let gate=text.lines().find(|line|line.starts_with("quality-gate ")).expect("quality gate line");
    assert!(gate.split_whitespace().any(|field|field=="unsafe_abstention_escape=0"),"unsafe reversal: {gate}");

    // Probability deviations are reported, not an F32-noise acceptance gate for INT8.
}
