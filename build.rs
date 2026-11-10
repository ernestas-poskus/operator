//! Bake the source revision into the binary for support/debuggability.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=OPERATOR_REVISION={}", git_revision());
}

fn git_revision() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|rev| !rev.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}
