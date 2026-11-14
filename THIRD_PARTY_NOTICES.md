# Third-party notices

`code-intelligence` is distributed under the MIT license; see [LICENSE](LICENSE).

This repository does not vendor third-party source code. Runtime and build
dependencies are resolved from crates.io and remain under their own licenses;
the exact versions are pinned by `Cargo.lock`. The only in-repository library is
`crates/crate-map`, which is first-party MIT code.

Notable direct dependencies include ast-grep, LanceDB, fastembed/candle,
xberg/anydoc, rmcp, and the Rust async ecosystem. Each crate's license and
copyright notice are available from its published package. No third-party
source code is vendored here.

The only third-party content committed to this repository is the small set of
Apache-2.0 test documents under `bench/doc-router/fixtures/`, used to test the
document router. They are attributed in `bench/doc-router/fixtures/ATTRIBUTIONS.md`.
