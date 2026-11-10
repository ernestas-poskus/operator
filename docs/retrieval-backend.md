# Retrieval backend: in-process fastembed on the GPU

`operator` embeds chunks and reranks queries **in-process** through
[fastembed](https://github.com/Anush008/fastembed-rs)'s candle backend. There is
no HTTP model server and no runtime backend switch: the models run on the
platform GPU and a missing GPU disables retrieval (the rest of the MCP server
keeps serving).

| Platform | Device | Build requirement |
|---|---|---|
| macOS (Apple silicon) | Metal | none beyond the Rust toolchain |
| Linux | NVIDIA CUDA | CUDA toolkit present at build time |
| Windows | NVIDIA CUDA | CUDA toolkit present at build time |

Models:

| Role | Hugging Face repo | Notes |
|---|---|---|
| Dense embedder | `mixedbread-ai/mxbai-embed-large-v1` | CLS pooling + L2, 1024-d, 512-token window |
| Cross-encoder reranker | `Qwen/Qwen3-Reranker-0.6B` | P("yes") score in `[0, 1]`, 512 tokens by default |

Both are loaded from an HF snapshot with fastembed's candle loaders; f16 is the
default dtype (`retrieval_model_dtype = "f32"` is the numerical escape hatch).

## Model cache discovery

Before downloading anything, `operator` probes these locations (first complete
snapshot wins; `models--<org>--<name>/snapshots/*{config.json,tokenizer.json,model.safetensors}`):

1. `retrieval_model_cache_dir` (explicit config)
2. `FASTEMBED_CACHE_DIR`
3. `HF_HUB_CACHE`
4. `HF_HOME`, then `HF_HOME/hub`
5. `XDG_CACHE_HOME/huggingface[/hub]`
6. `~/.cache/huggingface[/hub]`
7. `~/.cache/fastembed`, `~/.fastembed_cache`
8. `<workspace>/.fastembed_cache`
9. `<retrieval_cache_dir>/models` (created on demand; the only download target)

The selected root is pinned for the process (`HF_HOME` and
`FASTEMBED_CACHE_DIR` are set to it, because fastembed checks `HF_HOME` first).
MLX-converted copies (`mlx-community/...`) are not usable by candle and are
skipped with a diagnostic. The first run needs network access (~2.5 GB); later
runs work offline from the cache.

## Failure behaviour

- No GPU / no CUDA device: `operator serve` logs
  `(retrieval: fastembed backend unavailable — …)` and serves without
  `search_semantic`.
- Models missing and no network: the same line names the probed cache root and
  the expected files.
- CPU fallback does not exist by design: retrieval is GPU-only.

## Embedder loader families

`retrieval_embed_backend` selects the fastembed candle loader: `bert` (the
shipped mxbai path) or `qwen3` (Qwen3 embedding checkpoints, which do not have
a BERT config; last-token pooling + query instruction via
`retrieval_embed_query_prefix`). The backend is part of the index identity, so
switching reindexes; `retrieval_embed_dim` must match the model width
(1024 for Qwen3-Embedding-0.6B, 2560 for 4B).

## Shipped configuration

| input | shipped value |
|---|---|
| chunker | xberg, 50 lines, 10 overlap, `path` prefix |
| dense model | mxbai-embed-large-v1, 1024-d, F16, BERT loader |
| fusion | **RRF** (dense weight ignored) |
| rerank | Qwen3-Reranker-0.6B, code instruction, 512-token window |
| rerank pool | **20** |
| experimental knobs | blend 0, context off, gate off, expansion off, routing off, query log off |

The shipped values are asserted by `config::shipped_retrieval_defaults`; any
change to them is a deliberate product decision and must update that test.

## Dual index (opt-in)

Set `retrieval_dual_embed_model` (plus `retrieval_dual_embed_dim` and
`retrieval_dual_embed_backend` for non-1024/Qwen3 models) to build a secondary
index from the same corpus and route **symbol-shaped** queries to it
(`retrieval_dual_route`, default true when the model is set). It is a separate
table with its own identity; the primary index is untouched, and a failed
secondary build falls back to the primary. Cold start embeds the corpus twice
and memory holds a second embedder; leave it unset unless symbol-heavy search
is the priority.

## Query-time knobs

`retrieval_query_log` (default off) appends a local JSONL record per search —
query text, scope, samples, latency and returned paths, never source content —
so real-traffic relevance sets can be built later. `retrieval_fusion_gate`
routes the fusion to RRF only when the lexical stage returns a candidate the
dense stage did not (otherwise the configured fusion is used).
`retrieval_rerank_context` prefixes the cross-encoder's documents with a derived
`Module:` hint (the stored chunk text already carries the `File:` enrichment
prefix). `retrieval_rerank_blend` (0.0 = pure reranker order, 1.0 = reranker
only) blends the cross-encoder score with the fused retrieval score;
`retrieval_query_expansion` appends the code-aware identifier tokens to the
dense query. Both are query-time only (no reindex).

## Index identity and reindexing

The embedder identity is `fastembed:<repo>:candle-bert-1[:d<width>]` (see
`src/fastembed.rs`). It is baked into the LanceDB table name and the ingestion
manifest, so changing the embedder or its width creates a new table and
re-embeds the workspace once. The old table is left on disk untouched.

## Memory and throughput

With both models resident, expect ~2.8 GB RSS (the embedder alone is ~1.3 GB and
loads eagerly; the reranker loads on the first reranked query). On Apple silicon,
mxbai-embed-large-v1 embeds ~36 texts/s at batch 4 and a reranked query costs
roughly 0.9 s (30 candidates).

## Verification

- `cargo check`, `cargo nextest run --status-level fail -p operator`,
  `cargo clippy --all-targets`, `cargo fmt -p operator` on macOS.
- CPU-generic model paths are covered by unit tests (the attention/custom-op
  fallbacks are device-generic), so Linux/Windows CUDA builds are compile- and
  unit-tested without a GPU.
- GPU runtime behavior must be checked on Metal and NVIDIA hardware before a
  release that changes the backend.
