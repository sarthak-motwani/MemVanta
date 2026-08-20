# v0.7.2 performance and benchmark hardening

This change set targets the bottlenecks observed in the TinyStories 15M Q4_0 profile without weakening the low-memory design or changing model semantics.

## Runtime changes

- Reuse a persistent worker pool for decode-time GEMV calls that previously entered a fresh OpenMP region (or spawned threads) whenever `tensor_matvec` was called without an explicit pool.
- Reuse the Q8 activation scratch buffer for Q4_0 GEMV instead of allocating new vectors for every projection and output-head call.
- Remove the intermediate unpack/store/reload path from the AVX2 Q4_0 × Q8 block dot product and accumulate directly in registers.
- Replace scalar post-store horizontal reductions with AVX2 horizontal reductions in hot dot-product paths.
- Process eight FFN batch rows while loading/dequantizing each Q4_0/Q8_0 weight block once, rather than invoking the four-row kernel twice.

## Benchmark hardening

- Pin the llama.cpp reference used by CI so future results are comparable across MemVanta commits.
- Align llama-bench batch/ubatch and KV types with the MemVanta benchmark configuration.
- Parse llama-bench `samples_ts` rather than only `avg_ts`, so mean/SD/min/max reflect the actual repeated measurements.
- Emit coefficient-of-variation metrics and a high-variance warning when shared-runner noise exceeds 10%.
- Force llama-cli validation into non-conversation, single-turn mode and close stdin so the auxiliary greedy check cannot wait for interactive input.
- Record thread-binding environment controls and use stable OpenMP settings in the benchmark job.

## Validation expectation

CI remains the source of truth. The change should be accepted only if unit/tokenizer tests and trained-model generation pass, benchmark outputs remain numerically valid, and the same-model A/B artifacts contain real per-repetition statistics for both engines.
