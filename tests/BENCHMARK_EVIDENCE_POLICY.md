# Benchmark Evidence Publication Policy

MemVanta benchmark claims must be backed by reproducible CI evidence.

For trained-model validation:

- `Real Model Validation` is the source of truth for the repeated TinyStories 15M same-model A/B against pinned llama.cpp.
- `Medium Model Validation` is the source of truth for SmolLM2-360M trained-model smoke and benchmark evidence.
- Benchmark parameters must not be changed merely to improve a reported result.
- Successful workflow artifacts are automatically preserved under `results/` by `Publish Benchmark Results`.
- Published evidence must include workflow/run provenance and the exact MemVanta commit.
- Quantitative README claims may only use values present in committed `results/` evidence.
- A green workflow establishes successful execution; it does not by itself establish a throughput or memory advantage.

This file is intentionally under `tests/` so changes to the benchmark evidence policy exercise both trained-model validation gates.
