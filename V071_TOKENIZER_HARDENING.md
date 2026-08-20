# MemVanta CPU v0.7.1 — Tokenizer Hardening

## Goal

v0.7.1 hardens GGUF tokenization before publishing real-model correctness or perplexity comparisons. The implementation now supports both tokenizer families relevant to the first real-model targets:

- GPT-2 byte-level BPE (`tokenizer.ggml.model = gpt2`), used by SmolLM2-style GGUFs.
- SentencePiece/unigram (`tokenizer.ggml.model = llama`), used by TinyLlama-style GGUFs.

## Changes

- Added GGUF float/int array metadata parsing for token scores/types.
- Added GPT-2 byte-to-Unicode encoding/decoding and BPE merges.
- Added malformed UTF-8 and arbitrary-byte handling without crashes or silent byte loss.
- Added recognition of GGUF control/user-defined special tokens.
- Added SentencePiece-style Viterbi unigram segmentation using GGUF token scores.
- Added Llama metaspace handling and type-6 byte fallback pieces (`<0xXX>`).
- Added model-aware decoding and empty-input handling.
- Added `memvanta_tokenize` CLI with text and hexadecimal byte input.
- Added deterministic GPT-2 and SentencePiece tokenizer fixtures.
- Added a dedicated tokenizer regression/fuzz CTest suite.

## Tests actually executed

### Build / CTest

Clean CMake build: PASS

- `memvanta_tests`: PASS
- `memvanta_tokenizer_tests`: PASS
- Total: 2/2 tests passed

### SentencePiece reference parity

A locally trained SentencePiece unigram model was used as the external reference. MemVanta token IDs were compared against Python `sentencepiece 0.2.1` across 500 deterministic randomized multilingual/text cases.

- Cases: 500
- Exact token-ID matches: 500
- Mismatches: **0**

The randomized corpus contained ASCII, punctuation, newlines/tabs, accented Latin, Greek, CJK, Hindi and emoji.

### Arbitrary-byte round trip

Both tokenizer paths were exercised with binary input:

- complete byte set `00..FF`
- malformed UTF-8
- embedded NUL bytes
- 100 additional deterministic random byte strings

Result:

- GPT-2 fixture: **104/104 exact byte round trips**
- SentencePiece fixture: **104/104 exact byte round trips**

The C++ tokenizer CTest additionally contains 200 deterministic random binary fuzz cases per tokenizer family.

### Explicit SentencePiece vectors

Exact token-ID parity is also locked into CTest for English, Hindi, accented Latin, emoji, repeated whitespace/tab, and C++-like source text.

### Full inference-path smoke test

`memvanta_real` was run using a separate small GGUF fixture with a textual prompt after the compatibility fallback was added. Model loading, tokenization, transformer execution and generation completed successfully. Its weights are synthetic, so generated text is a path/correctness smoke test only, not a model-quality result.

## Real-model status

The actual SmolLM2/TinyLlama model binaries were not obtainable inside this execution container because their Hugging Face files are served through Xet/LFS and outbound access to those objects is unavailable here. We therefore do **not** claim exact SmolLM2/TinyLlama tokenizer parity yet and do not publish a fabricated llama.cpp comparison.

A useful fallback real-model target is a small Llama GGUF such as Stories 15M Q4_0. v0.7.1 can use the same comparison harness once that model file is present locally.

## Remaining parity caveats

1. SentencePiece GGUFs with non-empty `tokenizer.ggml.precompiled_charsmap` require full SentencePiece normalization-map execution for universal parity. The local fixture uses identity normalization; its parity is exact.
2. The dependency-free GPT-2 pre-tokenizer implements the important byte-BPE/SmolLM behavior, but exact token-ID parity against the *actual* SmolLM2 tokenizer remains a release gate until the complete production tokenizer metadata can be run side-by-side with llama.cpp/Hugging Face.
3. Consequently, perplexity and generated-token parity claims remain blocked until the same trained GGUF is available to both MemVanta and the reference runtime.

## Release gate

Call v0.7.1 tokenizer-hardened, but reserve the stronger **real-model parity** claim for a run satisfying all of the following:

- same trained GGUF bytes and SHA-256
- same text/corpus
- exact token-ID sequence comparison
- greedy next-token comparison
- cross-entropy/perplexity comparison
- `pp512`, `tg128`, TTFT, output TPS, load time and peak RSS on the same CPU
