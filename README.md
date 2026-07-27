# flashbpe

[![CI](https://github.com/Qwertyemma/flashbpe/actions/workflows/ci.yml/badge.svg)](https://github.com/Qwertyemma/flashbpe/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/flashbpe.svg)](https://pypi.org/project/flashbpe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Train a GPT-4 style BPE tokenizer in Rust. Encode with it directly, or hand the merges to `tiktoken` for inference.

**flashbpe** trains byte-level BPE tokenizers fast, in Rust, with a thin Python layer on top via PyO3. You point it at text — a list, a generator, a file iterator, anything Python can iterate — and it learns a vocabulary the same way GPT-2/GPT-4 does: start from 256 raw bytes, repeatedly merge whichever adjacent pair shows up most often, stop at your target vocab size.

## Why this exists

Training a real BPE vocabulary from scratch, correctly, on anything more than a toy corpus is slower than it should be in pure Python. `flashbpe` moves the hot path — pair counting and the merge loop — into Rust, releases the GIL while it works, and uses a heap + linked-list representation internally so a single `encode()` call resolves in `O(n log n)` rather than repeatedly rescanning the token sequence from scratch.

This project began as a fork of [karpathy/rustbpe](https://github.com/karpathy/rustbpe) and owes its overall shape to it — same core idea, same PyO3-based approach to exposing a Rust tokenizer to Python. From there it's been substantially reworked: a streaming training API, corrected error handling on decode, renamed internals, an expanded Rust test suite, and structured logging throughout training.

## Features

- Byte-level BPE training with parallelized pair-counting (rayon)
- `O(n log n)` encoding via a heap-driven merge loop, not a linear rescan
- Streams training data from any Python iterable — no need to materialize your whole corpus as a list first
- GPT-4 style regex pre-tokenization by default (`GPT4_PATTERN`), overridable
- Structured training logs (`log` + `pyo3-log`) forwarded straight into Python's `logging`
- Decode raises real `ValueError` / `UnicodeDecodeError` on bad input instead of silently dropping or mangling bytes
- Direct export to `tiktoken`'s `mergeable_ranks` format for fast inference
- 13 Rust-level unit tests covering the merge algorithm directly, independent of Python

## Installation

### From PyPI

```bash
pip install flashbpe
```

### From source

```bash
git clone https://github.com/Qwertyemma/flashbpe.git
cd flashbpe
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release
```

You'll need a working Rust toolchain for this path — see [rustup.rs](https://rustup.rs/) if you don't have one.

## Usage

### Train a tokenizer

```python
import flashbpe

tokenizer = flashbpe.Tokenizer()
tokenizer.train_from_iterator(
    ["your", "training", "texts", "here"],
    vocab_size=4096,
)

ids = tokenizer.encode("hello world")
text = tokenizer.decode(ids)   # "hello world"
print(tokenizer.vocab_size)    # 4096
```

### Train on a stream, not a list

Because `train_from_iterator` pulls from any Python iterable, you can hand it a generator over a large corpus without ever holding the whole thing in memory at once:

```python
def documents():
    with open("corpus.txt") as f:
        for line in f:
            yield line

tokenizer = flashbpe.Tokenizer()
tokenizer.train_from_iterator(documents(), vocab_size=32000, buffer_size=8192)
```

`buffer_size` controls how many strings get pulled from the iterator and handed to Rust's parallel counting pass at a time, before the GIL is reacquired to pull the next batch.

### Watch it train

```python
import logging
logging.basicConfig(level=logging.INFO)

tokenizer = flashbpe.Tokenizer()
tokenizer.train_from_iterator(texts, vocab_size=8192)
```

```
INFO:flashbpe:Starting BPE training: 7936 merges to compute (target vocab_size=8192)
INFO:flashbpe:Processing sequences from iterator (buffer_size: 8192)
INFO:flashbpe:Processed 50000 sequences total, 41823 unique
INFO:flashbpe:Computing initial pair counts from 41823 unique sequences
INFO:flashbpe:Building heap with 6104 unique pairs
INFO:flashbpe:Starting merge loop
INFO:flashbpe:Progress: 12% (1000/7936 merges) - Last merge: (256, 261) -> 1256 (frequency: 894)
...
INFO:flashbpe:Finished training: 7936 merges completed
```

### Export to tiktoken for inference

Train with `flashbpe`, then hand the result to `tiktoken` for fast encode/decode in production:

```python
import flashbpe
import tiktoken

tokenizer = flashbpe.Tokenizer()
tokenizer.train_from_iterator(open("corpus.txt"), vocab_size=8192)

enc = tiktoken.Encoding(
    name="my_tokenizer",
    pat_str=tokenizer.get_pattern(),
    mergeable_ranks={bytes(k): v for k, v in tokenizer.get_mergeable_ranks()},
    special_tokens={},
)

ids = enc.encode("hello world")
text = enc.decode(ids)
```

### Save and reload a vocabulary

```python
tokenizer.save("vocab.json", "merges.txt")
```

Writes a `token → id` JSON file plus a GPT-2 style `merges.txt`.

### Custom pattern

`GPT4_PATTERN` is the default split pattern. Supply your own if you want different pre-tokenization behavior:

```python
tokenizer.train_from_iterator(
    texts,
    vocab_size=4096,
    pattern=r"[a-zA-Z]+|[0-9]+|\s+",
)
```

## API reference

| Method | Description |
|---|---|
| `Tokenizer()` | Create a new, untrained tokenizer |
| `train_from_iterator(iterator, vocab_size, buffer_size=8192, pattern=None)` | Train on any Python iterable of strings |
| `encode(text)` | Encode a string to a list of token ids |
| `decode(ids)` | Decode token ids back to a string; raises on unknown ids or invalid UTF-8 |
| `batch_encode(texts, num_threads=8)` | Encode many strings in parallel |
| `vocab_size` | Property — current vocabulary size (`256 + merges learned`) |
| `get_pattern()` | The regex pattern currently in use |
| `get_mergeable_ranks()` | `list[tuple[bytes, int]]` — token bytes and rank, for `tiktoken` export |
| `save(vocab_path, merges_path)` | Write vocab + merges to disk |

## Development

### Prerequisites

- Rust ([rustup.rs](https://rustup.rs/))
- Python ≥ 3.9

### Setup

```bash
git clone https://github.com/Qwertyemma/flashbpe.git
cd flashbpe
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop
```

### Running tests

```bash
# Rust tests — exercise the algorithm directly, no Python involved.
# --no-default-features is required: cargo test can't link against
# pyo3's extension-module feature the way maturin's Python build does.
cargo test --no-default-features

# Python-level tests (requires `maturin develop` first)
pytest tests/python/ -v

# Both
cargo test --no-default-features && pytest tests/python/ -v
```

### Project layout

```
flashbpe/
├── Cargo.toml              # Rust package manifest
├── pyproject.toml          # Python package manifest (maturin build backend)
├── src/
│   └── lib.rs               # Training/encode/decode implementation, PyO3 bindings, Rust tests
└── tests/
    └── python/
        └── test_tokenizer.py
```

## How BPE works, briefly

1. Start with 256 tokens, one per possible byte value.
2. Count every adjacent pair of tokens across the corpus.
3. Merge whichever pair occurs most often into one new token.
4. Repeat until the vocabulary reaches the target size.

## Acknowledgements

This project is a fork of [karpathy/rustbpe](https://github.com/karpathy/rustbpe) by Andrej Karpathy, itself written to fill a real gap: [tiktoken](https://github.com/openai/tiktoken) is fast at inference but has no training code, the HuggingFace [tokenizers](https://github.com/huggingface/tokenizers) library trains but carries a lot of accumulated complexity, and Karpathy's own [minbpe](https://github.com/karpathy/minbpe) trains and infers but only in pure Python. Credit for the original architecture and the idea of pairing Rust training with `tiktoken` inference belongs there — this fork exists to fix a couple of things found along the way and to explore the implementation further.

## License

MIT — see [LICENSE](LICENSE).
