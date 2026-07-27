"""
Python-level tests for the flashbpe package.

These run against the compiled extension (after `maturin develop` or
`pip install target/wheels/*.whl`), as opposed to `src/lib.rs`'s own
`#[cfg(test)]` module, which exercises the Rust implementation directly
without going through Python at all.
"""
import pytest
import flashbpe


def test_train_and_roundtrip():
    tok = flashbpe.Tokenizer()
    tok.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog"],
        vocab_size=270,
    )
    text = "the quick brown fox"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_vocab_size_matches_target():
    tok = flashbpe.Tokenizer()
    tok.train_from_iterator(["hello world hello world"], vocab_size=260)
    assert tok.vocab_size == 260


def test_train_from_generator():
    """train_from_iterator must accept a real streaming iterator, not just a list."""
    def gen():
        for i in range(20):
            yield f"sequence number {i} with some shared words"

    tok = flashbpe.Tokenizer()
    tok.train_from_iterator(gen(), vocab_size=280, buffer_size=4)
    assert tok.vocab_size == 280

    text = "sequence number 3"
    assert tok.decode(tok.encode(text)) == text


def test_decode_unknown_id_raises_value_error():
    tok = flashbpe.Tokenizer()
    tok.train_from_iterator(["hello"], vocab_size=257)
    with pytest.raises(ValueError):
        tok.decode([999999])


def test_decode_invalid_utf8_raises_unicode_decode_error():
    tok = flashbpe.Tokenizer()
    tok.train_from_iterator(["hello"], vocab_size=257)
    ranks = tok.get_mergeable_ranks()
    lone_continuation_byte_id = next(
        i for tok_bytes, i in ranks if tok_bytes == bytes([0x80])
    )
    with pytest.raises(UnicodeDecodeError):
        tok.decode([lone_continuation_byte_id])


def test_custom_pattern():
    tok = flashbpe.Tokenizer()
    pattern = r"[a-zA-Z]+|[0-9]+|\s+"
    tok.train_from_iterator(["abc123 def456"], vocab_size=270, pattern=pattern)
    assert tok.get_pattern() == pattern


def test_batch_encode_matches_single_encode():
    tok = flashbpe.Tokenizer()
    tok.train_from_iterator(["hello world, this is a test"], vocab_size=280)
    texts = ["hello world", "this is a test"]
    batch_ids = tok.batch_encode(texts)
    single_ids = [tok.encode(t) for t in texts]
    assert batch_ids == single_ids


def test_save_and_reload_vocab(tmp_path):
    tok = flashbpe.Tokenizer()
    tok.train_from_iterator(["hello world hello world test"], vocab_size=280)

    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    tok.save(str(vocab_path), str(merges_path))

    assert vocab_path.exists()
    assert merges_path.exists()

    import json
    vocab = json.loads(vocab_path.read_text())
    # Note: vocab_size is a *target*, not a guarantee — a short training
    # corpus can run out of repeated pairs before reaching it, so the
    # tokenizer's actual learned vocab may end up smaller. What matters here
    # is that the saved file matches whatever it actually learned.
    assert len(vocab) == tok.vocab_size
    # Byte tokens are printable-first by design, so "!" should be id 0.
    assert vocab["!"] == 0


def test_get_mergeable_ranks_exports_cleanly_to_tiktoken_shape():
    """Sanity-check the shape expected by tiktoken.Encoding(mergeable_ranks=...)."""
    tok = flashbpe.Tokenizer()
    tok.train_from_iterator(["hello world"], vocab_size=260)
    ranks = tok.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in ranks}
    assert len(mergeable_ranks) == 260
    assert all(isinstance(k, bytes) for k in mergeable_ranks)
    assert all(isinstance(v, int) for v in mergeable_ranks.values())
