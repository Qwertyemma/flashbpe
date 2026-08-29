use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Default GPT-4 style text-splitting pattern.
/// Contractions → words → numbers → punctuation → whitespace — in that priority order.
const GPT4_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// A pair of adjacent token IDs.
type Pair = (u32, u32);

// =============================================================================
// Word
// =============================================================================

/// One pre-tokenized chunk stored as a flat sequence of token IDs.
///
/// After the initial byte-level encoding each `Word` is merged in-place as
/// BPE training proceeds.  `merge_pair` returns the minimal set of pair-count
/// deltas so the caller can update global statistics without re-scanning.
#[derive(Clone)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    fn new(bytes: &[u8], byte_to_rank: &[u32; 256]) -> Self {
        Self {
            ids: bytes.iter().map(|&b| byte_to_rank[b as usize]).collect(),
        }
    }

    fn pairs(&self) -> impl Iterator<Item = Pair> + '_ {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Replace every non-overlapping occurrence of `pair` with `new_id`.
    ///
    /// Returns `(pair, delta)` entries describing how the global pair counts
    /// change as a result: `-1` means a pair was consumed, `+1` means one
    /// was created at a boundary adjacent to the new token.
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(n);
        let mut deltas = Vec::with_capacity(8);
        let mut i = 0;

        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < n {
                    Some(self.ids[i + 2])
                } else {
                    None
                };

                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                out.push(new_id);
                i += 2;
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}

// =============================================================================
// MergeJob
// =============================================================================

/// One entry in the priority queue used during the merge loop.
///
/// The heap is a max-heap ordered by `count`.  Ties are broken by ascending
/// `pair` value so training is fully deterministic across runs.
///
/// Entries are never updated in-place; stale entries are detected lazily when
/// popped and either refreshed or discarded.
#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    /// Word indices that still contain this pair (used to limit re-scanning).
    pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    fn eq(&self, o: &Self) -> bool {
        self.count == o.count && self.pair == o.pair
    }
}
impl PartialOrd for MergeJob {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MergeJob {
    fn cmp(&self, o: &Self) -> Ordering {
        if self.count != o.count {
            self.count.cmp(&o.count)
        } else {
            o.pair.cmp(&self.pair)
        }
    }
}

// =============================================================================
// Pair counting
// =============================================================================

/// Count every adjacent pair across all words in parallel.
///
/// Returns two maps:
/// - `pair_counts`     — weighted frequency of each pair (weight = word count).
/// - `where_to_update` — set of word indices that contain each pair.
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let mut pc: AHashMap<Pair, i32> = AHashMap::new();
            let mut wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for p in w.pairs() {
                    *pc.entry(p).or_default() += counts[i];
                    wtu.entry(p).or_default().insert(i);
                }
            }
            (pc, wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut ap, mut aw), (pc, wtu)| {
                for (k, v) in pc {
                    *ap.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    aw.entry(k).or_default().extend(s);
                }
                (ap, aw)
            },
        )
}

// =============================================================================
// Byte encoder
// =============================================================================

/// Build rank→byte and byte→rank lookup tables for the 256 base tokens.
///
/// Printable bytes (`!`–`~`, `¡`–`¬`, `®`–`ÿ`) get ranks 0–187 in that order;
/// the remaining 68 non-printable bytes get ranks 188–255.  This makes
/// `vocab.json` start with `"!": 0, "\"": 1 …` as expected.
fn build_byte_rank_tables() -> ([u8; 256], [u32; 256]) {
    let printable: Vec<u8> = (b'!'..=b'~')
        .chain(b'\xa1'..=b'\xac')
        .chain(b'\xae'..=b'\xff')
        .collect();
    let mut rank_to_byte = [0u8; 256];
    let mut byte_to_rank = [0u32; 256];
    let mut rank = 0usize;
    for &b in &printable {
        rank_to_byte[rank] = b;
        byte_to_rank[b as usize] = rank as u32;
        rank += 1;
    }
    for b in 0u8..=255u8 {
        if !printable.contains(&b) {
            rank_to_byte[rank] = b;
            byte_to_rank[b as usize] = rank as u32;
            rank += 1;
        }
    }
    (rank_to_byte, byte_to_rank)
}

/// Build the GPT-2 byte-to-Unicode mapping.
///
/// The 188 printable bytes (`!`–`~`, `¡`–`¬`, `®`–`ÿ`) map to themselves.
/// The remaining 68 bytes (control characters, space, soft-hyphen) map to
/// `Ā`–`ń` (U+0100 onwards) so every token can be stored as valid UTF-8.
fn build_byte_encoder() -> AHashMap<u8, char> {
    let mut bs: Vec<u8> = (b'!'..=b'~')
        .chain(b'\xa1'..=b'\xac')
        .chain(b'\xae'..=b'\xff')
        .collect();
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs)
        .map(|(b, c)| (b, char::from_u32(c).unwrap()))
        .collect()
}

// =============================================================================
// Tokenizer
// =============================================================================

#[pyclass]
pub struct Tokenizer {
    /// Learned BPE merges: pair → merged token ID.
    merges: StdHashMap<Pair, u32>,
    /// The regex pattern string (stored for export via `get_pattern`).
    pattern: String,
    /// Compiled form of `pattern` used at encode / train time.
    compiled_pattern: Regex,
    /// Pre-computed byte→rank table for the 256 base tokens.
    /// Built once at construction and reused by every `encode` call.
    byte_to_rank: [u32; 256],
    /// Pre-computed rank→byte table for the 256 base tokens.
    rank_to_byte: [u8; 256],
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new() -> Self {
        let (rank_to_byte, byte_to_rank) = build_byte_rank_tables();
        Self {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").unwrap(),
            byte_to_rank,
            rank_to_byte,
        }
    }

    /// Train BPE from a streaming Python iterator (parallel ingestion).
    ///
    /// Unlike a plain `Vec<String>` parameter, this never requires the whole
    /// corpus to be materialized in memory at once. We refill a Rust buffer
    /// of up to `buffer_size` strings under the GIL, then release the GIL to
    /// do the heavy regex-splitting and counting work **in parallel** with
    /// rayon, before refilling again. This mirrors the streaming approach
    /// used by the upstream `karpathy/rustbpe` reference implementation, and
    /// lets training scale to corpora far larger than available RAM would
    /// otherwise allow if the whole thing had to be a Python list up front.
    ///
    /// # Arguments
    /// * `iterator`    — any Python iterable of strings (list, generator,
    ///                    file object line-by-line, etc.).
    /// * `vocab_size`  — target vocabulary size; must be ≥ 256.
    /// * `buffer_size` — how many strings to pull from the iterator at a
    ///                    time before releasing the GIL to process them.
    /// * `pattern`     — optional override for the splitting regex.
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None))]
    pub fn train_from_iterator(
        &mut self,
        py: Python<'_>,
        iterator: &Bound<'_, PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
    ) -> PyResult<()> {
        if vocab_size < 256 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "vocab_size must be >= 256",
            ));
        }

        let pat = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());
        self.pattern = pat.clone();
        self.compiled_pattern = Regex::new(&pat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let num_merges = vocab_size - 256;
        log::info!(
            "Starting BPE training: {} merges to compute (target vocab_size={})",
            num_merges,
            vocab_size
        );

        // Obtain a real Python iterator object from whatever iterable was passed in.
        let py_iter: Py<PyAny> = unsafe {
            Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        let compiled = self.compiled_pattern.clone();
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);
        let mut total_sequences = 0u64;

        log::info!(
            "Processing sequences from iterator (buffer_size: {})",
            buffer_size
        );

        // Refill `buf` with up to `buffer_size` strings from the Python iterator.
        // Returns Ok(true) once the iterator is exhausted.
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            Python::attach(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    let next_obj = unsafe {
                        Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if PyErr::occurred(py) {
                                return Err(PyErr::fetch(py));
                            }
                            return Ok(true); // exhausted
                        }
                    }
                }
            })
        };

        // Stream ingestion loop: refill under the GIL, process without the GIL.
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_sequences += buf.len() as u64;

            let local: AHashMap<CompactString, i32> = py.detach(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                        for mat in compiled.find_iter(s) {
                            let piece = mat.expect("regex match failed").as_str();
                            *m.entry(CompactString::from(piece)).or_default() += 1;
                        }
                        m
                    })
                    .reduce(AHashMap::new, |mut a, b| {
                        for (k, v) in b {
                            *a.entry(k).or_default() += v;
                        }
                        a
                    })
            });

            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }

            if exhausted {
                break;
            }
        }

        log::info!(
            "Processed {} sequences total, {} unique",
            total_sequences,
            counts.len()
        );

        // Convert unique chunks into byte-level Word sequences.
        let mut words = Vec::with_capacity(counts.len());
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, c) in &counts {
            words.push(Word::new(chunk.as_bytes(), &self.byte_to_rank));
            cvec.push(*c);
        }

        // Phase 2 — incremental merge loop.
        self.merges.clear();

        py.detach(|| {
            log::info!(
                "Computing initial pair counts from {} unique sequences",
                words.len()
            );
            let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &cvec);

            log::info!("Building heap with {} unique pairs", pair_counts.len());
            // Seed the heap once with every unique pair.
            let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
            for (pair, pos) in where_to_update.drain() {
                let c = *pair_counts.get(&pair).unwrap_or(&0);
                if c > 0 {
                    heap.push(MergeJob {
                        pair,
                        count: c as u64,
                        pos,
                    });
                }
            }

            log::info!("Starting merge loop");
            let mut merges_done = 0u32;
            while merges_done < num_merges {
                let Some(mut top) = heap.pop() else {
                    break;
                };

                // Lazy refresh: if the stored count is stale, correct it and
                // re-insert rather than doing eager heap updates after every merge.
                let current = *pair_counts.get(&top.pair).unwrap_or(&0);
                if top.count != current as u64 {
                    top.count = current as u64;
                    if top.count > 0 {
                        heap.push(top);
                    }
                    continue;
                }
                if top.count == 0 {
                    break;
                }

                // Record the merge and assign the next available token ID.
                let new_id = 256 + merges_done;
                self.merges.insert(top.pair, new_id);

                // Apply the merge to every affected word and propagate the
                // resulting pair-count deltas back into `pair_counts`.
                let mut local_pos: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
                for &wi in &top.pos {
                    for (pair, delta) in words[wi].merge_pair(top.pair, new_id) {
                        let dt = delta * cvec[wi];
                        if dt != 0 {
                            *pair_counts.entry(pair).or_default() += dt;
                            if delta > 0 {
                                local_pos.entry(pair).or_default().insert(wi);
                            }
                        }
                    }
                }
                for (pair, pos) in local_pos {
                    let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                    if cnt > 0 {
                        heap.push(MergeJob {
                            pair,
                            count: cnt as u64,
                            pos,
                        });
                    }
                }

                merges_done += 1;
                if merges_done.is_multiple_of(1000) || merges_done == num_merges {
                    log::info!(
                        "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {} (frequency: {})",
                        (merges_done * 100) / num_merges.max(1),
                        merges_done,
                        num_merges,
                        top.pair,
                        new_id,
                        top.count,
                    );
                }
            }

            log::info!("Finished training: {} merges completed", merges_done);
        });

        Ok(())
    }

    /// Encode a string into a sequence of token IDs.
    ///
    /// Uses a min-heap ordered by merge rank so the next merge is found in
    /// O(log n) rather than by scanning the full token list each step.
    /// Total cost per chunk: O(n log n) instead of O(n²).
    ///
    /// Algorithm:
    ///   1. Map each byte of the chunk to its base rank → ids Vec.
    ///   2. Build a doubly-linked list over ids (next/prev arrays) so deleted
    ///      slots can be skipped in O(1) without shifting the Vec.
    ///   3. Seed a min-heap with every valid adjacent pair (merge_rank, pos,
    ///      left_id, right_id). The expected ids serve as a stale-entry guard.
    ///   4. Pop the lowest-ranked entry. If ids at that position still match
    ///      the recorded left/right, apply the merge: write the merged id into
    ///      ids[pos], mark ids[rpos] deleted (u32::MAX), relink the list, and
    ///      push the two new neighbouring pairs. Otherwise discard as stale.
    ///   5. Collect live (non-sentinel) ids via the linked list.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        // Heap entry: (Reverse(merge_rank), left_pos, expected_left_id, expected_right_id).
        // Reverse makes BinaryHeap behave as a min-heap on merge_rank.
        type HeapEntry = (Reverse<u32>, usize, u32, u32);

        let mut all_ids = Vec::new();

        for mat in self.compiled_pattern.find_iter(text) {
            let mat = match mat {
                Ok(mat) => mat,
                Err(e) => {
                    log::warn!("Regex match error, skipping chunk: {}", e);
                    continue;
                }
            };
            let bytes: Vec<u8> = mat.as_str().bytes().collect();
            let n = bytes.len();
            if n == 0 {
                continue;
            }

            // Fast path: single byte, no merges possible.
            if n == 1 {
                all_ids.push(self.byte_to_rank[bytes[0] as usize]);
                continue;
            }

            // Step 1: byte → base rank.
            let mut ids: Vec<u32> = bytes
                .iter()
                .map(|&b| self.byte_to_rank[b as usize])
                .collect();

            // Step 2: doubly-linked list over indices.
            // next[i] = next live index after i  (n = "end of list" sentinel).
            // prev[i] = prev live index before i (n = "before start" sentinel).
            let mut next: Vec<usize> = (1..=n).collect();
            let mut prev: Vec<usize> = (0..n).map(|i| if i == 0 { n } else { i - 1 }).collect();

            // Step 3: seed heap.
            let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
            for i in 0..n - 1 {
                if let Some(&merged) = self.merges.get(&(ids[i], ids[i + 1])) {
                    heap.push((Reverse(merged), i, ids[i], ids[i + 1]));
                }
            }

            // Step 4: merge loop.
            while let Some((Reverse(rank), pos, left_id, right_id)) = heap.pop() {
                let rpos = next[pos];
                // Stale-entry checks.
                if rpos >= n {
                    continue;
                }
                if ids[pos] != left_id {
                    continue;
                }
                if ids[rpos] != right_id {
                    continue;
                }

                // Apply: merged token id == rank (by construction in train_from_iterator).
                ids[pos] = rank;
                ids[rpos] = u32::MAX; // deleted sentinel

                // Unlink rpos.
                let rright = next[rpos];
                next[pos] = rright;
                if rright < n {
                    prev[rright] = pos;
                }

                // Push new pairs at the left and right boundaries.
                let lpos = prev[pos];
                if lpos < n {
                    if let Some(&m) = self.merges.get(&(ids[lpos], ids[pos])) {
                        heap.push((Reverse(m), lpos, ids[lpos], ids[pos]));
                    }
                }
                if rright < n {
                    if let Some(&m) = self.merges.get(&(ids[pos], ids[rright])) {
                        heap.push((Reverse(m), pos, ids[pos], ids[rright]));
                    }
                }
            }

            // Step 5: collect live tokens by following the linked list.
            let mut i = 0usize;
            loop {
                if ids[i] != u32::MAX {
                    all_ids.push(ids[i]);
                }
                i = next[i];
                if i >= n {
                    break;
                }
            }
        }
        all_ids
    }

    /// Decode a sequence of token IDs back into a UTF-8 string.
    pub fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
        let tb = self.build_token_bytes();
        let mut bytes: Vec<u8> = Vec::new();
        for id in &ids {
            let token_bytes = tb.get(*id as usize).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unknown token id: {}", id))
            })?;
            bytes.extend(token_bytes);
        }
        String::from_utf8(bytes).map_err(|e| {
            let bad_bytes = e.into_bytes();
            let valid_up_to = String::from_utf8(bad_bytes.clone())
                .err()
                .map(|err| err.utf8_error().valid_up_to())
                .unwrap_or(0);
            match pyo3::exceptions::PyUnicodeDecodeError::new(
                py,
                c"utf-8",
                &bad_bytes,
                valid_up_to..(valid_up_to + 1).min(bad_bytes.len()),
                c"invalid utf-8 in decoded BPE token bytes",
            ) {
                Ok(err) => PyErr::from_value(err.into_any()),
                Err(fallback) => fallback,
            }
        })
    }

    /// Encode a batch of strings in parallel using a Rayon thread pool.
    ///
    /// Returns a Python RuntimeError instead of panicking if the thread pool
    /// cannot be created (e.g. num_threads exceeds the OS limit).
    #[pyo3(signature = (texts, num_threads=8))]
    pub fn batch_encode(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        num_threads: usize,
    ) -> PyResult<Vec<Vec<u32>>> {
        py.detach(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to build Rayon thread pool: {e}"
                    ))
                })
                .map(|pool| pool.install(|| texts.par_iter().map(|t| self.encode(t)).collect()))
        })
    }

    /// Return `[(token_bytes, rank), …]` sorted by rank.
    ///
    /// Pass the result directly to `tiktoken.Encoding(mergeable_ranks=…)`.
    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        let tb = self.build_token_bytes();
        tb.iter()
            .enumerate()
            .filter(|(_, bytes)| !bytes.is_empty())
            .map(|(rank, bytes)| (bytes.clone(), rank as u32))
            .collect()
    }

    /// Return the regex pattern used for pre-tokenisation.
    pub fn get_pattern(&self) -> &str {
        &self.pattern
    }

    /// Total number of tokens in the vocabulary (256 base + learned merges).
    #[getter]
    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }

    /// Write `vocab.json` and `merges.txt` to disk.
    ///
    /// Both files are written in ascending rank order.  `vocab.json` uses
    /// the GPT-2 byte encoder so every token appears as a printable string.
    pub fn save(&self, vocab_path: &str, merges_path: &str) -> PyResult<()> {
        let byte_enc = build_byte_encoder();
        let tb = self.build_token_bytes();
        let non_empty = tb.iter().filter(|b| !b.is_empty()).count();
        let mut written = 0usize;

        let mut vocab_json = String::from("{\n");
        for (rank, bytes) in tb.iter().enumerate() {
            if bytes.is_empty() {
                continue;
            }
            let token_str: String = bytes.iter().map(|b| byte_enc[b]).collect();
            let escaped = token_str.replace('\\', "\\\\").replace('"', "\\\"");
            written += 1;
            let comma = if written < non_empty { "," } else { "" };
            vocab_json.push_str(&format!("  \"{}\": {}{}\n", escaped, rank, comma));
        }
        vocab_json.push('}');
        std::fs::write(vocab_path, vocab_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        let mut sorted: Vec<_> = self.merges.iter().collect();
        sorted.sort_by_key(|(_, &id)| id);
        let mut merges_txt = String::from("#version: 0.2\n");
        for (&(a, b), _) in &sorted {
            let a_str: String = tb[a as usize].iter().map(|b| byte_enc[b]).collect();
            let b_str: String = tb[b as usize].iter().map(|b| byte_enc[b]).collect();
            merges_txt.push_str(&format!("{} {}\n", a_str, b_str));
        }
        std::fs::write(merges_path, merges_txt)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        println!("Saved {} and {}", vocab_path, merges_path);
        Ok(())
    }
}

impl Tokenizer {
    /// Reconstruct the raw byte sequence for every token rank.
    ///
    /// Ranks 0–255 are single bytes.  Merged tokens are built up
    /// incrementally in the order they were learned so each entry is
    /// the correct concatenation of its two source tokens.
    fn build_token_bytes(&self) -> Vec<Vec<u8>> {
        let vocab_size = 256 + self.merges.len();

        let mut tb: Vec<Vec<u8>> = (0..vocab_size)
            .map(|i| {
                if i < 256 {
                    vec![self.rank_to_byte[i]]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let mut sorted: Vec<_> = self.merges.iter().collect();
        sorted.sort_by_key(|(_, &id)| id);
        for (&(a, b), &id) in &sorted {
            let merged: Vec<u8> = tb[a as usize]
                .iter()
                .chain(tb[b as usize].iter())
                .copied()
                .collect();
            tb[id as usize] = merged;
        }
        tb
    }
}

// =============================================================================
// Module
// =============================================================================

#[pymodule]
fn flashbpe(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<Tokenizer>()?;
    Ok(())
}

// =============================================================================
// RUST TESTS
// =============================================================================
//
// Adapted from the upstream karpathy/rustbpe reference implementation's test
// suite, ported to chilomax's actual types: `Word::new` takes raw bytes plus
// the byte_to_rank table (rather than a pre-built Vec<u32>), and `decode`
// returns `PyResult<String>` rather than a plain `String`.

#[cfg(test)]
mod tests {
    use super::*;

    /// `cargo test` runs outside a real Python process, so there's no live
    /// interpreter for `Python::with_gil` to attach to by default. Any test
    /// that touches the GIL (directly, or indirectly via `decode`, which
    /// needs `py` to construct a `PyUnicodeDecodeError`) must call this
    /// first. Safe to call more than once — `pyo3` no-ops on repeat calls.
    fn ensure_python_initialized() {
        pyo3::Python::initialize();
    }

    fn new_tokenizer_with_pattern(pattern: &str) -> Tokenizer {
        let (rank_to_byte, byte_to_rank) = build_byte_rank_tables();
        Tokenizer {
            merges: StdHashMap::new(),
            pattern: pattern.to_string(),
            compiled_pattern: Regex::new(pattern).unwrap(),
            byte_to_rank,
            rank_to_byte,
        }
    }

    #[test]
    fn test_word_pairs() {
        let (_, byte_to_rank) = build_byte_rank_tables();
        let word = Word::new(&[1, 2, 3, 4], &byte_to_rank);
        // With an identity-like rank table this would be (1,2),(2,3),(3,4);
        // here we only check pair *count* since ranks are remapped.
        let pairs: Vec<Pair> = word.pairs().collect();
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn test_word_pairs_empty() {
        let (_, byte_to_rank) = build_byte_rank_tables();
        let word = Word::new(&[], &byte_to_rank);
        let pairs: Vec<Pair> = word.pairs().collect();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_word_merge_deltas_correctness() {
        // Word: [1, 2, 3, 1, 2] with merge (1, 2) -> 99
        // Before: pairs are (1,2), (2,3), (3,1), (1,2)
        // After:  [99, 3, 99],   pairs are (99,3), (3,99)
        let mut word = Word {
            ids: vec![1, 2, 3, 1, 2],
        };
        let deltas = word.merge_pair((1, 2), 99);

        let mut delta_map: StdHashMap<Pair, i32> = StdHashMap::new();
        for (pair, delta) in deltas {
            *delta_map.entry(pair).or_default() += delta;
        }

        assert_eq!(delta_map.get(&(1, 2)), Some(&-2)); // removed twice
        assert_eq!(delta_map.get(&(2, 3)), Some(&-1)); // removed once
        assert_eq!(delta_map.get(&(3, 1)), Some(&-1)); // removed once
        assert_eq!(delta_map.get(&(99, 3)), Some(&1)); // created once
        assert_eq!(delta_map.get(&(3, 99)), Some(&1)); // created once
    }

    #[test]
    fn test_encode_decode_roundtrip_simple() {
        ensure_python_initialized();
        let mut tok = new_tokenizer_with_pattern(r"\w+");
        tok.merges.insert(
            (
                tok.byte_to_rank[b'h' as usize],
                tok.byte_to_rank[b'i' as usize],
            ),
            256,
        );
        let text = "hi";
        let ids = tok.encode(text);
        let decoded = Python::attach(|py| tok.decode(py, ids)).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_decode_roundtrip_with_spaces() {
        ensure_python_initialized();
        let mut tok = new_tokenizer_with_pattern(r"\w+|\s+");
        let h = tok.byte_to_rank[b'h' as usize];
        let e = tok.byte_to_rank[b'e' as usize];
        let l = tok.byte_to_rank[b'l' as usize];
        tok.merges.insert((h, e), 256); // "he"
        tok.merges.insert((l, l), 257); // "ll"
        tok.merges.insert((256, 257), 258); // "hell"

        let text = "hello world";
        let ids = tok.encode(text);
        let decoded = Python::attach(|py| tok.decode(py, ids)).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_decode_byte_level() {
        ensure_python_initialized();
        // No merges: decoding should just invert the byte_to_rank mapping.
        let tok = new_tokenizer_with_pattern("");
        let h = tok.byte_to_rank[b'h' as usize];
        let i = tok.byte_to_rank[b'i' as usize];
        let decoded = Python::attach(|py| tok.decode(py, vec![h, i])).unwrap();
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn test_decode_invalid_token_errors() {
        ensure_python_initialized();
        // Token 9000 was never learned by this tokenizer -> must error,
        // not silently drop or lossily replace the byte sequence.
        let tok = Tokenizer::new();
        let result = Python::attach(|py| tok.decode(py, vec![9000]));
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_empty() {
        ensure_python_initialized();
        let tok = Tokenizer::new();
        let decoded = Python::attach(|py| tok.decode(py, vec![])).unwrap();
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_encode_empty_string() {
        let tok = new_tokenizer_with_pattern(r"\w+");
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_encode_no_matches() {
        // Pattern only matches words; input has no word characters at all.
        let tok = new_tokenizer_with_pattern(r"\w+");
        let ids = tok.encode("   ");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_count_pairs_parallel_empty() {
        let words: Vec<Word> = vec![];
        let counts: Vec<i32> = vec![];
        let (pair_counts, positions) = count_pairs_parallel(&words, &counts);
        assert!(pair_counts.is_empty());
        assert!(positions.is_empty());
    }

    #[test]
    fn test_count_pairs_parallel_zero_count() {
        let (_, byte_to_rank) = build_byte_rank_tables();
        let words = vec![Word::new(&[1, 2, 3], &byte_to_rank)];
        let counts = vec![0];
        let (pair_counts, _positions) = count_pairs_parallel(&words, &counts);
        assert!(pair_counts.is_empty());
    }

    #[test]
    fn test_train_creates_chained_merges() {
        // "aaa" -> bytes all identical -> first merge fuses the pair, second
        // merge fuses the result with the remaining byte.
        let (_, byte_to_rank) = build_byte_rank_tables();
        let mut tok = Tokenizer::new();
        let words = vec![Word::new(&[b'a', b'a', b'a'], &byte_to_rank)];
        let counts = vec![10];

        // vocab_size 258 => 2 merges expected.
        // Re-implemented inline since chilomax doesn't factor out
        // train_core_incremental as its own method the way upstream does.
        let a = byte_to_rank[b'a' as usize];
        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let c = *pair_counts.get(&pair).unwrap_or(&0);
            if c > 0 {
                heap.push(MergeJob {
                    pair,
                    count: c as u64,
                    pos,
                });
            }
        }
        let mut words = words;
        let mut merges_done = 0u32;
        let num_merges = 2u32;
        while merges_done < num_merges {
            let Some(top) = heap.pop() else {
                break;
            };
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if current <= 0 {
                continue;
            }
            let new_id = 256 + merges_done;
            tok.merges.insert(top.pair, new_id);
            // Newly created pairs must be re-pushed onto the heap so the next
            // iteration can find and merge them too (this is what the real
            // train_from_iterator loop does with `local_pos` re-insertion).
            let mut local_pos: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for &wi in &top.pos {
                for (pair, delta) in words[wi].merge_pair(top.pair, new_id) {
                    let dt = delta * counts[wi];
                    if dt != 0 {
                        *pair_counts.entry(pair).or_default() += dt;
                        if delta > 0 {
                            local_pos.entry(pair).or_default().insert(wi);
                        }
                    }
                }
            }
            for (pair, pos) in local_pos {
                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(MergeJob {
                        pair,
                        count: cnt as u64,
                        pos,
                    });
                }
            }
            merges_done += 1;
        }

        assert_eq!(tok.merges.len(), 2);
        assert_eq!(tok.merges.get(&(a, a)), Some(&256));
        assert_eq!(tok.merges.get(&(256, a)), Some(&257));
    }
}
