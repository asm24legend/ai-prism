"""Split a report into overlapping token windows that fit the encoder's trusted positions (Issue 2)."""


def window_ids(ids, window, overlap):
    if window <= overlap:
        raise ValueError("window must be larger than overlap")
    if not ids:
        return [[]]
    stride, out, i = window - overlap, [], 0
    while True:
        out.append(ids[i:i + window])
        if i + window >= len(ids):
            return out
        i += stride


def special_affixes(tok):
    """(prefix_ids, suffix_ids) the tokenizer adds around a sequence, found empirically so it works across
    transformers versions and tokenizer types (e.g. [CLS] ... [SEP])."""
    probe = tok("the", add_special_tokens=False)["input_ids"]
    full = tok("the")["input_ids"]
    k = next(i for i in range(len(full) - len(probe) + 1) if full[i:i + len(probe)] == probe)
    return full[:k], full[k + len(probe):]


class Chunker:
    def __init__(self, tokenizer, window, overlap, max_chunks):
        self.tok, self.window, self.overlap, self.max_chunks = tokenizer, window, overlap, max_chunks
        self.n_truncated = 0
        self.prefix, self.suffix = special_affixes(tokenizer)

    def encode(self, text):
        """-> list of id lists, each already wrapped in the tokenizer's special tokens (<= window + specials)."""
        ids = self.tok(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        chunks = window_ids(ids, self.window, self.overlap)
        if len(chunks) > self.max_chunks:
            self.n_truncated += 1
            chunks = chunks[:self.max_chunks]
        return [self.prefix + c + self.suffix for c in chunks]

    def encode_all(self, texts):
        self.n_truncated = 0
        out = [self.encode(t) for t in texts]
        n = [len(c) for c in out]
        print(f"[chunks] {len(out)} reports -> {sum(n)} windows (median {sorted(n)[len(n) // 2]}, max {max(n)}); "
              f"{self.n_truncated} reports exceeded max_chunks={self.max_chunks} and were truncated")
        return out
