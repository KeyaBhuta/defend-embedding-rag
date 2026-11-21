"""Paper-scoped retrieval. No model loading or network activity on import."""
import hashlib
import json
import re
from pathlib import Path

VERSION = 1
QUERY_PROMPT = ('Instruct: Given a scientific reviewer concern, retrieve original paper '
                'passages relevant to assessing the concern, including supporting or '
                'contradicting evidence.\nQuery: ')


def digest(value):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def context_hash(row):
    """Stable student context identity; never includes a reference rebuttal."""
    return digest({k: row[k] for k in ('paper', 'review_segment')})


def chunk_paper(text, tokenizer, size=500, overlap=75):
    """Token windows snapped to paragraph/line boundaries; exact character slices.

    This preserves text, not PDF layout. It cannot reliably detect tables in flat
    text. Offsets always refer to the unchanged source paper (Unicode characters).
    """
    if size < 2 or not 0 <= overlap < size:
        raise ValueError('Require chunk size >= 2 and 0 <= overlap < chunk size')
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    offsets = [(a, b) for a, b in offsets if b > a]
    if not offsets:
        raise ValueError('Paper produced no tokens')
    chunks, start = [], 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        if end < len(offsets):
            # Prefer a paragraph break in the latter half, then a newline.
            lower = start + max(overlap + 1, size // 2)
            for boundary in ('\n\n', '\n'):
                candidates = [i for i in range(lower, end + 1)
                              if text.endswith(boundary, 0, offsets[i][0])]
                if candidates:
                    end = candidates[-1]
                    break
        char_start = 0 if start == 0 else offsets[start][0]
        char_end = len(text) if end == len(offsets) else offsets[end][0]
        # Boundary retokenization can add tokens; shrink rather than truncate later.
        while token_count(tokenizer, text[char_start:char_end]) > size and end > start + 1:
            end -= 1
            char_end = offsets[end][0]
        piece = text[char_start:char_end]
        if token_count(tokenizer, piece) > size:
            raise ValueError('Cannot fit source slice in chunk token budget')
        chunks.append({'chunk_id': f'c{len(chunks):05d}', 'start': char_start,
                       'end': char_end, 'text': piece,
                       'tokens': token_count(tokenizer, piece)})
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return chunks


def evidence_text(chunks):
    return '\n\n'.join(f"[{c['chunk_id']} | chars {c['start']}:{c['end']}]\n{c['text']}"
                       for c in sorted(chunks, key=lambda c: c['start']))


# The prepared papers use Markdown headings and LaTeX display delimiters.
STRUCTURE = re.compile(r'(?m)^ {0,3}#{1,6}[ \t]+[^\r\n]+|\r?\n[ \t]*\r?\n|(?<!\\)\$\$|\\\[|\\\]')


def section_chunks(text, tokenizer, size=500):
    """Keep short sections whole and group intact paragraphs in longer sections."""
    chunks = []

    def append(start, end):
        piece = text[start:end]
        if piece.strip():
            chunks.append({'chunk_id': f'c{len(chunks):05d}', 'start': start,
                           'end': end, 'text': piece, 'tokens': token_count(tokenizer, piece)})

    def pack(start, end, breaks):
        if token_count(tokenizer, text[start:end]) <= size:
            append(start, end)
            return
        group_start = previous = start
        for boundary in [*breaks, end]:
            if previous > group_start and token_count(tokenizer, text[group_start:boundary]) > size:
                append(group_start, previous)
                group_start = previous
            previous = boundary
        append(group_start, end)

    start = heading_end = 0
    breaks = []
    closing = None
    for match in STRUCTURE.finditer(text):
        marker = match[0]
        if closing is not None:
            if marker == closing:
                closing = None
        elif marker in ('$$', r'\['):
            closing = '$$' if marker == '$$' else r'\]'
        elif marker.lstrip().startswith('#'):
            pack(start, match.start(), breaks)
            start, heading_end, breaks = match.start(), match.end(), []
        elif marker.startswith(('\r', '\n')) and text[heading_end:match.start()].strip():
            breaks.append(match.end())
            heading_end = match.end()
    pack(start, len(text), breaks)
    if not chunks:
        raise ValueError('Paper produced no chunks')
    return chunks


def select_chunks(chunks, scores, tokenizer, top_k=6, budget=3000, *, whole=False):
    """Rank by similarity; trim already selected overlaps using original offsets."""
    if top_k < 1 or budget < 1 or len(chunks) != len(scores):
        raise ValueError('Invalid retrieval selection settings')
    selected, covered = [], []
    ranked = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))
    for rank, i in enumerate(ranked, 1):
        c = chunks[i]
        intervals = [(c['start'], c['end'])]
        for left, right in (() if whole else covered):
            intervals = [(a, b) for s, e in intervals
                         for a, b in ((s, min(e, left)), (max(s, right), e)) if a < b]
        for start, end in intervals:
            piece = c['text'][start-c['start']:end-c['start']]
            if not piece.strip():
                continue
            candidate = {**c, 'start': start, 'end': end, 'text': piece,
                         'rank': rank, 'score': float(scores[i]),
                         'tokens': token_count(tokenizer, piece)}
            if token_count(tokenizer, evidence_text(selected + [candidate])) > budget:
                continue
            selected.append(candidate)
            covered.append((start, end))
            if len(selected) >= top_k:
                return sorted(selected, key=lambda x: x['start'])
    if not selected:
        raise ValueError('No passage fits evidence budget; increase budget or reduce chunk size')
    return sorted(selected, key=lambda x: x['start'])


def normalized(vectors):
    import numpy as np
    value = np.asarray(vectors, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError('Invalid embedding matrix')
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError('Embedding contains a zero vector')
    return value / norms


class Retriever:
    def __init__(self, tokenizer, encoder, cache, identity, chunk_size=500,
                 overlap=75, batch_size=4, show_progress=False, chunking='window'):
        self.tokenizer, self.encoder = tokenizer, encoder
        self.cache = Path(cache)
        self.identity = identity
        self.chunk_size, self.overlap, self.batch_size = chunk_size, overlap, batch_size
        self.show_progress = show_progress
        self.chunking = chunking

    def encode(self, texts, query=False):
        prompt = QUERY_PROMPT if query else ''
        # Sentence Transformers otherwise silently truncates overlong inputs.
        limit = self.encoder.max_seq_length
        for text in texts:
            n = len(self.encoder.tokenizer.encode(prompt + text, add_special_tokens=True))
            if n > limit:
                raise ValueError(f'Embedding input has {n} tokens, exceeding {limit}; no truncation allowed')
        return normalized(self.encoder.encode(
            texts, prompt=prompt, batch_size=self.batch_size,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=self.show_progress))

    def index(self, paper_id, paper):
        import numpy as np
        config = {'version': VERSION, 'paper_id': paper_id, 'paper_sha256': digest(paper),
                  'chunk_size': self.chunk_size, 'overlap': self.overlap,
                  'identity': self.identity}
        if self.chunking == 'section':
            config['chunking'] = 'section-v1'
            config['overlap'] = 0
        key = digest(config)
        root = self.cache / key
        meta_path, vectors_path = root / 'index.json', root / 'vectors.npy'
        if meta_path.exists() and vectors_path.exists():
            saved = json.loads(meta_path.read_text(encoding='utf-8'))
            if saved['config'] != config:
                raise ValueError('Cache configuration mismatch')
            chunks = saved['chunks']
            for c in chunks:
                if paper[c['start']:c['end']] != c['text']:
                    raise ValueError('Cached passage differs from source paper')
            raw = np.load(vectors_path, allow_pickle=False)
            if digest(raw.tobytes().hex()) != saved['vectors_sha256']:
                raise ValueError('Embedding cache checksum mismatch')
            vectors = normalized(raw)
            if len(vectors) != len(chunks):
                raise ValueError('Cache chunk/vector count mismatch')
            return chunks, vectors, key, True
        if self.chunking == 'section':
            chunks = section_chunks(paper, self.tokenizer, self.chunk_size)
        else:
            chunks = chunk_paper(paper, self.tokenizer, self.chunk_size, self.overlap)
        vectors = self.encode([c['text'] for c in chunks])
        root.mkdir(parents=True, exist_ok=True)
        # Commit metadata last; incomplete caches are recomputed on next invocation.
        np.save(vectors_path, vectors, allow_pickle=False)
        meta_path.write_text(json.dumps({'config': config, 'chunks': chunks,
            'vectors_sha256': digest(vectors.tobytes().hex())}, ensure_ascii=False, indent=2), encoding='utf-8')
        return chunks, vectors, key, False

    def retrieve(self, row, top_k=6, budget=3000):
        chunks, vectors, key, hit = self.index(row['paper_id'], row['paper'])
        query = self.encode([row['review_segment']], query=True)[0]
        scores = vectors @ query
        selected = select_chunks(chunks, scores, self.tokenizer, top_k, budget,
                                 whole=self.chunking == 'section')
        paper = evidence_text(selected)
        result = dict(row, paper=paper)
        result['retrieval'] = {'version': VERSION, 'context_kind': 'retrieved',
            'paper_sha256': digest(row['paper']), 'index_key': key, 'cache_hit': hit,
            'query_sha256': digest(row['review_segment']), 'query_prompt': QUERY_PROMPT,
            'identity': self.identity, 'chunk_size': self.chunk_size, 'chunking': self.chunking,
            'overlap': 0 if self.chunking == 'section' else self.overlap,
            'top_k': top_k, 'budget': budget, 'evidence_tokens': token_count(self.tokenizer, paper),
            'total_chunks': len(chunks), 'selected_chunks': selected}
        result['student_context_sha256'] = context_hash(result)
        return result


def validate_retrieved(row):
    if row.get('student_context_sha256') != context_hash(row):
        raise ValueError(f"{row['id']}: context hash missing or mismatched")
    meta = row.get('retrieval', {})
    if meta.get('context_kind') != 'retrieved':
        raise ValueError('Expected embedding-retrieved input')
    if evidence_text(meta['selected_chunks']) != row['paper']:
        raise ValueError('Evidence differs from saved passages')
    if meta['query_sha256'] != digest(row['review_segment']):
        raise ValueError('Retrieval query mismatch')
