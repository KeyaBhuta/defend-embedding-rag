"""Explicitly invoked embedding retrieval; --preview loads neither models nor tokenizers."""
import argparse
import json
import time
from pathlib import Path
from data_input import load_records
from rag import Retriever, digest


def select_inputs(rows, unit):
    seen = {}
    for row in rows:
        if row.get('retrieval'):
            raise ValueError('Input is already retrieved; supply original full paper records')
        pid = row.get('paper_id')
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError(f"{row['id']}: explicit paper_id is required")
        fingerprint = digest(row['paper'])
        if pid in seen and seen[pid] != fingerprint:
            raise ValueError(f'Conflicting paper text for {pid}')
        seen[pid] = fingerprint
        record_unit = row.get('record_unit')
        if unit == 'review' and record_unit != 'full_review':
            raise ValueError('DRG retrieval requires record_unit=full_review; do not relabel atomic concerns')
        if unit == 'concern' and record_unit == 'full_review':
            raise ValueError('Concern retrieval received a full review')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--split', choices=['train', 'eval'])
    p.add_argument('--limit', required=True, type=int, help='Explicit record cap, including JSONL')
    p.add_argument('--unit', choices=['concern', 'review'], default='concern')
    p.add_argument('--output', help='New directory containing retrieval.jsonl and inspection.md')
    p.add_argument('--cache', default='data/rag_cache')
    p.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-0.6B')
    p.add_argument('--embedding-revision', default='main', help='Pin commit for reproducible experiments')
    p.add_argument('--tokenizer', default='Qwen/Qwen3-8B', help='Generator tokenizer for chunk/evidence budgets')
    p.add_argument('--tokenizer-revision', default='main')
    p.add_argument('--device', default='cpu', help='cpu or cuda; start with small batches')
    p.add_argument('--dtype', choices=['float32','float16','bfloat16'], default='float32',
                   help='Embedding precision; float16 is useful on small CUDA GPUs')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--show-progress', action='store_true', help='Show embedding batch progress')
    p.add_argument('--chunk-size', type=int, default=500)
    p.add_argument('--chunking', choices=['window', 'section'], default='window',
                   help='Section mode keeps short sections and whole paragraphs intact')
    p.add_argument('--overlap', type=int, default=75)
    p.add_argument('--top-k', type=int, default=6)
    p.add_argument('--evidence-tokens', type=int, default=3000)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--local-files-only', action='store_true', help='Require models already cached')
    a = p.parse_args()
    if a.limit < 1 or a.batch_size < 1 or a.top_k < 1 or a.evidence_tokens < 1:
        p.error('Limits, batch size and top-k must be positive')
    if a.chunk_size < 2 or not 0 <= a.overlap < a.chunk_size:
        p.error('Require chunk size >= 2 and 0 <= overlap < chunk size')
    try:
        rows = select_inputs(load_records(a.input, a.split, a.limit), a.unit)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if a.preview:
        print(json.dumps({'records': len(rows), 'papers': len({r['paper_id'] for r in rows}),
                          'unit': a.unit, 'model_downloads': 0, 'model_calls': 0,
                          'examples': [{'id': r['id'], 'paper_id': r['paper_id'],
                                        'paper_characters': len(r['paper']),
                                        'query': r['review_segment']} for r in rows]}, indent=2))
        return
    if not a.output:
        p.error('--output is required unless --preview')
    root = Path(a.output)
    if root.exists():
        p.error('Output already exists; choose a new directory')
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import transformers
    import sentence_transformers
    import torch
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, revision=a.tokenizer_revision,
        use_fast=True, local_files_only=a.local_files_only)
    if not tokenizer.is_fast:
        p.error('A fast tokenizer is required for exact source offsets')
    encoder = SentenceTransformer(a.embedding_model, revision=a.embedding_revision,
                                 device=a.device, local_files_only=a.local_files_only,
                                 model_kwargs={'torch_dtype': getattr(torch, a.dtype)})
    identity = {'embedding_model': a.embedding_model, 'embedding_revision': a.embedding_revision,
                'embedding_commit': getattr(encoder[0].auto_model.config, '_commit_hash', None),
                'tokenizer': a.tokenizer, 'tokenizer_revision': a.tokenizer_revision,
                'tokenizer_backend_sha256': digest(tokenizer.backend_tokenizer.to_str()),
                'transformers': transformers.__version__,
                'sentence_transformers': sentence_transformers.__version__, 'torch': torch.__version__,
                'embedding_max_tokens': encoder.max_seq_length}
    identity['embedding_dtype'] = a.dtype
    retriever = Retriever(tokenizer, encoder, a.cache, identity, a.chunk_size, a.overlap, a.batch_size,
                          show_progress=a.show_progress, chunking=a.chunking)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'config.json').write_text(json.dumps(vars(a), indent=2), encoding='utf-8')
    load_seconds = time.perf_counter() - started
    if a.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(a.device)
    with (root / 'retrieval.jsonl').open('x', encoding='utf-8') as out, \
         (root / 'inspection.md').open('x', encoding='utf-8') as report:
        report.write('# Retrieval inspection\n\nOriginal evidence; similarity is not factual verification.\n')
        for row in rows:
            print(f"Retrieving evidence for {row['id']} on {a.device} ({a.dtype})", flush=True)
            tick = time.perf_counter()
            result = retriever.retrieve(row, a.top_k, a.evidence_tokens)
            result['retrieval']['source_split'] = a.split
            result['retrieval']['seconds'] = time.perf_counter() - tick
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            out.flush()
            report.write(f"\n## {row['id']}\n\n{row['review_segment']}\n")
            for c in result['retrieval']['selected_chunks']:
                report.write(f"\n### {c['chunk_id']} — rank {c['rank']}, score {c['score']:.4f}, "
                             f"characters {c['start']}:{c['end']}\n\n{c['text']}\n")
            print(f"Retrieved {row['id']}: {result['retrieval']['evidence_tokens']} tokens", flush=True)
    summary = {'records': len(rows), 'model_load_seconds': load_seconds,
               'total_seconds': time.perf_counter() - started,
               'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(a.device)
                   if a.device.startswith('cuda') else None,
               'identity': identity, 'generation_calls': 0}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
