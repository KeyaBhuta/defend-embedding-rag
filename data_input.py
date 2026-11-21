"""Stream prepared pairs from ZIP or JSONL without extracting the archive."""
import io
import json
import re
import zipfile
from itertools import islice
from pathlib import Path


def canonical_record(raw, location):
    if not isinstance(raw, dict):
        raise ValueError(f'{location}: expected a JSON object')
    row = dict(raw)
    for source, target in [('paper_text', 'paper'), ('review_text', 'review_segment'),
                           ('reference_rebuttal', 'author_rebuttal')]:
        if source in row:
            if target in row and row[target] != row[source]:
                raise ValueError(f'{location}: conflicting {source} and {target}')
            row[target] = row.pop(source)
    for key in ('id', 'paper', 'review_segment'):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f'{location}: missing/empty string {key}')
    if 'author_rebuttal' in row and not isinstance(row['author_rebuttal'], str):
        raise ValueError(f'{location}: author_rebuttal must be a string')
    row['input_location'] = location
    return row


def _lines(stream, location):
    for number, line in enumerate(stream, 1):
        if line.strip():
            yield canonical_record(json.loads(line), f'{location}:line{number}')


def _records(path, split):
    if path.suffix.lower() == '.zip':
        if split not in ('train', 'eval'):
            raise ValueError('ZIP input requires --split train or --split eval')
        with zipfile.ZipFile(path) as archive:
            pattern = re.compile(r'(?:^|/)prepared_' + split + r'/part-\d+\.jsonl$')
            entries = sorted((e for e in archive.infolist() if pattern.search(e.filename)),
                             key=lambda e: e.filename)
            if not entries:
                raise ValueError(f'No prepared_{split}/part-*.jsonl in {path}')
            if len({e.filename for e in entries}) != len(entries):
                raise ValueError('Duplicate ZIP member names')
            for entry in entries:
                with archive.open(entry) as binary, io.TextIOWrapper(binary, encoding='utf-8-sig') as stream:
                    yield from _lines(stream, f'{path}!{entry.filename}')
    else:
        if split is not None:
            raise ValueError('--split is only used with ZIP input')
        with path.open(encoding='utf-8-sig') as stream:
            yield from _lines(stream, str(path))


def load_records(path, split=None, limit=None):
    path = Path(path)
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    if path.suffix.lower() == '.zip' and limit is None:
        raise ValueError('ZIP input requires an explicit --limit; use --limit 1 for a trial')
    records = _records(path, split)
    try:
        rows = list(islice(records, limit)) if limit is not None else list(records)
    finally:
        records.close()
    if not rows:
        raise ValueError('No input records selected')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate IDs in selected records')
    return rows
