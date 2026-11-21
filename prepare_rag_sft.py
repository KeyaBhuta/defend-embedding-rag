"""Make review templates or export reviewed chat SFT data. Does not train a model."""
import argparse
import json
from pathlib import Path
from data_input import load_records
from rag import validate_retrieved, context_hash, digest
from rag_experiments import student_messages


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def unique_by_id(records):
    result = {}
    for item in records:
        if item['id'] in result:
            raise ValueError(f"Duplicate target/result ID {item['id']}; select exactly one repeat first")
        result[item['id']] = item
    return result


def review_template(rows, teacher=None, condition='without_rebuttal'):
    teacher = teacher or {}
    for row in rows:
        validate_retrieved(row)
        record = teacher.get(row['id'], {})
        arm = record.get(condition, {})
        synthesis = arm.get('synthesis', {})
        matches = record.get('student_context_sha256') == context_hash(row)
        yield {'id': row['id'], 'paper_id': row['paper_id'],
               'student_context_sha256': context_hash(row), 'approved': False,
               'review_note': '', 'coherent_reasoning': synthesis.get('coherent_reasoning',
                                                    synthesis.get('coherent_plan', '')),
               'rebuttal': synthesis.get('rebuttal', row.get('author_rebuttal', '')),
               'candidate_source': 'teacher synthesis (coherent_plan requires review/rewrite)'
                   if synthesis else 'reference rebuttal; reasoning not yet supplied',
               'teacher_condition': condition if record else None,
               'teacher_context_matches': matches if record else None,
               'teacher_accepted': arm.get('accepted'), 'teacher_error': record.get('error'),
               'instruction': 'Inspect exact retrieved evidence. Edit targets into a supported final '
                   'explanation and rebuttal (not a future plan). Set approved=true and add review_note '
                   'only after checking both. Teacher acceptance is not human approval.'}


def build_examples(rows, targets, mode, partition, heldout_ids=()):
    examples, rejected = [], []
    for row in rows:
        validate_retrieved(row)
        if row.get('record_unit') == 'full_review':
            raise ValueError('This SFT export compares concern-level targets only')
        if partition == 'train' and (row['paper_id'] in heldout_ids or
                                     row['retrieval'].get('source_split') == 'eval'):
            raise ValueError(f"{row['id']}: held-out/eval paper cannot enter training")
        target = targets.get(row['id'])
        if not target or target.get('approved') is not True:
            rejected.append({'id': row['id'], 'reason': 'missing or unapproved target'})
            continue
        if target.get('paper_id') != row['paper_id'] or target.get('student_context_sha256') != context_hash(row):
            raise ValueError(f"{row['id']}: target approval belongs to a different evidence context")
        if not isinstance(target.get('review_note'), str) or not target['review_note'].strip():
            raise ValueError(f"{row['id']}: approved target needs a review_note")
        reply = target.get('rebuttal')
        trace = target.get('coherent_reasoning')
        if not isinstance(reply, str) or not reply.strip() or reply.strip().lower() in ('na', 'none'):
            raise ValueError(f"{row['id']}: missing rebuttal target")
        if mode in ('both', 'reasoning') and (not isinstance(trace, str) or not trace.strip()):
            raise ValueError(f"{row['id']}: missing coherent explanation")
        common = {'id': row['id'], 'paper_id': row['paper_id'], 'partition': partition,
                  'student_context_sha256': context_hash(row),
                  'target_review_sha256': digest(target)}
        pair = {}
        for name in (('plain', 'reasoning') if mode == 'both' else (mode,)):
            answer = {'rebuttal': reply}
            if name == 'reasoning':
                answer = {'coherent_reasoning': trace, 'rebuttal': reply}
            messages = student_messages(row, name == 'reasoning')
            messages.append({'role': 'assistant', 'content': json.dumps(answer, ensure_ascii=False)})
            pair[name] = {**common, 'messages': messages}
        examples.append(pair)
    return examples, rejected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Retrieved training/development JSONL')
    p.add_argument('--limit', required=True, type=int)
    p.add_argument('--output', required=True, help='New directory')
    p.add_argument('--make-review-template', action='store_true', help='No models required')
    p.add_argument('--teacher-results', help='Optional teacher results, one repeat per ID')
    p.add_argument('--condition', choices=['without_rebuttal','with_rebuttal'], default='without_rebuttal')
    p.add_argument('--targets', help='Reviewed JSONL with approved=true and matching context hashes')
    p.add_argument('--mode', choices=['plain','reasoning','both'], default='both')
    p.add_argument('--partition', choices=['train','validation'], help='Required for export')
    p.add_argument('--heldout-input', help='JSONL containing paper_id for held-out papers; blocks training overlap')
    a = p.parse_args()
    rows = load_records(a.input, limit=a.limit)
    root = Path(a.output)
    if root.exists():
        p.error('Choose a new output directory')
    if a.make_review_template:
        teacher = unique_by_id(read_jsonl(a.teacher_results)) if a.teacher_results else {}
        templates = list(review_template(rows, teacher, a.condition))
        root.mkdir(parents=True, exist_ok=False)
        with (root / 'review_targets.jsonl').open('x', encoding='utf-8') as stream:
            for item in templates:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'Created {len(templates)} UNAPPROVED targets. No training export or model calls.')
        return
    if not a.targets or not a.partition:
        p.error('Export requires --targets and --partition')
    heldout = {r['paper_id'] for r in read_jsonl(a.heldout_input)} if a.heldout_input else set()
    targets = unique_by_id(read_jsonl(a.targets))
    examples, rejected = build_examples(rows, targets, a.mode, a.partition, heldout)
    if not examples:
        p.error('No reviewed examples to export; review targets before setting approved=true')
    root.mkdir(parents=True, exist_ok=False)
    for name in (('plain', 'reasoning') if a.mode == 'both' else (a.mode,)):
        with (root / f'{name}.jsonl').open('x', encoding='utf-8') as stream:
            for pair in examples:
                stream.write(json.dumps(pair[name], ensure_ascii=False) + '\n')
    report = {'selected': len(rows), 'exported': len(examples), 'rejected': rejected,
              'mode': a.mode, 'partition': a.partition, 'heldout_check_performed': bool(a.heldout_input),
              'target_policy': 'Same reviewed rebuttal for paired plain/reasoning exports',
              'training_started': False,
              'trainer_requirements': 'Use messages, assistant-only loss; tokenize full examples and '
                  'reject overlength samples rather than silently truncating evidence/targets. '
                  'Use Qwen enable_thinking=False consistently with inference.'}
    (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
