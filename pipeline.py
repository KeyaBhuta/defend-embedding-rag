"""Run only when explicitly invoked. No API call on import."""
import argparse, json, os, hashlib
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from prompts import BASE, STAGES
import reference_taxonomy as tax
from validation import validate_evidence, audit_accepted
from data_input import load_records
from rag import context_hash, validate_retrieved

class Evidence(BaseModel):
    source: Literal['paper', 'review_segment', 'author_rebuttal']
    quote: str
class Decision(BaseModel):
    label: str
    explanation: str
    evidence: list[Evidence]
    uncertainty: str
class Draft(BaseModel):
    coherent_plan: str
    rebuttal: str
    author_checks: list[str]
class Audit(BaseModel):
    dp: int = Field(ge=0, le=2)
    etp: int = Field(ge=0, le=2)
    rap: int = Field(ge=0, le=2)
    synthesis: int = Field(ge=0, le=2)
    inconsistent_stages: list[Literal['dp', 'etp', 'rap', 'synthesis']]
    issues: list[str]
    passed: bool
class Comparison(BaseModel):
    dp_difference: str
    etp_difference: str
    rap_difference: str
    evidence_and_anchoring: str
    limitations: str

ERRORS = list(tax.mapping_error_type_rebuttal_action)[:-1]

def allowed(dp, etp):
    if dp == 'no':
        return tax.rebuttal_actions_nondeficient
    # Explicit fallback for empty mappings in the supplied implementation.
    return tax.mapping_error_type_rebuttal_action.get(etp) or tax.rebuttal_actions_deficient

def context(row, condition):
    # Whitelist: gold labels and rebuttals cannot accidentally leak into blind arm.
    result = {k: row[k] for k in ('paper', 'review_segment')}
    if condition == 'with_rebuttal':
        value = row.get('author_rebuttal', '')
        if not value.strip() or value.strip().lower() in ('na', 'none'):
            raise ValueError('Paired comparison requires a real author_rebuttal')
        result['author_rebuttal'] = value
    return result

class Runner:
    def __init__(self, args):
        from openai import OpenAI
        self.args = args
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                             base_url=os.getenv('OPENAI_BASE_URL') or None,
                             timeout=180, max_retries=3)
        self.root = Path(args.output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.counter = 0
    def call(self, stage, data, schema, judge=False):
        model = self.args.judge_model if judge else self.args.model
        messages = [{'role':'system','content':BASE + '\n' + STAGES[stage]},
                    {'role':'user','content':json.dumps(data, ensure_ascii=False)}]
        self.counter += 1
        prefix = self.root / f'call_{self.counter:05d}_{stage}'
        prefix.with_suffix('.request.json').write_text(json.dumps(
            {'model':model,'messages':messages}, ensure_ascii=False, indent=2), encoding='utf-8')
        response = self.client.responses.parse(model=model, input=messages,
                    text_format=schema, max_output_tokens=self.args.max_output_tokens,
                    store=False)
        # Serialize the concrete task schema, not the SDK's generic parsed
        # response wrapper (which produced Pydantic union serializer warnings).
        parsed = response.output_parsed
        usage = response.usage
        record = {
            'id': response.id, 'model': response.model, 'status': response.status,
            'output_text': response.output_text,
            'parsed': parsed.model_dump(mode='json') if parsed is not None else None,
            'usage': None if usage is None else {
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
                'total_tokens': usage.total_tokens,
                'cached_tokens': getattr(usage.input_tokens_details, 'cached_tokens', 0),
                'reasoning_tokens': getattr(usage.output_tokens_details, 'reasoning_tokens', 0)},
            'error': str(response.error) if response.error is not None else None,
            'incomplete_details': str(response.incomplete_details) if response.incomplete_details is not None else None,
            'refusals': [c.refusal for item in response.output
                         for c in getattr(item, 'content', [])
                         if getattr(c, 'type', None) == 'refusal']}
        prefix.with_suffix('.response.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        if response.status != 'completed' or response.output_parsed is None:
            raise RuntimeError(f'Refusal or incomplete response: {prefix}')
        return response.output_parsed.model_dump()
    def arm(self, row, condition):
        ctx = context(row, condition)
        state = {}
        definitions = {'deficiency':tax.DEFICIENT, 'errors':tax.ERROR_TYPES_DEFINITIONS,
                       'actions':tax.REBUTTAL_ACTIONS_DEFINITIONS}
        for stage in ('dp','etp','rap'):
            labels = ['yes','no'] if stage == 'dp' else (
                (ERRORS if state['dp']['label']=='yes' else ['none']) if stage=='etp'
                else allowed(state['dp']['label'],state['etp']['label']))
            d = self.call(stage, {**ctx,'definitions':definitions,
                                 'previous_predictions':state,'allowed_labels':labels}, Decision)
            if d['label'] not in labels:
                raise ValueError(f'Invalid {stage} label: {d["label"]}')
            d['evidence_adjustments'], d['unverified_evidence'] = validate_evidence(d, ctx, stage)
            state[stage] = d
        state['synthesis'] = self.call('synthesis',{**ctx,'predictions':state},Draft)
        state['verification'] = self.call('verify',{
            **ctx, 'definitions': definitions,
            'allowed_rap_actions': allowed(state['dp']['label'], state['etp']['label']),
            'candidate':state},Audit,True)
        a = state['verification']
        state['accepted'] = audit_accepted(
            a, any(state[s]['unverified_evidence'] for s in ('dp', 'etp', 'rap')))
        return state

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,help='Prepared data ZIP or JSONL')
    p.add_argument('--split',choices=['train','eval'],help='Required for prepared ZIP input')
    p.add_argument('--limit',type=int,help='Maximum records; required for ZIP trials')
    p.add_argument('--preview',action='store_true',help='Inspect selection; no API calls or output files')
    p.add_argument('--output',help='New directory; required unless --preview')
    p.add_argument('--model',default='gpt-5.4')
    p.add_argument('--judge-model',default=None)
    p.add_argument('--condition',choices=['without_rebuttal','with_rebuttal','paired'],default='paired')
    p.add_argument('--max-output-tokens',type=int,default=8000)
    p.add_argument('--repeats',type=int,default=1)
    args=p.parse_args()
    args.judge_model=args.judge_model or args.model
    if args.repeats<1: p.error('--repeats must be positive')
    if not args.preview and not args.output: p.error('--output is required for generation')
    try:
        rows = load_records(args.input, args.split, args.limit)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    for row in rows:
        if row.get('retrieval'):
            validate_retrieved(row)
        for k in ('id','paper','review_segment'):
            if not isinstance(row[k],str) or not row[k].strip(): raise ValueError(f'Empty {k}')
        if args.condition!='without_rebuttal': context(row,'with_rebuttal')
    if args.preview:
        print(json.dumps({'selected_records': len(rows), 'model': args.model,
                          'planned_calls': len(rows)*args.repeats*(11 if args.condition=='paired' else 5),
                          'api_calls_made': 0,
                          'records': [{'id': r['id'], 'paper_id': r.get('paper_id'),
                                       'record_unit': r.get('record_unit', 'review_segment'),
                                       'input_location': r['input_location'],
                                       'review_segment': r['review_segment']} for r in rows]}, indent=2))
        return
    runner=Runner(args)
    (runner.root/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    with (runner.root/'results.jsonl').open('w',encoding='utf-8') as out:
        for row in rows:
            for repeat in range(args.repeats):
                result={'id':row['id'],'paper_id':row.get('paper_id',row['id']), 'repeat':repeat,
                        'input_location': row['input_location'],
                        'record_unit': row.get('record_unit','review_segment'),
                        'student_context_sha256': context_hash(row),
                        'retrieval': row.get('retrieval'),
                        'input_sha256':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()}
                conditions=['without_rebuttal','with_rebuttal'] if args.condition=='paired' else [args.condition]
                # Alternate order across repeats; each API request has fresh context.
                if repeat%2: conditions.reverse()
                try:
                    for c in conditions: result[c]=runner.arm(row,c)
                    if args.condition=='paired':
                        result['comparison']=runner.call('compare',{
                            'source':context(row,'with_rebuttal'),'paired_outputs':result},Comparison,True)
                        result['label_changed']={s:result['without_rebuttal'][s]['label']!=result['with_rebuttal'][s]['label'] for s in ('dp','etp','rap')}
                except Exception as exc:
                    result['error']=f'{type(exc).__name__}: {exc}'
                out.write(json.dumps(result,ensure_ascii=False)+'\n'); out.flush()
if __name__=='__main__': main()
