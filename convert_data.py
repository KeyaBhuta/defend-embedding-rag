"""Convert supplied repository format; never executes its scripts."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reviews',required=True)
    p.add_argument('--papers',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--unit',choices=['concern','review'],default='concern')
    a=p.parse_args()
    papers=json.loads(Path(a.papers).read_text(encoding='utf-8-sig'))
    reviews=json.loads(Path(a.reviews).read_text(encoding='utf-8-sig'))
    with Path(a.output).open('x',encoding='utf-8') as f:
        for paper in reviews:
            pid=paper['paper_id'].strip()
            for ri,review in enumerate(paper['reviews']):
                review_id=f'{pid}:r{ri}'
                if a.unit == 'review':
                    text=review.get('review_content')
                    if not isinstance(text,str) or not text.strip():
                        raise ValueError(f'Missing full review for {review_id}')
                    # Segment responses are not a ground-truth full-review rebuttal.
                    row={'id':review_id,'paper_id':pid,'review_id':review_id,
                         'record_unit':'full_review','paper':papers[pid], 'review_segment':text}
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    continue
                for si,segment in enumerate(review['rebuttal_summary']):
                    row={'id':f'{pid}:{ri}:{si}','paper_id':pid,'paper':papers[pid],
                         'review_id':review_id,'segment_index':si,'record_unit':'atomic_concern',
                         'review_segment':segment['question'],
                         'author_rebuttal':segment.get('response','')}
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
