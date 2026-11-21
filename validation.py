"""Local quote provenance checks; no model calls or fuzzy semantic matching."""
import re

_REPAIRS = {}
for _char in "‘’“”–—…\u00a0":
    for _encoding in ('cp1252', 'latin1'):
        try:
            _REPAIRS[_char.encode('utf-8').decode(_encoding)] = _char
        except UnicodeDecodeError:
            pass
_PATTERNS = sorted(_REPAIRS, key=len, reverse=True)
_PUNCT = {'‘': "'", '’': "'", '“': '"', '”': '"'}


def _normalise_with_offsets(text):
    """Normalize known typography while mapping to ORIGINAL source spans."""
    chars, offsets = [], []
    i = 0
    while i < len(text):
        start = i
        broken = next((p for p in _PATTERNS if text.startswith(p, i)), None)
        if broken:
            char = _REPAIRS[broken]
            i += len(broken)
        else:
            char = text[i]
            i += 1
        char = _PUNCT.get(char, char)
        if char.isspace():
            if chars and chars[-1] == ' ':
                offsets[-1] = (offsets[-1][0], i)
                continue
            char = ' '
        for out in char.casefold():
            chars.append(out)
            offsets.append((start, i))
    return ''.join(chars), offsets


def validate_evidence(decision, context, stage):
    adjustments, unverified = [], []
    if not decision['evidence']:
        unverified.append({'stage': stage, 'reason': 'No supporting evidence supplied'})
    for evidence in decision['evidence']:
        quote = evidence['quote']
        source = context.get(evidence['source'], '')
        if quote.strip() and quote in source:
            continue
        norm_quote, _ = _normalise_with_offsets(quote.strip())
        norm_source, offsets = _normalise_with_offsets(source)
        matches = []
        if len(norm_quote) >= 20:
            pattern = re.escape(norm_quote)
            if norm_quote[-1:] in ('.', ';'):
                pattern = re.escape(norm_quote[:-1]) + r'[.;]'
            for match in re.finditer(pattern, norm_source):
                a, b = match.span()
                if a and norm_source[a-1].isalnum() and norm_source[a].isalnum():
                    continue
                if b < len(norm_source) and norm_source[b-1].isalnum() and norm_source[b].isalnum():
                    continue
                if a and offsets[a] == offsets[a-1]:
                    continue
                if b < len(offsets) and offsets[b-1] == offsets[b]:
                    continue
                matches.append(match)
        if len(matches) == 1:
            start = offsets[matches[0].start()][0]
            end = offsets[matches[0].end()-1][1]
            corrected = source[start:end]
            adjustments.append({'source': evidence['source'], 'original_quote': quote,
                                'corrected_quote': corrected,
                                'source_start': start, 'source_end': end,
                                'reason': 'case/typography/whitespace normalization'})
            evidence['quote'] = corrected
        else:
            unverified.append({'source': evidence['source'], 'quote': quote,
                               'stage': stage,
                               'match_count_after_normalization': len(matches)})
    return adjustments, unverified


def audit_accepted(audit, has_unverified_evidence=False):
    return (not has_unverified_evidence and audit['passed'] and not audit['issues']
            and not audit['inconsistent_stages']
            and all(audit[s] == 2 for s in ('dp', 'etp', 'rap', 'synthesis')))
