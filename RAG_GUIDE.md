# Embedding retrieval: first trial, experiments, and SFT

## Local trial completed

See the [README](README.md#one-paper-trial-results) for the retrieval comparisons
and the one-draft SWRG smoke test. Qwen3-Embedding-0.6B performed retrieval.
The SWRG draft used Qwen3-0.6B on CPU in FP32 and contained a convergence-rate
error. The Qwen3-8B generation experiment is pending.

`--dtype` records embedding precision in the cache identity, and `--show-progress`
prints embedding batch progress. `run_one_paper_trial.ps1` uses the installed
environment, pinned revisions, batch size 1, and optional `-Offline` operation.

## What is implemented

One paper-scoped retriever is shared across methods. It selects original source
passages; it does not create an overview or send the author rebuttal to the
embedding model. The generator sees only the selected evidence and review query.

| File | Purpose |
| --- | --- |
| `rag.py` | Exact source chunks, normalized similarity, overlap removal, cache and context hashes |
| `prepare_rag.py` | Read prepared ZIP/JSONL; save retrieved evidence and a human-readable inspection report |
| `rag_experiments.py` | Local Transformers inference: DRG, SWRG, existing SA-Pred-style pipeline, or merged SFT checkpoints |
| `prepare_rag_sft.py` | Review template and matched plain/reasoning chat SFT exports; no trainer |
| `requirements-rag.txt` | Optional embedding and local generation dependencies |
| `test_rag.py` | Offline correctness tests using fake embeddings/tokenizers |

`pipeline.py` now accepts and validates retrieved records, retains retrieval
provenance in results, and records a stable student-context hash. Its existing
paid GPT teacher behavior and paired comparison remain available.
`convert_data.py --unit review` supplies full-review inputs for DRG from the raw
dataset. The default concern conversion now preserves review grouping metadata.

## 1. Open the project and inspect one concern

Run commands in the NEW `outputs/defend_embedding_rag` folder. Example for this computer:

```powershell
Set-Location "C:\Users\KEYA BHUTA\Documents\Codex\2026-11-11\tasks-1-write-scripts-for-generating\outputs\defend_embedding_rag"
python prepare_rag.py --input "C:\Users\KEYA BHUTA\Downloads\data.zip" --split train --limit 1 --preview
```

This preview uses only Python's standard library. It does not load tokenizers,
download models, create output files, generate embeddings, or call a paid API.
Use the actual archive path on another computer. ZIPs must contain
`prepared_train/part-*.jsonl` or `prepared_eval/part-*.jsonl` as in this data.zip.
The older raw Dataset.zip is not the same format.

`--limit` always counts examples, not papers, GPU seconds, or dollars. First
records are selected in source order. `--limit 3` does not guarantee three concerns
from one paper: check the preview. No automatic random sampling or paper filtering.

## 2. Prepare a model environment (only when ready)

Use a separate environment from the existing GPT-only environment if preferred.
Python 3.10+ is needed. On a GPU machine install a PyTorch build appropriate for
its driver/CUDA setup using the official PyTorch installer, then install:

```powershell
python -m pip install -r requirements-rag.txt
```

The dependency ranges are starting compatibility constraints, not a tested lock
for every machine. Record your final environment with `python -m pip freeze` for
research reproducibility. Optional packages are imported only when needed.
Setup can require internet and substantial disk space. No GPT API key is needed
for embedding or local Qwen generation. Local compute is still a resource cost.

## 3. Retrieve evidence for exactly one training concern

This command can DOWNLOAD model/tokenizer files and RUN the embedding model:

```powershell
python prepare_rag.py --input "C:\Users\KEYA BHUTA\Downloads\data.zip" --split train --limit 1 --output runs/rag_train_01 --cache data/rag_cache --device cpu --batch-size 1
```

For a configured CUDA GPU, use `--device cuda --batch-size 1` initially. Increase
batch size only after checking memory. Use a new output directory for each run.
`--local-files-only` prevents downloads but requires cached files already present.

Defaults:

- Embeddings: `Qwen/Qwen3-Embedding-0.6B`; query instruction is explicit and saved.
- Chunk/budget tokenizer: `Qwen/Qwen3-8B`, tokenizer only (not generator weights).
- 500-token source windows, 75-token overlap, snapping to later paragraph/newline
  boundaries where available. Offsets refer to unchanged Unicode source text.
- Up to 6 evidence spans, within 3,000 generator-token evidence budget including
  chunk labels. Similarity chooses spans; they are presented in original order.
- Already selected overlap is removed using source offsets, not paraphrased.
- Only the current paper is searched. Similarity is not confidence or entailment.
- Overlong embedding queries/documents raise an error; no silent truncation.

To try section-first retrieval, add `--chunking section` or run
`./run_one_paper_trial.ps1 -Chunking section -Offline`. Short Markdown sections
stay whole. Longer sections split between paragraphs, with recognized LaTeX
display equations kept intact. The 500-token chunk size is a target in this mode,
so a single long paragraph can exceed it. Section mode selects whole passages
without overlap trimming and skips candidates that exceed the remaining evidence
budget. It builds a separate cache from the original window mode.

`--top-k` sets the maximum number of passages, with a default of 6.
Use `--top-k 8` (or `-TopK 8` with the PowerShell shortcut) to raise it.
The separate evidence budget still applies, so the actual count can be lower.

If the full-review query exceeds the embedding limit, stop and design/document a
multi-query protocol rather than silently truncating it. If no chunk fits the
evidence budget, increase the budget or reduce chunk size.

### Files produced

```text
runs/rag_train_01/
  config.json         requested configuration
  retrieval.jsonl     canonical input with paper replaced by retrieved evidence
  inspection.md       concern, evidence, ranks, scores and source offsets
  summary.json        timing, package/model identity and GPU memory (when CUDA)
data/rag_cache/<hash>/
  index.json          exact original chunks, paper fingerprint and configuration
  vectors.npy         cached document embeddings
```

The reference rebuttal is retained in saved JSONL for teacher comparisons/targets,
but is excluded from query encoding and student input. Do not blindly stringify
the full saved record into a model prompt; use the provided whitelisted functions.

Cache keys cover paper ID/text, chunk settings, tokenizer fingerprint, embedding
model/revision/resolved commit, and library versions. Paper embeddings are reused
across concerns and methods with identical configuration. Query embeddings are
computed per request. Models still load on each invocation; cache does not remove
loading time. Pin `--embedding-revision` and `--tokenizer-revision` to HF commit IDs
for final experiments rather than moving `main`. Avoid concurrent writers to the
same cache; use separate cache paths for parallel jobs.

### Inspect before scaling

Read `inspection.md`: is dataset/method/results evidence actually present? Check
tables manually: the text splitter cannot reconstruct lost table columns or
guarantee intact tables in flat text. The script preserves offsets, not inferred
section headings or PDF page numbers. If extraction is poor, fix it upstream.

Record evidence relevance/coverage on a development sample, compare chunk settings
there, and freeze them before evaluating the 20-paper test set. Retrieval is not
proof that the selected facts support the final response. Avoid claims of absence
from the complete paper when only excerpts were supplied.

## 4. Local Qwen SWRG and SA-Pred

Preview does NOT load generation weights or make calls:

```powershell
python rag_experiments.py --input runs/rag_train_01/retrieval.jsonl --limit 1 --method swrg
python rag_experiments.py --input runs/rag_train_01/retrieval.jsonl --limit 1 --method sa_pred
```

When ready, append `--run` and a new output directory. These commands can download
Qwen3-8B and require substantially more memory than the embedding stage:

```powershell
python rag_experiments.py --input runs/rag_train_01/retrieval.jsonl --limit 1 --method swrg --output runs/swrg_rag_01 --run
python rag_experiments.py --input runs/rag_train_01/retrieval.jsonl --limit 1 --method sa_pred --output runs/sa_rag_01 --run
```

- SWRG: one generation per concern, JSON `rebuttal` output.
- SA-Pred here reuses this repository's taxonomy and prompts: DP, ETP, RAP,
  synthesis, plus a separate audit = FIVE local calls. Audit does not revise
  predictions; log its acceptance separately. This is an implementation of the
  existing pipeline, not a verified reproduction of external baseline code.
- Both receive the same saved evidence. No student call receives reference text.
- Default local generation is deterministic (`do_sample=False`), with Qwen
  `enable_thinking=False`. Requested public explanations still appear in JSON.
- Default configured context is 16,384 tokens with 2,048 reserved output tokens.
  Every complete stage prompt is checked with the actual generator tokenizer.
  Exceeding the budget raises an error; context is never silently truncated.
- All requests/raw responses are saved. Invalid JSON or output truncation stops
  at the failed example with an error record. No automatic retry or paid fallback.
- `--device-map auto` uses Transformers/Accelerate placement; `--device-map cpu`
  requests CPU placement. No quantization or LoRA adapter loading is implemented.

The code has not been run with real Qwen weights in this workspace. Verify
hardware compatibility, output formatting and evidence quality on one example
before scaling. Model outputs and automatic audit scores are not guaranteed.

## 5. DRG needs actual full reviews

Prepared atomic ZIP entries do not contain the original full review. Do not label
a single concern as a full review or concatenate a partial selection of concerns.
For this supplied raw schema, create a JSONL file with `--unit review`:

```powershell
New-Item -ItemType Directory -Path data -Force
python convert_data.py --reviews "../defend_gpt54/data/Dataset/test/qa pair.json" --papers ../defend_gpt54/data/paper_data.json --unit review --output data/drg_reviews_20papers.jsonl
```

This is conversion only. It reads the original project's supplied 20-paper test files without modifying them and writes
full-review rows; it does not run inference. The converter refuses overwriting.
For development use a training/development raw review file and matching full
paper mapping instead. The converter loads those JSON mappings in memory; the
large processed_papers.json therefore needs sufficient RAM.

Each full-review row has `record_unit=full_review`, `review_id`, and the complete
`review_content` in the canonical `review_segment` field. It does not invent a
full reference rebuttal by combining segment responses.

After freezing retrieval settings, example evaluation commands are:

```powershell
python prepare_rag.py --input data/drg_reviews_20papers.jsonl --unit review --limit 1 --preview
python prepare_rag.py --input data/drg_reviews_20papers.jsonl --unit review --limit 1 --output runs/drg_evidence_01 --device cpu --batch-size 1
python rag_experiments.py --input runs/drg_evidence_01/retrieval.jsonl --limit 1 --method drg
```

Append `--output runs/drg_rag_01 --run` to the last command to generate explicitly.
DRG uses the whole review as its query and generates one full-review rebuttal.
Record this retrieval policy; it can underrepresent smaller concerns. Concern-wise
query expansion/merging would be a different protocol, not implemented here.

## 6. Existing GPT teacher can use retrieved context

Preview only:

```powershell
python pipeline.py --input runs/rag_train_01/retrieval.jsonl --limit 1 --condition paired --model gpt-4o-mini --preview
```

To actually generate teacher targets, use the existing API-key setup and replace
`--preview` with `--output runs/teacher_rag_01`. This is a PAID operation, normally
11 calls per paired record, and may have transport retries. GPT-5.4 can be selected
with `--model gpt-5.4` where accessible. Student-context hashes bind the targets to
the same evidence used for student training. The full-paper teacher trials remain
separate; their targets need review against the newly selected evidence.

## 7. Prepare matched plain and coherent-reasoning SFT examples

No SFT trainer was present in this workspace. These commands prepare portable
chat-format data for your existing trainer; they do not perform SFT or RL.

First make an UNAPPROVED review template (standard Python + existing Pydantic;
no models). Optional teacher results must have only one repeat per ID:

```powershell
python prepare_rag_sft.py --input runs/rag_train_01/retrieval.jsonl --limit 1 --make-review-template --output runs/sft_review_01
```

If available, add `--teacher-results runs/teacher_rag_01/results.jsonl --condition
without_rebuttal` (or `with_rebuttal`, explicitly recording that experimental
condition). Without results, the reference rebuttal is the initial reply and the
coherent explanation is left empty. Missing teacher records also fall back to
that template, visibly marked in `candidate_source`.

Edit `runs/sft_review_01/review_targets.jsonl`:

1. Read the exact evidence in `inspection.md`.
2. Check/rewrite `coherent_reasoning` as the final concise explanation, not a plan
   about what the authors will do. The existing teacher's `coherent_plan` is only
   a candidate and does not automatically meet this requirement.
3. Check/rewrite `rebuttal` so it makes no unsupported claims or commitments.
4. Add a meaningful `review_note` and set `approved` to `true` only after review.
5. Keep ID, paper ID and context hash unchanged. If evidence changes, review again.

The script does not assume teacher `accepted=true` establishes correctness. It
marks whether the teacher's context matches; old full-paper results can be shown
as candidates but require checking and editing against the retrieved context.

Export both conditions on EXACTLY the same approved examples and rebuttal targets:

```powershell
python prepare_rag_sft.py --input runs/rag_train_01/retrieval.jsonl --limit 1 --targets runs/sft_review_01/review_targets.jsonl --partition train --mode both --heldout-input ../defend_gpt54/data/dataset.jsonl --output data/sft_rag_01
```

`../defend_gpt54/data/dataset.jsonl` is the original project's earlier 20-paper test conversion. Use
the correct held-out JSONL for your own experiment. `--heldout-input` checks paper
IDs, not just concern IDs. Prepared ZIP `eval` sources cannot be exported as train.
For custom JSONL, declare partitions correctly: filenames alone cannot prove a
record is training data. Without `--heldout-input`, report.json explicitly records
that no cross-file held-out check was performed.

Output:

- `plain.jsonl`: evidence + concern -> reviewed rebuttal.
- `reasoning.jsonl`: identical evidence + concern -> one coherent explanation +
  the same rebuttal. No separate DP/ETP/RAP explanation fields in the target.
- `report.json`: exported/rejected counts and split-check status. Unapproved rows
  are listed as rejected; zero approved examples is an error.

Both files contain `messages` (system, user, assistant) plus provenance metadata.
The assistant content is JSON. References, verifier metadata and teacher outputs
are absent from the user messages. A generated teacher rebuttal vs original author
rebuttal is a separate research choice; paired exports hold the reviewed target
constant to isolate the effect of explanation supervision. Record any corrections.

### Requirements for the existing SFT trainer

- Read the `messages` field, not the whole metadata record.
- Train on assistant tokens only; mask system/user tokens from loss.
- Use the same chat template and `enable_thinking=False` as inference.
- Check full training-example token length, including targets. Reject or revisit
  overlength examples rather than truncating evidence/answers silently.
- Split by paper. Start plain and reasoning comparisons from the same checkpoint.
- Retain provenance separately; do not turn review_note or audit scores into targets.

After training, preview inference using an actual merged HF checkpoint:

```powershell
python rag_experiments.py --input runs/rag_train_01/retrieval.jsonl --limit 1 --method plain_sft --model "C:\models\qwen_plain_merged"
python rag_experiments.py --input runs/rag_train_01/retrieval.jsonl --limit 1 --method reasoning_sft --model "C:\models\qwen_reasoning_merged"
```

These paths are placeholders. Use held-out retrieved inputs for actual evaluation,
and append `--run --output NEW_DIRECTORY` only when ready. Adapter-only checkpoints
must be merged using your training tool first; adapter loading is not supplied.

## 8. Experimental boundaries

This addition implements the embedding arm. It does not replace any external
LLM-based retrieval workflow, reproduce unavailable historical baseline scripts,
extract abstracts automatically, aggregate segment replies into a full review,
or train models. Retain those external workflows and document input policies.
Prepared ZIP IDs are preserved verbatim; full review grouping is only available
where explicit metadata is supplied or the raw converter is used. Do not infer
complete review coverage from a limited concern selection.

Use the same concern evidence for SWRG, SA-Pred and matched SFT comparisons.
For abstract/full-paper runs, use corresponding existing experiments with the
same generator, decoding and output limits. Do not compare greedy local runs to
different sampled/API runs and attribute all differences to retrieval.

No real embedding quality, model-generation success, training outcome or GPU time
is established by the offline tests. The installed environment has now completed
one CPU embedding trial; its separate report documents results and limitations.
One local Qwen3-0.6B SWRG draft was also generated. GPU execution and Qwen3-8B
generation remain untested here. The results cover one example.

## 9. Offline verification and references

```powershell
python -m unittest test_rag test_data_input test_validation test_pipeline -v
```

Tests require NumPy and existing Pydantic, but not model packages or an API key.
They cover source offsets, coverage, budgets, overlap removal, paper/cache
isolation, corruption checks, reference exclusion, DRG unit checks, context-bound
SFT review and split guards, and malformed-output rejection.

Primary references consulted:

- [Qwen3 embedding model and query instructions](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3-8B generation and thinking controls](https://huggingface.co/Qwen/Qwen3-8B)
- [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard): use retrieval tasks to shortlist models; no rank is asserted here.

The current implementation uses its own tokenizer-offset splitter, rather than
adding LangChain or a vector database. This keeps exact source provenance and
works with a small per-paper embedding matrix.
