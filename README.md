# DEFEND embedding RAG — separate project

## Kaggle GPU trial on one training concern

Retrieval defaults to **Qwen3-Embedding-4B**. SWRG, DRG and SA-Pred generation
default to **Qwen3-8B**. The PowerShell shortcut replays the older 0.6B CPU
retrieval trial. On Kaggle, enable a GPU, open the repository directory with
`%cd` in the notebook, and attach the ZIP as a private dataset. Run these cells
in order. Replace the example ZIP path with the path shown by Kaggle.

```python
%pip install -r requirements-rag.txt bitsandbytes
dataset_zip = "/kaggle/input/YOUR-DATASET/data.zip"
!nvidia-smi
!python prepare_rag.py --input "{dataset_zip}" --split train --limit 1 --preview
```

Each command below runs in its own process so retrieval releases GPU memory
before generation starts. The first model run needs access to the weights.

```python
!python prepare_rag.py --input "{dataset_zip}" --split train --limit 1 --output /kaggle/working/rag_4b_01 --cache /kaggle/working/rag_cache --device cuda --dtype float16 --batch-size 1 --chunking section --top-k 10 --evidence-tokens 4000
```

Inspect `/kaggle/working/rag_4b_01/inspection.md`, then generate one SWRG draft:

```python
!python rag_experiments.py --input /kaggle/working/rag_4b_01/retrieval.jsonl --method swrg --limit 1 --dtype float16 --load-in-4bit --output /kaggle/working/swrg_8b_01 --run
```

The generation command uses 4-bit weights to reduce GPU memory use. Read
`/kaggle/working/swrg_8b_01/results.jsonl` for the result. Use fresh output
directory names if you repeat the trial.

## One-paper trial results

The trials used Qwen3-Embedding-0.6B and one atomic concern about logistic-regression
covariance formulas.

| Retrieval setting | Chunks indexed | Passages selected | Evidence tokens | Named results found |
| --- | ---: | ---: | ---: | --- |
| Token windows, up to 6 passages / 3,000 tokens | 37 | 6 | 2,333 | Theorem 4 |
| Section-first, up to 6 passages / 3,000 tokens | 41 | 6 | 2,244 | Theorem 4 |
| Section-first, up to 8 passages / 4,000 tokens | 41 | 8 | 2,928 | Theorems 2 and 4 |
| Section-first, up to 10 passages / 4,000 tokens | 41 | 10 | 3,688 | Theorems 1, 2, and 4 |

Section-first chunking kept recognized display equations intact in the selected
passages. Theorem 1 ranked tenth. The paragraph defining Theorem 2's covariance
matrix ranked nineteenth and was outside the retrieved context.

A local **Qwen3-0.6B SWRG smoke test** used the ten passages, produced parseable
JSON, and incorrectly substituted the usual rate `N^(-1/2)` for the rare-event
rate `N₁^(-1/2)`. It also left the covariance request unanswered. The intended
generator for the main experiments is **Qwen3-8B**, which is pending evaluation.
These trials used local inference without paid API calls.

Local trial artifacts are under `runs/`. Git ignores that directory, so the
summary above is available in a fresh clone while the raw outputs stay local.

The code and supporting modules are self-contained. Setup requires Python and
access to the external dataset. With an existing virtual environment on Windows,
use `.\.venv\Scripts\python.exe` if `python` is unavailable on PATH.

Start with [RAG_GUIDE.md](RAG_GUIDE.md) for installation, retrieval, DRG, SWRG,
SA-Pred, and plain/coherent-reasoning SFT data preparation.

## First step: preview exactly one training concern

Open a terminal in this repository and set the dataset path for your machine.

```powershell
$datasetZip = 'C:\path\to\data.zip'
python prepare_rag.py --input $datasetZip --split train --limit 1 --preview
```

Preview requires no model download or paid API call and writes no output files.
The dataset remains external. Change the input path when sharing this project.

Install optional model dependencies only when ready:

```powershell
python -m pip install -r requirements-rag.txt
```

Actual one-concern embedding retrieval (can download model files and use compute):

```powershell
python prepare_rag.py --input $datasetZip --split train --limit 1 --output runs/rag_train_01 --cache data/rag_cache --device cpu --batch-size 1 --chunking section --top-k 10 --evidence-tokens 4000
```

New `data/` caches, SFT exports and `runs/` outputs belong to THIS folder and are
ignored by the supplied `.gitignore`. The guide points to the original project's
data only for optional raw full-review conversion and held-out paper checks;
those are read-only inputs and can be replaced with independent dataset paths.

## Scope and verification

- Original-text chunking, saved embeddings, per-paper similarity retrieval.
- Local Qwen generation commands for DRG/SWRG/SA-Pred and merged SFT checkpoints.
- Reviewed chat-format SFT data export; no SFT/RL trainer is included.
- One paper was used for retrieval comparisons and one local SWRG smoke test.
- Offline tests use fake embeddings and do not establish model quality or GPU compatibility.

```powershell
python -m unittest test_rag test_data_input test_validation test_pipeline -v
```

The historical-trial replay test skips when old runs are absent, as intended in
this fresh folder. New model runs must be validated on one example before scaling.
