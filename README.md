# DEFEND embedding RAG — separate project

## Completed one-paper trial

A real Qwen3-Embedding-0.6B retrieval trial completed on CPU/FP32:
37 chunks, 6 selected passages, 2,333 evidence tokens. Recorded model loading plus
retrieval took about 6 minutes 10 seconds, excluding setup/imports/aborted attempts.
No paid API calls, rebuttal generation or training were performed.

Read [the trial report](runs/one_paper_trial_fp32_01/TRIAL_REPORT.md) and
[retrieved evidence](runs/one_paper_trial_fp32_01/inspection.md).
Evidence was relevant but some mathematical boundaries and coverage need work
before scaling. Integrity checks passed; this is not a semantic quality guarantee.

This computer now has an isolated `.venv` with CPU PyTorch. Use
`.\.venv\Scripts\python.exe` if `python` is not on PATH. A reproducible optional
rerun is `./run_one_paper_trial.ps1 -Offline` (one concern, pinned models, fresh
output folder, cached document embeddings reused). No rerun is needed to read
the existing results. Do not use `--device cuda` until CUDA PyTorch is installed.

This folder contains the embedding-retrieval implementation and its own copies
of all supporting Python modules. It does not import code from or modify the
original `defend_gpt54` project. No models, environments, historical runs, Git
history, or large datasets were copied.

Start with [RAG_GUIDE.md](RAG_GUIDE.md) for installation, retrieval, DRG, SWRG,
SA-Pred, and plain/coherent-reasoning SFT data preparation.

## First step: preview exactly one training concern

```powershell
Set-Location "C:\Users\KEYA BHUTA\Documents\Codex\2026-11-11\tasks-1-write-scripts-for-generating\outputs\defend_embedding_rag"
python prepare_rag.py --input "C:\Users\KEYA BHUTA\Downloads\data.zip" --split train --limit 1 --preview
```

Preview requires no model download or paid API call and writes no output files.
The dataset remains external. Change the input path when sharing this project.

Install optional model dependencies only when ready:

```powershell
python -m pip install -r requirements-rag.txt
```

Actual one-concern embedding retrieval (can download model files and use compute):

```powershell
python prepare_rag.py --input "C:\Users\KEYA BHUTA\Downloads\data.zip" --split train --limit 1 --output runs/rag_train_01 --cache data/rag_cache --device cpu --batch-size 1
```

New `data/` caches, SFT exports and `runs/` outputs belong to THIS folder and are
ignored by the supplied `.gitignore`. The guide points to the original project's
data only for optional raw full-review conversion and held-out paper checks;
those are read-only inputs and can be replaced with independent dataset paths.

## Scope and verification

- Original-text chunking, saved embeddings, per-paper similarity retrieval.
- Local Qwen generation commands for DRG/SWRG/SA-Pred and merged SFT checkpoints.
- Reviewed chat-format SFT data export; no SFT/RL trainer is included.
- One real embedding retrieval trial has completed; generation and training have not run.
- Offline tests use fake embeddings and do not establish model quality or GPU compatibility.

```powershell
python -m unittest test_rag test_data_input test_validation test_pipeline -v
```

The historical-trial replay test skips when old runs are absent, as intended in
this fresh folder. New model runs must be validated on one example before scaling.
