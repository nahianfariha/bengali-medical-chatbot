# Bengali Medical Dialogue Generation with Qwen2.5-3B (QLoRA)

Fine-tuning `Qwen/Qwen2.5-3B-Instruct` to generate doctor-style responses to
Bengali patient prompts, using QLoRA (4-bit) so the whole pipeline runs on a
single free-tier Colab GPU (T4, ~15GB VRAM).

This started as an entry for the **Nasenica AI Hackathon: Bengali Medical
Dialogue Generation**. I didn't end up submitting to the leaderboard, but the
pipeline works end-to-end (train → validate → merge LoRA → batch-generate a
`submission.csv`), so I'm sharing it as a portfolio project.

## Problem

Given a Bengali patient prompt describing symptoms, generate a clinically
sound, well-communicated doctor response — in Bengali, under a 3B-parameter
base model budget.

Original competition scoring (for reference): a weighted blend of
BERTScore F1 (50%), token-level F1 (30%), and ROUGE-L F1 (20%) against a
reference doctor response.

## Constraints this project was built around

- **No local GPU** — everything targets a single Colab T4/A100 session.
- **3B parameter cap** on the base model (per competition rules).
- Training the full 108,954-example set on a T4 is slow, so this repo
  supports both a **quick, subsampled run** (`--sample_size`, e.g. 5,000
  examples, ~1–2 hrs on a T4) for iterating, and a **full-dataset run** with
  checkpointing to Google Drive so a disconnected Colab session can resume.

## Approach

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`, loaded in 4-bit NF4
  (bitsandbytes) with `bfloat16` compute dtype.
- **Fine-tuning:** QLoRA via `peft` — rank 16, alpha 32, dropout 0.05,
  targeting all attention + MLP projection matrices
  (`q/k/v/o_proj`, `gate/up/down_proj`).
- **Prompt format:** Qwen's chat template with a Bengali system prompt that
  instructs the model to act as an experienced, empathetic doctor — explain
  the likely cause first, then give advice, avoid speculation, no filler.
- **Label masking:** loss is only computed on the assistant's response
  tokens; the prompt (system + patient message) is masked with `-100`.
- **Custom data collator:** pads `input_ids`, `attention_mask`, and `labels`
  together (the default causal-LM collator would overwrite the label mask).
- **Optional:** Liger kernel fused ops for the full-dataset run, to reduce
  memory/time on longer training jobs.
- **Inference:** LoRA adapter is merged into the base weights
  (`merge_and_unload`) for faster, simpler batched generation, then run
  greedily (no sampling) with a repetition penalty over the test set.

## Repo layout

```
.
├── src/
│   ├── train.py        # end-to-end training: data -> tokenize -> QLoRA -> Trainer
│   ├── inference.py     # merge adapter + batch-generate submission.csv
│   └── evaluate.py      # BERTScore / token-F1 / ROUGE-L on a held-out split
├── notebooks/
│   ├── 01_quick_run.ipynb     # subsampled run for fast iteration (Colab T4)
│   └── 02_full_run.ipynb      # full 108k-example run w/ Drive checkpointing
├── configs/
│   └── default.yaml     # hyperparameters used in the reported run
├── requirements.txt
└── README.md
```

## Data

The dataset (`train.csv`, `test.csv`) is from the Nasenica AI Hackathon and
is distributed under **CC BY-NC 4.0** with competition-specific redistribution
restrictions, so it is **not included in this repo**. To reproduce:

1. Get `train.csv` / `test.csv` from the competition page.
2. Place them in a `data/` folder locally (already gitignored).

Expected columns: `id`, `input` (patient prompt, Bengali), `output`
(reference doctor response, train only).

## Usage

### Quick, subsampled run (fast iteration)

```bash
python src/train.py \
  --data_dir data \
  --output_dir outputs/qwen_medical_bengali \
  --sample_size 5000 \
  --epochs 1
```

### Full-dataset run, with resume support

```bash
python src/train.py \
  --data_dir data \
  --output_dir outputs/qwen_medical_bengali_full \
  --epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --use_liger \
  --resume
```

Re-running with `--resume` picks up from the latest `checkpoint-*` in
`output_dir` automatically — useful when a Colab session disconnects mid-run.

### Generate predictions

```bash
python src/inference.py \
  --adapter_dir outputs/qwen_medical_bengali_final \
  --test_csv data/test.csv \
  --out_csv submission.csv
```

### Evaluate on a held-out split

```bash
python src/evaluate.py \
  --adapter_dir outputs/qwen_medical_bengali_final \
  --val_csv data/val.csv
```

Reports the same composite metric as the original competition (0.5·BERTScore
F1 + 0.3·token F1 + 0.2·ROUGE-L F1), plus the three sub-scores individually.

## Results

*(Fill in once you've run it — e.g. composite score on your own held-out
split, sample generations, training loss curve.)*

| Run | Train examples | Epochs | Composite score |
|---|---|---|---|
| Quick (subsampled) | 5,000 | 1 | — |
| Full | ~103,500 | 2 | — |

## Limitations & disclaimer

This is a research/portfolio project, not a clinical tool. Outputs are not
medically validated and should never be used for real diagnostic or
treatment decisions. See the original dataset's license for usage
restrictions.

## Acknowledgements

- Nasenica AI Hackathon for the dataset and task design.
- `Qwen/Qwen2.5-3B-Instruct` (Alibaba Cloud) as the base model.
