# Benchmark datasets

This repository contains GSM8K, MATH-500, HumanEval, sanitized MBPP, and
MT-Bench in a JSONL format that is directly readable by Speco-Bench. All
prepared files use the same `dataset/<name>/question.jsonl` layout. Every row
contains `prompt` and `max_tokens`; the four datasets produced by
`prepare_datasets.py` also retain `turns` for AngelSlim's
`tools/vllm_spec_benchmark.py`.

## Prepared datasets

| Dataset | Output | Records |
| --- | --- | ---: |
| GSM8K test | `dataset/gsm8k/question.jsonl` | 1,319 |
| MATH-500 | `dataset/math500/question.jsonl` | 500 |
| HumanEval | `dataset/humaneval/question.jsonl` | 164 |
| MBPP sanitized test | `dataset/mbpp/question.jsonl` | 257 |
| MT-Bench | `dataset/mt_bench/question.jsonl` | 80 |

Every output record has a non-empty `prompt`; the four converted datasets also
have a non-empty `turns` list. Reference answers, canonical solutions, and
tests are retained as metadata fields even though throughput benchmarks only
need the prompt.

## Rebuild

The checked-in `raw/` files are pinned and checksum-verified, so conversion
does not require Hugging Face, PyArrow, or any other third-party dependency:

```bash
python3 prepare_datasets.py
```

To fetch any missing raw files before conversion:

```bash
python3 prepare_datasets.py --download
```

To refresh every raw file from its pinned source:

```bash
python3 prepare_datasets.py --force-download
```

The source commits and SHA-256 checksums are recorded in
`prepare_datasets.py`.

## Use with Speco-Bench

Each file can be passed directly to the benchmark:

```bash
speco-bench run \
  --base-url http://127.0.0.1:8000 \
  --model "$MODEL_NAME" \
  --dataset-path dataset/math500/question.jsonl \
  --num-prompts 500 \
  --concurrency 8 \
  --output-dir results/math500
```

## Use with AngelSlim

Pass the generated files as comma-separated local paths:

```bash
python3 tools/vllm_spec_benchmark.py \
  --target_model "$MODEL_DIR" \
  --draft_model "$EAGLE_DIR" \
  --dataset "dataset/gsm8k/question.jsonl,dataset/math500/question.jsonl,dataset/humaneval/question.jsonl,dataset/mbpp/question.jsonl,dataset/mt_bench/question.jsonl" \
  --output_file benchmark_stats.jsonl \
  --method eagle3 \
  --output_len 1024 \
  --max_num_seqs 8
```

Alternatively, generate directly into AngelSlim's dataset directory so the
short names work:

```bash
python3 prepare_datasets.py \
  --output-dir /Users/lijie/Documents/dflash/AngelSlim/dataset
```

Then use:

```bash
--dataset "math500,humaneval,mbpp,mt_bench"
```

MATH-500 and HumanEval prompts follow the formatting used by AngelSlim's
`tools/dflash_benchmark.py`. MT-Bench preserves both turns. MBPP uses the same
plain task prompt as that benchmark and filters the official sanitized test
split (`task_id` 11 through 510).

## Image VLM datasets

`prepare_multimodal_datasets.py` supports four complementary image-only VLM
workloads. It deliberately excludes audio and video:

| Short name | Hugging Face dataset | Split | Workload |
| --- | --- | --- | --- |
| `ai2d` | `lmms-lab/ai2d` | test | Scientific diagrams and multiple choice |
| `chartqa` | `HuggingFaceM4/ChartQA` | test | Chart reading and visual reasoning |
| `textvqa` | `lmms-lab/textvqa` | validation | OCR and text understanding in natural images |
| `mmmu` | `MMMU/MMMU` | validation | Multi-discipline, single- and multi-image reasoning |

AI2D is a compact starting point, ChartQA isolates chart workloads, TextVQA
stresses OCR, and MMMU supplies the broadest and most difficult mix. Review each
dataset card and license before redistributing downloaded files.

Install the optional preparation dependencies and download one or more datasets:

```bash
pip install -e '.[multimodal]'
python3 prepare_multimodal_datasets.py --datasets ai2d chartqa
```

For a quick smoke dataset, cap each selected dataset independently:

```bash
python3 prepare_multimodal_datasets.py \
  --datasets ai2d chartqa textvqa mmmu \
  --limit 100 \
  --overwrite
```

The resulting layout is the same layout discovered by the CLI and Web console:

```text
dataset/ai2d/
├── images/
│   ├── 000000_1.jpg
│   └── ...
└── question.jsonl
```

Each JSONL row uses `prompt`, `images`, `max_tokens`, and reference metadata.
Image paths are relative to `question.jsonl`; Speco-Bench converts them to data
URLs immediately before benchmarking. Downloaded `dataset/*/images/` directories
are ignored by Git because the datasets are large and have their own licenses.
