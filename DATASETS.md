# Additional benchmark datasets

This repository contains MATH-500, HumanEval, sanitized MBPP, and MT-Bench in
a JSONL format that is directly readable by Speco-Bench. Every row contains
`prompt` and `max_tokens`; the original `turns` field is also retained for
AngelSlim's `tools/vllm_spec_benchmark.py`.

## Prepared datasets

| Dataset | Output | Records |
| --- | --- | ---: |
| MATH-500 | `dataset/math500/question.jsonl` | 500 |
| HumanEval | `dataset/humaneval/question.jsonl` | 164 |
| MBPP sanitized test | `dataset/mbpp/question.jsonl` | 257 |
| MT-Bench | `dataset/mt_bench/question.jsonl` | 80 |

Every output record has a non-empty `prompt` and `turns` list. Reference
answers, canonical solutions, and tests are retained as metadata fields even
though throughput benchmarks only need the prompt.

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
  --dataset "/Users/lijie/Documents/Speco-Bench/dataset/math500/question.jsonl,/Users/lijie/Documents/Speco-Bench/dataset/humaneval/question.jsonl,/Users/lijie/Documents/Speco-Bench/dataset/mbpp/question.jsonl,/Users/lijie/Documents/Speco-Bench/dataset/mt_bench/question.jsonl" \
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
