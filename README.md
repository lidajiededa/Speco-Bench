# Speco-Bench

一个面向 vLLM / vLLM-Ascend OpenAI 兼容服务的轻量投机解码压测工具。第一版只做一件事：读取指定数据集，以固定并发发送流式请求，并输出 TTFT、TPOT、吞吐和投机接受指标。

## 已支持

- `/v1/chat/completions` 与 `/v1/completions`
- JSON / JSONL 数据集
- 固定并发、请求总数、输出长度、温度等参数
- 实时进度、成功/失败数、请求速率和预计剩余时间
- TTFT、TPOT、请求吞吐、输出 Token 吞吐
- `/metrics` 前后快照增量统计
- 投机平均接受长度、总接受率、各位置接受率
- 请求级 `requests.jsonl` 与汇总 `summary.json`
- ShareGPT、Alpaca、OpenAI messages、prompt 格式转换

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 准备数据集

统一格式为每行一条 JSON：

```json
{"messages":[{"role":"user","content":"请介绍投机解码"}],"max_tokens":1024}
```

也可以使用纯 prompt：

```json
{"prompt":"请介绍投机解码","max_tokens":1024}
```

转换常见格式：

```bash
speco-bench prepare \
  --input raw_dataset.json \
  --output dataset/custom/question.jsonl \
  --format auto \
  --default-max-tokens 1024
```

`--format` 支持 `auto`、`prompt`、`messages`、`sharegpt`、`alpaca`。ShareGPT 数据末尾已有的 assistant 参考答案会被移除，只保留待模型回答的上下文。

仓库内置的五个评测数据集统一位于
`dataset/<dataset_name>/question.jsonl`，具体列表和重建方式见
[`DATASETS.md`](DATASETS.md)。

## 运行压测

```bash
speco-bench run \
  --base-url http://127.0.0.1:8000 \
  --model glm52 \
  --dataset-path dataset/gsm8k/question.jsonl \
  --num-prompts 100 \
  --concurrency 16 \
  --max-tokens 1024 \
  --temperature 0 \
  --output-dir results/glm52-dflash
```

运行时会显示预热和正式压测进度：

```text
Benchmark [############------------]  50.00% 50/100 (ok=49, fail=1) elapsed=00:18 rate=2.78 req/s eta=00:18
```

进度默认每秒刷新。可调整刷新间隔，或在日志/自动化任务中关闭：

```bash
speco-bench run ... --progress-interval 2
speco-bench run ... --no-progress
```

进度写入标准错误流，最终报告仍写入标准输出，因此不会影响脚本读取最终结果。

如果服务的指标地址不是由 `base-url` 推导出的 `/metrics`：

```bash
speco-bench run \
  ... \
  --metrics-url http://127.0.0.1:8000/metrics
```

额外 OpenAI 请求参数可以通过 JSON 传入：

```bash
speco-bench run \
  ... \
  --extra-body '{"ignore_eos":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 指标口径

- `TTFT = 第一个有效文本增量到达时间 - 请求发出时间`
- `TPOT = (端到端耗时 - TTFT) / (输出 Token 数 - 1)`
- `请求吞吐 = 成功请求数 / 正式压测时长`
- `输出吞吐 = 成功请求输出 Token 总数 / 正式压测时长`
- `投机 Token 接受率 = accepted draft tokens / draft tokens`
- `平均接受长度 = 1 + accepted draft tokens / draft rounds`
- `位置 i 接受率 = position i accepted tokens / draft rounds`

客户端请求 `stream_options.include_usage=true`，优先采用服务端返回的 token 数。服务不返回 streaming usage 时会退回按非空流式 chunk 计数，并在结果中产生警告；该退回值不应视为精确 token 数。

投机指标取正式压测前后的 `/metrics` counter 增量。压测期间不要让其他客户端访问同一服务，否则其请求也会被计入。

## 输出

```text
results/glm52-dflash/
├── summary.json
└── requests.jsonl
```

`summary.json` 是稳定的结构化输出，供后续网页、数据库或报告模块直接消费。`requests.jsonl` 保存每个请求的成功状态、token 数、TTFT、TPOT、E2E 和错误信息。

## 网页化适配

代码把界面与执行逻辑分开：

- `BenchmarkService.run(config, progress_callback=...)` 是应用层入口；
- `BenchmarkConfig` 是可序列化配置模型；
- `BenchmarkReport` 是结构化结果；
- `ProgressUpdate` 提供阶段、完成数、成功/失败数、速率、耗时和 ETA；
- CLI 只负责参数解析和文件输出。

后续做网页时，可将 `progress_callback` 收到的 `ProgressUpdate.to_dict()` 写入任务状态或消息队列，再通过 WebSocket/SSE 推送给前端；前端不需要解析终端文本。

## 测试

项目测试只依赖 Python 标准库：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
