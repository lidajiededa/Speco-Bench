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
- 数据集与 tokenizer 定长随机输入两种压测模式
- 本地/远程单图与多图 VLM 数据集、指定分辨率随机图像
- 本地 Web 控制台、批量任务进度与结果下载

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

图片数据集可以使用简写格式。`images` 中的相对路径以 JSONL 文件所在目录为
基准，也可以使用绝对路径、`file://`、HTTP(S) 或 `data:` URL：

```json
{"prompt":"比较两张图中的趋势","images":["images/a.jpg","images/b.jpg"],"max_tokens":256}
```

需要精确控制图文顺序或图片 `detail` 时，可以直接写 OpenAI messages：

```json
{"messages":[{"role":"user","content":[{"type":"text","text":"先看这张图"},{"type":"image_url","image_url":{"url":"images/chart.png","detail":"high"}},{"type":"text","text":"最高点是多少？"}]}],"max_tokens":128}
```

本地图片会在加载数据集时编码成 data URL。多模态请求只支持
`--endpoint-type chat`。推荐的图片 VLM 数据集和下载方式见
[`DATASETS.md`](DATASETS.md)。

仓库内置的五个评测数据集统一位于
`dataset/<dataset_name>/question.jsonl`，具体列表和重建方式见
[`DATASETS.md`](DATASETS.md)。

## 随机定长输入

随机模式不需要数据集文件。先安装 tokenizer 依赖：

```bash
pip install -e '.[random]'
```

下面的命令生成 1,000 个输入 1,024 token、期望输出 128 token 的请求：

```bash
speco-bench run \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/model \
  --dataset-name random \
  --endpoint-type completions \
  --random-input-len 1024 \
  --random-output-len 128 \
  --random-range-ratio 0 \
  --num-prompts 1000 \
  --concurrency 16 \
  --ignore-eos \
  --output-dir results/random-1024-128
```

`--tokenizer` 默认使用 `--model`；如果服务端模型名不是客户端可读取的
Hugging Face 名称或本地路径，可另外传入 `--tokenizer /path/to/tokenizer`。
随机输入由 `--seed` 控制，可重复生成。

`--random-range-ratio 0` 表示固定长度。设为 `0.2` 时，输入和输出长度
分别在目标值的 80% 至 120% 之间按整数均匀采样。

`--random-output-len` 和数据集模式的 `--max-tokens` 都会设置请求的生成
上限；模型仍可能遇到 EOS 提前结束。对 vLLM / vLLM-Ascend 使用
`--ignore-eos`，才能让服务端持续生成到请求长度。严格比较输入长度时建议
使用 `--endpoint-type completions`；chat 端点还会由服务端加入 chat
template token，因此服务端统计的总输入 token 会高于随机 prompt 本身。

随机 VLM 请求在上述参数之外指定图片宽高即可：

```bash
speco-bench run \
  --base-url http://127.0.0.1:8000 \
  --model qwen-vl-serving \
  --tokenizer /path/to/qwen-vl \
  --dataset-name random \
  --endpoint-type chat \
  --random-input-len 256 \
  --random-output-len 128 \
  --random-image-width 1280 \
  --random-image-height 720 \
  --random-images-per-prompt 1 \
  --num-prompts 100 \
  --concurrency 8
```

每条请求都会生成内容不同但可由 `--seed` 复现的 RGB PNG；不需要 Pillow。
`--random-input-len` 只约束文字部分。服务端上报的 prompt token 还会包含视觉
token 和 chat template token，因此跨模型比较时应固定模型、分辨率和图片数。

## 多数据集并发矩阵

`speco-bench matrix` 可一次输入多个数据集和多个并发数，并运行它们的
笛卡尔积：

```bash
speco-bench matrix \
  --base-url http://127.0.0.1:8000 \
  --model glm52 \
  --datasets gsm8k math500 humaneval mbpp mt_bench \
  --concurrencies 1 4 8 16 \
  --num-prompts 20 80 160 320 \
  --max-tokens 1024 \
  --output-dir results/matrix
```

数据集和并发数均支持逗号写法，例如
`--datasets gsm8k,math500 --concurrencies 1,4,8`。数据集短名默认解析为
`dataset/<name>/question.jsonl`，也可以直接传入 JSON/JSONL 文件路径。

`--num-prompts` 按位置与 `--concurrencies` 一一对应。上例依次运行
`(并发 1, 20 条)`、`(并发 4, 80 条)`、`(并发 8, 160 条)` 和
`(并发 16, 320 条)`；两个参数的数量不一致时脚本会报错。不传
`--num-prompts` 时，每种并发都使用完整数据集。

汇总结果会持续写入 `results/matrix/matrix.csv`。每个组合的完整
`summary.json` 和 `requests.jsonl` 分别保存在
`results/matrix/<dataset>/concurrency-<N>-prompts-<M>/`。CSV 包含请求
数量、请求吞吐、输出及总 token 吞吐、TTFT/TPOT/E2E 分位数、投机接受率
和错误信息。某个组合失败时默认记录错误后继续；传入 `--fail-fast`
可立即停止。

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

- `TTFT = 第一个有效文本增量到达时间 - 请求发出时间`。有效文本包括
  `content`、vLLM 当前使用的 `reasoning`，以及兼容旧服务的
  `reasoning_content`
- `TPOT = (端到端耗时 - TTFT) / (输出 Token 数 - 1)`
- `请求吞吐 = 成功请求数 / 正式压测时长`
- `输出吞吐 = 成功请求输出 Token 总数 / 正式压测时长`
- `投机 Token 接受率 = accepted draft tokens / draft tokens`
- `平均接受长度 = 1 + accepted draft tokens / draft rounds`
- `位置 i 接受率 = position i accepted tokens / draft rounds`

客户端请求 `stream_options.include_usage=true`，优先采用服务端返回的 token 数
（包括 reasoning parser 产生的推理 token）。服务不返回 streaming usage 时会退回
按非空的正文或推理流式 chunk 计数，并在结果中产生警告；该退回值不应视为精确
token 数。

投机指标取正式压测前后的 `/metrics` counter 增量。压测期间不要让其他客户端访问同一服务，否则其请求也会被计入。

## 输出

```text
results/glm52-dflash/
├── summary.json
└── requests.jsonl
```

`summary.json` 是稳定的结构化输出，供后续网页、数据库或报告模块直接消费。`requests.jsonl` 保存每个请求的成功状态、token 数、TTFT、TPOT、E2E 和错误信息。

## Web 控制台

从项目根目录启动：

```bash
speco-bench web \
  --host 127.0.0.1 \
  --port 8080 \
  --dataset-root dataset \
  --output-dir results/web
```

然后访问 `http://127.0.0.1:8080`。页面支持：

- 选择一个或多个内置数据集，也可以填写服务器本地 JSON/JSONL 路径；
- 使用随机定长输入，并分别填写服务模型名和 tokenizer / 模型路径；
- 为随机输入开启图片，并设置宽度、高度与每条请求图片数；
- 一次填写多个并发数，以及与其一一对应的请求数；
- 点击任意组合切换该组合的吞吐与独立量程延迟分布；
- 查看投机接受率、平均接受长度和所有草稿位置的平均接受率；
- 停止活动任务、查看最近任务，并下载每个任务的 `matrix.csv`。

网页后台一次只执行一个任务，任务内部按顺序运行数据集和并发组合，避免并行
任务污染同一服务的 `/metrics` counter 增量。结果保存在
`results/web/<job-id>/`。如需在网页中使用随机输入，安装时仍需启用 tokenizer
依赖：

```bash
pip install -e '.[random]'
```

默认只监听 `127.0.0.1`。如果使用 `--host 0.0.0.0` 暴露给其他机器，请在受信
网络或反向代理鉴权之后使用，因为网页可以发起压测并读取服务器本地数据集路径。

## 测试

项目测试只依赖 Python 标准库：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
