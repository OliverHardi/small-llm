# small-llm

A GPT-style language model written from scratch in C++ (CPU and CUDA): tensor library, tokenizer, data pipeline,
bf16 training, instruction tuning, and a chat interface. The trained model also runs entirely in the browser with WebGPU.

**Live demo:** https://oliverhardi.github.io/small-llm

## The model

| | |
|---|---|
| Parameters | 100.3M |
| Layers | 16 |
| Model width | 640 |
| Attention | 10 heads (head dim 64), causal, RoPE |
| Feed-forward | SwiGLU, hidden size 1728 |
| Normalization | RMSNorm (pre-norm) |
| Embeddings | tied input/output, vocabulary 32,768 |
| Context length | 2048 tokens |


## What is implemented from scratch

- **Tensor library and ops** with matching CPU and CUDA paths (matmul, RMSNorm, RoPE, softmax, cross-entropy, embedding, SwiGLU, attention), each with a hand-written backward pass. The two paths are checked against each other in parity tests.
- **CUDA training path:** bf16 matmuls through cuBLAS with fp32 accumulation, fp32 master weights, an optional cuDNN flash-attention path, a caching GPU allocator, gradient accumulation and a single-sync gradient clip.
- **AdamW**, a cosine learning-rate schedule with warmup, and resumable checkpoints (weights, optimizer state, data position).
- **Byte-level BPE tokenizer** with a GPT-2-style pretokenizer (PCRE2)
- **Data pipeline:** parquet reading, tokenization to flat binary shards, and mixing several corpora to a target ratio.
- **Sampling:** temperature, top-p and repetition penalty.
- **Instruction tuning (SFT):** a chat template, loss masked to the assistant's tokens, whole-dialogue packing, validation on held-out dialogues.

## Training

**Pretraining**

- About 5B tokens, mixed from FineWeb (web), FineWeb-Edu (educational) and Cosmopedia stories, targeting 63% / 10% / 27%.
- AdamW, peak learning rate 3e-4, weight decay 0.1, cosine decay, 3% warmup, gradient clip 1.0.
- bf16 mixed precision, about 48k tokens/s. Final training loss about 3.0.
- Final training took about 23 hours on a single RTX 4070.

**Instruction tuning**

- Filtered [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) plus filtered
  [everyday-conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) dialogues, with
  code, math, science-heavy and "I'm an AI model" turns removed.
    -- About 80% No Robots and 20% everyday-conversations.
- Chat template: `<|user|> message <|end|> <|assistant|> reply <|end|>`. The loss is computed on the reply and its `<|end|>` only.

## Results

Pretraining loss and benchmarks for the base (pretrained) model.

| Model | Params | Tokens | HellaSwag | ARC-Easy  | OpenBookQA |
|---|---|---|---|---|---|
| **small-llm (this repo)** | 100M | 5B | 30.37 | **45.37** | **29.0** |
| EleutherAI/pythia-160m | 160M | 300B | **30.64** | 43.90 | 27.40 |
| gpt2 | 124M | 100B | 29.56 | 43.90 | 26.6 |

Random guessing is 25% on all four. Differences of a point or less are within noise on the smaller tasks.

## Limitations

This is a 100M-parameter model trained on about 5B tokens, so it is fluent but not knowledgeable:

- It hallucinates facts a lot:

      A whale is a type of fish that has been found throughout the ocean. They are known for their powerful stomachs
      and large numbers of jellyfish, which they eat when they feed. Some whales have even eaten seals before but never
      became extinct.
- It can repeat itself. A repetition penalty (about 1.1–1.2) helps.
- Open-ended creative requests (poems, stories) lose the thread after a few lines.
- It has seen little multi-turn data, so replies get shorter and less reliable in long conversations.

## Run it in the browser

The page in `docs/` runs the model locally with [onnxruntime-web](https://onnxruntime.ai/) on WebGPU. Nothing is sent to a server.

The model file (`model.onnx`, about 400 MB in fp32) is too large for GitHub, so it is hosted on Hugging Face and cached in the browser after the first load.

## Building and running the C++ code

**Requirements:** a C++20 compiler (MSVC on Windows), CMake with Ninja, Apache Arrow/Parquet and PCRE2 (via vcpkg). For the GPU path, CUDA 13.x
and an NVIDIA GPU; cuDNN 9.x is optional and enables the flash-attention path (`USE_CUDNN=ON`).

```
.vscode\msvc.bat configure Release      # or: build / rebuild, Debug / Release
build-Release\main
```

**Pipeline, in order:**

1. Train the tokenizer on a text sample (about 100 MB is enough).
2. Tokenize the parquet shards (documents end with an EOS token).
3. Mix the corpora into training shards.
4. Pretrain (resumable from checkpoints).
5. Tokenize SFT data and train (masked)

## Repository layout

<!-- TODO: adjust to the actual layout. -->

```
src/            main.cpp and the source files
include/        headers: model, tokenizer, checkpoint, inference
cuda/           CUDA kernels and cuBLAS/cuDNN wrappers
docs/           browser demo: index.html, script.js, style.css, tokenizer.json
```

## Data and licenses

Training data: FineWeb and FineWeb-Edu (ODC-By 1.0), Cosmopedia (Apache 2.0). Fine-tuning data: No Robots (CC BY-NC 4.0, non-commercial,
attribution required) and everyday-conversations (Apache 2.0). Check each dataset's page for the current terms before reusing the model or data.

Copyright (c) 2026 Oliver Hardi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
