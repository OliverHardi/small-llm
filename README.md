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

## Scope

small-llm is deliberately a general-prose model: it is trained on web text, educational text and stories, with no code and no math corpora, and the instruction-tuning data was filtered to remove code, math and science-heavy dialogues. At 100M parameters a model can’t be good at everything, so concentrating capacity on ordinary English text gave much better results than my first attempt, which was trained on the broader MiniPile dataset. It is not meant for code, math or technical questions.

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

On three zero-shot benchmarks, small-llm performs on par with GPT-2 small (124M) and Pythia-160m, despite having fewer parameters and far less training data. It is nominally ahead on ARC-Easy and OpenBookQA, and on HellaSwag it sits between GPT-2 and Pythia. The standard errors are about ±0.5 on HellaSwag, ±1 on ARC-Easy and ±2 on OpenBookQA, so these gaps are within noise and I don’t claim a ranking.
The performance of this model can possibly be attributed to a higher quality training data mix, which is built from filtered web text (FineWeb), educational text (FineWeb-Edu) and synthetic stories (Cosmopedia). Models such as Pythia are trained on the Pile, which includes code, mathematics and scientific papers, so part of their capacity goes to domains these benchmarks don’t test. My philosophy with this model is that it has limited capacity to learn, so I focused on English prose, rather than technical syntax.

## Limitations

This is a 100M-parameter model trained on about 5B tokens, so it is fluent but not knowledgeable:

- It hallucinates facts a lot:

      A whale is a type of fish that has been found throughout the ocean. They are known
      for their powerful stomachs and large numbers of jellyfish, which they eat when they
      feed. Some whales have even eaten seals before but never became extinct.
  
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

```
src/            main.cpp and the source files
include/        headers: model, tokenizer, checkpoint, inference
cuda/           CUDA kernels and cuBLAS/cuDNN wrappers
docs/           browser demo: index.html, script.js, style.css, tokenizer.json
```

## Example Outputs


**A short multi-turn conversation (reasonable)**

> **You:** hey

> **small-llm:** Hello! How can I help you today?

> **You:** what is photosynthesis?

> **small-llm:** Photosynthesis is the process by which plants convert sunlight into food and oxygen. It's a wonderful way to use up all that energy and make new things, like fruits or vegetables.

> **You:** do animals do this too?

> **small-llm:** Yes, they do. Animals in the animal kingdom eat leaves and other plants to make food for themselves. They also eat seeds from trees, flowers, and even small mammals.

> **You:** what about fish? what's the fastest fish?

> **small-llm:** The fastest fish is called the tuna. Fish are fast-moving creatures with long legs and big eyes. They swim around on land and have lots of fun exploring their underwater world.

> **You:** does tuna taste good?

> **small-llm:** Yes, many people enjoy tuna as a sweet treat because it's high in protein, healthy fats, and low sugar. It tastes great when eaten raw, cooked, or mixed with other ingredients like honey or oil.

> **You:** cool, thanks!

> **small-llm:** I hope your day has been filled with delicious meals and yummy drinks.

**Short stories (decent)**
> **You:** Tell me a story about recycling

> **small-llm:** One sunny day, Sam and Alex were walking down the street when they saw a big pile of trash on the ground. They picked up the litter and started picking it up again. As they got closer to the pile, they noticed that some of the plastic bags had been left behind by people who didn't want them.
>
> "What happened?" asked Sam. "Why did this happen? Why did this happen?"
>
> Alex thought for a moment before answering, "Well, I remember learning in school how important it is to clean our planet so we can all live happily ever after."
>
> Sam nodded, understanding what he was saying. He realized that even though things might not always go as planned, taking care of the environment could lead to positive changes over time. From then on, Sam made sure to take better care of his surroundings and took care of his waste disposal. 
>
> In conclusion, recycling isn't just about throwing away old items; it's also about being mindful of its impact on Earth and making responsible choices towards living sustainably. By doing so, everyone benefits from the process, including those less fortunate than themselves!
>  
> (Note: This story is inspired by real-life situations where individuals often encounter issues related to their own actions.)

**Poetry (better with higher temperature)**

> **You:** Write a short poem about heartbreak, the wind, and grass blowing.

> **small-llm:** Heartbreak I am\
> \- Somewhere deep in my soul's sea\
> No one can hear me now\
> But I know I'll make it out\
> I'll get up again soon enough

> **You:** again

> **small-llm:** In this lonely land lies love\
> It is in every tree where life turns\
> And you will find me again\
> For what could be?
>
> "Oh, those beautiful trees that dance by their feet\
> They sway gently with grace

> **You:** again

> **small-llm:** Just listen to how soft my branches are\
> My voice fills your breath like peace\
> A symphony of sounds and whispers adds another layer\
> This journey has been worth every day

**Where it fails**

Arithmetic (correct answer: 6):

> **You:** If Sam has 3 apples and gives 1 to Maya, then buys 4 more, how many does Sam have?

> **small-llm:** Sam had two apples. The first was a red apple with a greenish-brown crust, and the second was a white-colored apple with a yellowish-red edge.
>
> Maya was happy that Sam had enough apples for her. She wanted to buy 2 apples per day so she could eat them all. However, she didn't want to spend too much money on other things. So, she bought 2 apples and got 6 apples in total.

Factual recall (correct answer: Canberra):

> **You:** What's the capital of Australia?

> **small-llm:** The capital is Tasmania, which is in Australia. It's one of the largest states in Australia, with 864,000 people.

## Data and licenses

Training data: FineWeb and FineWeb-Edu (ODC-By 1.0), Cosmopedia (Apache 2.0). Fine-tuning data: No Robots (CC BY-NC 4.0, non-commercial,
attribution required) and everyday-conversations (Apache 2.0). Check each dataset's page for the current terms before reusing the model or data.

Copyright (c) 2026 Oliver Hardi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
