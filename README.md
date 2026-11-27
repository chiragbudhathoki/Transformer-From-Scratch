# Transformer from scratch

![Project image](https://i.pinimg.com/736x/73/5e/a8/735ea80c20eef49c97b5d9852e0f2549.jpg)

## Overview

This project is a from-scratch implementation of a compact
language model. It follows the GPT family at a broad level: a decoder-only
Transformer learns to predict the next token, using only the tokens that came
before it. It is not a reproduction of GPT-3 or a claim to match the scale of a
production LLM.

This project was completed for Stanford's CS336: Language Modeling from Scratch.

## Model architecture

The model uses causal multi-head self-attention, with RoPE (Rotary Positional
Embeddings) to represent token positions. Its Transformer blocks use
pre-normalization with RMSNorm, residual connections, and SwiGLU feed-forward
layers. These choices make it a modern GPT-style language model, rather than an
exact copy of GPT-3's architecture.

## Training and generation

The project includes BPE tokenization and data-preparation utilities, a
training loop with validation and checkpoint saving, and a separate inference
script. For inference, the script loads a trained checkpoint and tokenizer,
encodes a text prompt, then generates tokens one at a time. Temperature and
top-p (nucleus) sampling control randomness; setting temperature to zero uses
greedy selection. Generation stops at the token limit or the end-of-text token
and keeps inputs within the model's context window.

## Setup

For the assignment specification and course-specific instructions on running
the project, see the [CS336 Assignment 1 handout](./cs336_assignment1_basics.pdf).

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

