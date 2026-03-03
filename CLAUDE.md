# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repo contains two components:
- **Megatron Core** (`megatron/core/`): A pip-installable library of GPU-optimized building blocks for distributed transformer training (tensor/pipeline/data/expert/context parallelism, optimizers, checkpointing, inference).
- **Megatron-LM** (`megatron/training/`, top-level `pretrain_*.py`): Reference training scripts built on top of Megatron Core.

## Build and Installation

```bash
# Full install with training dependencies and dev tools
pip install --no-build-isolation .[mlm,dev]

# Build C++ dataset helpers (needed for data loading)
cd megatron/core/datasets && make
```

## Linting and Formatting

```bash
# Format with black (line length 100, no string normalization)
black megatron/

# Sort imports
isort megatron/

# Run flake8
flake8 megatron/

# Run ruff (checks for unsafe eval via S506)
ruff check megatron/
```

Config lives in `pyproject.toml`: black line length is 100, `skip_string_normalization = true`, isort profile is "black".

## Testing

```bash
# Run all unit tests
pytest tests/unit_tests/

# Run a single test file
pytest tests/unit_tests/test_optimizer.py

# Run a single test
pytest tests/unit_tests/test_optimizer.py::TestDistributedOptimizer::test_state_dict

# Run with coverage
pytest --cov=megatron tests/unit_tests/
```

Tests use `pytest` with flags `--durations=15 -s -rA -x` by default (stop on first failure). Test markers: `internal`, `flaky`, `flaky_in_dev`.

Functional tests require golden values. To update them after CI runs:
```bash
python tests/test_utils/python_scripts/download_golden_values.py --source github --pipeline-id <run-id>
```

## Architecture

### Parallelism (`megatron/core/parallel_state.py`)

The central module managing all process groups. Supports composable parallelism strategies:
- **TP** (Tensor Parallel): splits tensors within a layer across GPUs
- **PP** (Pipeline Parallel): splits layers across GPUs; supports virtual pipeline (VPP/interleaved)
- **DP** (Data Parallel): replicates model, shards optimizer state (ZeRO-1 via `--use-distributed-optimizer`)
- **CP** (Context Parallel): splits sequences across GPUs for long-context training
- **EP** (Expert Parallel): splits MoE experts across GPUs

Call `parallel_state.initialize_model_parallel(...)` before using any distributed features.

### Configuration (`megatron/core/transformer/transformer_config.py`, `megatron/core/model_parallel_config.py`)

All model and parallelism settings are passed via `TransformerConfig` (subclass of `ModelParallelConfig`). These dataclasses carry every hyperparameter needed to build a model. The Megatron-LM training scripts populate them from CLI args via `core_transformer_config_from_args()` in `megatron/training/arguments.py`.

### Module Spec System (`megatron/core/transformer/spec_utils.py`)

Models are assembled via `ModuleSpec` — a dataclass specifying which class implements each submodule and optional params. This enables swapping between local implementations and TransformerEngine (TE) fused implementations without changing model code. Key specs in `megatron/core/models/gpt/gpt_layer_specs.py`:
- `get_gpt_layer_local_spec()` — pure PyTorch implementation
- `get_gpt_layer_with_te_and_normalization_spec()` — TransformerEngine fused ops

### Forward/Backward Scheduling (`megatron/core/pipeline_parallel/schedules.py`)

`get_forward_backward_func()` returns the appropriate schedule (1F1B, interleaved, no-pipelining) based on `parallel_state` configuration. Training scripts call this function rather than running forward/backward directly. Schedules live in `combined_1f1b.py` and `hybrid_cp_schedule.py`.

### Distributed Checkpointing (`megatron/core/dist_checkpointing/`)

Megatron uses its own distributed checkpoint format (`torch_dist`) that allows resharding between parallelism configurations. Models expose `sharded_state_dict()` which returns `ShardedTensor` objects describing how tensors are partitioned. Use `dist_checkpointing.save()` / `dist_checkpointing.load()`.

### MoE (`megatron/core/transformer/moe/`)

Mixture-of-Experts with pluggable token dispatchers (`alltoall`, `allgather`, `flex` with DeepEP/HybridEP backends), load balancing strategies, and grouped GEMM. See `megatron/core/transformer/moe/README.md` for detailed feature docs and argument references.

### Data Pipeline (`megatron/core/datasets/`)

Binary `.bin`/`.idx` format via `IndexedDataset`. Higher-level: `BlendedMegatronDatasetBuilder` combines datasets by weight. `GPTDataset`, `T5Dataset`, etc. extend `MegatronDataset`. The C++ helper (`helpers.cpp`) must be compiled for efficient data loading.

### Training Entry Points

- `pretrain_gpt.py` — GPT/LLaMA/MoE pre-training and SFT
- `pretrain_bert.py` — BERT pre-training
- `pretrain_t5.py` — T5 pre-training
- `pretrain_mamba.py` — Mamba SSM pre-training
- `pretrain_vlm.py` — Vision-Language model training
- `train_rl.py` — RLHF training

All use `megatron.training.pretrain()` as the main entry point, which handles initialization, the train loop, evaluation, and checkpointing. Training scripts are launched with `torchrun`.

### Megatron-LM vs. Megatron Core Usage

Custom training frameworks should import directly from `megatron.core`. The `megatron.training` package wraps core with argument parsing, logging, and a full training loop — useful for the reference scripts but not required.

## PR Submission Process

1. Add `Expert Review` label → auto-assigns reviewers based on changed files
2. After all approvals + CI passing, add `Final Review` label
3. Tag `@mcore-oncall` if no response after 2 business days
