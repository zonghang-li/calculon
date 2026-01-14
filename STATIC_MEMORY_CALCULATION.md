# Static Memory Calculation in Calculon

## Overview

This document explains how Calculon calculates **static memory** requirements for Large Language Model (LLM) training. Static memory consists of two main components:

1. **Model Weights** - The parameters of the neural network
2. **Optimizer State** - The state maintained by the optimizer (Adam)

## Model Weights (`_weight_space`)

Model weights represent the learnable parameters of the neural network. The total weight space is calculated as:

```
_weight_space = _block_weight_space * _blocks_per_proc + embedding_weights
```

Where:
- `_block_weight_space`: Sum of all parameter bytes for a single transformer block
- `_blocks_per_proc`: Number of transformer blocks assigned to this process/GPU
- `embedding_weights`: Additional weights for token embeddings and language model head

### Per-Block Weight Calculation

For each transformer block, the weight space is accumulated across all layers:

```python
self._block_weight_space += layer.get_weight()  # parameters
```

For a Linear layer (the most common type), the weight space is:

```
weight_space = n * k * bytes_per_element
```

Where:
- `n`: Input dimension (e.g., `c_in`)
- `k`: Output dimension (e.g., `c_out`)
- `bytes_per_element`: Size of the data type (2 bytes for FP16/BF16, 4 bytes for FP32)

### Embedding and LM Head Weights

Additional weights are added for:

1. **Token Embeddings** (owned by first stage, or last stage if tied):
   ```
   _weight_space += _embed_weight_elems_shard * bytes_per_element
   ```

2. **LM Head** (if embeddings are untied, owned by last stage):
   ```
   _weight_space += _lm_head_weight_elems_shard * bytes_per_element
   ```

These embeddings are already sharded across tensor parallelism.

## Optimizer State (`_optimizer_space`)

Calculon models the **Adam optimizer**, which maintains additional state for each parameter. The optimizer space is calculated as:

```
_optimizer_space = _block_optimizer_space * _blocks_per_proc + embedding_optimizer_state
```

### Per-Block Optimizer Calculation

For each transformer block:

```python
self._block_optimizer_space += layer.get_optimizer()  # optimizer state
```

The optimizer state includes:

1. **Two Moments (m, v)**: Adam maintains first and second moment estimates
2. **Master Copy of Weights** (if training in mixed precision): FP32 copy when training dtype < FP32

For a layer with `weight_space` parameters:

```python
def get_optimizer(self):
    # Adam moments: 2 moments * weight_space * 4 bytes (FP32)
    moments_size = self.optim_space * 4
    
    # Master copy: only if training in lower precision than FP32
    if self.bytes_per_element < 4:
        master_copy_size = self.weight_space * 4
    else:
        master_copy_size = 0
    
    return (master_copy_size + moments_size) / self.optim_sharding_num_proc
```

Where `optim_space = 2 * weight_space` (for the two Adam moments).

### Optimizer Sharding

If **optimizer sharding** (ZeRO-style optimization) is enabled:

```python
opt_shard = self.exe.data_par if self.exe.optimizer_sharding else 1
```

The optimizer state is divided by `opt_shard`, reducing per-GPU memory:

```
actual_optimizer_space = optimizer_space / opt_shard
```

### Embedding Optimizer State

For token embeddings and LM head (if applicable):

```python
# Adam moments (2 * FP32)
_optimizer_space += (2 * embed_weight_elems_shard * 4) / opt_shard

# Master copy (if mixed precision)
if datatype != 'float32':
    _optimizer_space += (embed_weight_elems_shard * 4) / opt_shard
```

## Memory Calculation Example

For a Linear layer with `c_in=1024`, `c_out=4096` in BF16 training:

### Weights:
```
weight_params = 1024 * 4096 = 4,194,304 parameters
weight_space = 4,194,304 * 2 bytes = 8,388,608 bytes (~8 MB)
```

### Optimizer State (no sharding):
```
optim_space = 2 * weight_params = 8,388,608 elements (for m, v)
moments_size = 8,388,608 * 4 bytes = 33,554,432 bytes (~32 MB)
master_copy_size = 4,194,304 * 4 bytes = 16,777,216 bytes (~16 MB)
total_optimizer = 33,554,432 + 16,777,216 = 50,331,648 bytes (~48 MB)
```

### Total Static Memory:
```
static_memory = weight_space + optimizer_space
              = 8 MB + 48 MB
              = 56 MB per layer
```

## Impact on Total Memory Requirements

The static memory calculation directly affects:

1. **GPU Memory Capacity Requirements** (`get_mem_tier1_cap_req()`):
   - Must accommodate weights + optimizer + activations + gradients
   
2. **Memory Tier Planning** (for offloading scenarios):
   - Tier 1 (GPU memory): Minimum needed for active blocks
   - Tier 2 (CPU/NVMe): Additional capacity for offloaded weights/optimizer

3. **Model Parallelism Decisions**:
   - Tensor Parallelism: Shards weights and optimizer across GPUs
   - Pipeline Parallelism: Distributes blocks across stages
   - Data Parallelism + Optimizer Sharding: Reduces optimizer memory per GPU

## Key Code Locations

- **Weight space calculation**: `calculon/llm/llm.py`, lines 2331-2434 (blocks) and 2434-2453 (embeddings)
- **Optimizer space calculation**: `calculon/llm/llm.py`, lines 2421-2422 (blocks) and 2443-2445, 2451-2453 (embeddings)
- **Per-layer optimizer**: `calculon/llm/layers.py`, lines 283-291
- **Memory capacity check**: `calculon/llm/llm.py`, lines 2455-2465

## Notes

- All optimizer state is stored in **FP32** for numerical stability, regardless of training dtype
- Optimizer sharding (ZeRO) can significantly reduce per-GPU memory by splitting optimizer state
- Weight offload adds double-buffering (2x minimum weight space) for overlap
- The `optim_space` parameter for each layer is initialized to `2 * weight_space` to account for Adam's two moments (m and v)
