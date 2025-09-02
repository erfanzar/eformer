# eformer (EasyDel Former)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Compatible-brightgreen)](https://github.com/google/jax)
[![PyPI version](https://badge.fury.io/py/eformer.svg)](https://badge.fury.io/py/eformer)

**eformer** (EasyDel Former) is a utility library designed to simplify and enhance the development of machine learning models using JAX. It provides a comprehensive collection of tools for distributed computing, custom data structures, numerical optimization, and high-performance operations. Eformer aims to make it easier to build, scale, and optimize models efficiently while leveraging JAX's capabilities for high-performance computing.

## Project Structure Overview

The library is organized into several core modules:

- **`aparser`**: Advanced argument parsing utilities with dataclass integration
- **`callib`**: Custom function calling and Triton kernel integration
- **`common_types`**: Shared type definitions and sharding constants
- **`escale`**: Distributed sharding and parallelism utilities
- **`executor`**: Execution management and hardware-specific optimizations
- **`jaximus`**: Custom PyTree implementations and structured array utilities
- **`mpric`**: Mixed precision training and dynamic scaling infrastructure
- **`ops`**: Optimized operations including Flash Attention and quantization
- **`optimizers`**: Flexible optimizer configuration and factory patterns
- **`pytree`**: Enhanced tree manipulation and transformation utilities

## Key Features

### 1. Mixed Precision Training (`mpric`)

Advanced mixed precision utilities supporting float8, float16, and bfloat16 with dynamic loss scaling, enabling faster training and reduced memory footprint.

### 2. Distributed Sharding (`escale`)

Tools for efficient sharding and distributed computation in JAX, allowing you to scale your models across multiple devices with various sharding strategies:

- Data Parallelism (`DP`)
- Fully Sharded Data Parallel (`FSDP`)
- Tensor Parallelism (`TP`)
- Expert Parallelism (`EP`)
- Sequence Parallelism (`SP`)

### 3. Custom PyTrees (`jaximus`)

Enhanced utilities for creating custom PyTrees and `ArrayValue` objects, updated from Equinox, providing flexible data structures for your models.

### 4. Triton Integration (`callib`)

Custom function calling utilities with direct integration of Triton kernels in JAX, allowing you to optimize performance-critical operations.

### 5. Optimizer Factory

A flexible factory for creating and configuring optimizers like AdamW, Adafactor, Lion, and RMSProp, making it easy to experiment with different optimization strategies.

### 6. Optimized Operations (`ops`)

- Flash Attention 2 implementation for GPUs/TPUs (via Triton and Pallas) for faster attention computations
- 8-bit and NF4 quantization for efficient model deployment
- Additional optimized operations under active development

## API Documentation

For detailed API references and usage examples, see:

- [Argument Parser (`aparser`)](docs/api_docs/aparser.rst)
- [Triton Integration (`callib`)](docs/api_docs/callib.rst)
- [Sharding Utilities (`escale`)](docs/api_docs/escale.rst)
- [Execution Management (`executor`)](docs/api_docs/executor.rst)
- [Mixed Precision Infrastructure (`mpric`)](docs/api_docs/mpric.rst)
- [Custom Operations (`ops`)](docs/api_docs/ops.rst)

## Installation

You can install `eformer` via pip:

```bash
pip install eformer
```

## Getting Started

### Mixed Precision Handler with mpric

```python
from eformer.mpric import PrecisionHandler

# Create a handler with float8 compute precision
handler = PrecisionHandler(
    policy="p=f32,c=f8_e4m3,o=f32",  # params in f32, compute in float8, output in f32
    use_dynamic_scale=True
)
```

### Custom PyTree Implementation

```python
import jax
from eformer.jaximus import ArrayValue, implicit
from eformer.ops.quantization.quantization_functions import dequantize_row_q8_0, quantize_row_q8_0

class Array8B(ArrayValue):
    scale: jax.Array
    weight: jax.Array
    
    def __init__(self, array: jax.Array):
        self.weight, self.scale = quantize_row_q8_0(array)
    
    def materialize(self):
        return dequantize_row_q8_0(self.weight, self.scale)

array = jax.random.normal(jax.random.key(0), (256, 64), "f2")
qarray = Array8B(array)
```

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
