# Cache Simulator Project

This project implements a cache simulator for analyzing and optimizing cache performance through the simulation of various replacement policies, block sizes, and cache configurations. It's designed for a computer architecture course project focused on cache optimization.

## Overview

The project consists of two main components:

1. **Single-Level Cache Simulator**: Implements and analyzes direct-mapped, fully associative, and set-associative caches with different replacement policies.

2. **Multi-Level Cache Simulator**: Extends the analysis to multi-level cache hierarchies (L1, L2, L3) with different configurations and inclusion policies.

## Dependencies

- Python 3.6+
- NumPy
- Matplotlib

Install dependencies with:
```bash
pip install numpy matplotlib
```

## Phase 1: Single-Level Cache Simulator

### Features

- Support for direct-mapped, fully associative, and set-associative caches
- Implementation of various replacement policies: LRU, FIFO, LFU, Random
- Analysis of hit/miss rates for different cache configurations
- Visualization of memory access patterns
- Comparison tools for evaluating different configurations

### Usage

The single-level cache simulator (`cache_simulator.py`) can be run in different modes:

#### Basic Usage

```bash
python cache_simulator.py --cache-size 8192 --block-size 64 --associativity 2 --policy LRU
```

#### Available Arguments

- `--cache-size`: Cache size in bytes (default: 8192)
- `--block-size`: Block size in bytes (default: 64)
- `--associativity`: Associativity factor (1 for direct-mapped, -1 for fully associative, >1 for set-associative) (default: 2)
- `--policy`: Replacement policy (LRU, FIFO, LFU, RANDOM) (default: LRU)
- `--num-accesses`: Number of memory accesses to simulate (default: 10000)
- `--address-range`: Range of addresses to access (default: 32768)
- `--pattern`: Memory access pattern (sequential, random, loop, locality) (default: random)
- `--mode`: Simulation mode (single, compare-policies, compare-associativities, compare-block-sizes, comprehensive) (default: single)

#### Example: Comparing Replacement Policies

```bash
python cache_simulator.py --mode compare-policies --pattern locality
```

This will simulate memory accesses with temporal and spatial locality, comparing the performance of different replacement policies.

#### Example: Running Comprehensive Analysis

```bash
python cache_simulator.py --mode comprehensive
```

This will run a comprehensive analysis with various cache configurations and access patterns, identifying the optimal configuration for each access pattern.

## Phase 2: Multi-Level Cache Simulator

### Features

- Simulation of L1, L2, and L3 cache levels with configurable parameters
- Support for inclusive and exclusive cache hierarchies
- Analysis of hierarchical cache performance
- Visualization of hit rates and access distribution across cache levels
- Comparison of different cache hierarchy designs

### Usage

The multi-level cache simulator (`multilevel_cache_simulator.py`) can be run in different modes:

#### Basic Usage

```bash
python multilevel_cache_simulator.py --l1-size 32768 --l2-size 262144 --l3-size 4194304 --block-size 64 --associativity 4 --policy LRU
```

#### Available Arguments

- `--l1-size`: L1 cache size in bytes (default: 32768)
- `--l2-size`: L2 cache size in bytes (default: 262144)
- `--l3-size`: L3 cache size in bytes (default: 4194304)
- `--block-size`: Block size in bytes (default: 64)
- `--associativity`: Cache associativity (default: 4)
- `--policy`: Replacement policy (LRU, FIFO, LFU, RANDOM) (default: LRU)
- `--inclusive`: Use inclusive cache hierarchy (default: False)
- `--pattern`: Memory access pattern (sequential, random, loop, locality, mixed) (default: mixed)
- `--num-accesses`: Number of memory accesses to simulate (default: 100000)
- `--address-range`: Range of addresses to access (default: 1048576)
- `--mode`: Simulation mode (single, compare-hierarchies, compare-policies, comprehensive) (default: single)

#### Example: Comparing Cache Hierarchies

```bash
python multilevel_cache_simulator.py --mode compare-hierarchies --pattern locality
```

This will compare different cache hierarchy configurations with memory accesses exhibiting locality.

#### Example: Comparing Inclusion Policies

```bash
python multilevel_cache_simulator.py --mode compare-policies --pattern mixed
```

This will compare inclusive vs exclusive cache hierarchies with a mixed access pattern.

## Running Comprehensive Analysis

For both simulators, the `--mode comprehensive` option runs a comprehensive analysis with various configurations and access patterns:

```bash
python cache_simulator.py --mode comprehensive
```

or

```bash
python multilevel_cache_simulator.py --mode comprehensive
```

## Generated Visualizations

Both simulators generate visualizations to help analyze cache performance:

- Hit/miss rates for different configurations
- Memory access patterns with hits and misses
- Access distribution across cache levels (for multi-level simulator)
- Performance comparisons between different policies and hierarchies

## Tips for Analysis

1. **For sequential access patterns**: Look for caches with higher associativity or clever block sizes.
2. **For random access patterns**: Focus on larger cache sizes rather than sophisticated replacement policies.
3. **For locality patterns**: LRU typically performs best due to temporal locality.
4. **For multi-level hierarchies**: Consider the tradeoffs between inclusive vs exclusive policies.
5. **For optimal block size**: Balance spatial locality benefits with increased miss penalty from larger blocks.

## Example Analysis Workflow

1. Run the comprehensive analysis for both simulators.
2. Identify the access patterns most similar to your target workload.
3. Compare the performance of different cache configurations for that pattern.