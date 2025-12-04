# Value Function Iteration (VFI)

An educational implementation of Value Function Iteration demonstrating JAX fundamentals: `jit`, `vmap`, `pmap`, and `lax.scan`.

## Purpose

This module is designed to teach JAX patterns through a familiar economic problem. It progresses through increasingly sophisticated parallelization strategies:

1. **Manual vectorization** — Array broadcasting
2. **Automatic vectorization** — `jax.vmap`
3. **Multi-device parallelization** — `jax.pmap` (TPU/multi-GPU)
4. **Efficient loops** — `jax.lax.scan`

## Quick Start

### Google Colab

1. Create a new Colab notebook
2. Copy the contents of `vfi.py` into a cell
3. Run — the script auto-detects the environment

### Local

```bash
source .venv/bin/activate
python VFI/vfi.py
```

## The Economic Model

Classic income fluctuation problem:

$$\max \mathbb{E} \sum_{t=0}^{\infty} \beta^t u(c_t)$$

Subject to:
$$c_t + a_{t+1} \leq R \cdot a_t + y_t, \quad c_t \geq 0, \quad a_t \geq 0$$

Where:
- $c_t$ = consumption
- $a_t$ = assets
- $y_t$ = stochastic income (Markov chain)
- $R$ = gross interest rate
- $\beta$ = discount factor
- $u(c) = c^{1-\gamma}/(1-\gamma)$ (CRRA utility)

**Bellman equation:**
$$v(a,y) = \max_{0 \leq a' \leq Ra+y} \left\{ u(Ra+y-a') + \beta \sum_{y'} v(a',y') P(y,y') \right\}$$

## JAX Concepts Demonstrated

### 1. `jax.jit` — Just-In-Time Compilation

Compiles Python functions to optimized XLA code:

```python
T_manual_jit = jax.jit(T_manual)
```

### 2. `jax.vmap` — Automatic Vectorization

Transforms functions over single elements into batched operations:

```python
# Function for single (a, y, ap) combination
def action_value(a_idx, y_idx, ap_idx): ...

# Vectorize over ap
batched = jax.vmap(action_value, in_axes=(None, None, 0))
```

### 3. `jax.pmap` — Multi-Device Parallelization

Distributes computation across TPU cores or GPUs:

```python
# Partition data across devices
a_partitions = indices["a"].reshape(n_devices, -1)

# pmap distributes first axis across devices
T_pmapped = jax.pmap(T_partition, in_axes=(0, None))
```

### 4. `jax.lax.scan` — Efficient Loops

Compiles entire loop into XLA (no Python overhead per iteration):

```python
def bellman_step(v, _):
    new_v = ...
    error = jnp.max(jnp.abs(new_v - v))
    return new_v, error

# Runs n_iterations as single compiled operation
final_v, errors = jax.lax.scan(bellman_step, v_init, None, length=n_iterations)
```

## Performance Results

Typical results on different hardware (times in ms per Bellman update):

| State Space | CPU | GPU | TPU (1 core) | TPU (8 cores) |
|-------------|-----|-----|--------------|---------------|
| 131K | ~500 | ~4 | ~3 | ~3 |
| 2M | - | ~200 | ~100 | ~56 |
| 8M | - | ~1,600 | ~800 | ~210 |
| 33M | - | ~12,000 | ~6,200 | ~1,300 |

TPU multi-core (`pmap`) provides ~5-10x speedup over single-device for large problems.

## Configuration

Edit the `config` dictionary in `vfi.py`:

```python
config = {
    # Grid scale (1=small, 32=large)
    "scale": 4,
    
    # Model parameters
    "R": 1.1,        # Interest rate
    "beta": 0.99,    # Discount factor
    "gamma": 2.5,    # Risk aversion
    
    # VFI settings
    "max_iter": 500,
    "tol": 1e-6,
    
    # Benchmarks
    "run_benchmarks": True,
    "benchmark_scales": [1, 2, 4, 8],
}
```

## Files

| File | Description |
|------|-------------|
| `vfi.py` | Main script (local + Colab) |
| `Value_Function_Iteration_with_TPUs.ipynb` | Original notebook (archived) |

## References

- [QuantEcon JAX/GPU Tutorial](https://notes.quantecon.org/submission/622ed4daf57192000f918c61)
- [JAX Documentation](https://jax.readthedocs.io/)
