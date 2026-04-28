# SSD (Structured State Space Duality) CuTe DSL Kernels

CuTe DSL implementation of the Mamba2 SSD chunk-scan prefill kernel for
TensorRT Edge-LLM. Prebuilt artifacts (static library + headers) are checked
into the repo; CMake links them directly — no Python or GPU needed at build time.

Adapted from:
- [Mamba SSM Triton kernels](https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops/triton/) (Apache-2.0) — SM80+ chunk scan pipeline
- [FlashInfer Mamba2 Blackwell kernel](https://github.com/flashinfer-ai/flashinfer/pull/2709) (Apache-2.0) — SM100+ persistent kernel

Local modifications:
- **CuTe DSL port** — rewrote Triton kernels as CuTe DSL with SM80 warp MMA + cp.async, removing PyTorch/Triton dependency
- **Multi-variant AOT compilation** — 4 SM80 variants (D×N ∈ {64,128}²) + 2 Blackwell variants, each a compile-time specialization
- **Runtime parameter flexibility** — batch, nheads, ngroups, seq_len are runtime arguments
- **Blackwell N=64 support** — fixed TMA partition shape mismatch in FlashInfer kernel to support dstate=64
- **C++ plugin integration** — CuteDslSSDRunner with multi-module dispatch, AOT static library pattern matching FMHA/GDN
- **Dependency removal** — removed PyTorch; uses CuPy/NumPy for standalone testing

## Kernel Variants

`DIM` (headdim) and `DSTATE` (state dim) are **compile-time** constants that
define each AOT variant. All other parameters (batch, heads, groups, seq_len)
are **runtime** arguments.

### Non-Blackwell (SM80+)

| Variant | DIM | DSTATE | Notes |
|---|---|---|---|
| `ssd_prefill_d128_n128` | 128 | 128 | Nemotron-Nano-9B-v2 |
| `ssd_prefill_d64_n128` | 64 | 128 | Nemotron-3-Nano-4B, 30B-A3B |
| `ssd_prefill_d128_n64` | 128 | 64 | — |
| `ssd_prefill_d64_n64` | 64 | 64 | — |

### Blackwell (SM100-110 — TMA + TMEM + WGMMA)

SM120+ (GB10/GB20) lacks TMEM/wgmma and uses the non-Blackwell fallback.

| Variant | DIM | DSTATE | Notes |
|---|---|---|---|
| `ssd_prefill_blackwell_d64_n128` | 64 | 128 | Nemotron-3-Nano-4B, 30B-A3B |
| `ssd_prefill_blackwell_d64_n64` | 64 | 64 | — |

Blackwell native kernels are limited to DIM=64 due to SM100 TMEM capacity
(512 columns). DIM=128 models use the non-Blackwell fallback on Blackwell GPUs.

### Compile-time vs Runtime Parameters

| Parameter | Compile / Runtime | Affects variant? |
|---|---|---|
| `DIM` (headdim) | Compile-time | Yes |
| `DSTATE` (state dim) | Compile-time | Yes |
| `CHUNK_SIZE` | Compile-time (fixed 128) | No (same for all) |
| `batch` (n) | Runtime | No |
| `nheads` | Runtime | No |
| `ngroups` | Runtime | No |
| `seq_len` | Runtime | No |

## Chunk Scan Pipeline

5-kernel tiled matmul pipeline (CHUNK_SIZE=128):

| Step | Kernel | Operation |
|------|--------|-----------|
| 1 | `cumsum` | Prefix sum of `A*dt` → decay factors per chunk |
| 2 | `chunk_state` | `B^T @ (decay*dt*X)` → per-chunk state contribution |
| 3 | `state_passing` | Sequential scan over chunks (`nchunks` only) |
| 4 | `bmm` | `C @ B^T` → CB matrix |
| 5 | `chunk_scan` | `(CB*mask)@X + C@state` → output |

Features: FP16 I/O, FP32 accumulation, D skip connection (`has_D`),
z-gating/SiLU (`has_z`), AOT export for C++ plugin integration.

## Building Prebuilt Artifacts

Run on a machine with the target GPU and CUDA 12.x/13.x:

```bash
pip install nvidia-cutlass-dsl==4.4.1
pip install cupy-cuda12x==12.3.0   # or cupy-cuda13x==13.6.0 for CUDA 13

cd tensorrt-edge-llm

# Non-Blackwell variants (all 4 D×N combos):
python kernelSrcs/build_cutedsl.py --kernels ssd --gpu_arch sm_87 [--clean]

# Blackwell variants (D=64 only):
python kernelSrcs/build_cutedsl.py --kernels ssd --gpu_arch sm_100 [--clean]
```

Output under `cpp/kernels/cuteDSLArtifact/{arch}/`:

```
libcutedsl_{arch}.a    — all variant .o files + libcuda_dialect_runtime_static.a
metadata.json          — build provenance (groups, variants, CUDA ver, DSL ver)
include/
    cutedsl_all.h      — umbrella header
    ssd_prefill_d128_n128.h
    ssd_prefill_d64_n128.h
    ...
```

## CMake

```bash
cmake -DENABLE_CUTE_DSL=ssd ...
```

To enable SSD with other kernel groups:

```bash
cmake -DENABLE_CUTE_DSL=ALL ...
```

## Standalone Test / Export

```bash
cd kernelSrcs/ssd_cutedsl

# Non-Blackwell accuracy check (default D=128, N=128)
python3 ssd_prefill.py --n 1 --nheads 8 --dim 128 --dstate 128 --seq_len 1024

# Non-Blackwell with D=64, N=128
python3 ssd_prefill.py --n 1 --nheads 8 --dim 64 --dstate 128 --seq_len 1024

# Blackwell accuracy check
python3 ssd_prefill_blackwell.py --n 1 --nheads 8 --dim 64 --dstate 128 --seq_len 1024

# AOT export (single variant)
python3 ssd_prefill.py --export_only --dim 64 --dstate 128 \
    --output_dir ./out --file_name ssd_prefill_d64_n128 --function_prefix ssd_prefill_d64_n128
```

## C++ Integration

`CuteDslSSDRunner` (`cpp/kernels/mamba/cuteDslSSDRunner.{h,cpp}`): call
`loadKernelModules()` once, then `runPrefill(SSDParams, stream)`. Dispatches
to the correct D×N variant at runtime. On SM100+, D=64 uses the Blackwell
native kernel; all other configs fall back to non-Blackwell.

Plugin (`cpp/plugins/mamba/mambaPlugin.cpp`): integrates via the SSD runner.

## Tensor Shapes

| Tensor | Shape | Dtype |
|---|---|---|
| `x` | `(N, seq_len, nheads, dim)` | FP16 |
| `dt` | `(N, seq_len, nheads)` | FP16 |
| `A` | `(nheads,)` | FP32 |
| `dt_bias` | `(nheads,)` | FP32 |
| `B` | `(N, seq_len, ngroups, dstate)` | FP16 |
| `C` | `(N, seq_len, ngroups, dstate)` | FP16 |
| `D` | `(nheads,)` | FP32 |
| `z` | `(N, seq_len, nheads, dim)` | FP16 |
| `output` | `(N, seq_len, nheads, dim)` | FP16 |
| `state` | `(N, nheads, dim, dstate)` | FP32 |

## Files

| File | Description |
|------|-------------|
| `ssd_prefill.py` | Non-Blackwell 5-kernel chunk scan pipeline + AOT export |
| `ssd_prefill_blackwell.py` | Blackwell native kernel (TMA/TMEM/WGMMA) |
