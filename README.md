# GgmlSharp

A from-scratch C# reimplementation of the [GGML](https://github.com/ggerganov/ggml) CPU
compute runtime (`ggml-cpu`), targeting x86-64 with AVX2/FMA + scalar fallback. It mirrors the
enums, structs, and op semantics of `ggml.h` / `ggml-cpu.c` 1:1, but replaces the linear C arena
with managed objects while keeping raw `byte*` data buffers for the compute kernels.

The kernel style follows `HyMT2Sharp.Kernels`. The project builds on **net7.0** (chosen to match
the locally available SDK; can be raised to net8+/net10 later).

---

## Project layout

```
GgmlSharp.sln
src/GgmlSharp/               # the library
  Graph/
    GgmlGraph.cs             # graph builder + executor (public high-level API)
  Kernel/                    # low-level tensor/op runtime (public API + internal kernels)
    GgmlContext.cs           # arena allocator + tensor factory
    GgmlTensor.cs            # tensor descriptor + GgmlConst
    GgmlType.cs              # enum ggml_type (42 values)
    GgmlOp.cs                # enum ggml_op (96 values)
    GgmlUnaryOp.cs           # enum ggml_unary_op (22 values)
    GgmlTypeTraits.cs        # per-type traits + to/from-float row codecs
    GgmlCompute.cs           # top-level forward dispatch
    ComputeParams.cs         # per-thread compute params + thread row range
    GgmlMath.cs              # fp16/bf16 conversions, erf, etc.
    GgmlDequant.cs           # dequantize (to_float) kernels for standard block types
    GgmlGluOp.cs             # enum ggml_glu_op
    Ops.*.cs                 # per-op kernels (Unary/Binary/MatMul/Norm/Reduce/Remap/Rope/Simple/Softmax/Upscale)
tests/GgmlSharp.Tests/       # xUnit tests (Binary/Graph/MatMul/NormReduce/Remap/Unary)
```

## Building & running tests

```sh
dotnet build GgmlSharp.sln
dotnet test  tests/GgmlSharp.Tests/GgmlSharp.Tests.csproj
```

## Quick example

```csharp
using GgmlSharp.Graph;
using GgmlSharp.Kernel;

using var g = new GgmlGraph();

// leaf tensors: F32 with shape [rows, cols] (innermost = ne[0])
var a = g.NewTensor(GgmlType.F32, 4, 4);
var b = g.NewTensor(GgmlType.F32, 4, 4);

// build ops on the graph
var c = g.Add(a, b);          // elementwise add
var s = g.RmsNorm(c, 1e-5f);  // RMS norm over each row
var m = g.MulMat(a, b);       // a^T · b  (a:[K,N], b:[K,M] → [N,M])

// run the whole graph (nth = number of threads)
g.Execute(nth: 1);
```

Graph execution runs the full node list once per thread; each kernel threads internally via
`ComputeParams.Ith`/`Nth`, mirroring GGML's per-thread graph compute.

---

## Public API

### `GgmlSharp.Graph` — `GgmlGraph`

The high-level, graph-oriented entry point. Wraps a `GgmlContext` and an ordered node list;
op builders allocate a tensor node, wire `src`/`op_params`, and append it to the node list.
`Execute` runs the whole graph.

| Member | Description |
|---|---|
| `GgmlGraph(long reserveBytes = 1<<20)` | Create a graph with a fresh arena of the given size. |
| `Ctx` (`GgmlContext`) | The underlying context. |
| `Nodes` (`IReadOnlyList<GgmlTensor>`) | The ordered list of graph nodes. |
| `NewTensor(GgmlType, params long[] ne)` | Allocate a leaf tensor of shape `ne`. |
| `NewTensor(GgmlType, long[] ne, params GgmlTensor[] srcs)` | Allocate a tensor with sources. |
| `View(GgmlTensor a, long[] ne)` | Alias a source tensor's base data (structural; no compute). |
| `Add` / `Sub` | Elementwise add / subtract (`F32`). |
| `Mul` / `Div` | Elementwise multiply / divide with broadcast shape. |
| `MulMat(GgmlTensor a, GgmlTensor b)` | `dst = aᵀ·b`; `a:[K,N]`, `b:[K,M]` → `[N,M]` (batches broadcast). |
| `RmsNorm(GgmlTensor a, float eps)` | `x / sqrt(mean(x²) + eps)` per row. |
| `Norm(GgmlTensor a, float eps)` | `(x − mean) / sqrt(var + eps)` per row. |
| `SoftMax(GgmlTensor a, float scale = 1f)` | Row softmax with optional scale. |
| `Silu` / `Relu` / `Gelu` / `Tanh` / `Exp` | Unary activations. |
| `Scale(GgmlTensor a, float s)` | `dst = a·s`. |
| `Rope(a, positions, nDims, mode, freqBase=10000f)` | RoPE (NORMAL/NEOX) with YaRN yarn scaling. |
| `Repmat(GgmlTensor a, long[] dstShape)` | Tile `a` into a larger shape (`REPEAT`). |
| `Execute(int nth = 1)` | Run the whole graph with `nth` threads. |
| `Dispose()` | Free the underlying arena. |

### `GgmlSharp.Kernel` — contexts, tensors, compute

| Type | Description |
|---|---|
| `GgmlContext` | Arena allocator mirroring `ggml_context`: one growable aligned buffer, tensors carved off the tail. `NewTensor`/`NewOp`/`NewUnaryOp`/`NewBinaryOp` factories; `AllocBytes`; `Dispose` frees the arena. |
| `GgmlTensor` | Managed tensor descriptor mirroring `ggml_tensor`: `Type`, `Ne[]` (element counts), `Nb[]` (byte strides, column-major), `Data` (`byte*`), `Op`, `OpParams[]`, `Src[]`, plus shape queries (`NElements`, `NRows`, `RowSize`, `SetShape`, `ComputeStrides`) and `Get/SetOpParams{Int32,F32}`. |
| `GgmlConst` | GGML constants: `MaxDims=4`, `MaxSrc=10`, `MaxOpParams=64`, `MaxName=64`, `MemAlign=16`. |
| `GgmlCompute` | `Forward(in ComputeParams, GgmlTensor dst)` — top-level dispatch on `dst.Op` to the per-op kernel. |
| `ComputeParams` | `Ith`/`Nth` + `WorkData`; `ThreadRowRange(long nrows)` returns the `[ir0, ir1)` row range for this thread. |
| `GgmlTypeTraits` | Per-type block size, type size, quantization flag, and to/from-float row functions. `Get(GgmlType)` and `Name(GgmlType)`. |
| `GgmlMath` | fp16/bf16 ↔ fp32 conversions, `erf`, and math constants. |
| `GgmlDequant` | Dequantize (to_float) row kernels: `Q1_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q6_K`. |

#### Enums & flags (mirror GGML exactly)

| Type | Values |
|---|---|
| `GgmlType` | `F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K…Q8_K, IQ2_XXS…IQ1_M, I8, I16, I32, I64, F64, BF16, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0` (42 values; deprecated/removed slots kept for 1:1 alignment). |
| `GgmlOp` | 96 values from `ggml.h` (`NONE…GLU`), exact order. |
| `GgmlUnaryOp` | 22 values (`ABS…TRUNC`), exact order. |
| `GgmlGluOp` | `REGLU, GEGLU, SWIGLU, SWIGLU_OAI, GEGLU_ERF, GEGLU_QUICK`. |
| `GgmlRopeType` | `NORMAL=0, NEOX=1, VISION=2, MROPE=4, IMROPE=24` (bit flags). |
| `GgmlTriType` | `LOWER, LOWER_DIAG, UPPER, UPPER_DIAG`. |
| `GgmlSortOrder` | `ASC, DESC`. |
| `GgmlScaleMode` | `NEAREST=0, BILINEAR=1, BICUBIC=2`. |
| `GgmlScaleFlag` | `AlignCorners = 1<<16`, `Antialias = 1<<17`. |

#### Row codecs (delegates)

| Type | Description |
|---|---|
| `GgmlToFloat` | `(byte* x, float* y, long k)` — dequantize a row. |
| `GgmlFromFloat` | `(float* x, byte* y, long k)` — quantize a row. |

---

## Implemented ops

`GgmlCompute.Forward` dispatches to the following kernels (grouped by `Ops.*` source file):

| Category | Ops |
|---|---|
| **Unary** | `UNARY` (ABS, SGN, NEG, STEP, TANH, ELU, RELU, SIGMOID, GELU, GELU_QUICK, SILU, HARDSWISH, HARDSIGMOID, EXP, EXPM1, SOFTPLUS, GELU_ERF, XIELU, FLOOR, CEIL, ROUND, TRUNC) |
| **Binary** | `ADD`, `SUB`, `MUL`, `DIV` |
| **Reduce** | `SUM`, `SUM_ROWS`, `MEAN`, `ARGMAX`, `COUNT_EQUAL`, `CUMSUM` |
| **Norm** | `NORM`, `RMS_NORM`, `L2_NORM`, `GROUP_NORM` |
| **Softmax** | `SOFT_MAX` |
| **Simple** | `LEAKY_RELU`, `SILU_BACK`, `SCALE` |
| **Remap** | `DUP`/`CPY`/`CONT`, `REPEAT`, `REPEAT_BACK`, `SET`, `PAD`, `PAD_REFLECT_1D`, `ROLL`, `ARANGE`, `FILL`, `TRI`, `ARGSORT`, `TOP_K`, `TIMESTEP_EMBEDDING`, `RMS_NORM_BACK` |
| **Upscale** | `UPSCALE` |
| **MatMul** | `MUL_MAT`, `OUT_PROD` |
| **Rope** | `ROPE` (NORMAL/NEOX with YaRN) |

Any `GgmlOp` without a kernel throws `NotImplementedException`.

---

## Status / roadmap

- **M0** — enums, tensor/context, unary ops, binary ops, reduce, norm, softmax, simple, remap,
  upscale, rope. Wired in `GgmlCompute.Forward`.
- **M3** — `MUL_MAT`/`OUT_PROD` for `F32`, dequantize row kernels for standard block types
  (`Q1_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q6_K`).
- **Pending** — quantize (`from_float`) kernels, full `Q2_K`/`Q3_K`/`Q5_K`/`Q8_K`/`IQ*`/`MXFP4`/
  `NVFP4`/`TQ*` layouts, MROPE/IMROPE/VISION rope, `GLU` and the remaining `ggml_op` cases,
  higher target frameworks (net8+/net10).
