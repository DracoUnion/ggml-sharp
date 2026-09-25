# GgmlSharp API Reference

Complete API reference for GgmlSharp — a from-scratch C# reimplementation of the GGML CPU compute runtime.

---

## Table of Contents

- [Namespaces](#namespaces)
- [GgmlSharp.Graph](#ggmlsharpgraph)
  - [GgmlGraph](#ggmlgraph)
- [GgmlSharp.Kernel](#ggmlsharpkernel)
  - [Core Types](#core-types)
  - [Tensor Operations](#tensor-operations)
  - [Compute & Dispatch](#compute--dispatch)
  - [Type System](#type-system)
  - [Quantization](#quantization)
  - [Math Utilities](#math-utilities)
  - [Enums](#enums)

---

## Namespaces

| Namespace | Description |
|-----------|-------------|
| `GgmlSharp.Graph` | High-level graph builder and executor API |
| `GgmlSharp.Kernel` | Low-level tensor, context, and compute kernel API |

---

## GgmlSharp.Graph

### GgmlGraph

High-level, graph-oriented entry point. Wraps a `GgmlContext` and an ordered node list. Op builders allocate a tensor node, wire `src`/`op_params`, and append it to the node list. `Execute` runs the whole graph.

```csharp
public sealed unsafe class GgmlGraph : IDisposable
```

#### Constructors

| Constructor | Description |
|-------------|-------------|
| `GgmlGraph(long reserveBytes = 1 << 20)` | Create a graph with a fresh arena of the given size (default 1 MiB). |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Ctx` | `GgmlContext` | The underlying context (arena allocator). |
| `Nodes` | `IReadOnlyList<GgmlTensor>` | The ordered list of graph nodes. |

#### Tensor Creation (Leaf Nodes)

| Method | Description |
|--------|-------------|
| `NewTensor(GgmlType type, params long[] ne)` | Allocate a leaf tensor of shape `ne` (row-major, innermost = `ne[0]`). |
| `NewTensor(GgmlType type, long[] ne, params GgmlTensor[] srcs)` | Allocate a tensor with source dependencies. |

#### Structural Operations (No Compute)

| Method | Description |
|--------|-------------|
| `View(GgmlTensor a, long[] ne)` | Alias a source tensor's base data (VIEW/RESHAPE are structural; no compute). |

#### Elementwise Binary Ops

| Method | Description |
|--------|-------------|
| `Add(GgmlTensor a, GgmlTensor b)` | Elementwise add (`F32`). |
| `Sub(GgmlTensor a, GgmlTensor b)` | Elementwise subtract (`F32`). |
| `Mul(GgmlTensor a, GgmlTensor b)` | Elementwise multiply with broadcast shape (`F32`). |
| `Div(GgmlTensor a, GgmlTensor b)` | Elementwise divide with broadcast shape (`F32`). |

#### Matrix Multiplication

| Method | Description |
|--------|-------------|
| `MulMat(GgmlTensor a, GgmlTensor b)` | `dst = aᵀ·b`; `a:[K,N]`, `b:[K,M]` → `[N,M]` (batches broadcast). |

#### Normalization

| Method | Description |
|--------|-------------|
| `RmsNorm(GgmlTensor a, float eps)` | RMS norm: `x / sqrt(mean(x²) + eps)` per row. |
| `Norm(GgmlTensor a, float eps)` | Layer norm: `(x - mean) / sqrt(var + eps)` per row. |

#### Activation & Pointwise

| Method | Description |
|--------|-------------|
| `SoftMax(GgmlTensor a, float scale = 1f)` | Row softmax with optional scale. |
| `Silu(GgmlTensor a)` | SiLU activation: `x * sigmoid(x)`. |
| `Relu(GgmlTensor a)` | ReLU activation: `max(0, x)`. |
| `Gelu(GgmlTensor a)` | GELU activation (erf-based). |
| `Tanh(GgmlTensor a)` | Hyperbolic tangent. |
| `Exp(GgmlTensor a)` | Exponential. |
| `Scale(GgmlTensor a, float s)` | Scale: `dst = a * s`. |

#### RoPE (Rotary Positional Embedding)

| Method | Description |
|--------|-------------|
| `Rope(GgmlTensor a, int[] positions, int nDims, GgmlRopeType mode, float freqBase = 10000f)` | RoPE with YaRN yarn scaling. Supports NORMAL, NEOX, MROPE, IMROPE, VISION modes. |

#### Remap

| Method | Description |
|--------|-------------|
| `Repmat(GgmlTensor a, long[] dstShape)` | Tile `a` into a larger shape (`REPEAT` op). |

#### Execution

| Method | Description |
|--------|-------------|
| `Execute(int nth = 1)` | Run the whole graph with `nth` threads. Each kernel threads internally via `ComputeParams.Ith`/`Nth`. |
| `Dispose()` | Free the underlying arena. |

---

## GgmlSharp.Kernel

### Core Types

#### GgmlContext

Arena allocator mirroring `ggml_context`: one growable aligned buffer, tensors carved off the tail.

```csharp
public unsafe class GgmlContext : IDisposable
```

| Method | Description |
|--------|-------------|
| `GgmlContext(long reserveBytes = 1 << 20)` | Create context with initial arena size. |
| `AllocBytes(long nbytes)` | Allocate aligned bytes from arena. |
| `NewTensor(GgmlType type, int nDims, long[] ne, GgmlTensor?[]? src = null, bool noAlloc = false, byte* data = null)` | Create tensor with optional external data. |
| `NewOp(GgmlOp op, GgmlType type, long[] ne, params GgmlTensor[] srcs)` | Generic op node (builder pattern). |
| `NewUnaryOp(GgmlType type, long[] ne, GgmlTensor src, GgmlUnaryOp unaryOp)` | Unary op node with `op_params[0] = unaryOp`. |
| `NewBinaryOp(GgmlType type, long[] ne, GgmlTensor src0, GgmlTensor src1, GgmlOp op)` | Binary op node with two sources. |
| `Dispose()` | Free the arena. |

#### GgmlTensor

Managed tensor descriptor mirroring `ggml_tensor`.

```csharp
public sealed unsafe class GgmlTensor
```

##### Fields

| Field | Type | Description |
|-------|------|-------------|
| `Type` | `GgmlType` | Data type (enum). |
| `Ne` | `long[4]` | Element counts per dimension (column-major: `ne[0]` = innermost). |
| `Nb` | `long[4]` | Byte strides per dimension. |
| `Data` | `byte*` | Raw data pointer (allocated by context). |
| `Op` | `GgmlOp` | Operation type. |
| `OpParams` | `byte[64]` | Operation parameters (raw bytes). |
| `Src` | `GgmlTensor?[10]` | Source tensors (up to 10). |
| `ViewSrc` | `GgmlTensor?` | View source for VIEW/RESHAPE. |
| `ViewOffs` | `long` | View offset in bytes. |
| `Name` | `string?` | Optional debug name. |
| `Flags` | `int` | Flags. |
| `Extra` | `void*` | Extra pointer. |

##### Shape Queries

| Method | Description |
|--------|-------------|
| `NElements()` | Total elements: `ne[0] * ne[1] * ne[2] * ne[3]`. |
| `NRows()` | Row count: `ne[1] * ne[2] * ne[3]`. |
| `RowSize()` | Bytes per row (innermost dimension). |
| `SetShape(int nDims, params long[] ne)` | Set shape (unused dims = 1). |
| `ComputeStrides()` | Recompute `Nb[]` from `Ne[]` and `Type`. |

##### OpParams Accessors

| Method | Description |
|--------|-------------|
| `GetOpParamsI32(GgmlTensor t, int index)` | Read 32-bit int from `OpParams` (little-endian). |
| `GetOpParamsF32(GgmlTensor t, int index)` | Read 32-bit float from `OpParams`. |
| `SetOpParamsI32(GgmlTensor t, int index, int value)` | Write 32-bit int to `OpParams`. |
| `SetOpParamsF32(GgmlTensor t, int index, float value)` | Write 32-bit float to `OpParams`. |

#### GgmlConst

GGML constants (mirroring `ggml.h`):

| Constant | Value |
|----------|-------|
| `MaxDims` | 4 |
| `MaxSrc` | 10 |
| `MaxOpParams` | 64 |
| `MaxName` | 64 |
| `MemAlign` | 16 |

#### GgmlCompute

Top-level forward dispatch mirroring `ggml_compute_forward()`.

```csharp
public static unsafe class GgmlCompute
```

| Method | Description |
|--------|-------------|
| `Forward(in ComputeParams p, GgmlTensor dst)` | Dispatch on `dst.Op` to per-op kernel. Throws `NotImplementedException` for unported ops. |

#### ComputeParams

Per-thread compute parameters + thread row range.

```csharp
public readonly unsafe struct ComputeParams
```

| Field | Description |
|-------|-------------|
| `Ith` | Thread index (0-based). |
| `Nth` | Total thread count. |
| `WorkSize` | Scratch buffer size. |
| `WorkData` | Scratch buffer pointer. |

| Method | Description |
|--------|-------------|
| `ThreadRowRange(long nrows)` | Returns `(ir0, ir1)` row range for this thread: `dr = ceil(nrows / Nth)`, `ir0 = dr * Ith`, `ir1 = min(ir0 + dr, nrows)`. |

---

### Tensor Operations (Ops.*.cs)

Each `Ops*` class contains `Forward` methods dispatched by `GgmlCompute.Forward`.

| Class | Ops Implemented |
|-------|-----------------|
| `OpsUnary` | `UNARY` (ABS, SGN, NEG, STEP, TANH, ELU, RELU, SIGMOID, GELU, GELU_QUICK, SILU, HARDSWISH, HARDSIGMOID, EXP, EXPM1, SOFTPLUS, GELU_ERF, XIELU, FLOOR, CEIL, ROUND, TRUNC) |
| `OpsBinary` | `ADD`, `SUB`, `MUL`, `DIV` (with broadcast) |
| `OpsReduce` | `SUM`, `SUM_ROWS`, `MEAN`, `ARGMAX`, `COUNT_EQUAL`, `CUMSUM` |
| `OpsNorm` | `NORM`, `RMS_NORM`, `L2_NORM`, `GROUP_NORM` |
| `OpsSoftmax` | `SOFT_MAX` |
| `OpsSimple` | `LEAKY_RELU`, `SILU_BACK`, `SCALE` |
| `OpsRemap` | `DUP`/`CPY`/`CONT`, `REPEAT`, `REPEAT_BACK`, `SET`, `PAD`, `PAD_REFLECT_1D`, `ROLL`, `ARANGE`, `FILL`, `TRI`, `ARGSORT`, `TOP_K`, `TIMESTEP_EMBEDDING`, `RMS_NORM_BACK` |
| `OpsUpscale` | `UPSCALE` (nearest/bilinear/bicubic) |
| `OpsMatMul` | `MUL_MAT`, `OUT_PROD` (F32) |
| `OpsRope` | `ROPE` (NORMAL, NEOX, MROPE, IMROPE, VISION with YaRN) |
| `OpsGlu` | `GLU` (REGLU, GEGLU, SWIGLU, SWIGLU_OAI, GEGLU_ERF, GEGLU_QUICK) |

---

### Type System

#### GgmlTypeTraits

Per-type metadata and row conversion functions.

```csharp
public readonly unsafe struct GgmlTypeTraits
```

| Property | Description |
|----------|-------------|
| `TypeName` | Human-readable name (e.g., "f32", "q4_K"). |
| `BlckSize` | Block size (elements per block). |
| `BlckSizeInterleave` | Interleave factor (for K-quants). |
| `TypeSize` | Block size in bytes. |
| `IsQuantized` | Whether type is quantized. |
| `ToFloat` | Dequantize row: `(byte* x, float* y, long k)`. |
| `FromFloatRef` | Quantize row: `(float* x, byte* y, long k)`. |

| Static Method | Description |
|---------------|-------------|
| `Get(GgmlType t)` | Get traits for a type. |
| `Name(GgmlType t)` | Get type name string. |

#### GgmlType (enum ggml_type)

42 values, exact numeric alignment with GGML:

```csharp
public enum GgmlType
{
    F32 = 0, F16, Q4_0, Q4_1, Q4_2, Q4_3, Q5_0, Q5_1, Q8_0, Q8_1,
    Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
    IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M,
    I8, I16, I32, I64, F64, BF16,
    Q4_0_4_4, Q4_0_4_8, Q4_0_8_8,  // REMOVED
    TQ1_0, TQ2_0,
    IQ4_NL_4_4, IQ4_NL_4_8, IQ4_NL_8_8,  // REMOVED
    MXFP4, NVFP4, Q1_0,
    Count = 42
}
```

> **Note**: Deprecated (`Q4_2`, `Q4_3`) and removed slots are kept for 1:1 enum alignment with GGML.

---

### Quantization

#### GgmlDequant (to_float)

Dequantize row kernels. Input `x` points at row data (multiple of `BlckSize`), output `y` receives `k` floats.

| Method | Type | Block | Bytes | Status |
|--------|------|-------|-------|--------|
| `Q1_0` | `Q1_0` | 128 | 18 | ✅ |
| `Q4_0` | `Q4_0` | 32 | 18 | ✅ |
| `Q4_1` | `Q4_1` | 32 | 20 | ✅ |
| `Q5_0` | `Q5_0` | 32 | 22 | ✅ |
| `Q5_1` | `Q5_1` | 32 | 24 | ✅ |
| `Q8_0` | `Q8_0` | 32 | 34 | ✅ |
| `Q4_K` | `Q4_K` | 256 | 144 | ✅ |
| `Q6_K` | `Q6_K` | 256 | 210 | ✅ |
| `Q2_K`..`Q8_K`, `IQ*`, `MXFP4`, `NVFP4`, `TQ*` | — | 256/32/64 | — | Stubbed (throw `NotImplementedException`) |

#### GgmlQuant (from_float)

Quantize row kernels. Input `x` = `k` floats, output `y` = quantized blocks.

| Method | Type | Block | Bytes | Status |
|--------|------|-------|-------|--------|
| `Q1_0` | `Q1_0` | 128 | 18 | ✅ |
| `Q4_0` | `Q4_0` | 32 | 18 | ✅ |
| `Q4_1` | `Q4_1` | 32 | 20 | ✅ |
| `Q5_0` | `Q5_0` | 32 | 22 | ✅ |
| `Q5_1` | `Q5_1` | 32 | 24 | ✅ |
| `Q8_0` | `Q8_0` | 32 | 34 | ✅ |
| `Q8_1` | `Q8_1` | 32 | 36 | ✅ |
| K-quants, IQ, MXFP4, NVFP4, TQ* | — | — | — | Stubbed |

#### TypeSize Reference (bytes per block)

| Type | BlckSize | TypeSize | Bytes/element |
|------|----------|----------|---------------|
| F32 | 1 | 4 | 4.00 |
| F16 | 1 | 2 | 2.00 |
| BF16 | 1 | 2 | 2.00 |
| Q1_0 | 128 | 18 | 0.14 |
| Q4_0 | 32 | 18 | 0.56 |
| Q4_1 | 32 | 20 | 0.63 |
| Q5_0 | 32 | 22 | 0.69 |
| Q5_1 | 32 | 24 | 0.75 |
| Q8_0 | 32 | 34 | 1.06 |
| Q8_1 | 32 | 36 | 1.13 |
| Q2_K | 256 | 84 | 0.33 |
| Q3_K | 256 | 110 | 0.43 |
| Q4_K | 256 | 144 | 0.56 |
| Q5_K | 256 | 176 | 0.69 |
| Q6_K | 256 | 210 | 0.82 |
| Q8_K | 256 | 292 | 1.14 |
| IQ2_XXS | 256 | 66 | 0.26 |
| IQ2_XS | 256 | 74 | 0.29 |
| IQ3_XXS | 256 | 98 | 0.38 |
| IQ1_S | 256 | 50 | 0.20 |
| IQ4_NL | 32 | 18 | 0.56 |
| IQ3_S | 256 | 110 | 0.43 |
| IQ2_S | 256 | 82 | 0.32 |
| IQ4_XS | 256 | 136 | 0.53 |
| IQ1_M | 256 | 56 | 0.22 |
| TQ1_0 | 256 | 54 | 0.21 |
| TQ2_0 | 256 | 66 | 0.26 |
| MXFP4 | 32 | 17 | 0.53 |
| NVFP4 | 64 | 36 | 0.56 |

---

### Math Utilities

#### GgmlMath

Scalar helpers not first-class in .NET.

```csharp
public static unsafe class GgmlMath
```

| Method | Description |
|--------|-------------|
| `F16ToF32(ushort h)` | IEEE-754 binary16 → float (handles subnormals, NaN, Inf). |
| `F32ToF16(float f)` | Float → binary16 (round-to-nearest-even). |
| `Bf16ToF32(ushort h)` | bfloat16 → float. |
| `F32ToBf16(float f)` | Float → bfloat16. |
| `Erf(float x)` | Error function (Abramowitz & Stegun 7.1.26). |

| Constant | Value |
|----------|-------|
| `Sqrt2OverPi` | 0.7978845608028654f |
| `Sqrt2Inv` | 0.7071067811865476f |

---

### Enums

#### GgmlOp (enum ggml_op)

96 values, exact order from `ggml.h`:

```csharp
NONE, DUP, ADD, ADD_ID, ADD1, ACC, SUB, MUL, DIV, SQR, SQRT, LOG, SIN, COS,
SUM, SUM_ROWS, CUMSUM, MEAN, ARGMAX, COUNT_EQUAL, REPEAT, REPEAT_BACK,
CONCAT, SILU_BACK, NORM, RMS_NORM, RMS_NORM_BACK, GROUP_NORM, L2_NORM,
MUL_MAT, MUL_MAT_ID, OUT_PROD, SCALE, SET, CPY, CONT, RESHAPE, VIEW,
PERMUTE, TRANSPOSE, GET_ROWS, GET_ROWS_BACK, SET_ROWS, DIAG, DIAG_MASK_INF,
DIAG_MASK_ZERO, SOFT_MAX, SOFT_MAX_BACK, ROPE, ROPE_BACK, CLAMP,
CONV_TRANSPOSE_1D, IM2COL, IM2COL_BACK, IM2COL_3D, CONV_2D, CONV_3D,
CONV_2D_DW, CONV_TRANSPOSE_2D, POOL_1D, POOL_2D, POOL_2D_BACK, UPSCALE,
PAD, PAD_REFLECT_1D, ROLL, ARANGE, TIMESTEP_EMBEDDING, ARGSORT, TOP_K,
LEAKY_RELU, TRI, FILL, FLASH_ATTN_EXT, FLASH_ATTN_BACK, SSM_CONV, SSM_SCAN,
WIN_PART, WIN_UNPART, GET_REL_POS, ADD_REL_POS, RWKV_WKV6, GATED_LINEAR_ATTN,
RWKV_WKV7, SOLVE_TRI, GATED_DELTA_NET, UNARY, MAP_CUSTOM1, MAP_CUSTOM2,
MAP_CUSTOM3, CUSTOM, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_BACK,
OPT_STEP_ADAMW, OPT_STEP_SGD, GLU, Count = 96
```

#### GgmlUnaryOp (enum ggml_unary_op)

22 values:

```csharp
ABS, SGN, NEG, STEP, TANH, ELU, RELU, SIGMOID, GELU, GELU_QUICK, SILU,
HARDSWISH, HARDSIGMOID, EXP, EXPM1, SOFTPLUS, GELU_ERF, XIELU,
FLOOR, CEIL, ROUND, TRUNC, Count = 22
```

#### GgmlGluOp (enum ggml_glu_op)

6 values, stored in `op_params[0]` for `GGML_OP_GLU`:

```csharp
REGLU = 0, GEGLU, SWIGLU, SWIGLU_OAI, GEGLU_ERF, GEGLU_QUICK, Count = 6
```

| Op | Formula |
|----|---------|
| REGLU | `ReLU(x) * g` |
| GEGLU | `GELU(x) * g` |
| SWIGLU | `SiLU(x) * g` = `x * sigmoid(x) * g` |
| SWIGLU_OAI | `(x / (1 + exp(α·-x))) * (clamp(g, -L, L) + 1)` |
| GEGLU_ERF | `0.5 * x * (1 + erf(x/√2)) * g` |
| GEGLU_QUICK | `x * sigmoid(1.702 * x) * g` |

#### GgmlRopeType

Bit flags for RoPE variants:

```csharp
NORMAL = 0, NEOX = 1, VISION = 2, MROPE = 4, IMROPE = 24
```

| Mode | Description | Positions Layout |
|------|-------------|------------------|
| NORMAL | Standard RoPE | `[ne2]` (position per sequence) |
| NEOX | GPT-NeoX style (interleaved) | `[ne2]` |
| MROPE | Multi-modal RoPE (LLaVA) | `[ne2 * 4]` (t, h, w, e) |
| IMROPE | Interleaved MROPE (Qwen3-VL) | `[ne2 * 4]` (interleaved t/h/w) |
| VISION | Vision encoder RoPE | `[ne2 * 4]` (t, h, w, e) |

#### GgmlTriType

```csharp
LOWER, LOWER_DIAG, UPPER, UPPER_DIAG
```

#### GgmlSortOrder

```csharp
ASC, DESC
```

#### GgmlScaleMode

```csharp
NEAREST = 0, BILINEAR = 1, BICUBIC = 2
```

#### GgmlScaleFlag (bit flags)

```csharp
AlignCorners = 1 << 16, Antialias = 1 << 17
```

---

### Row Codecs (Delegates)

```csharp
public unsafe delegate void GgmlToFloat(byte* x, float* y, long k);      // dequantize
public unsafe delegate void GgmlFromFloat(float* x, byte* y, long k);   // quantize
```

- `x`: source pointer (quantized blocks or floats)
- `y`: destination pointer
- `k`: number of elements (must be multiple of `BlckSize`)

Used by `GgmlTypeTraits.ToFloat` / `FromFloatRef`.