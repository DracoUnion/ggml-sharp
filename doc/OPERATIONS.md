# Operations Reference

Complete reference of all implemented operations in GgmlSharp, grouped by category.

---

## Operation Categories

| Category | Ops | Source File |
|----------|-----|-------------|
| **Unary** | 22 ops | `OpsUnary.cs` |
| **Binary** | 4 ops | `OpsBinary.cs` |
| **Reduce** | 6 ops | `OpsReduce.cs` |
| **Normalization** | 4 ops | `OpsNorm.cs` |
| **Softmax** | 1 op | `OpsSoftmax.cs` |
| **Simple/Pointwise** | 3 ops | `OpsSimple.cs` |
| **Remap** | 14 ops | `OpsRemap.cs` |
| **Upscale** | 1 op | `OpsUpscale.cs` |
| **Matrix Multiply** | 2 ops | `OpsMatMul.cs` |
| **RoPE** | 1 op (5 modes) | `OpsRope.cs` |
| **GLU** | 1 op (6 variants) | `OpsGlu.cs` |

---

## Unary Operations (GGML_OP_UNARY)

All 22 unary ops are dispatched via `GGML_OP_UNARY` with `op_params[0] = GgmlUnaryOp`.

| Op | Enum | Formula | Notes |
|----|------|---------|-------|
| ABS | `ABS` | `x ≥ 0 ? x : -x` | |
| SGN | `SGN` | `x > 0 ? 1 : (x < 0 ? -1 : 0)` | |
| NEG | `NEG` | `-x` | |
| STEP | `STEP` | `x > 0 ? 1 : 0` | Heaviside step |
| TANH | `TANH` | `tanh(x)` | |
| ELU | `ELU` | `x ≥ 0 ? x : exp(x) - 1` | α=1 |
| RELU | `RELU` | `max(0, x)` | |
| SIGMOID | `SIGMOID` | `1 / (1 + exp(-x))` | |
| GELU | `GELU` | `0.5 * x * (1 + erf(x/√2))` | Exact (erf) |
| GELU_QUICK | `GELU_QUICK` | `x * sigmoid(1.702 * x)` | Fast approx |
| SILU | `SILU` | `x * sigmoid(x)` | Swish |
| HARDSWISH | `HARDSWISH` | `x * clamp(x+3, 0, 6) / 6` | |
| HARDSIGMOID | `HARDSIGMOID` | `clamp(x/6 + 0.5, 0, 1)` | |
| EXP | `EXP` | `exp(x)` | |
| EXPM1 | `EXPM1` | `exp(x) - 1` | |
| SOFTPLUS | `SOFTPLUS` | `ln(1 + exp(x))` | |
| GELU_ERF | `GELU_ERF` | Same as GELU | Alias |
| XIELU | `XIELU` | `x ≥ 0 ? x : (exp(x) - 1) / ln(2)` | |
| FLOOR | `FLOOR` | `floor(x)` | |
| CEIL | `CEIL` | `ceil(x)` | |
| ROUND | `ROUND` | `round(x)` | Banker's rounding |
| TRUNC | `TRUNC` | `trunc(x)` | Toward zero |

**Implementation**: `OpsUnary.Forward` — AVX2 fast paths for F32/F16, scalar fallback.

**Usage**:
```csharp
var unary = ctx.NewUnaryOp(GgmlType.F32, shape, src, GgmlUnaryOp.GELU);
```

---

## Binary Operations (Elementwise)

| Op | Enum | Formula | Broadcast |
|----|------|---------|-----------|
| ADD | `ADD` | `a + b` | ✅ |
| SUB | `SUB` | `a - b` | ✅ |
| MUL | `MUL` | `a * b` | ✅ |
| DIV | `DIV` | `a / b` | ✅ |

**Implementation**: `OpsBinary.Forward` — F32 SIMD + generic codec path.

**Broadcast Rules** (per dimension):
- Dimensions equal, or
- One dimension is 1 (broadcast), or
- Missing dimensions treated as 1 (right-aligned)

```csharp
// [256, 64] + [256, 1] → [256, 64]
// [256, 64] + [1, 64]  → [256, 64]
// [256, 64] + [64]     → [256, 64] (right-aligned)
```

---

## Reduction Operations

| Op | Enum | Output Shape | Description |
|----|------|--------------|-------------|
| SUM | `SUM` | `[1, 1, 1, 1]` | Sum all elements to scalar |
| SUM_ROWS | `SUM_ROWS` | `[1, ne1, ne2, ne3]` | Sum per row (over ne[0]) |
| MEAN | `MEAN` | `[1, ne1, ne2, ne3]` | Mean per row (sum / ne[0]) |
| ARGMAX | `ARGMAX` | `[1, ne1, ne2, ne3]` (I32) | Index of max per row |
| COUNT_EQUAL | `COUNT_EQUAL` | `[1, 1, 1, 1]` (I64) | Count equal elements (src0 == src1) |
| CUMSUM | `CUMSUM` | Same as input | Prefix sum per row |

**Implementation**: `OpsReduce.Forward*` — threaded over rows.

```csharp
// Sum all
var sum = ctx.NewOp(GgmlOp.SUM, GgmlType.F32, new[]{1,1,1,1}, x);

// Sum per row
var sumRows = ctx.NewOp(GgmlOp.SUM_ROWS, GgmlType.F32, new[]{1, ne1, ne2, ne3}, x);

// Argmax per row (output I32)
var argmax = ctx.NewOp(GgmlOp.ARGMAX, GgmlType.I32, new[]{1, ne1, ne2, ne3}, x);
```

---

## Normalization Operations

| Op | Enum | Formula | Parameters |
|----|------|---------|------------|
| NORM | `NORM` | `(x - mean) / sqrt(var + eps)` | `eps` in op_params[0] |
| RMS_NORM | `RMS_NORM` | `x / sqrt(mean(x²) + eps)` | `eps` in op_params[0] |
| L2_NORM | `L2_NORM` | `x / max(sqrt(sum(x²)), eps)` | `eps` in op_params[0] |
| GROUP_NORM | `GROUP_NORM` | Per-group norm | `num_groups` in op_params[0] |

**Implementation**: `OpsNorm.Forward*` — threaded over rows (groups).

```csharp
// RMS Norm (LLaMA style)
var rms = ctx.NewOp(GgmlOp.RMS_NORM, GgmlType.F32, shape, x);
GgmlTensor.SetOpParamsF32(rms, 0, 1e-6f);

// Group Norm
var gn = ctx.NewOp(GgmlOp.GROUP_NORM, GgmlType.F32, shape, x);
GgmlTensor.SetOpParamsI32(gn, 0, 32); // 32 groups
```

---

## Softmax

| Op | Enum | Description | Parameters |
|----|------|-------------|------------|
| SOFT_MAX | `SOFT_MAX` | Row-wise softmax | `scale` in op_params[0], `mask` in op_params[1] |

```csharp
var sm = ctx.NewOp(GgmlOp.SOFT_MAX, GgmlType.F32, shape, x);
GgmlTensor.SetOpParamsF32(sm, 0, 1.0f); // scale
GgmlTensor.SetOpParamsF32(sm, 1, 0.0f); // no mask
```

**Implementation**: `OpsSoftmax.Forward` — numerically stable (max subtraction), threaded over rows.

---

## Simple Pointwise Operations

| Op | Enum | Formula | Parameters |
|----|------|---------|------------|
| LEAKY_RELU | `LEAKY_RELU` | `x > 0 ? x : α*x` | `α` in op_params[0] |
| SILU_BACK | `SILU_BACK` | `dz * silu'(x)` | Used in backward pass |
| SCALE | `SCALE` | `x * s + b` | `s` in op_params[0], `b` in op_params[1] |

```csharp
// Leaky ReLU
var lr = ctx.NewOp(GgmlOp.LEAKY_RELU, GgmlType.F32, shape, x);
GgmlTensor.SetOpParamsF32(lr, 0, 0.01f);

// Scale + bias
var scale = ctx.NewOp(GgmlOp.SCALE, GgmlType.F32, shape, x);
GgmlTensor.SetOpParamsF32(scale, 0, 2.0f); // scale
GgmlTensor.SetOpParamsF32(scale, 1, 0.5f); // bias
```

---

## Remap Operations

| Op | Enum | Description |
|----|------|-------------|
| DUP | `DUP` | Copy src0 → dst with type conversion |
| CPY | `CPY` | Alias for DUP |
| CONT | `CONT` | Make contiguous (copy if strided) |
| REPEAT | `REPEAT` | Tile src0 to fill dst shape |
| REPEAT_BACK | `REPEAT_BACK` | Gradient: sum tiles into dst |
| SET | `SET` | Copy src0, then overwrite slice from src1 |
| PAD | `PAD` | Zero or circular padding per dimension |
| PAD_REFLECT_1D | `PAD_REFLECT_1D` | Reflect padding (1D) |
| ROLL | `ROLL` | Circular shift along dimension |
| ARANGE | `ARANGE` | Fill with 0, 1, 2, ... |
| FILL | `FILL` | Fill with scalar value |
| TRI | `TRI` | Triangular matrix (lower/upper ± diagonal) |
| ARGSORT | `ARGSORT` | Sort indices per row (ASC/DESC) |
| TOP_K | `TOP_K` | Top-K values + indices per row |
| TIMESTEP_EMBEDDING | `TIMESTEP_EMBEDDING` | Sinusoidal timestep embedding |
| RMS_NORM_BACK | `RMS_NORM_BACK` | RMS Norm gradient |

### Key Examples

```csharp
// REPEAT: tile [64, 32] → [64, 64]
var rep = ctx.NewOp(GgmlOp.REPEAT, GgmlType.F32, new[]{64, 64}, src);

// PAD: zero-pad [64, 64] → [64, 72] (pad 4 on each side)
var pad = ctx.NewOp(GgmlOp.PAD, GgmlType.F32, new[]{64, 72}, src);
GgmlTensor.SetOpParamsI32(pad, 0, 0); // pad_left
GgmlTensor.SetOpParamsI32(pad, 1, 4); // pad_right
GgmlTensor.SetOpParamsI32(pad, 2, 0); // pad_top
GgmlTensor.SetOpParamsI32(pad, 3, 4); // pad_bottom
GgmlTensor.SetOpParamsI32(pad, 8, 0); // circular = 0

// ROLL: shift by 2 along dim 1
var roll = ctx.NewOp(GgmlOp.ROLL, GgmlType.F32, shape, src);
GgmlTensor.SetOpParamsI32(roll, 0, 2); // shift

// ARGSORT: descending
var argsort = ctx.NewOp(GgmlOp.ARGSORT, GgmlType.I32, shape, src);
GgmlTensor.SetOpParamsI32(argsort, 0, (int)GgmlSortOrder.DESC);

// TOP_K: k=5
var topk = ctx.NewOp(GgmlOp.TOP_K, GgmlType.F32, new[]{5, ne1, ne2, ne3}, src);
GgmlTensor.SetOpParamsI32(topk, 0, 5); // k

// TIMESTEP_EMBEDDING: sinusoidal
var t = ctx.NewOp(GgmlOp.TIMESTEP_EMBEDDING, GgmlType.F32, new[]{dim, 1}, timestep);
// timestep: scalar tensor (I32 or F32)
```

---

## Upscale

| Op | Enum | Description |
|----|------|-------------|
| UPSCALE | `UPSCALE` | Nearest / Bilinear / Bicubic upscaling |

```csharp
var up = ctx.NewOp(GgmlOp.UPSCALE, GgmlType.F32, dstShape, src);
GgmlTensor.SetOpParamsI32(up, 0, (int)GgmlScaleMode.BILINEAR);
GgmlTensor.SetOpParamsI32(up, 1, (int)GgmlScaleFlag.AlignCorners); // flags
```

---

## Matrix Multiply

| Op | Enum | Formula | Shapes |
|----|------|---------|--------|
| MUL_MAT | `MUL_MAT` | `dst = aᵀ @ b` | `a:[K,N]`, `b:[K,M]` → `[N,M]` |
| OUT_PROD | `OUT_PROD` | `dst[i,j] = Σ_k a[i,k] * b[j,k]` | `a:[K,N]`, `b:[K,M]` → `[N,M]` |

**Broadcast**: Batch dims (ne[2], ne[3]) broadcast.

```csharp
// MUL_MAT: standard attention projection
var y = ctx.NewOp(GgmlOp.MUL_MAT, GgmlType.F32, new[]{outDim, batch}, w, x);
// w: [inDim, outDim], x: [inDim, batch] → y: [outDim, batch]

// OUT_PROD: outer product
var op = ctx.NewOp(GgmlOp.OUT_PROD, GgmlType.F32, new[]{N, M}, a, b);
```

**Implementation**: `OpsMatMul.ForwardMulMat/ForwardOutProd` — F32 only currently. Quantized (Q4_K × F32) dequantizes on-the-fly.

---

## RoPE (Rotary Positional Embedding)

| Op | Enum | Modes |
|----|------|-------|
| ROPE | `ROPE` | NORMAL, NEOX, MROPE, IMROPE, VISION |

### Parameters (op_params)

| Index | Type | Name | Description |
|-------|------|------|-------------|
| 0 | i32 | n_past | Past sequence length (unused) |
| 1 | i32 | n_dims | Number of dims to rotate (must be even) |
| 2 | i32 | mode | `GgmlRopeType` (bit flags) |
| 4 | i32 | n_ctx_orig | Original context length (YaRN) |
| 5 | f32 | freq_base | Base frequency (default 10000) |
| 6 | f32 | freq_scale | YaRN frequency scale |
| 7 | f32 | ext_factor | YaRN extrapolation factor |
| 8 | f32 | attn_factor | YaRN attention factor |
| 9 | f32 | beta_fast | YaRN beta fast |
| 10 | f32 | beta_slow | YaRN beta slow |
| 11-14 | i32 | sections[4] | MROPE/IMROPE section dims |

### Sources

| Index | Type | Description |
|-------|------|-------------|
| src[0] | Input | Tensor to rotate `[dim, seq, heads, batch]` |
| src[1] | I32 | Positions — layout depends on mode |
| src[2] | F32 (opt) | Frequency factors per dim |

### Mode Details

| Mode | Value | Positions Layout | Rotation Pattern |
|------|-------|------------------|------------------|
| NORMAL | 0 | `[seq]` | Adjacent pairs: (0,1), (2,3), ... |
| NEOX | 1 | `[seq]` | Interleaved: (0,dim/2), (1,dim/2+1), ... |
| MROPE | 4 | `[seq*4]` | t,h,w,e per position; sections define dims |
| IMROPE | 24 | `[seq*4]` | Interleaved t/h/w; sections define dims |
| VISION | 2 | `[seq*4]` | t,h,w,e; n_dims = ne[0]/2 |

```csharp
// NEOX (GPT-NeoX, LLaMA)
var rope = ctx.NewOp(GgmlOp.ROPE, GgmlType.F32, shape, x, positions);
GgmlTensor.SetOpParamsI32(rope, 1, 64);   // n_dims
GgmlTensor.SetOpParamsI32(rope, 2, (int)GgmlRopeType.NEOX);
GgmlTensor.SetOpParamsI32(rope, 4, 2048); // n_ctx_orig
GgmlTensor.SetOpParamsF32(rope, 5, 10000f); // freq_base

// MROPE (LLaVA)
// positions: [t0,h0,w0,e0, t1,h1,w1,e1, ...]
// sections: [time_dims, height_dims, width_dims, extra_dims]
GgmlTensor.SetOpParamsI32(rope, 2, (int)GgmlRopeType.MROPE);
for (int i = 0; i < 4; i++) 
    GgmlTensor.SetOpParamsI32(rope, 11 + i, sections[i]);
```

---

## GLU (Gated Linear Unit)

| Op | Enum | Variants |
|----|------|----------|
| GLU | `GLU` | REGLU, GEGLU, SWIGLU, SWIGLU_OAI, GEGLU_ERF, GEGLU_QUICK |

### Parameters (op_params)

| Index | Type | Name | Description |
|-------|------|------|-------------|
| 0 | i32 | glu_op | `GgmlGluOp` variant |
| 1 | i32 | swapped | 0: src0=x, src1=gate; 1: swapped |
| 2 | f32 | alpha | SWIGLU_OAI only |
| 3 | f32 | limit | SWIGLU_OAI only |

### Variants

| Variant | Formula | Description |
|---------|---------|-------------|
| REGLU | `ReLU(x) * g` | ReLU gate |
| GEGLU | `GELU(x) * g` | GELU gate (erf) |
| SWIGLU | `SiLU(x) * g` | `x * sigmoid(x) * g` |
| SWIGLU_OAI | `x / (1+exp(α·-x)) * (clamp(g,-L,L)+1)` | OpenAI SwiGLU |
| GEGLU_ERF | `0.5*x*(1+erf(x/√2)) * g` | Explicit erf GELU |
| GEGLU_QUICK | `x * sigmoid(1.702*x) * g` | Fast GELU approx |

```csharp
// SWIGLU (LLaMA FFN)
var glu = ctx.NewOp(GgmlOp.GLU, GgmlType.F32, new[]{4096, seq}, up, gate);
GgmlTensor.SetOpParamsI32(glu, 0, (int)GgmlGluOp.SWIGLU);
GgmlTensor.SetOpParamsI32(glu, 1, 0); // swapped = 0

// SWIGLU_OAI (OpenAI)
GgmlTensor.SetOpParamsI32(glu, 0, (int)GgmlGluOp.SWIGLU_OAI);
GgmlTensor.SetOpParamsF32(glu, 2, 1.0f); // alpha
GgmlTensor.SetOpParamsF32(glu, 3, 10.0f); // limit
```

---

## Not Yet Implemented (Stubs)

These ops exist in `GgmlOp` enum but throw `NotImplementedException`:

| Category | Ops |
|----------|-----|
| Convolution | `CONV_1D`, `CONV_2D`, `CONV_3D`, `CONV_TRANSPOSE_*`, `IM2COL*`, `POOL_*`, `CONV_2D_DW` |
| Attention | `FLASH_ATTN_EXT`, `FLASH_ATTN_BACK` |
| RNN/SSM | `SSM_CONV`, `SSM_SCAN`, `RWKV_WKV6`, `RWKV_WKV7`, `GATED_LINEAR_ATTN`, `GATED_DELTA_NET` |
| Custom | `MAP_CUSTOM1/2/3`, `CUSTOM` |
| Loss | `CROSS_ENTROPY_LOSS`, `CROSS_ENTROPY_LOSS_BACK` |
| Optimizer | `OPT_STEP_ADAMW`, `OPT_STEP_SGD` |
| Other | `SQR`, `SQRT`, `LOG`, `SIN`, `COS`, `ADD_ID`, `ADD1`, `ACC`, `SILU_BACK`, `CONCAT`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`, `GET_ROWS*`, `SET_ROWS`, `DIAG*`, `CLAMP`, `WIN_PART`, `WIN_UNPART`, `GET_REL_POS`, `ADD_REL_POS`, `SOLVE_TRI`, `ARGSORT`, `TOP_K` (partially), `TIMESTEP_EMBEDDING` (partially) |

---

## Op Parameter Quick Reference

```csharp
// Common pattern: set params after creating op node
var op = ctx.NewOp(GgmlOp.OP_NAME, type, shape, src...);
GgmlTensor.SetOpParamsI32(op, index, intValue);
GgmlTensor.SetOpParamsF32(op, index, floatValue);

// Read back
int iv = GgmlTensor.GetOpParamsI32(op, index);
float fv = GgmlTensor.GetOpParamsF32(op, index);
```

---

## Threading Behavior

All ops respect `ComputeParams.Ith`/`Nth`:

```csharp
var p = new ComputeParams(ith, nth);
GgmlCompute.Forward(p, op);
// Kernel processes rows [ir0, ir1) where:
// dr = ceil(nrows / nth)
// ir0 = dr * ith
// ir1 = min(ir0 + dr, nrows)
```

- **Row-parallel**: Reduce, Norm, Softmax, Unary, Binary, Simple, Remap
- **Column-parallel**: MatMul (splits output columns)
- **Position-parallel**: RoPE (splits sequence positions)

---

## Quantization Support by Op

| Op | F32 | F16 | BF16 | Q4_K | Q4_0 | Q8_0 | Other Quant |
|----|-----|-----|------|------|------|------|-------------|
| ADD/SUB/MUL/DIV | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| MUL_MAT | ✅ | ❌ | ❌ | ✅* | ❌ | ❌ | ❌ |
| Unary (all) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Reduce (all) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Norm (all) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| SoftMax | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| RoPE | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GLU | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Remap (DUP/CPY) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

* MUL_MAT with Q4_K weights dequantizes on-the-fly to F32.