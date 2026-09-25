# Quantization Guide

Deep dive into GGML quantization types supported by GgmlSharp.

---

## Overview

GGML uses **block-wise quantization**: weights are divided into fixed-size blocks, each quantized independently with its own scale/min parameters. This enables:

- **Low memory footprint** — 2-8 bits per weight vs 32 bits (FP32)
- **Fast inference** — Dequantize on-the-fly in compute kernels (SIMD-optimized)
- **Minimal accuracy loss** — Per-block scaling preserves dynamic range

GgmlSharp mirrors GGML's quantization exactly: same block layouts, same enum values, same dequantize/quantize algorithms.

---

## Quantization Type Categories

| Category | Types | Block Size | Use Case |
|----------|-------|------------|----------|
| **Non-K (legacy)** | Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | 32 or 128 | Older models, simple quantization |
| **K-quants (recommended)** | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | 256 | **Default for modern models** (LLaMA, Mistral, etc.) |
| **IQ (Importance Quantized)** | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | 32 or 256 | Extreme compression (1-4 bpw) |
| **Ternary** | TQ1_0, TQ2_0 | 256 | BitNet / 1.58-bit models |
| **Microscaling** | MXFP4, NVFP4 | 32, 64 | Blackwell / Hopper FP4 formats |

---

## Type Reference

### Non-K Types (Block Size 32/128)

| Type | Bits | Block | Bytes | Elements | Description |
|------|------|-------|-------|----------|-------------|
| `Q1_0` | 1.0 | 128 | 18 | 128 | Binary (sign only), delta per block |
| `Q4_0` | 4.0 | 32 | 18 | 32 | 4-bit, symmetric, delta per block |
| `Q4_1` | 4.0 | 32 | 20 | 32 | 4-bit, asymmetric (min + delta) |
| `Q5_0` | 5.0 | 32 | 22 | 32 | 5-bit, symmetric, delta + high bit |
| `Q5_1` | 5.0 | 32 | 24 | 32 | 5-bit, asymmetric (min + delta + high bit) |
| `Q8_0` | 8.0 | 32 | 34 | 32 | 8-bit, symmetric, delta per block |
| `Q8_1` | 8.0 | 32 | 36 | 32 | 8-bit, asymmetric (sum compensation) |

### K-Quants (Super-Block 256)

All K-quants use **super-block = 256 elements** (16 sub-blocks of 16, or 8 of 32).

| Type | Bits | Bytes | Sub-Blocks | Scales | Description |
|------|------|-------|------------|--------|-------------|
| `Q2_K` | 2.6 | 84 | 16×16 | 4-bit | 2-bit quants + 4-bit scales/mins |
| `Q3_K` | 3.4 | 110 | 16×16 | 6-bit | 3-bit quants (2+1 high bit) + 6-bit scales |
| `Q4_K` | 4.5 | 144 | 8×32 | 6-bit | 4-bit quants + 6-bit scales/mins |
| `Q5_K` | 5.5 | 176 | 8×32 | 6-bit | 5-bit quants (4+1 high bit) + scales/mins |
| `Q6_K` | 6.6 | 210 | 16×16 | 8-bit | 6-bit quants (4+2 high bits) + 8-bit scales |
| `Q8_K` | 8.0 | 292 | 16×16 | FP32 | 8-bit quants + FP32 delta + int16 bsums |

### IQ Types (Importance Quantized)

| Type | Bits | Bytes | Block | Description |
|------|------|-------|-------|-------------|
| `IQ1_S` | 1.56 | 50 | 256 | 1-bit + grid (256 grids) + 3-bit scales |
| `IQ1_M` | 1.75 | 56 | 256 | 1-bit + grid (1024 grids) + 3-bit scales + shift |
| `IQ2_XXS` | 2.0 | 66 | 256 | 2-bit + 5-ary grid + per-block scale |
| `IQ2_XS` | 2.3 | 74 | 256 | 2-bit + 5-ary grid + per-32 scale |
| `IQ2_S` | 2.56 | 82 | 256 | 2-bit + grid + signs + per-32 scale |
| `IQ3_XXS` | 3.06 | 98 | 256 | 3-bit + 5-ary grid + per-block scale |
| `IQ3_S` | 3.44 | 110 | 256 | 3-bit + grid + signs + per-64 scale |
| `IQ4_NL` | 4.0 | 18 | 32 | Non-linear 4-bit (custom 16-level codebook) |
| `IQ4_XS` | 4.0 | 136 | 256 | 4-bit + per-32 scale (high/low 4 bits) |

### Ternary Types

| Type | Bits | Bytes | Block | Description |
|------|------|-------|-------|-------------|
| `TQ1_0` | 1.69 | 54 | 256 | Ternary {-1,0,1} 5-per-byte (3⁵=243) + 4 per byte (high) + FP16 delta |
| `TQ2_0` | 2.06 | 66 | 256 | Ternary 2-bit per element + FP16 delta |

### Microscaling (FP4)

| Type | Bits | Bytes | Block | Description |
|------|------|-------|-------|-------------|
| `MXFP4` | 4.0 (E2M1) | 17 | 32 | OCP MXFP4: 1 E8M0 scale + 32 packed E2M1 |
| `NVFP4` | 4.0 (E2M1) | 36 | 64 | NVIDIA FP4: 4 UE4M3 scales + 64 packed E2M1 |

---

## Block Layout Details

### Q4_0 (18 bytes, 32 elements)
```
struct { fp16 delta; uint8_t qs[16]; }  // 2 + 16 = 18
// qs: 2 elements per byte (4 bits each), values 0..15 → -8..7
// dequant: (qs - 8) * delta
```

### Q4_1 (20 bytes, 32 elements)
```
struct { fp16 delta; fp16 min; uint8_t qs[16]; }  // 2+2+16 = 20
// dequant: qs * delta + min
```

### Q4_K (144 bytes, 256 elements)
```
struct {
    fp16 delta;      // super-block scale
    fp16 min;        // super-block min scale
    uint8_t scales[12];  // 6-bit scales (8) + 6-bit mins (8) packed in 12 bytes
    uint8_t qs[128];     // 4-bit quants, 256 elements
}
// 256 = 8 sub-blocks × 32 elements
// dequant per sub-block: qs * (delta * scale) - (min * min_scale)
```

### Q8_K (292 bytes, 256 elements)
```
struct {
    float delta;         // FP32 super-block scale
    int8_t qs[256];      // 8-bit quants
    int16_t bsums[16];   // sum of qs per 16-element group
}
// Used for intermediate quantization and dot products
```

### IQ4_NL (18 bytes, 32 elements)
```
struct { fp16 delta; uint8_t qs[16]; }  // 2 + 16 = 18
// Non-linear: qs values index into 16-level codebook
// codebook: {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
// dequant: codebook[qs] * delta
```

---

## API Usage

### Dequantize (to_float) — In Kernels

Kernels automatically call `GgmlTypeTraits.Get(type).ToFloat`:

```csharp
// Inside a kernel, for a quantized tensor 'src':
var traits = GgmlTypeTraits.Get(src.Type);
traits.ToFloat(src.Data, floatOutput, nElements);
```

Available `ToFloat` implementations:

| Type | Method | Status |
|------|--------|--------|
| Q1_0 | `GgmlDequant.Q1_0` | ✅ |
| Q4_0 | `GgmlDequant.Q4_0` | ✅ |
| Q4_1 | `GgmlDequant.Q4_1` | ✅ |
| Q5_0 | `GgmlDequant.Q5_0` | ✅ |
| Q5_1 | `GgmlDequant.Q5_1` | ✅ |
| Q8_0 | `GgmlDequant.Q8_0` | ✅ |
| Q4_K | `GgmlDequant.Q4_K` | ✅ |
| Q6_K | `GgmlDequant.Q6_K` | ✅ |
| Others | — | ❌ Stub (throws) |

### Quantize (from_float) — Host Side

Use `GgmlQuant` or `GgmlTypeTraits.FromFloatRef`:

```csharp
// Quantize 4096 F32 elements to Q4_K
int n = 4096;
float[] src = GetWeights();
byte[] dst = new byte[(n / 256) * 144];

unsafe {
    fixed (float* s = src)
    fixed (byte* d = dst) {
        GgmlQuant.Q4_K(s, d, n);
    }
}

// Or via TypeTraits:
var traits = GgmlTypeTraits.Get(GgmlType.Q4_K);
traits.FromFloatRef(srcPtr, dstPtr, n);
```

Available `FromFloatRef` implementations:

| Type | Method | Status |
|------|--------|--------|
| Q1_0 | `GgmlQuant.Q1_0` | ✅ |
| Q4_0 | `GgmlQuant.Q4_0` | ✅ |
| Q4_1 | `GgmlQuant.Q4_1` | ✅ |
| Q5_0 | `GgmlQuant.Q5_0` | ✅ |
| Q5_1 | `GgmlQuant.Q5_1` | ✅ |
| Q8_0 | `GgmlQuant.Q8_0` | ✅ |
| Q8_1 | `GgmlQuant.Q8_1` | ✅ |
| K-quants, IQ, MXFP4, NVFP4, TQ* | — | ❌ Stub (throws) |

### Creating Quantized Tensors

```csharp
using var ctx = new GgmlContext();

// Option 1: Allocate and fill manually
var q = ctx.NewTensor(GgmlType.Q4_K, 2, new long[]{4096, 1024}, noAlloc: true);
unsafe {
    // q.Data points to uninitialized arena memory
    // Fill with pre-quantized data:
    Buffer.MemoryCopy(quantizedData, q.Data, quantizedData.Length, quantizedData.Length);
}

// Option 2: Quantize on the fly (F32 → quantized)
var f = ctx.NewTensor(GgmlType.F32, 2, new long[]{4096, 1024});
// ... fill f.Data with float weights ...

var q = ctx.NewTensor(GgmlType.Q4_K, 2, new long[]{4096, 1024});
unsafe {
    // Quantize row by row (each row = 4096 = 16 blocks)
    for (int row = 0; row < 1024; row++) {
        float* srcRow = (float*)f.Data + row * 4096;
        byte* dstRow = q.Data + row * 16 * 144;
        GgmlQuant.Q4_K(srcRow, dstRow, 4096);
    }
}
```

---

## Choosing Quantization

| Scenario | Recommended Type | Reason |
|----------|------------------|--------|
| **LLM weights (default)** | `Q4_K` | Best quality/size tradeoff; standard for LLaMA, Mistral, Gemma |
| **Maximum quality** | `Q5_K`, `Q6_K` | Higher bits, larger size |
| **Minimum size** | `Q2_K`, `Q3_K`, `IQ2_XS` | For edge deployment |
| **Extreme compression** | `IQ1_S`, `IQ1_M` | Research / 1-bit models |
| **BitNet / 1.58-bit** | `TQ1_0`, `TQ2_0` | Ternary models |
| **Blackwell/Hopper FP4** | `MXFP4`, `NVFP4` | Native FP4 hardware |
| **Legacy models** | `Q4_0`, `Q4_1`, `Q8_0` | Older GGML checkpoints |

### Quality vs Size (Approximate)

```
Q8_0 (8-bit)     ████████████████  ~1.0 bpw  → Near FP16 quality
Q6_K (6.6-bit)   ██████████████    ~0.8 bpw  → Excellent
Q5_K (5.5-bit)   ████████████      ~0.7 bpw  → Very good
Q4_K (4.5-bit)   ██████████        ~0.6 bpw  → **Recommended default**
Q3_K (3.4-bit)   ███████           ~0.5 bpw  → Good
Q2_K (2.6-bit)   █████             ~0.3 bpw  → Acceptable
IQ2_XS (2.3-bit) ████              ~0.3 bpw  → Better than Q2_K at same size
IQ1_S (1.56-bit) ██                ~0.2 bpw  → Significant degradation
TQ1_0 (1.69-bit) ███               ~0.2 bpw  → Ternary, research use
```

---

## Implementation Status in GgmlSharp

### Dequantize (to_float) — Kernels

| Type | Implemented | Notes |
|------|-------------|-------|
| Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | ✅ | Full SIMD + scalar |
| Q4_K, Q6_K | ✅ | Scalar reference |
| Q2_K, Q3_K, Q5_K, Q8_K | ❌ | Stub (throws) |
| All IQ, MXFP4, NVFP4, TQ* | ❌ | Stub (throws) |

### Quantize (from_float) — Host

| Type | Implemented | Notes |
|------|-------------|-------|
| Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | ✅ | Reference algorithm |
| K-quants, IQ, MXFP4, NVFP4, TQ* | ❌ | Stub (throws) |

### Type Metadata (Complete for All)

| Property | Status |
|----------|--------|
| `TypeSize` (bytes/block) | ✅ All 42 types |
| `BlckSize` (elements/block) | ✅ All 42 types |
| `BlckSizeInterleave` | ✅ All 42 types |
| `IsQuantized` | ✅ All 42 types |
| `TypeName` | ✅ All 42 types |

---

## Example: Quantize a Linear Layer

```csharp
// FP32 weights: [out=11008, in=4096] (LLaMA 7B FFN up_proj)
// Quantize to Q4_K for inference

using var ctx = new GgmlContext(1L << 30); // Large arena for quantized weights

// Load FP32 weights (row-major: out_dim × in_dim)
float[] fp32Weights = LoadWeights("up_proj.bin"); // 11008 * 4096 floats

// Create FP32 tensor (GGML is column-major: [in_dim, out_dim])
var wFp32 = ctx.NewTensor(GgmlType.F32, 2, new long[]{4096, 11008});
unsafe {
    // Transpose during copy: row-major → column-major
    fixed (float* src = fp32Weights)
    fixed (float* dst = (float*)wFp32.Data) {
        for (int outIdx = 0; outIdx < 11008; outIdx++) {
            for (int inIdx = 0; inIdx < 4096; inIdx++) {
                dst[inIdx * 11008 + outIdx] = src[outIdx * 4096 + inIdx];
            }
        }
    }
}

// Create Q4_K output tensor
var wQ4k = ctx.NewTensor(GgmlType.Q4_K, 2, new long[]{4096, 11008});

// Quantize column by column (each column = 4096 elements = 16 blocks)
unsafe {
    for (int col = 0; col < 11008; col++) {
        float* srcCol = (float*)wFp32.Data + col; // Stride = 11008
        byte* dstCol = wQ4k.Data + col * 16 * 144; // Each column = 16 blocks × 144 bytes
        
        // Gather column into contiguous buffer
        float* temp = stackalloc float[4096];
        for (int i = 0; i < 4096; i++) temp[i] = srcCol[i * 11008];
        
        // Quantize
        GgmlQuant.Q4_K(temp, dstCol, 4096);
    }
}

// wQ4k now ready for inference
// Use in MulMat: GgmlCompute.Forward handles Q4_K dequantization automatically
```

---

## Dot Products with Quantized Types

GGML provides optimized dot products between quantized types. GgmlSharp exposes these via `GgmlTypeTraits` but the actual implementations are in the compute kernels.

| Pair | Kernel | Use Case |
|------|--------|----------|
| Q4_K × Q8_K | `ggml_vec_dot_q4_K_q8_K` | Weight × activation |
| Q5_K × Q8_K | `ggml_vec_dot_q5_K_q8_K` | Weight × activation |
| Q6_K × Q8_K | `ggml_vec_dot_q6_K_q8_K` | Weight × activation |
| IQ4_NL × Q8_0 | `ggml_vec_dot_iq4_nl_q8_0` | IQ weight × FP32 activation |
| MXFP4 × Q8_0 | `ggml_vec_dot_mxfp4_q8_0` | FP4 weight × FP32 activation |

In GgmlSharp, these are invoked automatically by `OpsMatMul.ForwardMulMat` when source types match.

---

## Migration from GGML C

| GGML C | GgmlSharp |
|--------|-----------|
| `ggml_type` | `GgmlType` (same values) |
| `ggml_type_traits` | `GgmlTypeTraits.Get(type)` |
| `ggml_to_float` | `traits.ToFloat` |
| `ggml_from_float` | `traits.FromFloatRef` / `GgmlQuant.*` |
| `block_q4_K` | Layout identical (byte-for-byte) |
| `quantize_row_q4_K` | `GgmlQuant.Q4_K` |
| `dequantize_row_q4_K` | `GgmlDequant.Q4_K` |
| `GGML_TYPE_Q4_K` | `GgmlType.Q4_K` |

All block layouts are **bit-for-bit compatible** with GGML C.