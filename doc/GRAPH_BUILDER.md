# Graph Builder API Guide

Comprehensive guide to `GgmlGraph` — the high-level, graph-oriented API for building and executing computation graphs.

---

## Overview

`GgmlGraph` is the primary entry point for most users. It provides:

- **Automatic memory management** — wraps `GgmlContext` (arena allocator)
- **Declarative op builders** — methods like `Add`, `MulMat`, `RmsNorm` that create and wire nodes
- **Ordered execution** — nodes execute in construction order via `Execute(nth)`
- **Threaded execution** — automatic work splitting across threads

```csharp
using var g = new GgmlGraph();           // 1 MiB arena
using var g = new GgmlGraph(1 << 25);    // 32 MiB arena for large models
```

---

## Graph Construction Patterns

### 1. Leaf Tensors (Inputs)

```csharp
// Shape: [ne0, ne1, ne2, ne3] — innermost = ne[0]
// Column-major: ne[0] contiguous, ne[1] rows, ne[2] depth, ne[3] batch

var x = g.NewTensor(GgmlType.F32, 512);              // [512] vector
var x = g.NewTensor(GgmlType.F32, 512, 128);         // [512, 128] matrix
var x = g.NewTensor(GgmlType.F32, 512, 128, 8);      // [512, 128, 8] 3D
var x = g.NewTensor(GgmlType.F32, 512, 128, 8, 2);   // [512, 128, 8, 2] 4D (batch=2)

// With source dependencies (for ops that need them)
var y = g.NewTensor(GgmlType.F32, 512, 128, x);      // Same shape, src[0] = x
```

### 2. Elementwise Operations

```csharp
var a = g.NewTensor(GgmlType.F32, 256, 64);
var b = g.NewTensor(GgmlType.F32, 256, 64);

// Broadcast rules: dimensions must match or be 1
var add = g.Add(a, b);      // [256,64] + [256,64] → [256,64]
var sub = g.Sub(a, b);
var mul = g.Mul(a, b);      // Hadamard product
var div = g.Div(a, b);

// Broadcast examples:
// [256,64] + [256,1] → [256,64] (broadcast over rows)
// [256,64] + [1,64]  → [256,64] (broadcast over cols)
// [256,64] + [1,1]   → [256,64] (broadcast both)
```

### 3. Matrix Multiplication

```csharp
// MulMat: dst = aᵀ @ b
// a: [K, N], b: [K, M] → dst: [N, M]
// Batches (ne[2], ne[3]) broadcast

var a = g.NewTensor(GgmlType.F32, 64, 256);   // [K=64, N=256]
var b = g.NewTensor(GgmlType.F32, 64, 128);   // [K=64, M=128]
var c = g.MulMat(a, b);                        // [N=256, M=128]

// With batches:
// a: [64, 256, 8, 2], b: [64, 128, 8, 2] → c: [256, 128, 8, 2]

// OUT_PROD: dst[i,j] = sum_k a[i,k] * b[j,k] (outer product summed)
var outProd = g.Ctx.NewOp(GgmlOp.OUT_PROD, GgmlType.F32, new[]{N, M}, a, b);
```

### 4. Normalization

```csharp
var x = g.NewTensor(GgmlType.F32, 512, 128, 8);

// RMS Norm (LLaMA, Gemma, Mistral, etc.)
// eps in op_params[0]
var rms = g.RmsNorm(x, 1e-6f);

// Layer Norm (BERT, GPT-2, etc.)
// eps in op_params[0]
var ln = g.Norm(x, 1e-5f);

// Group Norm (num_groups in op_params[0] - low-level only)
var gn = g.Ctx.NewOp(GgmlOp.GROUP_NORM, GgmlType.F32, x.Shape, x);
GgmlTensor.SetOpParamsI32(gn, 0, 32); // 32 groups
```

### 5. Activations

```csharp
var x = g.NewTensor(GgmlType.F32, 256, 64);

// Graph-level (F32 only)
var silu = g.Silu(x);
var relu = g.Relu(x);
var gelu = g.Gelu(x);
var tanh = g.Tanh(x);
var exp  = g.Exp(x);

// Low-level: all GgmlUnaryOp variants
var unary = g.Ctx.NewUnaryOp(GgmlType.F32, x.Shape, x, GgmlUnaryOp.GELU_QUICK);
var unary = g.Ctx.NewUnaryOp(GgmlType.F32, x.Shape, x, GgmlUnaryOp.HARDSWISH);
```

### 6. SoftMax

```csharp
var x = g.NewTensor(GgmlType.F32, 128, 64); // [seq=128, heads=64]

// scale in op_params[0], mask in op_params[1] (0 = no mask)
var sm = g.SoftMax(x, scale: 1.0f);

// Masked softmax (causal):
// Requires low-level op_params setup
var smMasked = g.Ctx.NewOp(GgmlOp.SOFT_MAX, GgmlType.F32, x.Shape, x);
GgmlTensor.SetOpParamsF32(smMasked, 0, 1.0f); // scale
GgmlTensor.SetOpParamsF32(smMasked, 1, 0.0f); // no mask (or implement mask separately)
```

### 7. RoPE (Rotary Positional Embedding)

```csharp
var x = g.NewTensor(GgmlType.F32, 64, 128, 8, 1); // [dim, seq, heads, batch]

int[] positions = Enumerable.Range(0, 128).ToArray();

// NORMAL: standard RoPE (adjacent pairs)
var rope = g.Rope(x, positions, nDims: 64, mode: GgmlRopeType.NORMAL);

// NEOX: GPT-NeoX style (interleaved pairs: 0,32,1,33,...)
var ropeNeox = g.Rope(x, positions, nDims: 64, mode: GgmlRopeType.NEOX);

// MROPE: Multi-modal (LLaVA) - positions = [t,h,w,e] interleaved
// IMROPE: Interleaved MROPE (Qwen3-VL)
// VISION: Vision encoder RoPE
// See EXAMPLES.md for MROPE position array construction
```

### 8. GLU (Gated Linear Units)

```csharp
var x = g.NewTensor(GgmlType.F32, 512, 64);
var gate = g.NewTensor(GgmlType.F32, 512, 64);

// Low-level: Graph doesn't expose GLU builder yet
var glu = g.Ctx.NewOp(GgmlOp.GLU, GgmlType.F32, new[]{512, 64}, x, gate);
GgmlTensor.SetOpParamsI32(glu, 0, (int)GgmlGluOp.SWIGLU); // REGLU, GEGLU, SWIGLU, etc.
GgmlTensor.SetOpParamsI32(glu, 1, 0); // swapped = 0 (x is gate, gate is value)

// SWIGLU_OAI parameters (alpha, limit)
GgmlTensor.SetOpParamsF32(glu, 2, 1.0f); // alpha
GgmlTensor.SetOpParamsF32(glu, 3, 10.0f); // limit
```

### 9. Remap Operations

```csharp
var src = g.NewTensor(GgmlType.F32, 64, 32);

// VIEW: alias with new shape (no copy)
var view = g.View(src, new long[]{32, 64}); // Transpose via view

// REPEAT: tile to larger shape
var repeat = g.Repmat(src, new long[]{64, 64}); // [64,32] → [64,64]

// DUP/CPY/CONT: copy with type conversion
var dup = g.Ctx.NewOp(GgmlOp.DUP, GgmlType.F16, src.Shape, src); // F32 → F16

// SET: copy src0 then overwrite slice from src1
var set = g.Ctx.NewOp(GgmlOp.SET, GgmlType.F32, dstShape, src0, src1);

// PAD: zero or circular padding
var pad = g.Ctx.NewOp(GgmlOp.PAD, GgmlType.F32, dstShape, src);
GgmlTensor.SetOpParamsI32(pad, 8, 1); // circular = 1

// ROLL: shift elements
var roll = g.Ctx.NewOp(GgmlOp.ROLL, GgmlType.F32, src.Shape, src);
GgmlTensor.SetOpParamsI32(roll, 0, shift); // shift amount

// ARANGE: fill with 0,1,2...
var arange = g.Ctx.NewOp(GgmlOp.ARANGE, GgmlType.F32, new[]{100});

// FILL: fill with scalar
var fill = g.Ctx.NewOp(GgmlOp.FILL, GgmlType.F32, new[]{64,64});
GgmlTensor.SetOpParamsF32(fill, 0, 3.14f); // value

// TRI: triangular matrix
var tri = g.Ctx.NewOp(GgmlOp.TRI, GgmlType.F32, new[]{64,64});
GgmlTensor.SetOpParamsI32(tri, 0, (int)GgmlTriType.LOWER);
```

---

## OpParams Reference

Each op stores parameters in `tensor.OpParams` (byte[64]). Use `GgmlTensor.SetOpParamsI32/F32` / `GetOpParamsI32/F32`.

| Op | Param Index | Type | Meaning |
|------|-------------|------|---------|
| `RMS_NORM` | 0 | f32 | `eps` |
| `NORM` | 0 | f32 | `eps` |
| `SOFT_MAX` | 0 | f32 | `scale` |
| `SOFT_MAX` | 1 | f32 | `mask` (0/1) |
| `SCALE` | 0 | f32 | `scale` |
| `SCALE` | 1 | f32 | `bias` |
| `LEAKY_RELU` | 0 | f32 | `negative_slope` |
| `ROPE` | 0 | i32 | `n_past` (unused) |
| `ROPE` | 1 | i32 | `n_dims` (rotation dims) |
| `ROPE` | 2 | i32 | `mode` (GgmlRopeType) |
| `ROPE` | 4 | i32 | `n_ctx_orig` |
| `ROPE` | 5 | f32 | `freq_base` |
| `ROPE` | 6 | f32 | `freq_scale` |
| `ROPE` | 7 | f32 | `ext_factor` |
| `ROPE` | 8 | f32 | `attn_factor` |
| `ROPE` | 9 | f32 | `beta_fast` |
| `ROPE` | 10 | f32 | `beta_slow` |
| `ROPE` | 11-14 | i32 | `sections[4]` (MROPE/IMROPE) |
| `GLU` | 0 | i32 | `glu_op` (GgmlGluOp) |
| `GLU` | 1 | i32 | `swapped` (0/1) |
| `GLU` (SWIGLU_OAI) | 2 | f32 | `alpha` |
| `GLU` (SWIGLU_OAI) | 3 | f32 | `limit` |
| `GROUP_NORM` | 0 | i32 | `num_groups` |
| `PAD` | 8 | i32 | `circular` (0/1) |
| `ROLL` | 0 | i32 | `shift` |
| `FILL` | 0 | f32 | `value` |
| `TRI` | 0 | i32 | `GgmlTriType` |
| `UPSCALE` | 0 | i32 | `GgmlScaleMode` |
| `UPSCALE` | 1 | i32 | `flags` (GgmlScaleFlag) |

---

## Execution

### Single-Threaded

```csharp
g.Execute(); // nth = 1 (default)
```

### Multi-Threaded

```csharp
// Use all available cores
g.Execute(nth: Environment.ProcessorCount);

// Or fixed count
g.Execute(nth: 8);
```

**Threading model**: Each call to `Execute(nth)` runs the **entire graph** `nth` times (once per thread). Each kernel splits its output rows across threads using `ComputeParams.ThreadRowRange`. This matches GGML's per-thread graph compute.

```csharp
// Equivalent manual execution:
for (int ith = 0; ith < nth; ith++) {
    var p = new ComputeParams(ith, nth);
    foreach (var node in g.Nodes) {
        if (node.Op != GgmlOp.NONE) {
            GgmlCompute.Forward(p, node);
        }
    }
}
```

### Thread-Local Scratch

Some ops (quantized MulMat) need scratch space:

```csharp
int nth = 8;
long workSize = 1024 * 1024; // 1 MB per thread
var workBuffers = new byte[nth][];

Parallel.For(0, nth, ith => {
    workBuffers[ith] = new byte[workSize];
    unsafe {
        fixed (byte* workPtr = workBuffers[ith]) {
            var p = new ComputeParams(ith, nth, workSize, workPtr);
            foreach (var node in g.Nodes) {
                if (node.Op != GgmlOp.NONE) {
                    GgmlCompute.Forward(p, node);
                }
            }
        }
    }
});
```

---

## Advanced Patterns

### 1. Sharing Context Between Graphs

```csharp
// Graph owns its context by default
using var g1 = new GgmlGraph();
using var g2 = new GgmlGraph();

// To share weights: create context separately, then assign
using var sharedCtx = new GgmlContext(1L << 30);
var w = sharedCtx.NewTensor(GgmlType.F32, 4096, 11008);

// Graph 1
using var g1 = new GgmlGraph();
g1.Ctx = sharedCtx; // Replace graph's context
var x1 = g1.NewTensor(GgmlType.F32, 4096, 1);
var y1 = g1.MulMat(w, x1);

// Graph 2 (same weights)
using var g2 = new GgmlGraph();
g2.Ctx = sharedCtx;
var x2 = g2.NewTensor(GgmlType.F32, 4096, 1);
var y2 = g2.MulMat(w, x2);
```

### 2. External Data (No Alloc)

```csharp
// Pre-allocated buffer (e.g., from memory-mapped file)
byte[] weightData = File.ReadAllBytes("weights.bin");
unsafe {
    fixed (byte* ptr = weightData) {
        var w = g.NewTensor(GgmlType.F32, 4096, 11008, noAlloc: true, data: ptr);
        // w.Data == ptr, no arena allocation
    }
}
```

### 3. Dynamic Shapes

```csharp
// Reshape via VIEW (no copy)
var x = g.NewTensor(GgmlType.F32, 512, 64); // [512, 64]
var y = g.View(x, new long[]{64, 512});     // [64, 512] — transposed view

// SetShape on existing tensor (updates Ne, recomputes Nb)
var z = g.NewTensor(GgmlType.F32, 256, 32);
z.SetShape(2, 128, 64); // Now [128, 64]
z.ComputeStrides();
```

### 4. Conditional Graph Building

```csharp
// Build different paths based on config
using var g = new GgmlGraph();
var x = g.NewTensor(GgmlType.F32, 512, 128);

GgmlTensor output;
if (useRMSNorm) {
    output = g.RmsNorm(x, 1e-6f);
} else {
    output = g.Norm(x, 1e-5f);
}

if (useActivation) {
    output = g.Silu(output);
}

// Both paths exist in graph; only executed path computes
// (unexecuted nodes have Op = NONE or are skipped)
g.Execute();
```

### 5. Gradient / Backward Pass (Manual)

```csharp
// Forward
var x = g.NewTensor(GgmlType.F32, 256, 64);
var w = g.NewTensor(GgmlType.F32, 256, 512);
var y = g.MulMat(w, x);
var loss = g.Ctx.NewOp(GgmlOp.SUM, GgmlType.F32, new[]{1}, y);

g.Execute();

// Backward (manual — no autograd)
// 1. Set loss gradient = 1
// 2. Traverse graph in reverse, call backward kernels
// 3. Accumulate gradients into weight tensors
// (Requires implementing backward kernels for each op)
// See: GGML's ggml_backward API for reference
```

---

## Debugging

### Inspect Graph

```csharp
using var g = new GgmlGraph();
var a = g.NewTensor(GgmlType.F32, 64, 32);
var b = g.NewTensor(GgmlType.F32, 64, 32);
var c = g.Add(a, b);
var d = g.MulMat(a, b);

Console.WriteLine($"Nodes: {g.Nodes.Count}");
foreach (var (i, node) in g.Nodes.Select((n, i) => (i, n))) {
    Console.WriteLine($"  [{i}] {node.Op} {node.Type} [{string.Join(",", node.Ne)}]");
    if (node.Src[0] != null) {
        Console.WriteLine($"       src0: {node.Src[0].Op} [{string.Join(",", node.Src[0].Ne)}]");
    }
}
```

### Inspect Tensor Data

```csharp
unsafe {
    var ptr = (float*)tensor.Data;
    for (int i = 0; i < Math.Min(10, tensor.NElements()); i++) {
        Console.Write($"{ptr[i]:F4} ");
    }
    Console.WriteLine();
}
```

### NaN/Inf Checking

```csharp
// Enable in DEBUG builds (already in some kernels)
#ifndef NDEBUG
for (int k = 0; k < nc; k++) {
    float x = ((float*)dstPtr)[k];
    Debug.Assert(!float.IsNaN(x) && !float.IsInfinity(x));
}
#endif
```

---

## Performance Tips

| Tip | Why |
|-----|-----|
| **Reuse `GgmlGraph`** | Arena allocation is expensive; create once, run many times |
| **Prefer `MulMat` over loop of small matmuls** | Large GEMM = better SIMD utilization |
| **Use `Q4_K` weights + `F32` activations** | Best quality/speed tradeoff; kernel dequantizes on-the-fly |
| **Set `nth` to physical cores** | Avoid hyperthreading contention |
| **Fuse pointwise ops** | `Add` + `Mul` → single kernel where possible |
| **Avoid `View` for transposes in hot path** | `MulMat` expects specific layouts; use `TRANSPOSE` op if needed |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `NotImplementedException: op XYZ` | Op not ported yet | Use low-level `Ctx.NewOp` + check `GgmlCompute.Forward` |
| `AccessViolationException` | Tensor data not allocated | Use `noAlloc: false` (default) or provide valid `data` pointer |
| `InvalidOperationException: shape mismatch` | Broadcast rules violated | Check `ne[]` compatibility |
| `ObjectDisposedException` | Used graph after `Dispose()` | Keep graph alive for execution lifetime |
| `OutOfMemoryException` | Arena too small | Increase `reserveBytes` in constructor |