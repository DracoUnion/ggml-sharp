# GgmlSharp Examples

Practical usage examples for common patterns and workflows.

---

## Table of Contents

1. [Basic Tensor Operations](#basic-tensor-operations)
2. [Neural Network Layers](#neural-network-layers)
3. [Attention Mechanisms](#attention-mechanisms)
4. [Quantization Workflows](#quantization-workflows)
4. [Multi-threaded Execution](#multi-threaded-execution)
5. [Custom Graph Construction](#custom-graph-construction)
6. [Interop with GGML](#interop-with-ggml)

---

## Basic Tensor Operations

### Elementwise Arithmetic

```csharp
using var g = new GgmlGraph();

// [batch=2, seq=8, dim=64]
var a = g.NewTensor(GgmlType.F32, 64, 8, 2);
var b = g.NewTensor(GgmlType.F32, 64, 8, 2);

// Broadcast: [64,8,2] + [64,1,2] → [64,8,2]
var bias = g.NewTensor(GgmlType.F32, 64, 1, 2);

var add = g.Add(a, bias);      // a + bias (broadcast)
var sub = g.Sub(a, b);         // a - b
var mul = g.Mul(a, b);         // a * b (Hadamard)
var div = g.Div(a, b);         // a / b
```

### Reductions

```csharp
using var g = new GgmlGraph();

// Input: [64, 16, 4, 1] (dim, seq, heads, batch)
var x = g.NewTensor(GgmlType.F32, 64, 16, 4, 1);

// Sum all elements → scalar [1,1,1,1]
var sum = g.NewTensor(GgmlType.F32, 1, 1, 1, 1);
// Use OpsReduce directly for more control
var sumNode = g.Ctx.NewOp(GgmlOp.SUM, GgmlType.F32, new long[]{1,1,1,1}, x);

// Sum per row → [1, 16, 4, 1]
var sumRows = g.Ctx.NewOp(GgmlOp.SUM_ROWS, GgmlType.F32, new long[]{1, 16, 4, 1}, x);

// Mean per row → [1, 16, 4, 1]
var mean = g.Ctx.NewOp(GgmlOp.MEAN, GgmlType.F32, new long[]{1, 16, 4, 1}, x);

// Argmax per row → [1, 16, 4, 1] (I32)
var argmax = g.Ctx.NewOp(GgmlOp.ARGMAX, GgmlType.I32, new long[]{1, 16, 4, 1}, x);
```

### Normalization

```csharp
using var g = new GgmlGraph();

// [dim=512, seq=128, batch=4]
var x = g.NewTensor(GgmlType.F32, 512, 128, 1, 4);

// RMS Norm (used in LLaMA, Gemma, etc.)
var rms = g.RmsNorm(x, eps: 1e-6f);

// Layer Norm
var ln = g.Norm(x, eps: 1e-5f);

// Group Norm (num_groups in op_params, not yet exposed in Graph)
// Use low-level:
// var gn = g.Ctx.NewOp(GgmlOp.GROUP_NORM, GgmlType.F32, shape, x);
// GgmlTensor.SetOpParamsI32(gn, 0, num_groups);
```

### Activations

```csharp
using var g = new GgmlGraph();
var x = g.NewTensor(GgmlType.F32, 256, 16);

// Pointwise activations (all F32)
var silu = g.Silu(x);      // x * sigmoid(x)
var relu = g.Relu(x);      // max(0, x)
var gelu = g.Gelu(x);      // GELU (erf-based)
var geluQ = g.Ctx.NewUnaryOp(GgmlType.F32, new[]{256,16}, x, GgmlUnaryOp.GELU_QUICK);
var tanh = g.Tanh(x);
var exp  = g.Exp(x);

// Leaky ReLU (alpha in op_params)
var leaky = g.Ctx.NewOp(GgmlOp.LEAKY_RELU, GgmlType.F32, new[]{256,16}, x);
GgmlTensor.SetOpParamsF32(leaky, 0, 0.01f); // negative_slope
```

---

## Neural Network Layers

### Linear / Dense Layer

```csharp
// y = x @ W^T + b
// x: [batch, in_dim], W: [out_dim, in_dim], b: [out_dim]

using var g = new GgmlGraph();
int batch = 32, inDim = 512, outDim = 1024;

var x = g.NewTensor(GgmlType.F32, inDim, batch);
var w = g.NewTensor(GgmlType.F32, inDim, outDim);  // Note: GGML is column-major!
var b = g.NewTensor(GgmlType.F32, outDim);

// MulMat: x [in_dim, batch] * w [in_dim, out_dim] → [out_dim, batch]
var y = g.MulMat(w, x);  // w^T @ x

// Add bias (broadcast over batch)
var yBias = g.Add(y, b);

// Result: [out_dim, batch]
```

### Embedding Layer

```csharp
// Embedding lookup: indices [seq_len, batch] → [dim, seq_len, batch]

using var g = new GgmlGraph();
int vocabSize = 32000, dim = 4096, seqLen = 128, batch = 4;

// Weight matrix: [vocab, dim] (stored as [dim, vocab] in GGML)
var w = g.NewTensor(GgmlType.F32, dim, vocabSize);

// Indices: I32 [seq_len, batch]
var indices = g.NewTensor(GgmlType.I32, seqLen, batch);

// GET_ROWS: gather rows from weight matrix
var embed = g.Ctx.NewOp(GgmlOp.GET_ROWS, GgmlType.F32, new long[]{dim, seqLen, batch}, w, indices);
```

### LayerNorm + Residual

```csharp
// residual + LayerNorm(x)

using var g = new GgmlGraph();
var x = g.NewTensor(GgmlType.F32, 512, 64, 8);  // [dim, seq, heads]
var residual = g.NewTensor(GgmlType.F32, 512, 64, 8);

// Add residual
var added = g.Add(x, residual);

// LayerNorm
var ln = g.Norm(added, 1e-5f);
```

---

## Attention Mechanisms

### Scaled Dot-Product Attention (Simplified)

```csharp
// Q, K, V: [dim, seq, heads, batch]
// attn = softmax(Q @ K^T / sqrt(d)) @ V

using var g = new GgmlGraph();
int dim = 64, seq = 128, heads = 8, batch = 2;
float scale = 1.0f / MathF.Sqrt(dim);

var q = g.NewTensor(GgmlType.F32, dim, seq, heads, batch);
var k = g.NewTensor(GgmlType.F32, dim, seq, heads, batch);
var v = g.NewTensor(GgmlType.F32, dim, seq, heads, batch);

// Q @ K^T → [seq, seq, heads, batch]
var qk = g.MulMat(q, k);

// Scale
var scaled = g.Scale(qk, scale);

// SoftMax per row (over key dimension)
var attn = g.SoftMax(scaled, scale: 1.0f);

// attn @ V → [dim, seq, heads, batch]
// Need to transpose attn for MulMat: [seq, seq] @ [dim, seq]^T
// This is simplified; real impl uses custom kernels
```

### RoPE-Enhanced Attention

```csharp
// Apply RoPE to Q and K before attention

using var g = new GgmlGraph();
int dim = 64, seq = 128, heads = 8, batch = 2;

var q = g.NewTensor(GgmlType.F32, dim, seq, heads, batch);
var k = g.NewTensor(GgmlType.F32, dim, seq, heads, batch);

// Positions: [seq] = 0, 1, 2, ...
int[] positions = Enumerable.Range(0, seq).ToArray();

// Apply RoPE (NEOX style, 64 dims)
var qRope = g.Rope(q, positions, nDims: dim, mode: GgmlRopeType.NEOX);
var kRope = g.Rope(k, positions, nDims: dim, mode: GgmlRopeType.NEOX);

// Continue with attention...
```

### Multi-Head Attention with MROPE (LLaVA-style)

```csharp
// MROPE: separate position encodings for time, height, width, extra

using var g = new GgmlGraph();
int dim = 64, seq = 576, heads = 8, batch = 1;  // 24x24 patches

var q = g.NewTensor(GgmlType.F32, dim, seq, heads, batch);
var k = g.NewTensor(GgmlType.F32, dim, seq, heads, batch);

// MROPE positions: [t, h, w, e] each [seq]
// For 24x24 grid: t = frame idx, h = row, w = col, e = 0
int[] posT = new int[seq];
int[] posH = new int[seq];
int[] posW = new int[seq];
int[] posE = new int[seq];
for (int i = 0; i < seq; i++) {
    posT[i] = 0;           // single frame
    posH[i] = i / 24;      // row
    posW[i] = i % 24;      // col
    posE[i] = 0;
}

// Interleave into single array: [t0, h0, w0, e0, t1, h1, w1, e1, ...]
int[] mropePos = new int[seq * 4];
for (int i = 0; i < seq; i++) {
    mropePos[i * 4 + 0] = posT[i];
    mropePos[i * 4 + 1] = posH[i];
    mropePos[i * 4 + 2] = posW[i];
    mropePos[i * 4 + 3] = posE[i];
}

// Sections: [time_dims, height_dims, width_dims, extra_dims]
// For standard MROPE with dim=64: 16 dims each
// Pass via Graph.Rope (sets op_params[11..14] = sections)
var qRope = g.Rope(q, mropePos, nDims: dim, mode: GgmlRopeType.MROPE, freqBase: 10000f);
// Note: sections must be set manually on the tensor op_params
// after Rope() returns, or use low-level API for full control
```

---

## Quantization Workflows

### Quantize Float Weights (Host Side)

```csharp
using GgmlSharp.Kernel;

// Float weights: [out_dim, in_dim] = [1024, 4096]
float[] floatWeights = LoadWeights(); // 1024 * 4096 floats

// Quantize to Q4_K (4-bit K-quant, 256 elements per block)
int blockSize = 256;
int blocks = (1024 * 4096) / blockSize;
byte[] q4kData = new byte[blocks * 144]; // 144 bytes per Q4_K block

unsafe {
    fixed (float* src = floatWeights)
    fixed (byte* dst = q4kData) {
        // Process row by row (each row = 4096 elements = 16 blocks)
        for (int row = 0; row < 1024; row++) {
            float* rowPtr = src + row * 4096;
            byte* dstPtr = dst + row * 16 * 144;
            GgmlQuant.Q4_K(rowPtr, dstPtr, 4096); // Quantize 4096 elements
        }
    }
}

// Create tensor with quantized data
using var ctx = new GgmlContext();
var w = ctx.NewTensor(GgmlType.Q4_K, 2, new long[]{4096, 1024}, noAlloc: true);
unsafe {
    Buffer.MemoryCopy(q4kData, w.Data, q4kData.Length, q4kData.Length);
}
// w.Data now points to quantized weights
```

### Dequantize for Computation

```csharp
// Kernels automatically dequantize via GgmlTypeTraits.ToFloat
// No manual dequantize needed for most ops

// But you can dequantize manually:
using var ctx = new GgmlContext();
var q = ctx.NewTensor(GgmlType.Q4_0, 1, new long[]{32}); // 1 block
// ... fill q.Data with Q4_0 data ...

var f = ctx.NewTensor(GgmlType.F32, 1, new long[]{32});
unsafe {
    GgmlTypeTraits.Get(GgmlType.Q4_0).ToFloat!(q.Data, (float*)f.Data, 32);
}
// f.Data now has 32 floats
```

### Mixed Precision: Quantized Weights, Float Activations

```csharp
// Common pattern: Q4_K weights, F32 activations
// MulMat kernel handles F32 @ Q4_K automatically (dequantizes on-the-fly)

using var g = new GgmlGraph();
int batch = 1, inDim = 4096, outDim = 11008;  // LLaMA 7B FFN

// Activation: F32 [in_dim, batch]
var x = g.NewTensor(GgmlType.F32, inDim, batch);

// Weight: Q4_K [in_dim, out_dim] (GGML layout: column-major)
var w = g.NewTensor(GgmlType.Q4_K, inDim, outDim, noAlloc: true);
// ... fill w.Data with Q4_K data ...

// MulMat: F32 @ Q4_K → F32 [out_dim, batch]
// Kernel dequantizes Q4_K blocks to F32 on-the-fly
var y = g.MulMat(w, x);
```

---

## Multi-threaded Execution

### Parallel Graph Execution

```csharp
using var g = new GgmlGraph();

// Large tensors
var a = g.NewTensor(GgmlType.F32, 4096, 1024);
var b = g.NewTensor(GgmlType.F32, 4096, 1024);

var c = g.Add(a, b);

// Execute with 8 threads
g.Execute(nth: 8);

// Each kernel splits its rows across 8 threads internally
```

### Thread-Local Scratch Buffers

```csharp
// Some ops (e.g., quantized MulMat) need scratch space
// Provide via ComputeParams.WorkData

using var ctx = new GgmlContext();
var a = ctx.NewTensor(GgmlType.F32, 4096, 1024);
var b = ctx.NewTensor(GgmlType.Q4_K, 4096, 1024);
var c = ctx.NewOp(GgmlOp.MUL_MAT, GgmlType.F32, new[]{1024, 1024}, a, b);

// Scratch buffer: typically 1-2 MB per thread
int nth = 8;
long workSize = 1024 * 1024; // 1 MB
var workBuffers = new byte[nth][];
for (int i = 0; i < nth; i++) workBuffers[i] = new byte[workSize];

// Execute each thread
Parallel.For(0, nth, ith => {
    unsafe {
        fixed (byte* workPtr = workBuffers[ith]) {
            var p = new ComputeParams(ith, nth, workSize, workPtr);
            GgmlCompute.Forward(p, c);
        }
    }
});
```

---

## Custom Graph Construction

### Building a Transformer Block

```csharp
// Pre-LN Transformer Block:
// x = x + Attention(LayerNorm(x))
// x = x + FFN(LayerNorm(x))

using var g = new GgmlGraph();
int dim = 512, seq = 128, heads = 8, batch = 2;

var x = g.NewTensor(GgmlType.F32, dim, seq, batch); // Input

// ---- Attention Block ----
var ln1 = g.Norm(x, 1e-5f);

// QKV projections (combined)
var qkvW = g.NewTensor(GgmlType.F32, dim, dim * 3);
var qkv = g.MulMat(qkvW, ln1); // [3*dim, seq, batch]

// Split Q, K, V
// (In practice, use GET_ROWS or VIEW ops)
// Simplified:
var q = g.View(qkv, new[]{dim, seq, batch});           // [dim, seq, batch]
var k = g.View(qkv, new[]{dim, seq, batch});           // offset = dim * seq
var v = g.View(qkv, new[]{dim, seq, batch});           // offset = 2*dim * seq

// RoPE on Q, K
int[] pos = Enumerable.Range(0, seq).ToArray();
var qRope = g.Rope(q, pos, nDims: dim/heads, mode: GgmlRopeType.NEOX);
var kRope = g.Rope(k, pos, nDims: dim/heads, mode: GgmlRopeType.NEOX);

// Attention scores: Q @ K^T
var scores = g.MulMat(qRope, kRope); // [seq, seq, batch]
var scaled = g.Scale(scores, 1.0f / MathF.Sqrt(dim / heads));
var attn = g.SoftMax(scaled);

// Output: attn @ V
var attnOut = g.MulMat(attn, v); // [dim, seq, batch]

// Residual
var x1 = g.Add(x, attnOut);

// ---- FFN Block ----
var ln2 = g.Norm(x1, 1e-5f);

// SwiGLU FFN (LLaMA style)
// up = x @ W_up, gate = x @ W_gate, down = SwiGLU(up, gate) @ W_down
var wUp = g.NewTensor(GgmlType.F32, dim, dim * 4);
var wGate = g.NewTensor(GgmlType.F32, dim, dim * 4);
var wDown = g.NewTensor(GgmlType.F32, dim * 4, dim);

var up = g.MulMat(wUp, ln2);    // [4*dim, seq, batch]
var gate = g.MulMat(wGate, ln2);

// Build GLU node for SwiGLU
var glu = g.Ctx.NewOp(GgmlOp.GLU, GgmlType.F32, new[]{4*dim, seq, batch}, up, gate);
GgmlTensor.SetOpParamsI32(glu, 0, (int)GgmlGluOp.SWIGLU);
GgmlTensor.SetOpParamsI32(glu, 1, 0);

var ffnOut = g.MulMat(wDown, glu); // [dim, seq, batch]

// Final residual
var output = g.Add(x1, ffnOut);

g.Execute(nth: Environment.ProcessorCount);
```

---

## Interop with GGML

### Loading GGML Model Weights

```csharp
// GGML model files (.gguf) can be parsed and weights loaded into GgmlSharp tensors
// This is a conceptual example — real implementation needs GGUF parser

public static void LoadGGMLWeights(string ggufPath, GgmlContext ctx, Dictionary<string, GgmlTensor> tensors) {
    // 1. Parse GGUF file (use GGUF library or custom parser)
    // 2. For each tensor:
    //    - Read name, shape, type (GGML_TYPE enum)
    //    - Map GGML_TYPE → GgmlType (same numeric values)
    //    - Create tensor: ctx.NewTensor(type, shape, noAlloc: true)
    //    - Copy raw data: Buffer.MemoryCopy(ggufData, tensor.Data, ...)
    // 3. Store in dictionary for graph building
}

// Usage
using var ctx = new GgmlContext(1L << 30); // 1 GB for large models
var weights = new Dictionary<string, GgmlTensor>();
LoadGGMLWeights("model.gguf", ctx, weights);

// Build graph using loaded weights
using var g = new GgmlGraph(1L << 30);
g.Ctx = ctx; // Share context (advanced: graph owns context by default)

var x = g.NewTensor(GgmlType.F32, 4096, 1);
var w = weights["blk.0.attn_q.weight"];
var q = g.MulMat(w, x);
// ...
```

### Exporting to GGML Format

```csharp
// To export GgmlSharp tensors to GGML-compatible format:
// 1. Ensure tensor data is in GGML layout (column-major, same quantization)
// 2. Write GGUF header + tensor metadata + raw data
// 3. Quantization types match exactly (same enum values, block layouts)

public static void SaveAsGGUF(GgmlContext ctx, Dictionary<string, GgmlTensor> tensors, string outputPath) {
    using var fs = File.Create(outputPath);
    using var bw = new BinaryWriter(fs);
    
    // Write GGUF header (magic, version, tensor count, kv count)
    // Write metadata (architecture, context length, etc.)
    // Write tensor info (name, dims, type, offset)
    // Write tensor data
    
    foreach (var (name, t) in tensors) {
        long nElements = t.NElements();
        long byteSize = 0;
        switch (t.Type) {
            case GgmlType.F32: byteSize = nElements * 4; break;
            case GgmlType.F16: byteSize = nElements * 2; break;
            case GgmlType.Q4_0: byteSize = (nElements / 32) * 18; break;
            // ... other types
        }
        
        // Write tensor metadata
        bw.Write(name);
        bw.Write(t.Ne.Length); // n_dims
        foreach (var dim in t.Ne) bw.Write(dim);
        bw.Write((uint)t.Type); // GGML_TYPE
        bw.Write(0); // offset (filled later)
        
        // Write data
        unsafe {
            byte[] buf = new byte[byteSize];
            Marshal.Copy((IntPtr)t.Data, buf, 0, (int)byteSize);
            bw.Write(buf);
        }
    }
}
```

---

## Performance Tips

1. **Reuse contexts/graphs** — Arena allocation is expensive; reuse for multiple forward passes
2. **Batch operations** — Larger tensors = better SIMD utilization
3. **Use appropriate quantization** — Q4_K for weights, F32 for activations
4. **Thread count** — Set `nth` to physical core count (not logical)
5. **Avoid small ops** — Fuse pointwise ops where possible (e.g., `Add` + `Mul` → `Scale` + `Add`)

```csharp
// Good: single large matmul
var y = g.MulMat(largeW, largeX);

// Avoid: many small matmuls in loop
foreach (var w in smallWeights) {
    var y = g.MulMat(w, x); // Slow!
}
```