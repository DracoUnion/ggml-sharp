# Getting Started with GgmlSharp

GgmlSharp is a from-scratch C# reimplementation of the [GGML](https://github.com/ggerganov/ggml) CPU compute runtime. It mirrors the enums, structs, and op semantics of `ggml.h` / `ggml-cpu.c` 1:1, but replaces the linear C arena with managed objects while keeping raw `byte*` data buffers for compute kernels.

---

## Requirements

- **.NET 7.0 SDK** (or later — the project targets `net7.0` but works on net8+/net10)
- **Windows x64 / Linux x64 / macOS x64** (AVX2/FMA kernels are x86-64 only; scalar fallback works everywhere)

---

## Installation

### Clone & Build

```bash
git clone https://github.com/DracoUnion/ggml-sharp.git
cd ggml-sharp
dotnet build GgmlSharp.sln
```

### Run Tests

```bash
dotnet test tests/GgmlSharp.Tests/GgmlSharp.Tests.csproj
```

Expected output: **117 tests passing**.

---

## Project Structure

```
GgmlSharp.sln
src/GgmlSharp/               # Library
  Graph/
    GgmlGraph.cs             # High-level graph builder + executor
  Kernel/                    # Low-level tensor/op runtime
    GgmlContext.cs           # Arena allocator + tensor factory
    GgmlTensor.cs            # Tensor descriptor
    GgmlType.cs              # enum ggml_type (42 values)
    GgmlOp.cs                # enum ggml_op (96 values)
    GgmlUnaryOp.cs           # enum ggml_unary_op (22 values)
    GgmlGluOp.cs             # enum ggml_glu_op (6 values)
    GgmlTypeTraits.cs        # Per-type traits + to/from-float codecs
    GgmlCompute.cs           # Top-level forward dispatch
    ComputeParams.cs         # Per-thread compute params
    GgmlMath.cs              # fp16/bf16 conversions, erf
    GgmlDequant.cs           # Dequantize (to_float) row kernels
    GgmlQuant.cs             # Quantize (from_float) row kernels
    Ops.*.cs                 # Per-op kernels
tests/GgmlSharp.Tests/       # xUnit tests
```

---

## Quick Start

### 1. Basic Graph Construction

```csharp
using GgmlSharp.Graph;
using GgmlSharp.Kernel;

// Create a graph with 1 MiB arena
using var g = new GgmlGraph();

// Leaf tensors: F32, shape [rows=4, cols=4] (innermost = ne[0])
var a = g.NewTensor(GgmlType.F32, 4, 4);
var b = g.NewTensor(GgmlType.F32, 4, 4);

// Fill with data (example: a = 1..16, b = 16..1)
unsafe {
    var pa = (float*)a.Data;
    var pb = (float*)b.Data;
    for (int i = 0; i < 16; i++) {
        pa[i] = i + 1;
        pb[i] = 16 - i;
    }
}

// Build ops on the graph
var c = g.Add(a, b);              // Elementwise add
var s = g.RmsNorm(c, 1e-5f);      // RMS norm per row
var m = g.MulMat(a, b);           // aᵀ·b → [4,4]

// Execute (single-threaded)
g.Execute(nth: 1);

// Read results
unsafe {
    var ps = (float*)s.Data;
    var pm = (float*)m.Data;
    Console.WriteLine($"RMS norm output[0] = {ps[0]:F4}");
    Console.WriteLine($"MatMul output[0,0] = {pm[0]:F4}");
}
```

### 2. Using the Low-Level Kernel API

```csharp
using GgmlSharp.Kernel;

// Create context (arena)
using var ctx = new GgmlContext(1 << 20);

// Create tensors directly
var a = ctx.NewTensor(GgmlType.F32, 2, new long[] { 4, 4 });
var b = ctx.NewTensor(GgmlType.F32, 2, new long[] { 4, 4 });

// Fill data
unsafe {
    var pa = (float*)a.Data;
    var pb = (float*)b.Data;
    for (int i = 0; i < 16; i++) { pa[i] = i + 1; pb[i] = 16 - i; }
}

// Build op node: c = a + b
var c = ctx.NewOp(GgmlOp.ADD, GgmlType.F32, new long[] { 4, 4 }, a, b);

// Compute params: single-threaded
var p = new ComputeParams(ith: 0, nth: 1);

// Execute
GgmlCompute.Forward(p, c);

// Result in c.Data
```

### 3. RoPE (Rotary Positional Embedding)

```csharp
using var g = new GgmlGraph();

// Input: [head_dim=64, seq_len=16, heads=8, batch=1]
var x = g.NewTensor(GgmlType.F32, 64, 16, 8, 1);

// Positions: [seq_len] = 0..15
int[] positions = Enumerable.Range(0, 16).ToArray();

// Apply NEOX-style RoPE on 64 dims
var rope = g.Rope(x, positions, nDims: 64, mode: GgmlRopeType.NEOX, freqBase: 10000f);

g.Execute();
```

### 4. GLU (Gated Linear Units)

```csharp
using var ctx = new GgmlContext(1 << 20);

// Input: [64], gate: [64] (or single [128] split in half)
var x = ctx.NewTensor(GgmlType.F32, 1, new long[] { 64 });
var g = ctx.NewTensor(GgmlType.F32, 1, new long[] { 64 });

// Fill...

// Build GLU node (SWIGLU)
var glu = ctx.NewOp(GgmlOp.GLU, GgmlType.F32, new long[] { 64 }, x, g);
// op_params[0] = glu_op (GgmlGluOp), [1] = swapped (0/1)
GgmlTensor.SetOpParamsI32(glu, 0, (int)GgmlGluOp.SWIGLU);
GgmlTensor.SetOpParamsI32(glu, 1, 0);

var p = new ComputeParams(0, 1);
GgmlCompute.Forward(p, glu);
// Result in glu.Data
```

---

## Key Concepts

### Tensor Layout (Column-Major)

GGML uses **column-major** (Fortran-style) layout:

- `ne[0]` = innermost dimension (contiguous in memory)
- `ne[1]` = rows
- `ne[2]` = depth / channels
- `ne[3]` = batch

Strides (`Nb[]`) are in **bytes**:
- `Nb[0]` = element size (e.g., 4 for F32)
- `Nb[1]` = `Nb[0] * ne[0]`
- `Nb[2]` = `Nb[1] * ne[1]`
- `Nb[3]` = `Nb[2] * ne[2]`

### Arena Allocation

- `GgmlContext` owns a single growable aligned buffer (`NativeMemory.AlignedAlloc`)
- All tensor data is carved from this arena
- `Dispose()` frees the entire arena at once
- `GgmlGraph` wraps a context and manages node lifetimes

### Threading Model

- `GgmlGraph.Execute(nth)` runs the full graph `nth` times (once per thread)
- Each kernel splits its work across threads via `ComputeParams.Ith`/`Nth`
- `ComputeParams.ThreadRowRange(nrows)` gives the `[ir0, ir1)` row range for this thread
- Matches GGML's `get_thread_range` semantics

### Quantization

- Quantized types store data in compressed blocks (see [Quantization Guide](QUANTIZATION.md))
- Dequantize on-the-fly in kernels via `GgmlTypeTraits.ToFloat`
- Quantize (host-side) via `GgmlTypeTraits.FromFloatRef` or `GgmlQuant.*` methods
- Supported: `Q1_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q6_K` (dequantize); non-K types (quantize)

---

## Common Patterns

### Filling Tensor Data

```csharp
unsafe {
    var ptr = (float*)tensor.Data;
    for (long i = 0; i < tensor.NElements(); i++) {
        ptr[i] = ...;
    }
}
```

### Reading Results

```csharp
unsafe {
    var ptr = (float*)tensor.Data;
    for (long i = 0; i < tensor.NElements(); i++) {
        Console.WriteLine(ptr[i]);
    }
}
```

### Setting Op Parameters

```csharp
// RMS Norm epsilon
GgmlTensor.SetOpParamsF32(tensor, 0, 1e-5f);

// SoftMax scale
GgmlTensor.SetOpParamsF32(tensor, 0, 1.0f);
GgmlTensor.SetOpParamsF32(tensor, 1, 0.0f);

// GLU op type
GgmlTensor.SetOpParamsI32(tensor, 0, (int)GgmlGluOp.SWIGLU);
GgmlTensor.SetOpParamsI32(tensor, 1, 0); // swapped
```

---

## Next Steps

- [API Reference](API.md) — Complete API documentation
- [Examples](EXAMPLES.md) — More usage examples
- [Quantization Guide](QUANTIZATION.md) — Quantized types deep dive
- [Graph Builder](GRAPH_BUILDER.md) — Graph API patterns
- [Operations Reference](OPERATIONS.md) — All implemented ops