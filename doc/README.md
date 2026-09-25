# GgmlSharp Documentation

Welcome to the GgmlSharp documentation. GgmlSharp is a from-scratch C# reimplementation of the [GGML](https://github.com/ggerganov/ggml) CPU compute runtime.

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Getting Started](GETTING_STARTED.md) | Installation, quick start, key concepts |
| [API Reference](API.md) | Complete API documentation for all public types |
| [Examples](EXAMPLES.md) | Practical usage examples for common patterns |
| [Quantization Guide](QUANTIZATION.md) | Deep dive into GGML quantization types |
| [Graph Builder](GRAPH_BUILDER.md) | Comprehensive `GgmlGraph` API guide |
| [Operations Reference](OPERATIONS.md) | All implemented operations with parameters |

---

## Quick Links

### For New Users
1. Start with [Getting Started](GETTING_STARTED.md) — build, test, first graph
2. See [Examples](EXAMPLES.md) — neural network layers, attention, quantization workflows
3. Reference [API Reference](API.md) — complete type/method documentation

### For ML Engineers
- [Quantization Guide](QUANTIZATION.md) — Q4_K, IQ, MXFP4, choosing types
- [Graph Builder](GRAPH_BUILDER.md) — advanced patterns, debugging, performance
- [Operations Reference](OPERATIONS.md) — all ops with formulas and parameters

### For GGML Users
- Type mapping: `ggml_type` → `GgmlType` (same values)
- Block layouts: bit-for-bit compatible with GGML C
- Quantization: `quantize_row_q4_K` → `GgmlQuant.Q4_K`

---

## Project Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| **M0** | ✅ Done | Enums, tensor/context, unary/binary/reduce/norm/softmax/simple/remap/upscale, rope (NORMAL/NEOX) |
| **M3** | ✅ Done | `MUL_MAT`/`OUT_PROD` F32, dequantize kernels (Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K) |
| **M4** | ✅ Done | **Quantize (from_float) for non-K types**, complete type metadata for all 42 types, **MROPE/IMROPE/VISION rope**, **GLU (6 variants)** |
| **Pending** | 🔄 | Full quantize/dequantize for K-quants/IQ/MXFP4/NVFP4/TQ*, net8+/net10 |

---

## Support

- **GitHub**: [DracoUnion/ggml-sharp](https://github.com/DracoUnion/ggml-sharp)
- **Issues**: Report bugs or request features on GitHub Issues
- **GGML Reference**: [ggerganov/ggml](https://github.com/ggerganov/ggml)

---

## License

MIT License — see [LICENSE](../LICENSE) in the repository root.