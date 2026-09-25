using System.Runtime.CompilerServices;

namespace GgmlSharp.Kernel;

/// <summary>Row-level type conversion, mirroring <c>ggml_to_float_t</c>.</summary>
public unsafe delegate void GgmlToFloat(byte* x, float* y, long k);

/// <summary>Row-level type conversion, mirroring <c>ggml_from_float_t</c>.</summary>
public unsafe delegate void GgmlFromFloat(float* x, byte* y, long k);

/// <summary>
/// Mirrors <c>struct ggml_type_traits</c> from <c>ggml.c</c>: block size, type size, whether
/// the type is quantized, and the dequantize/quantize row functions. Non-quantized rows are
/// filled; quantized rows are stubbed (BlckSize known, TypeSize/ToFloat/FromFloatRef filled in
/// milestone M3 when the per-block kernels land). Elementwise ops do NOT use this table — they
/// convert via dedicated f32/f16/bf16/i32 codecs.
/// </summary>
public readonly unsafe struct GgmlTypeTraits
{
    public readonly string TypeName;
    public readonly long BlckSize;
    public readonly long BlckSizeInterleave;
    public readonly long TypeSize;
    public readonly bool IsQuantized;
    public readonly GgmlToFloat? ToFloat;
    public readonly GgmlFromFloat? FromFloatRef;

    public GgmlTypeTraits(string name, long blck, long interleave, long size, bool quantized,
        GgmlToFloat? toFloat = null, GgmlFromFloat? fromFloat = null)
    {
        TypeName = name;
        BlckSize = blck;
        BlckSizeInterleave = interleave;
        TypeSize = size;
        IsQuantized = quantized;
        ToFloat = toFloat;
        FromFloatRef = fromFloat;
    }

    public static GgmlTypeTraits Get(GgmlType t) => _table[(int)t];

    // ---- to/from float row kernels for the non-quantized types -----------------

    private static void ToF32Row(byte* x, float* y, long k) { for (long i = 0; i < k; i++) y[i] = *(float*)(x + i * 4); }
    private static void FromF32Row(float* x, byte* y, long k) { for (long i = 0; i < k; i++) *(float*)(y + i * 4) = x[i]; }

    private static void ToF16Row(byte* x, float* y, long k) { for (long i = 0; i < k; i++) y[i] = GgmlMath.F16ToF32(*(ushort*)(x + i * 2)); }
    private static void FromF16Row(float* x, byte* y, long k) { for (long i = 0; i < k; i++) *(ushort*)(y + i * 2) = GgmlMath.F32ToF16(x[i]); }

    private static void ToBf16Row(byte* x, float* y, long k) { for (long i = 0; i < k; i++) y[i] = GgmlMath.Bf16ToF32(*(ushort*)(x + i * 2)); }
    private static void FromBf16Row(float* x, byte* y, long k) { for (long i = 0; i < k; i++) *(ushort*)(y + i * 2) = GgmlMath.F32ToBf16(x[i]); }

    private static readonly GgmlTypeTraits[] _table = Build();

    private static GgmlTypeTraits[] Build()
    {
        var t = new GgmlTypeTraits[(int)GgmlType.Count];
        t[(int)GgmlType.F32]  = new("f32",  1, 0, 4, false, ToF32Row, FromF32Row);
        t[(int)GgmlType.F16]  = new("f16",  1, 0, 2, false, ToF16Row, FromF16Row);
        t[(int)GgmlType.BF16] = new("bf16", 1, 0, 2, false, ToBf16Row, FromBf16Row);
        t[(int)GgmlType.I8]   = new("i8",   1, 0, 1, false);
        t[(int)GgmlType.I16]  = new("i16",  1, 0, 2, false);
        t[(int)GgmlType.I32]  = new("i32",  1, 0, 4, false);
        t[(int)GgmlType.I64]  = new("i64",  1, 0, 8, false);
        t[(int)GgmlType.F64]  = new("f64",  1, 0, 8, false);

        // Quantized: blck_size fixed by GGML; TypeSize/convert fns filled in M3.
        t[(int)GgmlType.Q1_0]   = new("q1_0", 128, 0, 18, true, GgmlDequant.Q1_0);
        t[(int)GgmlType.Q4_0]   = new("q4_0", 32,  0, 18, true, GgmlDequant.Q4_0);
        t[(int)GgmlType.Q4_1]   = new("q4_1", 32,  0, 20, true, GgmlDequant.Q4_1);
        t[(int)GgmlType.Q5_0]   = new("q5_0", 32,  0, 22, true, GgmlDequant.Q5_0);
        t[(int)GgmlType.Q5_1]   = new("q5_1", 32,  0, 24, true, GgmlDequant.Q5_1);
        t[(int)GgmlType.Q8_0]   = new("q8_0", 32,  0, 34, true, GgmlDequant.Q8_0);
        t[(int)GgmlType.Q8_1]   = new("q8_1", 32,  0, 36, true);
        t[(int)GgmlType.Q2_K]   = new("q2_K", 256, 0, 0, true);
        t[(int)GgmlType.Q3_K]   = new("q3_K", 256, 0, 0, true);
        t[(int)GgmlType.Q4_K]   = new("q4_K", 256, 0, 144, true, GgmlDequant.Q4_K);
        t[(int)GgmlType.Q5_K]   = new("q5_K", 256, 0, 176, true);
        t[(int)GgmlType.Q6_K]   = new("q6_K", 256, 0, 210, true, GgmlDequant.Q6_K);
        t[(int)GgmlType.Q8_K]   = new("q8_K", 256, 0, 292, true);
        t[(int)GgmlType.IQ2_XXS] = new("iq2_xxs", 256, 0, 0, true);
        t[(int)GgmlType.IQ2_XS]  = new("iq2_xs",  256, 0, 0, true);
        t[(int)GgmlType.IQ3_XXS] = new("iq3_xxs", 256, 0, 0, true);
        t[(int)GgmlType.IQ1_S]   = new("iq1_s",   256, 0, 0, true);
        t[(int)GgmlType.IQ4_NL]  = new("iq4_nl",  32,  0, 0, true);
        t[(int)GgmlType.IQ3_S]   = new("iq3_s",   256, 0, 0, true);
        t[(int)GgmlType.IQ2_S]   = new("iq2_s",   256, 0, 0, true);
        t[(int)GgmlType.IQ4_XS]  = new("iq4_xs",  256, 0, 0, true);
        t[(int)GgmlType.IQ1_M]   = new("iq1_m",   256, 0, 0, true);
        t[(int)GgmlType.MXFP4]   = new("mxfp4",   32,  0, 0, true);
        t[(int)GgmlType.NVFP4]   = new("nvfp4",   64,  0, 0, true);
        t[(int)GgmlType.TQ1_0]   = new("tq1_0",   256, 0, 0, true);
        t[(int)GgmlType.TQ2_0]   = new("tq2_0",   256, 0, 0, true);

        // Deprecated / removed slots.
        t[(int)GgmlType.Q4_2]    = new("DEPRECATED", 0, 0, 0, false);
        t[(int)GgmlType.Q4_3]    = new("DEPRECATED", 0, 0, 0, false);
        t[(int)GgmlType.Q4_0_4_4] = new("REMOVED", 0, 0, 0, false);
        t[(int)GgmlType.Q4_0_4_8] = new("REMOVED", 0, 0, 0, false);
        t[(int)GgmlType.Q4_0_8_8] = new("REMOVED", 0, 0, 0, false);
        t[(int)GgmlType.IQ4_NL_4_4] = new("REMOVED", 0, 0, 0, false);
        t[(int)GgmlType.IQ4_NL_4_8] = new("REMOVED", 0, 0, 0, false);
        t[(int)GgmlType.IQ4_NL_8_8] = new("REMOVED", 0, 0, 0, false);

        return t;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static string Name(GgmlType t) => _table[(int)t].TypeName;
}
