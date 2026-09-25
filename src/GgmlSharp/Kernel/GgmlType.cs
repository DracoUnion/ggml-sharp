namespace GgmlSharp.Kernel;

/// <summary>
/// Mirrors <c>enum ggml_type</c> from <c>ggml.h</c> (42 values, exact numeric order).
/// Deprecated / removed slots (Q4_2, Q4_3, Q4_0_4_4 … IQ4_NL_8_8) are kept so the enum
/// values line up 1:1 with GGML.
/// </summary>
public enum GgmlType
{
    F32 = 0,
    F16,
    Q4_0,
    Q4_1,
    Q4_2,          // DEPRECATED
    Q4_3,          // DEPRECATED
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    Q4_0_4_4,      // REMOVED
    Q4_0_4_8,      // REMOVED
    Q4_0_8_8,      // REMOVED
    TQ1_0,
    TQ2_0,
    IQ4_NL_4_4,    // REMOVED
    IQ4_NL_4_8,    // REMOVED
    IQ4_NL_8_8,    // REMOVED
    MXFP4,
    NVFP4,
    Q1_0,
    Count = 42,
}
