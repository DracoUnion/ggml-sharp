using System.Runtime.CompilerServices;

namespace GgmlSharp.Kernel;

/// <summary>
/// GLU (Gated Linear Unit) operation kernels, ported from <c>ggml_compute_forward_glu</c> in ops.cpp.
/// Supported GLU types: REGLU, GEGLU, SWIGLU, SWIGLU_OAI, GEGLU_ERF, GEGLU_QUICK.
/// op_params layout: [0]=glu_op (GgmlGluOp), [1]=swapped (bool), [2]=alpha (SWIGLU_OAI), [3]=limit (SWIGLU_OAI).
/// src0 and src1 are the two gating inputs (src1 can be null, meaning split src0 in half).
/// </summary>
public static unsafe class OpsGlu
{
    public static void Forward(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        var src1 = dst.Src[1]; // can be null

        int gluOp = GgmlTensor.GetOpParamsI32(dst, 0); // GgmlGluOp enum
        int swapped = GgmlTensor.GetOpParamsI32(dst, 1);
        float alpha = GgmlTensor.GetOpParamsF32(dst, 2); // SWIGLU_OAI
        float limit = GgmlTensor.GetOpParamsF32(dst, 3); // SWIGLU_OAI

        int nc = src1 != null ? (int)src0.Ne[0] : (int)(src0.Ne[0] / 2);
        int nr = (int)(src0.Ne[1] * src0.Ne[2] * src0.Ne[3]);
        var (ir0L, ir1L) = p.ThreadRowRange(nr);
        int ir0 = (int)ir0L, ir1 = (int)ir1L;

        long src0Stride = src0.Nb[1];
        long src1Stride = src1 != null ? src1.Nb[1] : src0.Nb[1];
        long dstStride = dst.Nb[1];

        bool isF16 = src0.Type == GgmlType.F16;
        bool isF32 = src0.Type == GgmlType.F32;

        if (!isF16 && !isF32)
            throw new NotSupportedException($"GLU only supports F32/F16, got {src0.Type}");

        for (int i1 = ir0; i1 < ir1; i1++)
        {
            byte* src0Ptr = src0.Data + i1 * src0Stride;
            byte* src1Ptr = src1 != null ? src1.Data + i1 * src1Stride : src0.Data + i1 * src1Stride;
            byte* dstPtr = dst.Data + i1 * dstStride;

            if (src1 == null)
            {
                // split src0 in half
                int offset = nc * (isF16 ? 2 : 4);
                if (swapped != 0)
                {
                    src0Ptr += offset;
                }
                else
                {
                    src1Ptr += offset;
                }
            }

            switch ((GgmlGluOp)gluOp)
            {
                case GgmlGluOp.REGLU:
                    if (isF32) RegLU_F32(nc, (float*)dstPtr, (float*)src0Ptr, (float*)src1Ptr);
                    else RegLU_F16(nc, (ushort*)dstPtr, (ushort*)src0Ptr, (ushort*)src1Ptr);
                    break;
                case GgmlGluOp.GEGLU:
                    if (isF32) GeGLU_F32(nc, (float*)dstPtr, (float*)src0Ptr, (float*)src1Ptr);
                    else GeGLU_F16(nc, (ushort*)dstPtr, (ushort*)src0Ptr, (ushort*)src1Ptr);
                    break;
                case GgmlGluOp.SWIGLU:
                    if (isF32) SiGLU_F32(nc, (float*)dstPtr, (float*)src0Ptr, (float*)src1Ptr);
                    else SiGLU_F16(nc, (ushort*)dstPtr, (ushort*)src0Ptr, (ushort*)src1Ptr);
                    break;
                case GgmlGluOp.SWIGLU_OAI:
                    if (isF32) SiGLU_OAI_F32(nc, (float*)dstPtr, (float*)src0Ptr, (float*)src1Ptr, alpha, limit);
                    else SiGLU_OAI_F16(nc, (ushort*)dstPtr, (ushort*)src0Ptr, (ushort*)src1Ptr, alpha, limit);
                    break;
                case GgmlGluOp.GEGLU_ERF:
                    if (isF32) GeGLU_Erf_F32(nc, (float*)dstPtr, (float*)src0Ptr, (float*)src1Ptr);
                    else GeGLU_Erf_F16(nc, (ushort*)dstPtr, (ushort*)src0Ptr, (ushort*)src1Ptr);
                    break;
                case GgmlGluOp.GEGLU_QUICK:
                    if (isF32) GeGLU_Quick_F32(nc, (float*)dstPtr, (float*)src0Ptr, (float*)src1Ptr);
                    else GeGLU_Quick_F16(nc, (ushort*)dstPtr, (ushort*)src0Ptr, (ushort*)src1Ptr);
                    break;
                default:
                    throw new NotImplementedException($"GLU op {(GgmlGluOp)gluOp} not implemented");
            }
        }
    }

    // ---- Math helpers ----------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Erf(float x)
    {
        // Abramowitz & Stegun approximation 7.1.26
        // .NET 7 doesn't have Math.Erf, .NET 8+ does
        // Coefficients from https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;
        const float p = 0.3275911f;

        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);

        float t = 1f / (1f + p * x);
        float y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);

        return sign * y;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Clamp(float value, float min, float max)
    {
        return value < min ? min : (value > max ? max : value);
    }

    // ---- REGLU: ReLU(x) * g ----------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RegLU_F32(int n, float* y, float* x, float* g)
    {
        for (int i = 0; i < n; i++)
            y[i] = x[i] > 0f ? x[i] * g[i] : 0f;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RegLU_F16(int n, ushort* y, ushort* x, ushort* g)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = GgmlMath.F16ToF32(x[i]);
            float gi = GgmlMath.F16ToF32(g[i]);
            y[i] = GgmlMath.F32ToF16(xi > 0f ? xi * gi : 0f);
        }
    }

    // ---- GEGLU: GELU(x) * g ----------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GeGLU_F32(int n, float* y, float* x, float* g)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = x[i];
            // GELU with erf: 0.5 * x * (1 + erf(x / sqrt(2)))
            float gelu = 0.5f * xi * (1f + Erf(xi * GgmlMath.Sqrt2Inv));
            y[i] = gelu * g[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GeGLU_F16(int n, ushort* y, ushort* x, ushort* g)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = GgmlMath.F16ToF32(x[i]);
            float gi = GgmlMath.F16ToF32(g[i]);
            float gelu = 0.5f * xi * (1f + Erf(xi * GgmlMath.Sqrt2Inv));
            y[i] = GgmlMath.F32ToF16(gelu * gi);
        }
    }

    // ---- SWIGLU: SiLU(x) * g = x * sigmoid(x) * g ------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SiGLU_F32(int n, float* y, float* x, float* g)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = x[i];
            float silu = xi / (1f + MathF.Exp(-xi)); // SiLU = x * sigmoid(x)
            y[i] = silu * g[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SiGLU_F16(int n, ushort* y, ushort* x, ushort* g)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = GgmlMath.F16ToF32(x[i]);
            float gi = GgmlMath.F16ToF32(g[i]);
            float silu = xi / (1f + MathF.Exp(-xi));
            y[i] = GgmlMath.F32ToF16(silu * gi);
        }
    }

    // ---- SWIGLU_OAI: (x / (1 + exp(alpha * -x))) * (clamp(g, -limit, limit) + 1) --

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SiGLU_OAI_F32(int n, float* y, float* x, float* g, float alpha, float limit)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = MathF.Min(x[i], limit);
            float gi = Clamp(g[i], -limit, limit);
            float silu = xi / (1f + MathF.Exp(alpha * -xi));
            y[i] = silu * (gi + 1f);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SiGLU_OAI_F16(int n, ushort* y, ushort* x, ushort* g, float alpha, float limit)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = MathF.Min(GgmlMath.F16ToF32(x[i]), limit);
            float gi = Clamp(GgmlMath.F16ToF32(g[i]), -limit, limit);
            float silu = xi / (1f + MathF.Exp(alpha * -xi));
            y[i] = GgmlMath.F32ToF16(silu * (gi + 1f));
        }
    }

    // ---- GEGLU_ERF: GELU_ERF(x) * g --------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GeGLU_Erf_F32(int n, float* y, float* x, float* g)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = x[i];
            float gelu = 0.5f * xi * (1f + Erf(xi * GgmlMath.Sqrt2Inv));
            y[i] = gelu * g[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GeGLU_Erf_F16(int n, ushort* y, ushort* x, ushort* g)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = GgmlMath.F16ToF32(x[i]);
            float gi = GgmlMath.F16ToF32(g[i]);
            float gelu = 0.5f * xi * (1f + Erf(xi * GgmlMath.Sqrt2Inv));
            y[i] = GgmlMath.F32ToF16(gelu * gi);
        }
    }

    // ---- GEGLU_QUICK: GELU_QUICK(x) * g ----------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GeGLU_Quick_F32(int n, float* y, float* x, float* g)
    {
        // GELU quick approximation: x * sigmoid(1.702 * x)
        for (int i = 0; i < n; i++)
        {
            float xi = x[i];
            float gelu = xi / (1f + MathF.Exp(-1.702f * xi));
            y[i] = gelu * g[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GeGLU_Quick_F16(int n, ushort* y, ushort* x, ushort* g)
    {
        for (int i = 0; i < n; i++)
        {
            float xi = GgmlMath.F16ToF32(x[i]);
            float gi = GgmlMath.F16ToF32(g[i]);
            float gelu = xi / (1f + MathF.Exp(-1.702f * xi));
            y[i] = GgmlMath.F32ToF16(gelu * gi);
        }
    }
}