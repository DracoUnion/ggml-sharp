using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace GgmlSharp.Kernel;

/// <summary>
/// Ports of a few simple unary-with-param ops from <c>ops.cpp</c>: LEAKY_RELU, SILU_BACK,
/// SCALE (+bias). These use op_params for the extra scalar(s).
/// </summary>
public static unsafe class OpsSimple
{
    // ---- LEAKY_RELU: x if x&gt;0 else negative_slope*x -------------------------------

    public static void ForwardLeakyRelu(in ComputeParams p, GgmlTensor dst)
    {
        if (p.Ith != 0) return;
        var src0 = dst.Src[0]!;
        float slope = GgmlTensor.GetOpParamsF32(dst, 0);
        long ne0 = src0.Ne[0];
        long nrows = src0.NRows();
        long nb01 = src0.Nb[1], nb1 = dst.Nb[1];

        for (long r = 0; r < nrows; r++)
        {
            byte* src = src0.Data + r * nb01;
            byte* d = dst.Data + r * nb1;
            if (src0.Type == GgmlType.F32) ApplyLeakyF32((float*)d, (float*)src, (int)ne0, slope);
            else ApplyLeakyF16((ushort*)d, (ushort*)src, (int)ne0, slope);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ApplyLeakyF32(float* y, float* x, int n, float slope)
    {
        int i = 0;
        if (Avx2.IsSupported)
        {
            var sv = Vector256.Create(slope);
            var zero = Vector256<float>.Zero;
            for (; i <= n - 8; i += 8)
            {
                var v = Avx.LoadVector256(x + i);
                var gt = Avx.CompareGreaterThan(v, zero);   // all-ones where x>0
                // where x>0 keep x, else x*slope  =>  blend(slope*x, x, gt)
                var lx = Avx.Multiply(v, sv);
                var sel = Avx.BlendVariable(lx, v, gt.AsSingle());
                Avx.Store(y + i, sel);
            }
        }
        for (; i < n; i++) y[i] = x[i] > 0 ? x[i] : x[i] * slope;
    }

    private static void ApplyLeakyF16(ushort* y, ushort* x, int n, float slope)
    {
        for (int i = 0; i < n; i++)
        {
            float v = GgmlMath.F16ToF32(x[i]);
            y[i] = GgmlMath.F32ToF16(v > 0 ? v : v * slope);
        }
    }

    // ---- SILU_BACK: dz * silu'(x) where silu'(x)=s*(1+x*(1-s)), s=sigmoid(x) ---------

    public static void ForwardSiluBack(in ComputeParams p, GgmlTensor dst)
    {
        var grad = dst.Src[0]!;
        var src1 = dst.Src[1]!;
        long ne0 = src1.Ne[0];
        long nb1 = dst.Nb[1], nb01 = src1.Nb[1], nbg = grad.Nb[1];

        var (ir0, ir1) = p.ThreadRowRange(src1.NRows());
        for (long i1 = ir0; i1 < ir1; i1++)
        {
            float* d = (float*)(dst.Data + i1 * nb1);
            float* x = (float*)(src1.Data + i1 * nb01);
            float* dz = (float*)(grad.Data + i1 * nbg);
            for (int i = 0; i < (int)ne0; i++)
            {
                float xi = x[i];
                float s = 1f / (1f + MathF.Exp(-xi));
                d[i] = dz[i] * (s * (1f + xi * (1f - s)));
            }
        }
    }

    // ---- SCALE: dst = src0 * s + b (s,b from op_params) ------------------------------

    public static void ForwardScale(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        float s = GgmlTensor.GetOpParamsF32(dst, 0);
        float b = GgmlTensor.GetOpParamsF32(dst, 1);
        int nc = (int)src0.Ne[0];
        long nb01 = src0.Nb[1], nb1 = dst.Nb[1];

        var (ir0, ir1) = p.ThreadRowRange(src0.NRows());
        if (b == 0f)
        {
            for (long i1 = ir0; i1 < ir1; i1++)
            {
                float* y = (float*)(dst.Data + i1 * nb1);
                if (dst.Data != src0.Data)
                    for (int i = 0; i < nc; i++) y[i] = ((float*)(src0.Data + i1 * nb01))[i];
                OpsReduce.Scale(y, nc, s);
            }
        }
        else
        {
            for (long i1 = ir0; i1 < ir1; i1++)
                OpsReduce.Mad1((float*)(dst.Data + i1 * nb1), (float*)(src0.Data + i1 * nb01), nc, s, b);
        }
    }
}