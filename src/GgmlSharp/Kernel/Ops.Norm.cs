using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace GgmlSharp.Kernel;

/// <summary>
/// Ports of the GGML normalization ops from <c>ops.cpp</c>: NORM, RMS_NORM, L2_NORM,
/// GROUP_NORM. Each normalizes a row (dim0) independently and threads by the ith-stride
/// pattern (<c>for i01 = ith; i01 &lt; ne01; i01 += nth</c>) exactly as the C kernels do.
/// </summary>
public static unsafe class OpsNorm
{
    // ---- NORM: (x - mean) / sqrt(var + eps) -------------------------------------

    public static void ForwardNorm(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        float eps = GgmlTensor.GetOpParamsF32(dst, 0);
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];

        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = p.Ith; i1 < ne1; i1 += p.Nth)
        {
            float* x = (float*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            float* y = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            int n = (int)ne0;
            float mean = OpsReduce.RowSum(x, n) / n;
            float variance = 0;
            for (int i = 0; i < n; i++)
            {
                float v = x[i] - mean;
                y[i] = v;
                variance += v * v;
            }
            float scale = 1f / MathF.Sqrt(variance / n + eps);
            OpsReduce.Scale(y, n, scale);
        }
    }

    // ---- RMS_NORM: x / sqrt(mean(x^2) + eps) --------------------------------------

    public static void ForwardRmsNorm(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        float eps = GgmlTensor.GetOpParamsF32(dst, 0);
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];

        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = p.Ith; i1 < ne1; i1 += p.Nth)
        {
            float* x = (float*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            float* y = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            int n = (int)ne0;
            double sum = 0;
            for (int i = 0; i < n; i++) sum += (double)x[i] * x[i];
            float mean = (float)(sum / n);
            float scale = 1f / MathF.Sqrt(mean + eps);
            for (int i = 0; i < n; i++) y[i] = x[i] * scale;
        }
    }

    // ---- L2_NORM: x / max(sqrt(sum(x^2)), eps) -------------------------------------

    public static void ForwardL2Norm(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        float eps = GgmlTensor.GetOpParamsF32(dst, 0);
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];

        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = p.Ith; i1 < ne1; i1 += p.Nth)
        {
            float* x = (float*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            float* y = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            int n = (int)ne0;
            double sum = 0;
            for (int i = 0; i < n; i++) sum += (double)x[i] * x[i];
            float scale = 1f / MathF.Max(MathF.Sqrt((float)sum), eps);
            for (int i = 0; i < n; i++) y[i] = x[i] * scale;
        }
    }

    // ---- GROUP_NORM: group channels (ne02), normalize each group -------------------

    public static void ForwardGroupNorm(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        int nGroups = dst.OpParams[0];
        float eps = GgmlTensor.GetOpParamsF32(dst, 1);
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        int nChannels = (int)src0.Ne[2];
        int nPerGroup = (nChannels + nGroups - 1) / nGroups;

        for (int g = p.Ith; g < nGroups; g += p.Nth)
        {
            int start = g * nPerGroup;
            int end = Math.Min(start + nPerGroup, nChannels);
            int step = end - start;
            long count = ne0 * ne1 * step;

            for (long i3 = 0; i3 < ne3; i3++)
            {
                double sum = 0;
                for (int i2 = start; i2 < end; i2++)
                for (long i1 = 0; i1 < ne1; i1++)
                {
                    float* x = (float*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    for (int i = 0; i < (int)ne0; i++) sum += x[i];
                }
                float mean = (float)(sum / count);

                double sum2 = 0;
                for (int i2 = start; i2 < end; i2++)
                for (long i1 = 0; i1 < ne1; i1++)
                {
                    float* x = (float*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    float* y = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    for (int i = 0; i < (int)ne0; i++)
                    {
                        float v = x[i] - mean;
                        y[i] = v;
                        sum2 += (double)v * v;
                    }
                }
                float variance = (float)(sum2 / count);
                float scale = 1f / MathF.Sqrt(variance + eps);

                for (int i2 = start; i2 < end; i2++)
                for (long i1 = 0; i1 < ne1; i1++)
                    OpsReduce.Scale((float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1), (int)ne0, scale);
            }
        }
    }
}