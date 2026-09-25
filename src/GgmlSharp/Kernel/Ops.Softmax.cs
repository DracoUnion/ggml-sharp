using System.Runtime.InteropServices;

namespace GgmlSharp.Kernel;

/// <summary>
/// Port of <c>ggml_compute_forward_soft_max_f32</c> (ops.cpp). Supports the scale and
/// max_bias (ALiBi) op_params, an optional f32/f16 mask (src1, broadcast across rows) and
/// optional "sink" tokens (src2). A per-call scratch holds the scaled+masked row.
/// </summary>
public static unsafe class OpsSoftmax
{
    public static void Forward(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        var src1 = dst.Src.Length > 1 ? dst.Src[1] : null;
        var src2 = dst.Src.Length > 2 ? dst.Src[2] : null;

        float scale = GgmlTensor.GetOpParamsF32(dst, 0);
        float maxBias = GgmlTensor.GetOpParamsF32(dst, 1);

        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb11 = src1?.Nb[1] ?? 1, nb12 = src1?.Nb[2] ?? 1, nb13 = src1?.Nb[3] ?? 1;
        long ne12 = src1?.Ne[2] ?? 1, ne13 = src1?.Ne[3] ?? 1;

        int nHead = (int)ne2;
        int nHeadLog2 = 1 << (int)MathF.Floor(MathF.Log2(nHead));
        float m0 = MathF.Pow(2f, -maxBias / nHeadLog2);
        float m1 = MathF.Pow(2f, -maxBias / 2f / nHeadLog2);
        bool useF16 = src1 != null && src1.Type == GgmlType.F16;
        float* sk = src2 != null ? (float*)src2.Data : null;

        int n = (int)ne0;
        var scratch = (float*)NativeMemory.AlignedAlloc((nuint)(n * 4), 64);
        try
        {
            for (long i3 = 0; i3 < ne3; i3++)
            for (long i2 = 0; i2 < ne2; i2++)
            for (long i1 = p.Ith; i1 < ne1; i1 += p.Nth)
            {
                long i11 = i1, i12 = i2 % ne12, i13 = i3 % ne13;

                // ALiBi slope
                int h = (int)i2;
                float slope = maxBias > 0f
                    ? (h < nHeadLog2 ? MathF.Pow(m0, h + 1) : MathF.Pow(m1, 2 * (h - nHeadLog2) + 1))
                    : 1f;

                float* sp = (float*)(src0.Data + i1 * nb01 + i2 * nb02 + i3 * nb03);
                float* dp = (float*)(dst.Data + i1 * nb1 + i2 * nb2 + i3 * nb3);
                byte* maskRow = src1 != null ? src1.Data + i11 * nb11 + i12 * nb12 + i13 * nb13 : null;

                for (int i = 0; i < n; i++) scratch[i] = sp[i] * scale;
                if (maskRow != null)
                {
                    if (useF16)
                        for (int i = 0; i < n; i++) scratch[i] += slope * GgmlMath.F16ToF32(((ushort*)maskRow)[i]);
                    else
                        for (int i = 0; i < n; i++) scratch[i] += slope * ((float*)maskRow)[i];
                }

                float max = float.NegativeInfinity;
                for (int i = 0; i < n; i++) if (scratch[i] > max) max = scratch[i];
                if (sk != null) max = MathF.Max(max, sk[i2]);

                float sum = 0;
                for (int i = 0; i < n; i++)
                {
                    float e = MathF.Exp(scratch[i] - max);
                    dp[i] = e;
                    sum += e;
                }
                if (sk != null) sum += MathF.Exp(sk[i2] - max);

                float inv = 1f / sum;
                for (int i = 0; i < n; i++) dp[i] *= inv;
            }
        }
        finally
        {
            NativeMemory.AlignedFree(scratch);
        }
    }
}