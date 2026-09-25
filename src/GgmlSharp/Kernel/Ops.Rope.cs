using System.Runtime.InteropServices;

namespace GgmlSharp.Kernel;

/// <summary>
/// Mirrors <c>enum ggml_rope_type</c> (bit flags).
/// </summary>
public enum GgmlRopeType
{
    NORMAL = 0,
    NEOX = 1,
    VISION = 2,
    MROPE = 4,
    IMROPE = 24,
}

/// <summary>
/// Port of <c>ggml_compute_forward_rope_flt</c> (ops.cpp): NORMAL and NEOX rotation modes
/// with YaRN yarn scaling (freq_scale/ext_factor/attn_factor/beta_*). op_params layout:
/// [0]=n_past, [1]=n_dims, [2]=mode, [4]=n_ctx_orig, [5..10]=freq_base/freq_scale/
/// ext_factor/attn_factor/beta_fast/beta_slow (f32). src1 = positions (i32), src2 optional
/// freq_factors (f32). MROPE/IMROPE/VISION are not yet ported.
/// </summary>
public static unsafe class OpsRope
{
    public static void Forward(in ComputeParams p, GgmlTensor dst, bool forward = true)
    {
        var src0 = dst.Src[0]!;
        var src1 = dst.Src[1]!;        // positions, i32
        var src2 = dst.Src.Length > 2 ? dst.Src[2] : null; // freq factors

        int nDims = GgmlTensor.GetOpParamsI32(dst, 1);
        int mode = GgmlTensor.GetOpParamsI32(dst, 2);
        int nCtxOrig = GgmlTensor.GetOpParamsI32(dst, 4);
        float freqBase = GgmlTensor.GetOpParamsF32(dst, 5);
        float freqScale = GgmlTensor.GetOpParamsF32(dst, 6);
        float extFactor = GgmlTensor.GetOpParamsF32(dst, 7);
        float attnFactor = GgmlTensor.GetOpParamsF32(dst, 8);
        float betaFast = GgmlTensor.GetOpParamsF32(dst, 9);
        float betaSlow = GgmlTensor.GetOpParamsF32(dst, 10);

        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb0 = dst.Nb[0], nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb00 = src0.Nb[0], nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        int n = (int)ne0;
        bool isF16 = src0.Type == GgmlType.F16;

        float thetaScale = MathF.Pow(freqBase, -2f / nDims);
        float[] corr = new float[2];
        YarnCorrDims(nDims, nCtxOrig, freqBase, betaFast, betaSlow, corr);
        float* freqFactors = src2 != null ? (float*)src2.Data : null;

        var cache = (float*)NativeMemory.AlignedAlloc((nuint)(n * 4), 64);
        try
        {
            int rows = (int)(ne1 * ne2 * ne3);
            var (ir0, ir1) = p.ThreadRowRange(rows);
            int lastI2 = int.MinValue;
            for (int irrow = 0; irrow < rows; irrow++) // cheap full pass: only range [ir0,ir1) does work
            {
                long i3 = irrow / (ne2 * ne1);
                long i2 = (irrow - i3 * ne2 * ne1) / ne1;
                long i1 = irrow - i3 * ne2 * ne1 - i2 * ne1;
                if (irrow < ir0 || irrow >= ir1) continue;

                if ((int)i2 != lastI2)
                {
                    int pos = ((int*)src1.Data)[i2];
                    CacheInit(pos, freqScale, freqFactors, corr, n, extFactor, attnFactor, cache, 1f, thetaScale);
                    lastI2 = (int)i2;
                }

                byte* src = src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01;
                byte* dstp = dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1;

                if (mode == (int)GgmlRopeType.NORMAL)
                {
                    if (isF16) RotatePairsNormal((ushort*)dstp, (ushort*)src, nDims, cache);
                    else RotatePairsNormal((float*)dstp, (float*)src, nDims, cache);
                }
                else if (mode == (int)GgmlRopeType.NEOX)
                {
                    if (isF16) RotatePairsNeox((ushort*)dstp, (ushort*)src, nDims, cache);
                    else RotatePairsNeox((float*)dstp, (float*)src, nDims, cache);
                }
                else throw new NotImplementedException($"rope mode {mode}");

                // copy remaining (non-rotated) channels
                for (long i0 = nDims; i0 + 1 < ne0; i0 += 2)
                {
                    byte* s = src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00;
                    byte* d = dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0;
                    if (isF16) { *(ushort*)d = *(ushort*)s; *(ushort*)(d + 2) = *(ushort*)(s + 2); }
                    else { *(float*)d = *(float*)s; *(float*)(d + 4) = *(float*)(s + 4); }
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(cache);
        }
    }

    private static void RotatePairsNormal(float* d, float* s, int n, float* cache)
    {
        for (int i = 0; i < n; i += 2)
        {
            float cos = cache[i + 0], sin = cache[i + 1];
            float x0 = s[i], x1 = s[i + 1];
            d[i] = x0 * cos - x1 * sin;
            d[i + 1] = x0 * sin + x1 * cos;
        }
    }

    private static void RotatePairsNeox(float* d, float* s, int n, float* cache)
    {
        int half = n / 2;
        for (int i = 0; i < n; i += 2)
        {
            int ic = i / 2;
            float cos = cache[i + 0], sin = cache[i + 1];
            float x0 = s[ic], x1 = s[ic + half];
            d[ic] = x0 * cos - x1 * sin;
            d[ic + half] = x0 * sin + x1 * cos;
        }
    }

    private static void RotatePairsNormal(ushort* d, ushort* s, int n, float* cache)
    {
        for (int i = 0; i < n; i += 2)
        {
            float cos = cache[i], sin = cache[i + 1];
            float x0 = GgmlMath.F16ToF32(s[i]), x1 = GgmlMath.F16ToF32(s[i + 1]);
            d[i] = GgmlMath.F32ToF16(x0 * cos - x1 * sin);
            d[i + 1] = GgmlMath.F32ToF16(x0 * sin + x1 * cos);
        }
    }

    private static void RotatePairsNeox(ushort* d, ushort* s, int n, float* cache)
    {
        int half = n / 2;
        for (int i = 0; i < n; i += 2)
        {
            int ic = i / 2;
            float cos = cache[i], sin = cache[i + 1];
            float x0 = GgmlMath.F16ToF32(s[ic]), x1 = GgmlMath.F16ToF32(s[ic + half]);
            d[ic] = GgmlMath.F32ToF16(x0 * cos - x1 * sin);
            d[ic + half] = GgmlMath.F32ToF16(x0 * sin + x1 * cos);
        }
    }

    private static void CacheInit(float thetaBase, float freqScale, float* freqFactors, float[] corr, int ne0,
        float extFactor, float mscale, float* cache, float sinSign, float thetaScale)
    {
        float theta = thetaBase;
        for (int i0 = 0; i0 < ne0; i0 += 2)
        {
            float ff = freqFactors != null ? freqFactors[i0 / 2] : 1f;
            Yarn(theta / ff, freqScale, corr, i0, extFactor, mscale, out float cos, out float sin);
            cache[i0] = cos;
            cache[i0 + 1] = sin * sinSign;
            theta *= thetaScale;
        }
    }

    private static void Yarn(float thetaExtrap, float freqScale, float[] corr, int i0, float extFactor,
        float mscale, out float cosTheta, out float sinTheta)
    {
        float thetaInterp = freqScale * thetaExtrap;
        float theta = thetaInterp;
        float ms = mscale;
        if (extFactor != 0f)
        {
            float rampMix = YarnRamp(corr[0], corr[1], i0) * extFactor;
            theta = thetaInterp * (1 - rampMix) + thetaExtrap * rampMix;
            ms *= 1f + 0.1f * MathF.Log(1f / freqScale);
        }
        cosTheta = MathF.Cos(theta) * ms;
        sinTheta = MathF.Sin(theta) * ms;
    }

    private static float YarnRamp(float low, float high, int i0)
    {
        float y = (i0 / 2 - low) / MathF.Max(0.001f, high - low);
        return 1 - MathF.Min(1, MathF.Max(0, y));
    }

    private static void YarnCorrDims(int nDims, int nCtxOrig, float freqBase, float betaFast, float betaSlow, float[] out_)
    {
        out_[0] = MathF.Floor(nCtxOrig * MathF.Log(betaFast / freqBase) / MathF.Log(2f * MathF.PI));
        out_[1] = MathF.Min(nDims, MathF.Ceiling(nCtxOrig * MathF.Log(betaSlow / freqBase) / MathF.Log(2f * MathF.PI)));
    }
}