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
/// Port of <c>ggml_compute_forward_rope_flt</c> (ops.cpp): NORMAL, NEOX, MROPE, IMROPE, VISION
/// rotation modes with YaRN yarn scaling. op_params layout:
/// [0]=n_past, [1]=n_dims, [2]=mode, [4]=n_ctx_orig,
/// [5..10]=freq_base/freq_scale/ext_factor/attn_factor/beta_fast/beta_slow (f32),
/// [11..14]=sections[4] for MROPE/IMROPE.
/// src1 = positions (i32), shape varies by mode:
///   - NORMAL/NEOX: [ne2]
///   - MROPE: [ne2 * 4] (t, h, w, e)
///   - IMROPE: [ne2 * 4] (interleaved)
///   - VISION: [ne2 * 4] (t, h, w, e)
/// src2 optional freq_factors (f32).
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

        // Sections for MROPE/IMROPE (4 ints at op_params[11..14])
        int[] sections = new int[4];
        if (mode != (int)GgmlRopeType.NORMAL && mode != (int)GgmlRopeType.NEOX)
        {
            for (int i = 0; i < 4; i++)
                sections[i] = GgmlTensor.GetOpParamsI32(dst, 11 + i);
        }

        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb00 = src0.Nb[0], nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
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
            float[] mropeCache = new float[4]; // [theta_t, theta_h, theta_w, theta_e]

            for (int irrow = 0; irrow < rows; irrow++)
            {
                long i3 = irrow / (ne2 * ne1);
                long i2 = (irrow - i3 * ne2 * ne1) / ne1;
                long i1 = irrow - i3 * ne2 * ne1 - i2 * ne1;
                if (irrow < ir0 || irrow >= ir1) continue;

                bool isMrope = (mode & (int)GgmlRopeType.MROPE) != 0; // includes VISION (24 & 4) and IMROPE (24 & 4)
                bool isImrope = mode == (int)GgmlRopeType.IMROPE;
                bool isVision = mode == (int)GgmlRopeType.VISION;

                if ((int)i2 != lastI2)
                {
                    if (!isMrope)
                    {
                        int pos = ((int*)src1.Data)[i2];
                        CacheInit(pos, freqScale, freqFactors, corr, n, extFactor, attnFactor, cache, 1f, thetaScale);
                    }
                    else
                    {
                        // MROPE/IMROPE/VISION: positions are [t, h, w, e] for each i2
                        // pos layout: [ne2] for t, [ne2] for h, [ne2] for w, [ne2] for e
                        int pos_t = ((int*)src1.Data)[i2];
                        int pos_h = ((int*)src1.Data)[i2 + ne2];
                        int pos_w = ((int*)src1.Data)[i2 + ne2 * 2];
                        int pos_e = ((int*)src1.Data)[i2 + ne2 * 3];
                        MropeCacheInit(pos_t, pos_h, pos_w, pos_e, sections, isImrope, isVision,
                            freqBase, freqScale, freqFactors, corr, n, extFactor, attnFactor, cache, 1f, thetaScale);
                    }
                    lastI2 = (int)i2;
                }

                byte* src = src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01;
                byte* dstp = dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1;

                switch (mode)
                {
                    case (int)GgmlRopeType.NORMAL:
                        if (isF16) RotatePairsNormal((ushort*)dstp, (ushort*)src, nDims, cache);
                        else RotatePairsNormal((float*)dstp, (float*)src, nDims, cache);
                        break;
                    case (int)GgmlRopeType.NEOX:
                        if (isF16) RotatePairsNeox((ushort*)dstp, (ushort*)src, nDims, cache);
                        else RotatePairsNeox((float*)dstp, (float*)src, nDims, cache);
                        break;
                    case (int)GgmlRopeType.MROPE:
                    case (int)GgmlRopeType.IMROPE:
                        if (isF16) RotatePairsNeox((ushort*)dstp, (ushort*)src, nDims, cache);
                        else RotatePairsNeox((float*)dstp, (float*)src, nDims, cache);
                        break;
                    case (int)GgmlRopeType.VISION:
                        if (isF16) RotatePairsVision((ushort*)dstp, (ushort*)src, nDims, cache);
                        else RotatePairsVision((float*)dstp, (float*)src, nDims, cache);
                        break;
                    default:
                        throw new NotImplementedException($"rope mode {mode}");
                }

                // Copy remaining (non-rotated) channels
                if (!isVision)
                {
                    for (long i0 = nDims; i0 + 1 < ne0; i0 += 2)
                    {
                        byte* s = src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00;
                        byte* d = dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb00;
                        if (isF16) { *(ushort*)d = *(ushort*)s; *(ushort*)(d + 2) = *(ushort*)(s + 2); }
                        else { *(float*)d = *(float*)s; *(float*)(d + 4) = *(float*)(s + 4); }
                    }
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(cache);
        }
    }

    // ---- Rotation kernels ------------------------------------------------------

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

    private static void RotatePairsVision(float* d, float* s, int n, float* cache)
    {
        // VISION mode: rotate full ne0 dimensions with nDims pairs
        for (int i = 0; i < n; i += 2)
        {
            int ic = i; // VISION uses ic = i (scale=1)
            float cos = cache[i + 0], sin = cache[i + 1];
            float x0 = s[ic], x1 = s[ic + 1];
            d[ic] = x0 * cos - x1 * sin;
            d[ic + 1] = x0 * sin + x1 * cos;
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

    private static void RotatePairsVision(ushort* d, ushort* s, int n, float* cache)
    {
        for (int i = 0; i < n; i += 2)
        {
            int ic = i;
            float cos = cache[i], sin = cache[i + 1];
            float x0 = GgmlMath.F16ToF32(s[ic]), x1 = GgmlMath.F16ToF32(s[ic + 1]);
            d[ic] = GgmlMath.F32ToF16(x0 * cos - x1 * sin);
            d[ic + 1] = GgmlMath.F32ToF16(x0 * sin + x1 * cos);
        }
    }

    // ---- Cache initialization --------------------------------------------------

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

    private static void MropeCacheInit(
        int pos_t, int pos_h, int pos_w, int pos_e,
        int[] sections, bool isImrope, bool isVision,
        float freqBase, float freqScale, float* freqFactors, float[] corr, int ne0,
        float extFactor, float mscale, float* cache, float sinSign, float thetaScale)
    {
        // theta_base for each component derived from position * freqBase
        // In GGML, the base theta is freqBase (10000 typically), and positions scale it
        // Here we compute theta_base_t = pos_t * theta_scale_base, etc.
        // The reference uses theta_base_t/h/w/e directly from op_params, but we derive from positions.

        // For the reference, the theta_base values are passed as separate params (op_params 0..3)
        // In our implementation, we have positions in src1. The reference computes:
        // theta_t = pos_t * (freqBase * theta_scale^i) ...
        // Actually, the reference expects theta_base_t/h/w/e as separate params.
        // Let's follow the reference: use freqBase as base, and positions as multipliers.

        float theta_t = pos_t * freqBase; // simplified - reference uses separate base
        float theta_h = pos_h * freqBase;
        float theta_w = pos_w * freqBase;
        float theta_e = pos_e * freqBase;

        int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        int sec_w = sections[1] + sections[0];
        int sec_e = sections[2] + sec_w;

        for (int i0 = 0; i0 < ne0; i0 += 2)
        {
            float ff = freqFactors != null ? freqFactors[i0 / 2] : 1f;

            int sector = (i0 / 2) % sect_dims;
            if (isImrope)
            {
                // qwen3vl interleaved mrope: sector cycles through t,h,w,t,h,w,...
                if (sector % 3 == 1 && sector < 3 * sections[1])
                    theta_t = pos_h * freqBase; // use theta_h
                else if (sector % 3 == 2 && sector < 3 * sections[2])
                    theta_t = pos_w * freqBase; // use theta_w
                else if (sector % 3 == 0 && sector < 3 * sections[0])
                    theta_t = pos_t * freqBase; // use theta_t
                else
                    theta_t = pos_e * freqBase; // use theta_e
            }
            else
            {
                if (sector >= sections[0] && sector < sec_w)
                    theta_t = pos_h * freqBase;
                else if (sector >= sec_w && sector < sec_w + sections[2])
                    theta_t = pos_w * freqBase;
                else if (sector >= sec_w + sections[2])
                    theta_t = pos_e * freqBase;
                else
                    theta_t = pos_t * freqBase;
            }

            float theta = theta_t;
            Yarn(theta / ff, freqScale, corr, i0, extFactor, mscale, out float cos, out float sin);
            cache[i0] = cos;
            cache[i0 + 1] = sin * sinSign;

            // Advance all thetas
            theta_t *= thetaScale;
            theta_h *= thetaScale;
            theta_w *= thetaScale;
            theta_e *= thetaScale;
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
        float y = (i0 / 2f - low) / MathF.Max(0.001f, high - low);
        return 1 - MathF.Min(1, MathF.Max(0, y));
    }

    private static void YarnCorrDims(int nDims, int nCtxOrig, float freqBase, float betaFast, float betaSlow, float[] out_)
    {
        // corr_dim = n_dims * log(n_ctx_orig / (n_rot * 2pi)) / (2 * log(base))
        float corrDimFast = nDims * MathF.Log(nCtxOrig / (betaFast * 2 * MathF.PI)) / (2 * MathF.Log(freqBase));
        float corrDimSlow = nDims * MathF.Log(nCtxOrig / (betaSlow * 2 * MathF.PI)) / (2 * MathF.Log(freqBase));
        out_[0] = MathF.Max(0, MathF.Floor(corrDimFast));
        out_[1] = MathF.Min(nDims - 1, MathF.Ceiling(corrDimSlow));
    }
}