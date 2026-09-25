using System.Runtime.CompilerServices;

namespace GgmlSharp.Kernel;

/// <summary>
/// Dequantize (to_float) row kernels for the standard block types, ported from
/// <c>ggml-quants.c</c>. x points at the row (nb blocks of <c>BlckSize</c>), y receives
/// <c>k</c> floats. Fields are read by byte offset to avoid struct-packing assumptions.
/// Covered: Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K.
/// K-quants, IQ, MXFP4, NVFP4, TQ* have TypeSize/BlckSize in TypeTraits; ToFloat/FromFloatRef stubbed here.
/// </summary>
public static unsafe class GgmlDequant
{
    // ---- non-K types (block size 18..36, qk 32/128) ------------------------------

    public static void Q1_0(byte* x, float* y, long k)
    {
        const int qk = 128, bsz = 18;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            byte* b = x + (long)i * bsz;
            float d = GgmlMath.F16ToF32(*(ushort*)b);
            for (int j = 0; j < qk; j++)
            {
                byte bit = (byte)((b[2 + j / 8] >> (j % 8)) & 1);
                y[(long)i * qk + j] = bit != 0 ? d : -d;
            }
        }
    }

    public static void Q4_0(byte* x, float* y, long k)
    {
        const int qk = 32, bsz = 18;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            byte* b = x + (long)i * bsz;
            float d = GgmlMath.F16ToF32(*(ushort*)b);
            for (int j = 0; j < qk / 2; j++)
            {
                byte q = b[2 + j];
                y[(long)i * qk + j] = ((q & 0x0F) - 8) * d;
                y[(long)i * qk + j + qk / 2] = ((q >> 4) - 8) * d;
            }
        }
    }

    public static void Q4_1(byte* x, float* y, long k)
    {
        const int qk = 32, bsz = 20;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            byte* b = x + (long)i * bsz;
            float d = GgmlMath.F16ToF32(*(ushort*)b);
            float m = GgmlMath.F16ToF32(*(ushort*)(b + 2));
            for (int j = 0; j < qk / 2; j++)
            {
                byte q = b[4 + j];
                y[(long)i * qk + j] = (q & 0x0F) * d + m;
                y[(long)i * qk + j + qk / 2] = (q >> 4) * d + m;
            }
        }
    }

    public static void Q5_0(byte* x, float* y, long k)
    {
        const int qk = 32, bsz = 22;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            byte* b = x + (long)i * bsz;
            float d = GgmlMath.F16ToF32(*(ushort*)b);
            uint qh = Unsafe.ReadUnaligned<uint>(b + 2);
            for (int j = 0; j < qk / 2; j++)
            {
                int xh0 = (int)(((qh >> (j + 0)) << 4) & 0x10);
                int xh1 = (int)((qh >> (j + 12)) & 0x10);
                byte q = b[6 + j];
                y[(long)i * qk + j] = (((q & 0x0F) | xh0) - 16) * d;
                y[(long)i * qk + j + qk / 2] = (((q >> 4) | xh1) - 16) * d;
            }
        }
    }

    public static void Q5_1(byte* x, float* y, long k)
    {
        const int qk = 32, bsz = 24;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            byte* b = x + (long)i * bsz;
            float d = GgmlMath.F16ToF32(*(ushort*)b);
            float m = GgmlMath.F16ToF32(*(ushort*)(b + 2));
            uint qh = Unsafe.ReadUnaligned<uint>(b + 4);
            for (int j = 0; j < qk / 2; j++)
            {
                int xh0 = (int)(((qh >> (j + 0)) << 4) & 0x10);
                int xh1 = (int)((qh >> (j + 12)) & 0x10);
                byte q = b[8 + j];
                y[(long)i * qk + j] = ((q & 0x0F) | xh0) * d + m;
                y[(long)i * qk + j + qk / 2] = ((q >> 4) | xh1) * d + m;
            }
        }
    }

    public static void Q8_0(byte* x, float* y, long k)
    {
        const int qk = 32, bsz = 34;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            byte* b = x + (long)i * bsz;
            float d = GgmlMath.F16ToF32(*(ushort*)b);
            for (int j = 0; j < qk; j++)
                y[(long)i * qk + j] = *(sbyte*)(b + 2 + j) * d;
        }
    }

    public static void Q8_1(byte* x, float* y, long k)
    {
        throw new NotImplementedException("Q8_1 dequantize not yet ported");
    }

    // ---- Q4_K (block 144 B, qk 256) -----------------------------------------------

    public static void Q4_K(byte* x, float* y, long k)
    {
        const int qk = 256, bsz = 144;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            byte* b = x + (long)i * bsz;
            float d = GgmlMath.F16ToF32(*(ushort*)b);
            float min = GgmlMath.F16ToF32(*(ushort*)(b + 2));
            byte* scales = b + 4;
            byte* q = b + 16;
            int idx = 0;
            for (int j = 0; j < qk; j += 64)
            {
                GetScaleMinK4(idx + 0, scales, out byte sc0, out byte m0);
                GetScaleMinK4(idx + 1, scales, out byte sc1, out byte m1);
                float d1 = d * sc0, m1v = min * m0;
                float d2 = d * sc1, m2v = min * m1;
                for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1v;
                for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2v;
                q += 32; idx += 2;
            }
        }
    }

    private static void GetScaleMinK4(int j, byte* q, out byte d, out byte m)
    {
        if (j < 4)
        {
            d = (byte)(q[j] & 63);
            m = (byte)(q[j + 4] & 63);
        }
        else
        {
            d = (byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j - 4] >> 2) << 4));
        }
    }

    // ---- Q6_K (block 210 B, qk 256) ------------------------------------------------

    public static void Q6_K(byte* x, float* y, long k)
    {
        const int qk = 256, bsz = 210;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            byte* b = x + (long)i * bsz;
            float d = GgmlMath.F16ToF32(*(ushort*)(b + 208));
            byte* ql = b;
            byte* qh = b + 128;
            sbyte* sc = (sbyte*)(b + 192);
            for (int n = 0; n < qk; n += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    int si = l / 16;
                    int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l + 0] = d * sc[si + 0] * q1;
                    y[l + 32] = d * sc[si + 2] * q2;
                    y[l + 64] = d * sc[si + 4] * q3;
                    y[l + 96] = d * sc[si + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
    }

    // ---- Stubs for remaining quant types (ToFloat not yet ported) -------------------
    // These allow TypeTraits to have non-null FromFloatRef but ToFloat throws until implemented.

    public static void Q2_K(byte* x, float* y, long k) => throw new NotImplementedException("Q2_K dequantize not yet ported");
    public static void Q3_K(byte* x, float* y, long k) => throw new NotImplementedException("Q3_K dequantize not yet ported");
    public static void Q5_K(byte* x, float* y, long k) => throw new NotImplementedException("Q5_K dequantize not yet ported");
    public static void Q8_K(byte* x, float* y, long k) => throw new NotImplementedException("Q8_K dequantize not yet ported");
    public static void TQ1_0(byte* x, float* y, long k) => throw new NotImplementedException("TQ1_0 dequantize not yet ported");
    public static void TQ2_0(byte* x, float* y, long k) => throw new NotImplementedException("TQ2_0 dequantize not yet ported");
    public static void IQ2_XXS(byte* x, float* y, long k) => throw new NotImplementedException("IQ2_XXS dequantize not yet ported");
    public static void IQ2_XS(byte* x, float* y, long k) => throw new NotImplementedException("IQ2_XS dequantize not yet ported");
    public static void IQ3_XXS(byte* x, float* y, long k) => throw new NotImplementedException("IQ3_XXS dequantize not yet ported");
    public static void IQ1_S(byte* x, float* y, long k) => throw new NotImplementedException("IQ1_S dequantize not yet ported");
    public static void IQ4_NL(byte* x, float* y, long k) => throw new NotImplementedException("IQ4_NL dequantize not yet ported");
    public static void IQ3_S(byte* x, float* y, long k) => throw new NotImplementedException("IQ3_S dequantize not yet ported");
    public static void IQ2_S(byte* x, float* y, long k) => throw new NotImplementedException("IQ2_S dequantize not yet ported");
    public static void IQ4_XS(byte* x, float* y, long k) => throw new NotImplementedException("IQ4_XS dequantize not yet ported");
    public static void IQ1_M(byte* x, float* y, long k) => throw new NotImplementedException("IQ1_M dequantize not yet ported");
    public static void MXFP4(byte* x, float* y, long k) => throw new NotImplementedException("MXFP4 dequantize not yet ported");
    public static void NVFP4(byte* x, float* y, long k) => throw new NotImplementedException("NVFP4 dequantize not yet ported");
}