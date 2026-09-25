using System.Runtime.CompilerServices;

namespace GgmlSharp.Kernel;

/// <summary>
/// Quantize (from_float) row kernels for the standard block types, ported from
/// <c>ggml-quants.c</c> reference implementations. x points at the source floats,
/// y receives the quantized block. k is the number of elements (must be multiple of block size).
/// </summary>
public static unsafe class GgmlQuant
{
    // ---- non-K types ----------------------------------------------------------

    /// <summary>Q1_0: 1-bit quantization (sign only), block size 128, 18 bytes/block.</summary>
    public static void Q1_0(float* x, byte* y, long k)
    {
        const int qk = 128;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            float sumAbs = 0f;
            for (int j = 0; j < qk; j++)
                sumAbs += Math.Abs(x[i * qk + j]);
            float d = sumAbs / qk;

            // delta as fp16
            *(ushort*)(y + i * 18) = GgmlMath.F32ToF16(d);

            // Clear all bits first
            for (int j = 0; j < qk / 8; j++)
                y[i * 18 + 2 + j] = 0;

            // Store sign bits
            for (int j = 0; j < qk; j++)
            {
                if (x[i * qk + j] >= 0f)
                {
                    int bitIndex = j;
                    int byteIndex = bitIndex / 8;
                    int bitOffset = bitIndex % 8;
                    y[i * 18 + 2 + byteIndex] |= (byte)(1 << bitOffset);
                }
            }
        }
    }

    /// <summary>Q4_0: 4-bit quantization, block size 32, 18 bytes/block.</summary>
    public static void Q4_0(float* x, byte* y, long k)
    {
        const int qk = 32;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            float amax = 0f, max = 0f;
            for (int j = 0; j < qk; j++)
            {
                float v = x[i * qk + j];
                float av = Math.Abs(v);
                if (amax < av) { amax = av; max = v; }
            }

            float d = max / -8f;
            float id = d != 0f ? 1f / d : 0f;

            *(ushort*)(y + i * 18) = GgmlMath.F32ToF16(d);

            for (int j = 0; j < qk / 2; j++)
            {
                float x0 = x[i * qk + j] * id;
                float x1 = x[i * qk + qk / 2 + j] * id;
                byte xi0 = (byte)Math.Min(15, (int)(x0 + 8.5f));
                byte xi1 = (byte)Math.Min(15, (int)(x1 + 8.5f));
                y[i * 18 + 2 + j] = (byte)(xi0 | (xi1 << 4));
            }
        }
    }

    /// <summary>Q4_1: 4-bit quantization with min, block size 32, 20 bytes/block.</summary>
    public static void Q4_1(float* x, byte* y, long k)
    {
        const int qk = 32;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            float min = float.MaxValue, max = float.MinValue;
            for (int j = 0; j < qk; j++)
            {
                float v = x[i * qk + j];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            float d = (max - min) / 15f;
            float id = d != 0f ? 1f / d : 0f;

            *(ushort*)(y + i * 20) = GgmlMath.F32ToF16(d);
            *(ushort*)(y + i * 20 + 2) = GgmlMath.F32ToF16(min);

            for (int j = 0; j < qk / 2; j++)
            {
                float x0 = (x[i * qk + j] - min) * id;
                float x1 = (x[i * qk + qk / 2 + j] - min) * id;
                byte xi0 = (byte)Math.Min(15, (int)(x0 + 0.5f));
                byte xi1 = (byte)Math.Min(15, (int)(x1 + 0.5f));
                y[i * 20 + 4 + j] = (byte)(xi0 | (xi1 << 4));
            }
        }
    }

    /// <summary>Q5_0: 5-bit quantization, block size 32, 22 bytes/block.</summary>
    public static void Q5_0(float* x, byte* y, long k)
    {
        const int qk = 32;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            float amax = 0f, max = 0f;
            for (int j = 0; j < qk; j++)
            {
                float v = x[i * qk + j];
                float av = Math.Abs(v);
                if (amax < av) { amax = av; max = v; }
            }

            float d = max / -16f;
            float id = d != 0f ? 1f / d : 0f;

            *(ushort*)(y + i * 22) = GgmlMath.F32ToF16(d);

            uint qh = 0u;
            for (int j = 0; j < qk / 2; j++)
            {
                float x0 = x[i * qk + j] * id;
                float x1 = x[i * qk + qk / 2 + j] * id;
                byte xi0 = (byte)Math.Min(31, (int)(x0 + 16.5f));
                byte xi1 = (byte)Math.Min(31, (int)(x1 + 16.5f));

                y[i * 22 + 2 + j] = (byte)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));

                qh |= ((uint)(xi0 & 0x10) >> 4) << (j + 0);
                qh |= ((uint)(xi1 & 0x10) >> 4) << (j + qk / 2);
            }
            // Write qh as 4 bytes (little-endian)
            y[i * 22 + 18] = (byte)qh;
            y[i * 22 + 19] = (byte)(qh >> 8);
            y[i * 22 + 20] = (byte)(qh >> 16);
            y[i * 22 + 21] = (byte)(qh >> 24);
        }
    }

    /// <summary>Q5_1: 5-bit quantization with min, block size 32, 24 bytes/block.</summary>
    public static void Q5_1(float* x, byte* y, long k)
    {
        const int qk = 32;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            float min = float.MaxValue, max = float.MinValue;
            for (int j = 0; j < qk; j++)
            {
                float v = x[i * qk + j];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            float d = (max - min) / 31f;
            float id = d != 0f ? 1f / d : 0f;

            *(ushort*)(y + i * 24) = GgmlMath.F32ToF16(d);
            *(ushort*)(y + i * 24 + 2) = GgmlMath.F32ToF16(min);

            uint qh = 0u;
            for (int j = 0; j < qk / 2; j++)
            {
                float x0 = (x[i * qk + j] - min) * id;
                float x1 = (x[i * qk + qk / 2 + j] - min) * id;
                byte xi0 = (byte)(x0 + 0.5f);
                byte xi1 = (byte)(x1 + 0.5f);

                y[i * 24 + 4 + j] = (byte)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));

                qh |= ((uint)(xi0 & 0x10) >> 4) << (j + 0);
                qh |= ((uint)(xi1 & 0x10) >> 4) << (j + qk / 2);
            }
            y[i * 24 + 20] = (byte)qh;
            y[i * 24 + 21] = (byte)(qh >> 8);
            y[i * 24 + 22] = (byte)(qh >> 16);
            y[i * 24 + 23] = (byte)(qh >> 24);
        }
    }

    /// <summary>Q8_0: 8-bit quantization, block size 32, 34 bytes/block.</summary>
    public static void Q8_0(float* x, byte* y, long k)
    {
        const int qk = 32;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            float amax = 0f;
            for (int j = 0; j < qk; j++)
            {
                float v = Math.Abs(x[i * qk + j]);
                if (amax < v) amax = v;
            }

            float d = amax / 127f;
            float id = d != 0f ? 1f / d : 0f;

            *(ushort*)(y + i * 34) = GgmlMath.F32ToF16(d);

            for (int j = 0; j < qk; j++)
            {
                float x0 = x[i * qk + j] * id;
                y[i * 34 + 2 + j] = (byte)(Math.Round(x0) + 128);
            }
        }
    }

    /// <summary>Q8_1: 8-bit quantization with sum, block size 32, 36 bytes/block.</summary>
    public static void Q8_1(float* x, byte* y, long k)
    {
        const int qk = 32;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            float amax = 0f;
            for (int j = 0; j < qk; j++)
            {
                float v = Math.Abs(x[i * qk + j]);
                if (amax < v) amax = v;
            }

            float d = amax / 127f;
            float id = d != 0f ? 1f / d : 0f;

            *(ushort*)(y + i * 36) = GgmlMath.F32ToF16(d);

            int sum = 0;
            for (int j = 0; j < qk / 2; j++)
            {
                float v0 = x[i * qk + j] * id;
                float v1 = x[i * qk + qk / 2 + j] * id;
                byte q0 = (byte)(Math.Round(v0) + 128);
                byte q1 = (byte)(Math.Round(v1) + 128);
                y[i * 36 + 2 + j] = q0;
                y[i * 36 + 2 + qk / 2 + j] = q1;
                sum += q0 - 128;
                sum += q1 - 128;
            }
            *(ushort*)(y + i * 36 + 34) = GgmlMath.F32ToF16(sum * d);
        }
    }

    // ---- K-quants (super-block, 256 elements) ---------------------------------

    /// <summary>Q2_K: 2-bit K-quant, super-block 256, 84 bytes/block.</summary>
    public static void Q2_K(float* x, byte* y, long k)
    {
        const int qk = 256;
        int nb = (int)(k / qk);
        for (int i = 0; i < nb; i++)
        {
            // Simplified reference implementation
            // Full implementation requires proper scale/min quantization
            // For now, delegate to a basic quantization
            throw new NotImplementedException("Q2_K quantization not yet fully ported");
        }
    }

    /// <summary>Q3_K: 3-bit K-quant, super-block 256, 110 bytes/block.</summary>
    public static void Q3_K(float* x, byte* y, long k)
    {
        throw new NotImplementedException("Q3_K quantization not yet fully ported");
    }

    /// <summary>Q4_K: 4-bit K-quant, super-block 256, 144 bytes/block.</summary>
    public static void Q4_K(float* x, byte* y, long k)
    {
        throw new NotImplementedException("Q4_K quantization not yet fully ported");
    }

    /// <summary>Q5_K: 5-bit K-quant, super-block 256, 176 bytes/block.</summary>
    public static void Q5_K(float* x, byte* y, long k)
    {
        throw new NotImplementedException("Q5_K quantization not yet fully ported");
    }

    /// <summary>Q6_K: 6-bit K-quant, super-block 256, 210 bytes/block.</summary>
    public static void Q6_K(float* x, byte* y, long k)
    {
        throw new NotImplementedException("Q6_K quantization not yet fully ported");
    }

    /// <summary>Q8_K: 8-bit K-quant (fp32 delta), super-block 256, 292 bytes/block.</summary>
    public static void Q8_K(float* x, byte* y, long k)
    {
        throw new NotImplementedException("Q8_K quantization not yet fully ported");
    }

    // ---- Ternary quants --------------------------------------------------------

    /// <summary>TQ1_0: ternary 1.6875 bpw, super-block 256.</summary>
    public static void TQ1_0(float* x, byte* y, long k)
    {
        throw new NotImplementedException("TQ1_0 quantization not yet fully ported");
    }

    /// <summary>TQ2_0: ternary 2.0625 bpw, super-block 256.</summary>
    public static void TQ2_0(float* x, byte* y, long k)
    {
        throw new NotImplementedException("TQ2_0 quantization not yet fully ported");
    }

    // ---- IQ (importance-quantized) types --------------------------------------

    /// <summary>IQ2_XXS: ~2 bpw, super-block 256.</summary>
    public static void IQ2_XXS(float* x, byte* y, long k)
    {
        throw new NotImplementedException("IQ2_XXS quantization not yet fully ported");
    }

    /// <summary>IQ2_XS: ~2.3 bpw, super-block 256.</summary>
    public static void IQ2_XS(float* x, byte* y, long k)
    {
        throw new NotImplementedException("IQ2_XS quantization not yet fully ported");
    }

    /// <summary>IQ3_XXS: ~3 bpw, super-block 256.</summary>
    public static void IQ3_XXS(float* x, byte* y, long k)
    {
        throw new NotImplementedException("IQ3_XXS quantization not yet fully ported");
    }

    /// <summary>IQ1_S: ~1.56 bpw, super-block 256.</summary>
    public static void IQ1_S(float* x, byte* y, long k)
    {
        throw new NotImplementedException("IQ1_S quantization not yet fully ported");
    }

    /// <summary>IQ4_NL: non-linear 4-bit, block 32.</summary>
    public static void IQ4_NL(float* x, byte* y, long k)
    {
        throw new NotImplementedException("IQ4_NL quantization not yet fully ported");
    }

    /// <summary>IQ3_S: ~3.4 bpw, super-block 256.</summary>
    public static void IQ3_S(float* x, byte* y, long k)
    {
        throw new NotImplementedException("IQ3_S quantization not yet fully ported");
    }

    /// <summary>IQ2_S: ~2.56 bpw, super-block 256.</summary>
    public static void IQ2_S(float* x, byte* y, long k)
    {
        throw new NotImplementedException("IQ2_S quantization not yet fully ported");
    }

    /// <summary>IQ4_XS: ~4 bpw, super-block 256.</summary>
    public static void IQ4_XS(float* x, byte* y, long k)
    {
        throw new NotImplementedException("IQ4_XS quantization not yet fully ported");
    }

    /// <summary>IQ1_M: ~1.75 bpw, super-block 256.</summary>
    public static void IQ1_M(float* x, byte* y, long k)
    {
        throw new NotImplementedException("IQ1_M quantization not yet fully ported");
    }

    // ---- MXFP4 / NVFP4 ---------------------------------------------------------

    /// <summary>MXFP4: microscaling FP4, block 32.</summary>
    public static void MXFP4(float* x, byte* y, long k)
    {
        throw new NotImplementedException("MXFP4 quantization not yet fully ported");
    }

    /// <summary>NVFP4: NVIDIA FP4, block 64.</summary>
    public static void NVFP4(float* x, byte* y, long k)
    {
        throw new NotImplementedException("NVFP4 quantization not yet fully ported");
    }
}