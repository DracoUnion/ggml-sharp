namespace GgmlSharp.Kernel;

/// <summary>
/// Scalar helpers the GGML kernels need that aren't first-class in .NET:
/// fp16 &lt;-&gt; fp32 (as <c>ushort</c>, matching HyMT2Sharp.Kernels' <c>HalfBits</c>),
/// bf16 &lt;-&gt; fp32, and <c>erf</c> (GGML's GELU_ERF uses <c>erff</c>, which .NET lacks).
/// </summary>
public static unsafe class GgmlMath
{
    public const float Sqrt2OverPi = 0.7978845608028654f;
    public const float Sqrt2Inv     = 0.7071067811865476f;

    // ---- fp16 (IEEE-754 binary16) ------------------------------------------

    /// <summary>fp16 bits -&gt; fp32, handling subnormals and NaN/Inf.</summary>
    public static float F16ToF32(ushort h)
    {
        uint sign = (uint)(h >> 15) & 1u;
        uint exp  = (uint)(h >> 10) & 0x1fu;
        uint mant = (uint)h & 0x3ffu;

        if (exp == 0)
        {
            if (mant == 0) return sign == 1 ? -0.0f : 0.0f;
            // subnormal: mant * 2^-24
            float v = mant * 5.960464477539063e-8f;
            return sign == 1 ? -v : v;
        }
        if (exp == 31)
        {
            if (mant == 0) return sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;
            return float.NaN;
        }
        uint bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13); // exp-15+127 = exp+112
        return BitConverter.UInt32BitsToSingle(bits);
    }

    /// <summary>fp32 -&gt; fp16 bits, round-to-nearest-even.</summary>
    public static ushort F32ToF16(float value)
    {
        uint b = BitConverter.SingleToUInt32Bits(value);
        uint sign = (b >> 16) & 0x8000u;
        uint biasedExp = (b >> 23) & 0xffu;
        uint mant = b & 0x7fffffu;

        if (biasedExp == 0xffu) // inf / NaN
        {
            uint half = 0x7c00u | (mant >> 13);
            if (mant != 0 && (mant & 0x1fffu) == 0 && (mant >> 13) == 0) half |= 1u; // keep a NaN bit
            return (ushort)(sign | half);
        }

        int exp = (int)biasedExp - 127 + 15;
        if (exp >= 31) return (ushort)(sign | 0x7c00u); // overflow -> inf
        if (exp <= 0)  // subnormal or zero
        {
            if (exp < -10) return (ushort)sign; // flush to signed zero
            mant |= 0x800000u;                  // implicit leading 1
            int shift = 14 - exp;
            uint half = mant >> shift;
            uint rem = mant & ((1u << shift) - 1);
            uint halfway = 1u << (shift - 1);
            if (rem > halfway || (rem == halfway && (half & 1) != 0)) half++;
            return (ushort)(sign | half);
        }
        uint halfExp = (uint)exp << 10;
        uint halfMant = mant >> 13;
        uint rem2 = mant & 0x1fffu;
        if (rem2 > 0x1000u || (rem2 == 0x1000u && (halfMant & 1) != 0))
        {
            halfMant++;
            if (halfMant == 0x400u) { halfMant = 0; halfExp += 0x400u; }
        }
        return (ushort)(sign | halfExp | halfMant);
    }

    // ---- bf16 (bfloat16 = top 16 bits of fp32) -----------------------------

    /// <summary>bf16 bits -&gt; fp32 (left-shift by 16).</summary>
    public static float Bf16ToF32(ushort h) => BitConverter.UInt32BitsToSingle((uint)h << 16);

    /// <summary>fp32 -&gt; bf16 bits, round-to-nearest-even (same as ggml's conversion).</summary>
    public static ushort F32ToBf16(float value)
    {
        uint b = BitConverter.SingleToUInt32Bits(value);
        uint lower = b & 0xffffu;
        uint rounded = lower == 0x8000u ? b + (b & 0x10000u) : b + 0x8000u;
        return (ushort)(rounded >> 16);
    }

    // ---- erf (Abramowitz &amp; Stegun 7.1.26, ~1.5e-7 max abs err) ------------

    public static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        float ax = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * ax);
        float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
        return sign * (1f - poly * MathF.Exp(-ax * ax));
    }

    // ---- expm1 (MathF.Expm1 landed after net7.0.0; provide a stable series fallback)

    public static float Expm1(float x)
    {
        if (float.IsNegativeInfinity(x)) return -1f;
        if (float.IsPositiveInfinity(x)) return float.PositiveInfinity;
        if (float.IsNaN(x)) return float.NaN;
        // Taylor series, accurate & cancellation-free for small |x|:
        // x + x^2/2 + x^3/6 + x^4/24 + x^5/120
        if (MathF.Abs(x) < 0.01f)
            return x * (1f + x * (0.5f + x * (1f / 6f + x * (1f / 24f + x * (1f / 120f)))));
        return MathF.Exp(x) - 1f;
    }
}
