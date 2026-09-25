using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace GgmlSharp.Kernel;

/// <summary>
/// Port of the GGML <c>unary-ops.cpp</c> compute kernels: all 22 <see cref="GgmlUnaryOp"/>
/// ops for F32/F16/BF16, mirroring <c>apply_unary_op</c>'s row-decompose loop and the
/// f32/f16/bf16 type-codec dispatch. AVX2 fast paths are provided only for ops whose vector
/// result is bit-exact (abs/neg/relu/step/sgn/floor/ceil/hardswish/hardsigmoid); the
/// transcendental ops delegate to scalar <c>MathF</c> so their output matches the reference
/// exactly (perf tuning via FastExp-style approximations is a later milestone).
/// </summary>
public static unsafe class OpsUnary
{
    private delegate float UnaryF(float x);
    private delegate Vector256<float> UnaryV(Vector256<float> v);

    public static void Forward(in ComputeParams p, GgmlTensor dst)
    {
        GgmlUnaryOp op = (GgmlUnaryOp)GgmlTensor.GetOpParamsI32(dst, 0);
        GgmlTensor src0 = dst.Src[0]!;

        UnaryF f = Scalar(op, dst);
        UnaryV? v = Vector(op);

        var (ir0, ir1) = p.ThreadRowRange(src0.NRows());
        Apply(dst, src0, f, v, ir0, ir1);
    }

    private static void Apply(GgmlTensor dst, GgmlTensor src0, UnaryF f, UnaryV? v, long ir0, long ir1)
    {
        long ne0 = src0.Ne[0];
        long ne01 = src0.Ne[1];
        long ne02 = src0.Ne[2];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long plane = ne02 * ne01;

        for (long ir = ir0; ir < ir1; ir++)
        {
            long i03 = ir / plane;
            long i02 = (ir - i03 * plane) / ne01;
            long i01 = ir - i03 * plane - i02 * ne01;
            byte* dstPtr = dst.Data + i03 * nb3 + i02 * nb2 + i01 * nb1;
            byte* srcPtr = src0.Data + i03 * nb03 + i02 * nb02 + i01 * nb01;

            switch ((src0.Type, dst.Type))
            {
                case (GgmlType.F32, GgmlType.F32):     ApplyF32((float*)dstPtr, (float*)srcPtr, (int)ne0, f, v); break;
                case (GgmlType.F16, GgmlType.F16):     ApplyF16((ushort*)dstPtr, (ushort*)srcPtr, (int)ne0, f); break;
                case (GgmlType.BF16, GgmlType.BF16):   ApplyBf16((ushort*)dstPtr, (ushort*)srcPtr, (int)ne0, f); break;
                case (GgmlType.F16, GgmlType.F32):     ApplyF16ToF32((float*)dstPtr, (ushort*)srcPtr, (int)ne0, f); break;
                case (GgmlType.BF16, GgmlType.F32):    ApplyBf16ToF32((float*)dstPtr, (ushort*)srcPtr, (int)ne0, f); break;
                default:
                    throw new NotSupportedException(
                        $"unary: unsupported types dst={GgmlTypeTraits.Name(dst.Type)}, src0={GgmlTypeTraits.Name(src0.Type)}");
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ApplyF32(float* y, float* x, int n, UnaryF f, UnaryV? v)
    {
        int i = 0;
        if (v != null && Avx2.IsSupported)
            for (; i <= n - 8; i += 8)
                Avx.Store(y + i, v(Avx.LoadVector256(x + i)));
        for (; i < n; i++) y[i] = f(x[i]);
    }

    private static void ApplyF16(ushort* y, ushort* x, int n, UnaryF f)
    {
        for (int i = 0; i < n; i++) y[i] = GgmlMath.F32ToF16(f(GgmlMath.F16ToF32(x[i])));
    }

    private static void ApplyBf16(ushort* y, ushort* x, int n, UnaryF f)
    {
        for (int i = 0; i < n; i++) y[i] = GgmlMath.F32ToBf16(f(GgmlMath.Bf16ToF32(x[i])));
    }

    private static void ApplyF16ToF32(float* y, ushort* x, int n, UnaryF f)
    {
        for (int i = 0; i < n; i++) y[i] = f(GgmlMath.F16ToF32(x[i]));
    }

    private static void ApplyBf16ToF32(float* y, ushort* x, int n, UnaryF f)
    {
        for (int i = 0; i < n; i++) y[i] = f(GgmlMath.Bf16ToF32(x[i]));
    }

    // ---- per-op scalar formulas (bit-identical to ggml's op_* helpers) ----------

    private static float F_abs(float x) => MathF.Abs(x);
    private static float F_sgn(float x) => x > 0 ? 1f : (x < 0 ? -1f : 0f);
    private static float F_neg(float x) => -x;
    private static float F_step(float x) => x > 0 ? 1f : 0f;
    private static float F_tanh(float x) => MathF.Tanh(x);
    private static float F_elu(float x) => x > 0 ? x : GgmlMath.Expm1(x);
    private static float F_relu(float x) => x > 0 ? x : 0f;
    private static float F_sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
    private static float F_gelu(float x) => 0.5f * x * (1f + MathF.Tanh(GgmlMath.Sqrt2OverPi * x * (1f + 0.044715f * x * x)));
    private static float F_gelu_quick(float x) => x / (1f + MathF.Exp(-1.702f * x));
    private static float F_silu(float x) => x / (1f + MathF.Exp(-x));
    private static float F_hardswish(float x) => x * Hardsigmoid(x);
    private static float F_hardsigmoid(float x) => Hardsigmoid(x);
    private static float F_exp(float x) => MathF.Exp(x);
    private static float F_expm1(float x) => MathF.Exp(x) - 1f;
    private static float F_softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
    private static float F_gelu_erf(float x) => 0.5f * x * (1f + GgmlMath.Erf(x * GgmlMath.Sqrt2Inv));
    private static float F_xielu(float x, float aN, float aP, float beta, float eps)
        => x > 0 ? aP * x * x + beta * x : (GgmlMath.Expm1(MathF.Min(x, eps)) - x) * aN + beta * x;
    private static float F_floor(float x) => MathF.Floor(x);
    private static float F_ceil(float x) => MathF.Ceiling(x);
    private static float F_round(float x) => MathF.Round(x, MidpointRounding.AwayFromZero);
    private static float F_trunc(float x) => MathF.Truncate(x);

    private static float Hardsigmoid(float x)
    {
        float t = (x + 3f) / 6f;
        return MathF.Min(1f, MathF.Max(0f, t));
    }

    private static UnaryF Scalar(GgmlUnaryOp op, GgmlTensor dst) => op switch
    {
        GgmlUnaryOp.ABS         => F_abs,
        GgmlUnaryOp.SGN         => F_sgn,
        GgmlUnaryOp.NEG         => F_neg,
        GgmlUnaryOp.STEP        => F_step,
        GgmlUnaryOp.TANH        => F_tanh,
        GgmlUnaryOp.ELU         => F_elu,
        GgmlUnaryOp.RELU        => F_relu,
        GgmlUnaryOp.SIGMOID     => F_sigmoid,
        GgmlUnaryOp.GELU        => F_gelu,
        GgmlUnaryOp.GELU_QUICK  => F_gelu_quick,
        GgmlUnaryOp.SILU        => F_silu,
        GgmlUnaryOp.HARDSWISH   => F_hardswish,
        GgmlUnaryOp.HARDSIGMOID => F_hardsigmoid,
        GgmlUnaryOp.EXP         => F_exp,
        GgmlUnaryOp.EXPM1       => F_expm1,
        GgmlUnaryOp.SOFTPLUS    => F_softplus,
        GgmlUnaryOp.GELU_ERF    => F_gelu_erf,
        GgmlUnaryOp.XIELU       => XieluClosure(dst),
        GgmlUnaryOp.FLOOR       => F_floor,
        GgmlUnaryOp.CEIL        => F_ceil,
        GgmlUnaryOp.ROUND       => F_round,
        GgmlUnaryOp.TRUNC       => F_trunc,
        _                       => throw new NotSupportedException($"unary op {op}"),
    };

    private static UnaryF XieluClosure(GgmlTensor dst)
    {
        float aN = GgmlTensor.GetOpParamsF32(dst, 1);
        float aP = GgmlTensor.GetOpParamsF32(dst, 2);
        float beta = GgmlTensor.GetOpParamsF32(dst, 3);
        float eps = GgmlTensor.GetOpParamsF32(dst, 4);
        return x => F_xielu(x, aN, aP, beta, eps);
    }

    // ---- AVX2 fast paths (bit-exact vs the scalar fallback) ---------------------

    private static Vector256<float> V_abs(Vector256<float> v)
    {
        // clear the sign bit (0x7fffffff) in the integer view, then reinterprete as float
        return Avx2.And(v.AsInt32(), Vector256.Create(0x7fffffff)).AsSingle();
    }

    private static Vector256<float> V_neg(Vector256<float> v)
        => Avx.Multiply(v, Vector256.Create(-1f));

    private static Vector256<float> V_relu(Vector256<float> v)
        => Avx.Max(v, Vector256<float>.Zero);

    private static Vector256<float> V_step(Vector256<float> v)
    {
        // CompareGreaterThan yields all-ones (0xFFFFFFFF) where x>0; AND the float 1.0f bit
        // pattern through the integer view so a true lane becomes exactly 1.0f.
        var one = Vector256.Create(1f);
        var zero = Vector256<float>.Zero;
        var cmp = Avx.CompareGreaterThan(v, zero);          // float bits: 0xFFFFFFFF or 0
        return Avx2.And(one.AsInt32(), cmp.AsInt32()).AsSingle();
    }

    private static Vector256<float> V_sgn(Vector256<float> v)
    {
        var one = Vector256.Create(1f);
        var zero = Vector256<float>.Zero;
        var gt = Avx2.And(one.AsInt32(), Avx.CompareGreaterThan(v, zero).AsInt32()).AsSingle();   // 1 where x>0
        var lt = Avx2.And(one.AsInt32(), Avx.CompareGreaterThan(zero, v).AsInt32()).AsSingle();   // 1 where x<0
        return Avx.Subtract(gt, lt);
    }

    private static Vector256<float> V_floor(Vector256<float> v) => Avx.Floor(v);
    private static Vector256<float> V_ceil(Vector256<float> v) => Avx.Ceiling(v);

    private static Vector256<float> V_hardsigmoid(Vector256<float> v)
    {
        var three = Vector256.Create(3f);
        var six = Vector256.Create(6f);
        var one = Vector256.Create(1f);
        var zero = Vector256<float>.Zero;
        var t = Avx.Divide(Avx.Add(v, three), six);
        return Avx.Min(one, Avx.Max(zero, t));
    }

    private static Vector256<float> V_hardswish(Vector256<float> v)
        => Avx.Multiply(V_hardsigmoid(v), v);

    private static UnaryV? Vector(GgmlUnaryOp op) => op switch
    {
        GgmlUnaryOp.ABS         => V_abs,
        GgmlUnaryOp.NEG         => V_neg,
        GgmlUnaryOp.RELU        => V_relu,
        GgmlUnaryOp.STEP        => V_step,
        GgmlUnaryOp.SGN         => V_sgn,
        GgmlUnaryOp.FLOOR       => V_floor,
        GgmlUnaryOp.CEIL        => V_ceil,
        GgmlUnaryOp.HARDSIGMOID => V_hardsigmoid,
        GgmlUnaryOp.HARDSWISH   => V_hardswish,
        _                       => null,
    };
}