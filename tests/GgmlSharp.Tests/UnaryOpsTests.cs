using System.Runtime.InteropServices;
using GgmlSharp.Kernel;
using Xunit;

namespace GgmlSharp.Tests;

public sealed unsafe class UnaryOpsTests
{
    private static readonly Random Rng = new(20240925);

    // Independent scalar reference for each unary op (same formulas as ggml, re-derived here
    // so the AVX2/scalar kernels and the threaded row-split are checked against something else).
    private static float Ref(GgmlUnaryOp op, float x, float[] prm)
    {
        return op switch
        {
            GgmlUnaryOp.ABS         => MathF.Abs(x),
            GgmlUnaryOp.SGN         => x > 0 ? 1f : (x < 0 ? -1f : 0f),
            GgmlUnaryOp.NEG         => -x,
            GgmlUnaryOp.STEP        => x > 0 ? 1f : 0f,
            GgmlUnaryOp.TANH        => MathF.Tanh(x),
            GgmlUnaryOp.ELU         => x > 0 ? x : GgmlMath.Expm1(x),
            GgmlUnaryOp.RELU        => x > 0 ? x : 0f,
            GgmlUnaryOp.SIGMOID     => 1f / (1f + MathF.Exp(-x)),
            GgmlUnaryOp.GELU        => 0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * x * (1f + 0.044715f * x * x))),
            GgmlUnaryOp.GELU_QUICK  => x / (1f + MathF.Exp(-1.702f * x)),
            GgmlUnaryOp.SILU        => x / (1f + MathF.Exp(-x)),
            GgmlUnaryOp.HARDSWISH   => x * Hardsig(x),
            GgmlUnaryOp.HARDSIGMOID => Hardsig(x),
            GgmlUnaryOp.EXP         => MathF.Exp(x),
            GgmlUnaryOp.EXPM1       => MathF.Exp(x) - 1f,
            GgmlUnaryOp.SOFTPLUS    => x > 20f ? x : MathF.Log(1f + MathF.Exp(x)),
            GgmlUnaryOp.GELU_ERF    => 0.5f * x * (1f + GgmlMath.Erf(x * 0.7071067811865476f)),
            GgmlUnaryOp.XIELU       => x > 0 ? prm[1] * x * x + prm[2] * x : (GgmlMath.Expm1(MathF.Min(x, prm[3])) - x) * prm[0] + prm[2] * x,
            GgmlUnaryOp.FLOOR       => MathF.Floor(x),
            GgmlUnaryOp.CEIL        => MathF.Ceiling(x),
            GgmlUnaryOp.ROUND       => MathF.Round(x, MidpointRounding.AwayFromZero),
            GgmlUnaryOp.TRUNC       => MathF.Truncate(x),
            _                       => throw new NotSupportedException(op.ToString()),
        };
    }

    private static float Hardsig(float x)
    {
        float t = (x + 3f) / 6f;
        return MathF.Min(1f, MathF.Max(0f, t));
    }

    private static float[] XieluParams() => new[] { 0.5f, 1.5f, 0.1f, 0.5f };

    private static void RunSingle(GgmlContext ctx, GgmlTensor dst)
    {
        var p = new ComputeParams(ith: 0, nth: 1);
        GgmlCompute.Forward(p, dst);
    }

    /// <summary>Simulate an nth-thread run by calling Forward once per thread with its row range.</summary>
    private static void RunThreaded(GgmlContext ctx, GgmlTensor dst, int nth)
    {
        for (int ith = 0; ith < nth; ith++)
        {
            var p = new ComputeParams(ith, nth);
            GgmlCompute.Forward(p, dst);
        }
    }

    private static long[] Shape => new long[] { 16, 3, 2 }; // 96 elements, 6 rows of 16

    private static void AssertAll(GgmlType type, GgmlUnaryOp op, float rel)
    {
        int n = 96;
        using var ctx = new GgmlContext();
        var x = ctx.NewTensor(type, 3, Shape);
        var y = ctx.NewUnaryOp(type, Shape, x, op);
        if (op == GgmlUnaryOp.XIELU)
        {
            GgmlTensor.SetOpParamsF32(y, 1, 0.5f); // alpha_n
            GgmlTensor.SetOpParamsF32(y, 2, 1.5f); // alpha_p
            GgmlTensor.SetOpParamsF32(y, 3, 0.1f); // beta
            GgmlTensor.SetOpParamsF32(y, 4, 0.5f); // eps
        }

        float[] prm = op == GgmlUnaryOp.XIELU ? XieluParams() : new float[4];

        void FillAndCheck(bool threaded)
        {
            float[] expected = new float[n];
            for (int i = 0; i < n; i++)
            {
                float v = (float)(Rng.NextDouble() * 10.0 - 5.0);
                // The kernel reads the stored input through the type's f32 codec, so the
                // reference must apply the same quantization to input and output.
                float vq = type switch
                {
                    GgmlType.F32  => v,
                    GgmlType.F16  => GgmlMath.F16ToF32(GgmlMath.F32ToF16(v)),
                    _             => GgmlMath.Bf16ToF32(GgmlMath.F32ToBf16(v)),
                };
                float refv = Ref(op, vq, prm);
                float exp = type switch
                {
                    GgmlType.F32  => refv,
                    GgmlType.F16  => GgmlMath.F16ToF32(GgmlMath.F32ToF16(refv)),
                    _             => GgmlMath.Bf16ToF32(GgmlMath.F32ToBf16(refv)),
                };
                if (type == GgmlType.F32) { ((float*)x.Data)[i] = v; expected[i] = exp; }
                else if (type == GgmlType.F16) { ((ushort*)x.Data)[i] = GgmlMath.F32ToF16(v); expected[i] = exp; }
                else { ((ushort*)x.Data)[i] = GgmlMath.F32ToBf16(v); expected[i] = exp; }
            }

            if (threaded) RunThreaded(ctx, y, 4);
            else RunSingle(ctx, y);

            for (int i = 0; i < n; i++)
            {
                float got = type == GgmlType.F32 ? ((float*)y.Data)[i]
                          : type == GgmlType.F16 ? GgmlMath.F16ToF32(((ushort*)y.Data)[i])
                          : GgmlMath.Bf16ToF32(((ushort*)y.Data)[i]);
                float exp = expected[i];
                float tol = rel * MathF.Max(1f, MathF.Abs(exp));
                Assert.True(MathF.Abs(got - exp) <= tol,
                    $"op={op} type={type} i={i}: got {got}, expected {exp}");
            }
        }

        FillAndCheck(threaded: false);
        FillAndCheck(threaded: true);
    }

    public static IEnumerable<object[]> AllOps() => Enum.GetValues<GgmlUnaryOp>()
        .Where(o => o != GgmlUnaryOp.Count)
        .Select(o => new object[] { o });

    [Theory]
    [MemberData(nameof(AllOps))]
    public void F32_MatchesReference(GgmlUnaryOp op) => AssertAll(GgmlType.F32, op, 1e-4f);

    [Theory]
    [MemberData(nameof(AllOps))]
    public void F16_MatchesReference(GgmlUnaryOp op) => AssertAll(GgmlType.F16, op, 2e-3f);

    [Theory]
    [MemberData(nameof(AllOps))]
    public void BF16_MatchesReference(GgmlUnaryOp op) => AssertAll(GgmlType.BF16, op, 2e-3f);
}