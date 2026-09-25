using GgmlSharp.Kernel;
using Xunit;

namespace GgmlSharp.Tests;

public sealed unsafe class BinaryOpsTests
{
    private static readonly Random Rng = new(777);

    private static float Ref(GgmlOp op, float a, float b) => op switch
    {
        GgmlOp.ADD => a + b,
        GgmlOp.SUB => a - b,
        GgmlOp.MUL => a * b,
        GgmlOp.DIV => a / b,
        _ => throw new NotSupportedException(op.ToString()),
    };

    private static float ReadElem(GgmlType t, byte* p) => t switch
    {
        GgmlType.F32  => *(float*)p,
        GgmlType.F16  => GgmlMath.F16ToF32(*(ushort*)p),
        _             => GgmlMath.Bf16ToF32(*(ushort*)p),
    };

    private static void WriteElem(GgmlType t, byte* p, float v)
    {
        switch (t)
        {
            case GgmlType.F32: *(float*)p = v; break;
            case GgmlType.F16: *(ushort*)p = GgmlMath.F32ToF16(v); break;
            default: *(ushort*)p = GgmlMath.F32ToBf16(v); break;
        }
    }

    private static void Run(GgmlTensor dst, int nth)
    {
        for (int ith = 0; ith < nth; ith++)
            GgmlCompute.Forward(new ComputeParams(ith, nth), dst);
    }

    private static void AssertBinary(GgmlType type, GgmlOp op, long[] ne0, long[] ne1)
    {
        long ne00 = ne0[0], ne01 = ne0[1], ne02 = ne0.Length > 2 ? ne0[2] : 1;
        long ne10 = ne1[0], ne11 = ne1[1], ne12 = ne1.Length > 2 ? ne1[2] : 1;
        int size = type == GgmlType.F32 ? 4 : 2;
        long total = ne00 * ne01 * ne02;

        using var ctx = new GgmlContext();
        var x = ctx.NewTensor(type, 3, ne0);
        var y = ctx.NewTensor(type, ne1.Length, ne1);
        var dst = ctx.NewBinaryOp(type, ne0, x, y, op);

        float[] expected = new float[total];
        // Fill src0/src1 and precompute the expected output.
        for (long i2 = 0; i2 < ne02; i2++)
        for (long i1 = 0; i1 < ne01; i1++)
        for (long i0 = 0; i0 < ne00; i0++)
        {
            float a = (float)(Rng.NextDouble() * 4.0 - 2.0);
            WriteElem(type, x.Data + (i2 * ne01 + i1) * ne00 * size + i0 * size, a);
        }
        for (long i2 = 0; i2 < ne12; i2++)
        for (long i1 = 0; i1 < ne11; i1++)
        for (long i0 = 0; i0 < ne10; i0++)
        {
            float b = (float)(Rng.NextDouble() * 4.0 - 2.0);
            WriteElem(type, y.Data + (i2 * ne11 + i1) * ne10 * size + i0 * size, b);
        }
        for (long i2 = 0; i2 < ne02; i2++)
        for (long i1 = 0; i1 < ne01; i1++)
        for (long i0 = 0; i0 < ne00; i0++)
        {
            float a = ReadElem(type, x.Data + (i2 * ne01 + i1) * ne00 * size + i0 * size);
            long j0 = i0 % ne10, j1 = i1 % ne11, j2 = i2 % ne12;
            float b = ReadElem(type, y.Data + (j2 * ne11 + j1) * ne10 * size + j0 * size);
            float e = Ref(op, a, b);
            if (type == GgmlType.F16) e = GgmlMath.F16ToF32(GgmlMath.F32ToF16(e));
            else if (type == GgmlType.BF16) e = GgmlMath.Bf16ToF32(GgmlMath.F32ToBf16(e));
            expected[(i2 * ne01 + i1) * ne00 + i0] = e;
        }

        foreach (int nth in new[] { 1, 4 })
        {
            Run(dst, nth);
            for (long i = 0; i < total; i++)
            {
                float got = ReadElem(type, dst.Data + i * size);
                float exp = expected[i];
                float tol = (type == GgmlType.F32 ? 1e-4f : 2e-3f) * MathF.Max(1f, MathF.Abs(exp));
                Assert.True(MathF.Abs(got - exp) <= tol,
                    $"op={op} type={type} nth={nth} i={i}: got {got}, expected {exp}");
            }
        }
    }

    public static IEnumerable<object[]> AllOps() => new[]
    {
        new object[] { GgmlOp.ADD }, new object[] { GgmlOp.SUB },
        new object[] { GgmlOp.MUL }, new object[] { GgmlOp.DIV },
    };

    private static long[] Same      => new long[] { 16, 3, 2 };
    private static long[] RowBroad  => new long[] { 16, 1, 1 };   // src1 repeats across rows
    private static long[] ScalarRow => new long[] { 1, 3, 2 };    // one scalar broadcast within each row

    [Theory] [MemberData(nameof(AllOps))]
    public void F32_SameShape(GgmlOp op) => AssertBinary(GgmlType.F32, op, Same, Same);

    [Theory] [MemberData(nameof(AllOps))]
    public void F32_RowBroadcast(GgmlOp op) => AssertBinary(GgmlType.F32, op, Same, RowBroad);

    [Theory] [MemberData(nameof(AllOps))]
    public void F32_ScalarPerRow(GgmlOp op) => AssertBinary(GgmlType.F32, op, Same, ScalarRow);

    [Theory] [MemberData(nameof(AllOps))]
    public void F16_SameShape(GgmlOp op) => AssertBinary(GgmlType.F16, op, Same, Same);

    [Theory] [MemberData(nameof(AllOps))]
    public void BF16_SameShape(GgmlOp op) => AssertBinary(GgmlType.BF16, op, Same, Same);
}