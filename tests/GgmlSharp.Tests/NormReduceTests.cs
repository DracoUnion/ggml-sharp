using GgmlSharp.Kernel;
using Xunit;

namespace GgmlSharp.Tests;

public sealed unsafe class NormReduceTests
{
    private static readonly Random Rng = new(555);

    private static void Fill(float* p, int n)
    {
        for (int i = 0; i < n; i++) p[i] = (float)(Rng.NextDouble() * 4.0 - 2.0);
    }

    private static void Run(GgmlTensor dst, int nth)
    {
        for (int ith = 0; ith < nth; ith++)
            GgmlCompute.Forward(new ComputeParams(ith, nth), dst);
    }

    private static (GgmlTensor src, GgmlTensor dst) MakeOp(GgmlContext ctx, GgmlType type, long[] ne,
        GgmlOp op, params GgmlTensor[] srcs)
    {
        var src = ctx.NewTensor(type, ne.Length, ne);
        var dst = ctx.NewTensor(type, ne.Length, ne, srcs);
        dst.Op = op;
        return (src, dst);
    }

    private static float Tol(float exp, float rel = 1e-4f) => rel * MathF.Max(1f, MathF.Abs(exp));

    [Fact]
    public void Sum_ReducesAllElements()
    {
        long[] ne = { 16, 3, 2 };
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)src.Data, (int)src.NElements());
        var dst = ctx.NewTensor(GgmlType.F32, 1, new long[] { 1 }, new[] { src });
        dst.Op = GgmlOp.SUM;

        double sum = 0;
        for (int i = 0; i < (int)src.NElements(); i++) sum += ((float*)src.Data)[i];

        Run(dst, 1);
        Assert.Equal((float)sum, ((float*)dst.Data)[0], 4);

        Run(dst, 4); // thread 0 does all; others no-op
        Assert.Equal((float)sum, ((float*)dst.Data)[0], 4);
    }

    [Fact]
    public void SumRows_And_Mean()
    {
        long[] ne = { 16, 3, 2 };
        long ne0 = ne[0], ne1 = ne[1], ne2 = ne[2];
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)src.Data, (int)src.NElements());

        foreach (GgmlOp op in new[] { GgmlOp.SUM_ROWS, GgmlOp.MEAN })
        {
            var dst = ctx.NewTensor(GgmlType.F32, 3, new long[] { 1, ne1, ne2 }, new[] { src });
            dst.Op = op;
            Run(dst, 4);
            for (long i2 = 0; i2 < ne2; i2++)
            for (long i1 = 0; i1 < ne1; i1++)
            {
                float* row = (float*)src.Data + (i2 * ne1 + i1) * ne0;
                float rowSum = 0;
                for (int i = 0; i < ne0; i++) rowSum += row[i];
                float expected = op == GgmlOp.SUM_ROWS ? rowSum : rowSum / ne0;
                float got = ((float*)dst.Data)[(i2 * ne1 + i1)];
                Assert.True(MathF.Abs(got - expected) <= Tol(expected), $"op={op} got {got} exp {expected}");
            }
        }
    }

    [Fact]
    public void Argmax_PerRow()
    {
        long[] ne = { 16, 3 };
        long ne0 = ne[0], ne1 = ne[1];
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 2, ne);
        Fill((float*)src.Data, (int)src.NElements());
        var dst = ctx.NewTensor(GgmlType.I32, 2, new long[] { 1, ne1 }, new[] { src });
        dst.Op = GgmlOp.ARGMAX;
        Run(dst, 1);

        for (long i1 = 0; i1 < ne1; i1++)
        {
            float* row = (float*)src.Data + i1 * ne0;
            int expected = 0;
            for (int i = 1; i < ne0; i++) if (row[i] > row[expected]) expected = i;
            Assert.Equal(expected, ((int*)dst.Data)[i1]);
        }
    }

    [Fact]
    public void CountEqual()
    {
        long[] ne = { 16, 3, 2 };
        using var ctx = new GgmlContext();
        var a = ctx.NewTensor(GgmlType.I32, 3, ne);
        var b = ctx.NewTensor(GgmlType.I32, 3, ne);
        int n = (int)(ne[0] * ne[1] * ne[2]);
        for (int i = 0; i < n; i++) { ((int*)a.Data)[i] = Rng.Next(3); ((int*)b.Data)[i] = Rng.Next(3); }
        var dst = ctx.NewTensor(GgmlType.I64, 1, new long[] { 1 }, new[] { a, b });
        dst.Op = GgmlOp.COUNT_EQUAL;

        long count = 0;
        for (int i = 0; i < n; i++) count += ((int*)a.Data)[i] == ((int*)b.Data)[i] ? 1 : 0;
        Run(dst, 1);
        Assert.Equal(count, ((long*)dst.Data)[0]);
    }

    [Fact]
    public void Cumsum()
    {
        long[] ne = { 16, 3, 2 };
        long ne0 = ne[0];
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)src.Data, (int)src.NElements());
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { src });
        dst.Op = GgmlOp.CUMSUM;

        Run(dst, 4);
        int rows = (int)(ne[1] * ne[2]);
        for (int r = 0; r < rows; r++)
        {
            float* srcRow = (float*)src.Data + r * ne0;
            float* dRow = (float*)dst.Data + r * ne0;
            float acc = 0;
            for (int i = 0; i < ne0; i++) { acc += srcRow[i]; Assert.True(MathF.Abs(dRow[i] - acc) <= Tol(acc)); }
        }
    }

    [Theory]
    [InlineData(GgmlOp.NORM)]
    [InlineData(GgmlOp.RMS_NORM)]
    [InlineData(GgmlOp.L2_NORM)]
    public void RowNorms(GgmlOp op)
    {
        long[] ne = { 16, 3, 2 };
        long ne0 = ne[0];
        float eps = 1e-5f;
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)src.Data, (int)src.NElements());
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { src });
        dst.Op = op;
        GgmlTensor.SetOpParamsF32(dst, 0, eps);

        Run(dst, 4);
        int rows = (int)(ne[1] * ne[2]);
        for (int r = 0; r < rows; r++)
        {
            float* x = (float*)src.Data + r * ne0;
            float* y = (float*)dst.Data + r * ne0;
            double sum = 0, sumSq = 0;
            for (int i = 0; i < ne0; i++) { sum += x[i]; sumSq += (double)x[i] * x[i]; }
            for (int i = 0; i < ne0; i++)
            {
                float expected = op switch
                {
                    GgmlOp.NORM     => (x[i] - (float)(sum / ne0)) / MathF.Sqrt((float)(sumSq / ne0 - (sum / ne0) * (sum / ne0)) + eps),
                    GgmlOp.RMS_NORM => x[i] / MathF.Sqrt((float)(sumSq / ne0) + eps),
                    _               => x[i] / MathF.Max(MathF.Sqrt((float)sumSq), eps),
                };
                Assert.True(MathF.Abs(y[i] - expected) <= Tol(expected, 1e-3f), $"op={op} got {y[i]} exp {expected}");
            }
        }
    }

    [Fact]
    public void GroupNorm()
    {
        long[] ne = { 16, 3, 4 }; // 4 channels, 2 groups of 2
        int nGroups = 2;
        float eps = 1e-5f;
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)src.Data, (int)src.NElements());
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { src });
        dst.Op = GgmlOp.GROUP_NORM;
        dst.OpParams[0] = (byte)nGroups;
        GgmlTensor.SetOpParamsF32(dst, 1, eps);

        Run(dst, 4);
        long ne0 = ne[0], ne1 = ne[1], ne2 = ne[2];
        for (long i2 = 0; i2 < ne2; i2++)
        {
            int g = (int)(i2 / (ne2 / nGroups));
            int start = g * (int)(ne2 / nGroups);
            int end = start + (int)(ne2 / nGroups);
            double sum = 0, sum2 = 0;
            long count = ne0 * ne1 * (end - start);
            for (int c = start; c < end; c++)
            for (long i1 = 0; i1 < ne1; i1++)
            for (long i0 = 0; i0 < ne0; i0++)
            {
                float v = ((float*)src.Data)[(c * ne1 + i1) * ne0 + i0];
                sum += v; sum2 += (double)v * v;
            }
            float mean = (float)(sum / count);
            float var = (float)(sum2 / count) - mean * mean;
            float scale = 1f / MathF.Sqrt(var + eps);
            for (long i1 = 0; i1 < ne1; i1++)
            for (long i0 = 0; i0 < ne0; i0++)
            {
                float expected = (((float*)src.Data)[(i2 * ne1 + i1) * ne0 + i0] - mean) * scale;
                float got = ((float*)dst.Data)[(i2 * ne1 + i1) * ne0 + i0];
                Assert.True(MathF.Abs(got - expected) <= Tol(expected, 1e-3f), $"got {got} exp {expected}");
            }
        }
    }

    [Fact]
    public void Softmax_WithScaleAndMask()
    {
        long[] ne = { 16, 3, 2 };
        long ne0 = ne[0];
        float scale = 0.5f;
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)src.Data, (int)src.NElements());
        var mask = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)mask.Data, (int)mask.NElements());
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { src, mask });
        dst.Op = GgmlOp.SOFT_MAX;
        GgmlTensor.SetOpParamsF32(dst, 0, scale);
        GgmlTensor.SetOpParamsF32(dst, 1, 0f); // no ALiBi

        Run(dst, 4);
        int rows = (int)(ne[1] * ne[2]);
        for (int r = 0; r < rows; r++)
        {
            float* x = (float*)src.Data + r * ne0;
            float* m = (float*)mask.Data + r * ne0;
            float* y = (float*)dst.Data + r * ne0;
            float max = float.NegativeInfinity;
            for (int i = 0; i < ne0; i++) max = MathF.Max(max, x[i] * scale + m[i]);
            float sum = 0;
            for (int i = 0; i < ne0; i++) sum += MathF.Exp(x[i] * scale + m[i] - max);
            for (int i = 0; i < ne0; i++)
            {
                float expected = MathF.Exp(x[i] * scale + m[i] - max) / sum;
                Assert.True(MathF.Abs(y[i] - expected) <= Tol(expected), $"got {y[i]} exp {expected}");
            }
        }
    }

    [Fact]
    public void LeakyRelu()
    {
        long[] ne = { 16, 3, 2 };
        long ne0 = ne[0];
        float slope = 0.1f;
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)src.Data, (int)src.NElements());
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { src });
        dst.Op = GgmlOp.LEAKY_RELU;
        GgmlTensor.SetOpParamsF32(dst, 0, slope);
        Run(dst, 1);
        for (int i = 0; i < (int)(ne[0] * ne[1] * ne[2]); i++)
        {
            float x = ((float*)src.Data)[i];
            float expected = x > 0 ? x : x * slope;
            Assert.Equal(expected, ((float*)dst.Data)[i], 4);
        }
    }

    [Fact]
    public void SiluBack()
    {
        long[] ne = { 16, 3, 2 };
        long ne0 = ne[0];
        using var ctx = new GgmlContext();
        var x = ctx.NewTensor(GgmlType.F32, 3, ne);
        var dz = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)x.Data, (int)x.NElements());
        Fill((float*)dz.Data, (int)dz.NElements());
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { dz, x });
        dst.Op = GgmlOp.SILU_BACK;
        Run(dst, 4);
        for (int i = 0; i < (int)ne0 * (int)(ne[1] * ne[2]); i++)
        {
            float xi = ((float*)x.Data)[i];
            float s = 1f / (1f + MathF.Exp(-xi));
            float expected = ((float*)dz.Data)[i] * (s * (1f + xi * (1f - s)));
            Assert.True(MathF.Abs(((float*)dst.Data)[i] - expected) <= Tol(expected));
        }
    }

    [Fact]
    public void Scale_WithBias()
    {
        long[] ne = { 16, 3, 2 };
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)src.Data, (int)src.NElements());
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { src });
        dst.Op = GgmlOp.SCALE;
        GgmlTensor.SetOpParamsF32(dst, 0, 2.5f);
        GgmlTensor.SetOpParamsF32(dst, 1, 0.5f);
        Run(dst, 4);
        for (int i = 0; i < (int)(ne[0] * ne[1] * ne[2]); i++)
        {
            float expected = ((float*)src.Data)[i] * 2.5f + 0.5f;
            Assert.True(MathF.Abs(((float*)dst.Data)[i] - expected) <= Tol(expected));
        }
    }
}