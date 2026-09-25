using GgmlSharp.Kernel;
using Xunit;

namespace GgmlSharp.Tests;

public sealed unsafe class RemapOpsTests
{
    private static readonly Random Rng = new(31337);

    private static void Fill(float* p, int n)
    {
        for (int i = 0; i < n; i++) p[i] = (float)(Rng.NextDouble() * 4.0 - 2.0);
    }

    private static void Run(GgmlTensor dst, int nth)
    {
        for (int ith = 0; ith < nth; ith++)
            GgmlCompute.Forward(new ComputeParams(ith, nth), dst);
    }

    private static float Tol(float exp) => 1e-4f * MathF.Max(1f, MathF.Abs(exp));

    [Fact]
    public void Dup_CastsF16ToF32()
    {
        long[] ne = { 16, 3, 2 };
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F16, 3, ne);
        for (int i = 0; i < (int)(ne[0] * ne[1] * ne[2]); i++)
            ((ushort*)src.Data)[i] = GgmlMath.F32ToF16((float)(Rng.NextDouble() * 4.0 - 2.0));
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { src });
        dst.Op = GgmlOp.CPY;
        Run(dst, 4);
        for (int i = 0; i < (int)(ne[0] * ne[1] * ne[2]); i++)
        {
            float expected = GgmlMath.F16ToF32(((ushort*)src.Data)[i]);
            Assert.True(MathF.Abs(((float*)dst.Data)[i] - expected) <= Tol(expected));
        }
    }

    [Fact]
    public void Repeat_Tiles()
    {
        long[] srcNe = { 4, 2 };
        long[] dstNe = { 8, 6 };
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 2, srcNe);
        Fill((float*)src.Data, (int)(srcNe[0] * srcNe[1]));
        var dst = ctx.NewTensor(GgmlType.F32, 2, dstNe, new[] { src });
        dst.Op = GgmlOp.REPEAT;
        Run(dst, 1);
        for (long i1 = 0; i1 < dstNe[1]; i1++)
        for (long i0 = 0; i0 < dstNe[0]; i0++)
        {
            float expected = ((float*)src.Data)[(i1 % srcNe[1]) * srcNe[0] + i0 % srcNe[0]];
            Assert.Equal(expected, ((float*)dst.Data)[i1 * dstNe[0] + i0], 5);
        }
    }

    [Fact]
    public void RepeatBack_SumsTiles()
    {
        long[] srcNe = { 8, 6 };
        long[] dstNe = { 4, 2 };
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 2, srcNe);
        Fill((float*)src.Data, (int)(srcNe[0] * srcNe[1]));
        var dst = ctx.NewTensor(GgmlType.F32, 2, dstNe, new[] { src });
        dst.Op = GgmlOp.REPEAT_BACK;
        Run(dst, 1);
        int nr0 = (int)(srcNe[0] / dstNe[0]), nr1 = (int)(srcNe[1] / dstNe[1]);
        for (long i1 = 0; i1 < dstNe[1]; i1++)
        for (long i0 = 0; i0 < dstNe[0]; i0++)
        {
            float acc = 0;
            for (int a = 0; a < nr1; a++)
            for (int b = 0; b < nr0; b++)
                acc += ((float*)src.Data)[(a * dstNe[1] + i1) * srcNe[0] + (b * dstNe[0] + i0)];
            Assert.True(MathF.Abs(((float*)dst.Data)[i1 * dstNe[0] + i0] - acc) <= Tol(acc));
        }
    }

    [Fact]
    public void Set_OverwritesSlice()
    {
        long[] ne = { 8, 3 };
        long[] src1Ne = { 4, 1 };
        using var ctx = new GgmlContext();
        var src0 = ctx.NewTensor(GgmlType.F32, 2, ne);
        Fill((float*)src0.Data, (int)(ne[0] * ne[1]));
        var src1 = ctx.NewTensor(GgmlType.F32, 2, src1Ne);
        Fill((float*)src1.Data, (int)(src1Ne[0] * src1Ne[1]));
        var dst = ctx.NewTensor(GgmlType.F32, 2, ne, new[] { src0, src1 });
        dst.Op = GgmlOp.SET;
        GgmlTensor.SetOpParamsI32(dst, 0, 4);   // nb1 view stride (bytes)
        GgmlTensor.SetOpParamsI32(dst, 1, 0);
        GgmlTensor.SetOpParamsI32(dst, 2, 0);
        GgmlTensor.SetOpParamsI32(dst, 3, 0);   // offset
        GgmlTensor.SetOpParamsI32(dst, 4, 0);   // inplace=false
        Run(dst, 1);
        // copy then overwrite first 4 elements of each row? src1 ne0=4, ne1=1 -> overwrites [0..4) of row 0
        for (long i1 = 0; i1 < ne[1]; i1++)
        for (long i0 = 0; i0 < ne[0]; i0++)
        {
            float expected = i1 == 0 && i0 < src1Ne[0] ? ((float*)src1.Data)[i0] : ((float*)src0.Data)[i1 * ne[0] + i0];
            Assert.Equal(expected, ((float*)dst.Data)[i1 * ne[0] + i0], 5);
        }
    }

    [Fact]
    public void Pad_ZeroAndCircular()
    {
        long[] srcNe = { 4, 3 };
        long[] dstNe = { 7, 3 }; // lp0=1, rp0=2
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 2, srcNe);
        Fill((float*)src.Data, (int)(srcNe[0] * srcNe[1]));
        foreach (bool circular in new[] { false, true })
        {
            var dst = ctx.NewTensor(GgmlType.F32, 2, dstNe, new[] { src });
            dst.Op = GgmlOp.PAD;
            GgmlTensor.SetOpParamsI32(dst, 0, 1); GgmlTensor.SetOpParamsI32(dst, 1, 2); // dim0 l,r
            GgmlTensor.SetOpParamsI32(dst, 2, 0); GgmlTensor.SetOpParamsI32(dst, 3, 0);
            GgmlTensor.SetOpParamsI32(dst, 4, 0); GgmlTensor.SetOpParamsI32(dst, 5, 0);
            GgmlTensor.SetOpParamsI32(dst, 6, 0); GgmlTensor.SetOpParamsI32(dst, 7, 0);
            GgmlTensor.SetOpParamsI32(dst, 8, circular ? 1 : 0);
            Run(dst, 4);
            for (long i1 = 0; i1 < dstNe[1]; i1++)
            for (long i0 = 0; i0 < dstNe[0]; i0++)
            {
                float expected;
                if (circular)
                    expected = ((float*)src.Data)[i1 * srcNe[0] + Wrap(i0 - 1, srcNe[0])];
                else if (i0 >= 1 && i0 < 5)
                    expected = ((float*)src.Data)[i1 * srcNe[0] + (i0 - 1)];
                else expected = 0;
                Assert.Equal(expected, ((float*)dst.Data)[i1 * dstNe[0] + i0], 5);
            }
        }
    }

    private static long Wrap(long i, long n) => ((i % n) + n) % n;

    [Fact]
    public void Roll()
    {
        long[] ne = { 16, 3, 2 };
        int s0 = 3;
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)src.Data, (int)(ne[0] * ne[1] * ne[2]));
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { src });
        dst.Op = GgmlOp.ROLL;
        GgmlTensor.SetOpParamsI32(dst, 0, s0);
        GgmlTensor.SetOpParamsI32(dst, 1, 0);
        GgmlTensor.SetOpParamsI32(dst, 2, 0);
        GgmlTensor.SetOpParamsI32(dst, 3, 0);
        Run(dst, 4);
        long rows = ne[1] * ne[2];
        for (long r = 0; r < rows; r++)
        for (long i0 = 0; i0 < ne[0]; i0++)
        {
            long si = ((i0 - s0) % ne[0] + ne[0]) % ne[0];
            float expected = ((float*)src.Data)[r * ne[0] + si];
            Assert.Equal(expected, ((float*)dst.Data)[r * ne[0] + i0], 5);
        }
    }

    [Fact]
    public void Arange()
    {
        using var ctx = new GgmlContext();
        var dst = ctx.NewTensor(GgmlType.F32, 1, new long[] { 5 });
        dst.Op = GgmlOp.ARANGE;
        GgmlTensor.SetOpParamsF32(dst, 0, 0f);
        GgmlTensor.SetOpParamsF32(dst, 1, 10f);
        GgmlTensor.SetOpParamsF32(dst, 2, 2f);
        Run(dst, 1);
        for (int i = 0; i < 5; i++) Assert.Equal(2f * i, ((float*)dst.Data)[i], 5);
    }

    [Fact]
    public void Fill_Constant()
    {
        using var ctx = new GgmlContext();
        var dst = ctx.NewTensor(GgmlType.F32, 2, new long[] { 8, 4 });
        dst.Op = GgmlOp.FILL;
        GgmlTensor.SetOpParamsF32(dst, 0, 3.25f);
        Run(dst, 4);
        for (int i = 0; i < 32; i++) Assert.Equal(3.25f, ((float*)dst.Data)[i], 5);
    }

    [Fact]
    public void Tri_LowerDiag()
    {
        long[] ne = { 8, 8 };
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 2, ne);
        Fill((float*)src.Data, 64);
        var dst = ctx.NewTensor(GgmlType.F32, 2, ne, new[] { src });
        dst.Op = GgmlOp.TRI;
        GgmlTensor.SetOpParamsI32(dst, 0, (int)GgmlTriType.LOWER_DIAG);
        Run(dst, 4);
        for (long i1 = 0; i1 < 8; i1++)
        for (long i0 = 0; i0 < 8; i0++)
        {
            float expected = i0 <= i1 ? ((float*)src.Data)[i1 * 8 + i0] : 0f;
            Assert.Equal(expected, ((float*)dst.Data)[i1 * 8 + i0], 5);
        }
    }

    [Fact]
    public void Argsort_Asc()
    {
        long[] ne = { 16, 3 };
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 2, ne);
        Fill((float*)src.Data, 48);
        var dst = ctx.NewTensor(GgmlType.I32, 2, ne, new[] { src });
        dst.Op = GgmlOp.ARGSORT;
        GgmlTensor.SetOpParamsI32(dst, 0, (int)GgmlSortOrder.ASC);
        Run(dst, 4);
        for (long i1 = 0; i1 < ne[1]; i1++)
        {
            var vals = new float[16];
            for (int j = 0; j < 16; j++) vals[j] = ((float*)src.Data)[i1 * 16 + j];
            var expected = Enumerable.Range(0, 16).OrderBy(j => vals[j]).ToArray();
            for (int j = 0; j < 16; j++)
                Assert.Equal(expected[j], ((int*)dst.Data)[i1 * 16 + j]);
        }
    }

    [Fact]
    public void TopK()
    {
        long[] ne = { 16, 3 };
        int k = 4;
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 2, ne);
        Fill((float*)src.Data, 48);
        var dst = ctx.NewTensor(GgmlType.I32, 2, new long[] { k, ne[1] }, new[] { src });
        dst.Op = GgmlOp.TOP_K;
        Run(dst, 4);
        for (long i1 = 0; i1 < ne[1]; i1++)
        {
            var vals = new float[16];
            for (int j = 0; j < 16; j++) vals[j] = ((float*)src.Data)[i1 * 16 + j];
            var topVals = Enumerable.Range(0, 16).OrderByDescending(j => vals[j]).Take(k).ToArray();
            var got = Enumerable.Range(0, k).Select(j => ((int*)dst.Data)[i1 * k + j]).ToArray();
            Assert.Equal(topVals.OrderBy(x => x), got.OrderBy(x => x)); // order within top-k is unspecified
        }
    }

    [Fact]
    public void TimestepEmbedding()
    {
        int dim = 6, maxPeriod = 10000;
        long[] srcNe = { 4 };
        using var ctx = new GgmlContext();
        var src = ctx.NewTensor(GgmlType.F32, 1, srcNe);
        for (int i = 0; i < 4; i++) ((float*)src.Data)[i] = i * 5f;
        var dst = ctx.NewTensor(GgmlType.F32, 2, new long[] { dim, srcNe[0] }, new[] { src }); // [ne0=dim, ne1=timesteps]
        dst.Op = GgmlOp.TIMESTEP_EMBEDDING;
        GgmlTensor.SetOpParamsI32(dst, 0, dim);
        GgmlTensor.SetOpParamsI32(dst, 1, maxPeriod);
        Run(dst, 4);
        int half = dim / 2;
        for (long i = 0; i < 4; i++)
        {
            float t = ((float*)src.Data)[i];
            for (int j = 0; j < half; j++)
            {
                float freq = MathF.Exp(-MathF.Log(maxPeriod) * j / half);
                float arg = t * freq;
                float cg = ((float*)dst.Data)[i * dim + j];
                float ce = MathF.Cos(arg);
                Assert.True(MathF.Abs(cg - ce) <= Tol(ce), $"cos i={i} j={j}: got {cg} exp {ce}");
                float sg = ((float*)dst.Data)[i * dim + j + half];
                float se = MathF.Sin(arg);
                Assert.True(MathF.Abs(sg - se) <= Tol(se), $"sin i={i} j={j}: got {sg} exp {se}");
            }
        }
    }

    [Fact]
    public void RmsNormBack()
    {
        long[] ne = { 16, 3, 2 };
        float eps = 1e-5f;
        using var ctx = new GgmlContext();
        var dz = ctx.NewTensor(GgmlType.F32, 3, ne);
        var x = ctx.NewTensor(GgmlType.F32, 3, ne);
        Fill((float*)dz.Data, (int)(ne[0] * ne[1] * ne[2]));
        Fill((float*)x.Data, (int)(ne[0] * ne[1] * ne[2]));
        var dst = ctx.NewTensor(GgmlType.F32, 3, ne, new[] { dz, x });
        dst.Op = GgmlOp.RMS_NORM_BACK;
        GgmlTensor.SetOpParamsF32(dst, 0, eps);
        Run(dst, 4);
        int rows = (int)(ne[1] * ne[2]);
        for (int r = 0; r < rows; r++)
        {
            float* xr = (float*)x.Data + r * ne[0];
            float* dzr = (float*)dz.Data + r * ne[0];
            float* dr = (float*)dst.Data + r * ne[0];
            double sumXx = 0, sumXdz = 0;
            for (int i = 0; i < ne[0]; i++) { sumXx += (double)xr[i] * xr[i]; sumXdz += (double)xr[i] * dzr[i]; }
            float sumEps = (float)sumXx + eps * ne[0];
            float rrms = 1f / MathF.Sqrt((float)(sumXx) / ne[0] + eps);
            for (int i = 0; i < ne[0]; i++)
            {
                float expected = (xr[i] * (-(float)sumXdz) / sumEps + dzr[i]) * rrms;
                Assert.True(MathF.Abs(dr[i] - expected) <= Tol(expected), $"got {dr[i]} exp {expected}");
            }
        }
    }
}