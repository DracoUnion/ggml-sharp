using GgmlSharp.Kernel;
using Xunit;

namespace GgmlSharp.Tests;

public sealed unsafe class MatMulTests
{
    private static readonly Random Rng = new(999);

    private static void Fill(float* p, int n)
    {
        for (int i = 0; i < n; i++) p[i] = (float)(Rng.NextDouble() * 2.0 - 1.0);
    }

    private static void Run(GgmlTensor dst, int nth)
    {
        for (int ith = 0; ith < nth; ith++)
            GgmlCompute.Forward(new ComputeParams(ith, nth), dst);
    }

    [Fact]
    public void MulMat_F32()
    {
        int K = 16, N = 4, M = 3;
        using var ctx = new GgmlContext();
        var src0 = ctx.NewTensor(GgmlType.F32, 2, new long[] { K, N });
        var src1 = ctx.NewTensor(GgmlType.F32, 2, new long[] { K, M });
        Fill((float*)src0.Data, K * N);
        Fill((float*)src1.Data, K * M);
        var dst = ctx.NewTensor(GgmlType.F32, 2, new long[] { N, M }, new[] { src0, src1 });
        dst.Op = GgmlOp.MUL_MAT;

        Run(dst, 4);
        for (int i1 = 0; i1 < M; i1++)
        for (int i0 = 0; i0 < N; i0++)
        {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += ((float*)src0.Data)[i0 * K + k] * ((float*)src1.Data)[i1 * K + k];
            float got = ((float*)dst.Data)[i1 * N + i0];
            Assert.True(MathF.Abs(got - (float)acc) <= 1e-3f * MathF.Max(1f, MathF.Abs((float)acc)), $"got {got} exp {acc}");
        }
    }

    [Fact]
    public void MulMat_Q4_0_Weights()
    {
        int K = 32, N = 4, M = 3;
        using var ctx = new GgmlContext();
        var src0 = ctx.NewTensor(GgmlType.Q4_0, 2, new long[] { K, N });
        var src1 = ctx.NewTensor(GgmlType.F32, 2, new long[] { K, M });
        Fill((float*)src1.Data, K * M);

        // Fill 4 Q4_0 blocks (18 B each): d (fp16) + 16 nibble bytes.
        var dq = new float[K * N];
        for (int i = 0; i < N; i++)
        {
            byte* block = src0.Data + i * 18;
            float d = 0.5f + i;
            *(ushort*)block = GgmlMath.F32ToF16(d);
            for (int j = 0; j < 16; j++)
            {
                byte low = (byte)((j * 7 + i * 3) % 16);
                byte high = (byte)((j * 11 + i) % 16);
                block[2 + j] = (byte)(low | (high << 4));
                dq[i * K + j] = (low - 8) * d;
                dq[i * K + j + 16] = (high - 8) * d;
            }
        }

        var dst = ctx.NewTensor(GgmlType.F32, 2, new long[] { N, M }, new[] { src0, src1 });
        dst.Op = GgmlOp.MUL_MAT;
        Run(dst, 4);

        for (int i1 = 0; i1 < M; i1++)
        for (int i0 = 0; i0 < N; i0++)
        {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += dq[i0 * K + k] * ((float*)src1.Data)[i1 * K + k];
            float got = ((float*)dst.Data)[i1 * N + i0];
            Assert.True(MathF.Abs(got - (float)acc) <= 1e-3f * MathF.Max(1f, MathF.Abs((float)acc)), $"got {got} exp {acc}");
        }
    }

    [Fact]
    public void OutProd_F32()
    {
        int i0Dim = 4, kDim = 16, i1Dim = 3;
        using var ctx = new GgmlContext();
        var src0 = ctx.NewTensor(GgmlType.F32, 2, new long[] { i0Dim, kDim });
        var src1 = ctx.NewTensor(GgmlType.F32, 2, new long[] { i1Dim, kDim });
        Fill((float*)src0.Data, i0Dim * kDim);
        Fill((float*)src1.Data, i1Dim * kDim);
        var dst = ctx.NewTensor(GgmlType.F32, 2, new long[] { i0Dim, i1Dim }, new[] { src0, src1 });
        dst.Op = GgmlOp.OUT_PROD;

        Run(dst, 4);
        for (int i1 = 0; i1 < i1Dim; i1++)
        for (int i0 = 0; i0 < i0Dim; i0++)
        {
            float acc = 0;
            for (int k = 0; k < kDim; k++) acc += ((float*)src0.Data)[k * i0Dim + i0] * ((float*)src1.Data)[k * i1Dim + i1];
            float got = ((float*)dst.Data)[i1 * i0Dim + i0];
            Assert.True(MathF.Abs(got - acc) <= 1e-3f * MathF.Max(1f, MathF.Abs(acc)), $"got {got} exp {acc}");
        }
    }
}