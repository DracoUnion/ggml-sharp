using GgmlSharp.Kernel;
using GgmlSharp.Graph;
using Xunit;

namespace GgmlSharp.Tests;

public sealed unsafe class GraphTests
{
    private static readonly Random Rng = new(2024);

    private static void Fill(float* p, int n)
    {
        for (int i = 0; i < n; i++) p[i] = (float)(Rng.NextDouble() * 2.0 - 1.0);
    }

    [Fact]
    public void Layer_MulMat_Add_Silu_RmsNorm()
    {
        int K = 16, N = 4, M = 3;
        using var g = new GgmlGraph();
        var x = g.NewTensor(GgmlType.F32, K, M);
        var w = g.NewTensor(GgmlType.F32, K, N);
        var bias = g.NewTensor(GgmlType.F32, N, M);
        Fill((float*)x.Data, K * M);
        Fill((float*)w.Data, K * N);
        Fill((float*)bias.Data, N * M);

        var mm = g.MulMat(w, x);
        var sum = g.Add(mm, bias);
        var act = g.Silu(sum);
        var outT = g.RmsNorm(act, 1e-5f);

        g.Execute(nth: 4);

        // Reference.
        var mmRef = new float[N * M];
        for (int i1 = 0; i1 < M; i1++)
        for (int i0 = 0; i0 < N; i0++)
        {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += ((float*)w.Data)[i0 * K + k] * ((float*)x.Data)[i1 * K + k];
            mmRef[i1 * N + i0] = (float)acc;
        }
        var silu = new float[N * M];
        for (int i = 0; i < N * M; i++)
        {
            float s = mmRef[i] + ((float*)bias.Data)[i];
            silu[i] = s / (1f + MathF.Exp(-s));
        }
        for (int i1 = 0; i1 < M; i1++)
        {
            float sumSq = 0;
            for (int i0 = 0; i0 < N; i0++) sumSq += silu[i1 * N + i0] * silu[i1 * N + i0];
            float scale = 1f / MathF.Sqrt(sumSq / N + 1e-5f);
            for (int i0 = 0; i0 < N; i0++)
            {
                float expected = silu[i1 * N + i0] * scale;
                float got = ((float*)outT.Data)[i1 * N + i0];
                Assert.True(MathF.Abs(got - expected) <= 1e-3f * MathF.Max(1f, MathF.Abs(expected)), $"got {got} exp {expected}");
            }
        }
    }

    [Fact]
    public void Graph_Rope_Neox()
    {
        int dim = 8, heads = 2, seq = 4;
        float freqBase = 10000f;
        using var g = new GgmlGraph();
        var a = g.NewTensor(GgmlType.F32, dim, heads, seq);
        Fill((float*)a.Data, dim * heads * seq);
        var pos = new int[seq];
        for (int i = 0; i < seq; i++) pos[i] = i * 3;
        var rope = g.Rope(a, pos, dim, GgmlRopeType.NEOX, freqBase);
        g.Execute(nth: 4);

        int half = dim / 2;
        for (int s = 0; s < seq; s++)
        for (int h = 0; h < heads; h++)
        {
            float* row = (float*)a.Data + (s * heads + h) * dim;
            float* outr = (float*)rope.Data + (s * heads + h) * dim;
            for (int p = 0; p < half; p++)
            {
                float theta = pos[s] * MathF.Pow(freqBase, -2.0f * p / dim);
                float cos = MathF.Cos(theta), sin = MathF.Sin(theta);
                float x0 = row[p], x1 = row[p + half];
                Assert.True(MathF.Abs(outr[p] - (x0 * cos - x1 * sin)) <= 1e-4f);
                Assert.True(MathF.Abs(outr[p + half] - (x0 * sin + x1 * cos)) <= 1e-4f);
            }
        }
    }
}