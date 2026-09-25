using System.Runtime.InteropServices;

namespace GgmlSharp.Kernel;

/// <summary>
/// Ports of the GGML "remap/reshape/utility" ops from <c>ops.cpp</c>: DUP/CPY/CONT,
/// REPEAT(+BACK), SET, PAD, PAD_REFLECT_1D, ROLL, ARANGE, FILL, TRI, ARGSORT, TOP_K,
/// TIMESTEP_EMBEDDING, RMS_NORM_BACK. Ops that need a shared memcpy/barrier (SET,
/// REPEAT_BACK, non-inplace SET) run on thread 0 only for correctness.
/// </summary>
public static unsafe class OpsRemap
{
    // ---- DUP / CPY / CONT: copy src0 to dst, converting type via codecs -----------------

    public static void ForwardDup(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];

        // Fast path: same type + contiguous rows on both sides -> plain memcpy.
        if (src0.Type == dst.Type)
        {
            if (src0.Nb[0] == dst.Nb[0] && src0.Nb[0] == GgmlTypeTraits.Get(src0.Type).TypeSize)
            {
                NativeMemory.Copy(src0.Data, dst.Data, (nuint)(dst.NElements() * dst.Nb[0]));
                return;
            }
        }

        var toF = GgmlTypeTraits.Get(src0.Type).ToFloat
            ?? throw new NotSupportedException($"dup src0 type {src0.Type} not yet supported");
        var fromF = GgmlTypeTraits.Get(dst.Type).FromFloatRef
            ?? throw new NotSupportedException($"dup dst type {dst.Type} not yet supported");

        int n = (int)ne0;
        var scratch = (float*)NativeMemory.AlignedAlloc((nuint)(n * 4), 64);
        try
        {
            var (ir0, ir1) = p.ThreadRowRange(src0.NRows());
            for (long ir = ir0; ir < ir1; ir++)
            {
                long i3 = ir / (ne2 * ne1);
                long i2 = (ir - i3 * ne2 * ne1) / ne1;
                long i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
                byte* s = src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01;
                byte* d = dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1;
                toF(s, scratch, n);
                fromF(scratch, d, n);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(scratch);
        }
    }

    // ---- REPEAT: tile src0 to fill dst ------------------------------------------------

    public static void ForwardRepeat(in ComputeParams p, GgmlTensor dst)
    {
        if (p.Ith != 0) return;
        var src0 = dst.Src[0]!;
        long ne0 = dst.Ne[0], ne1 = dst.Ne[1], ne2 = dst.Ne[2], ne3 = dst.Ne[3];
        long ne00 = src0.Ne[0], ne01 = src0.Ne[1], ne02 = src0.Ne[2], ne03 = src0.Ne[3];
        long nb0 = dst.Nb[0], nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        int es = (int)GgmlTypeTraits.Get(src0.Type).TypeSize;
        long nr0 = ne0 / ne00, nr1 = ne1 / ne01, nr2 = ne2 / ne02, nr3 = ne3 / ne03;
        long rowBytes = ne00 * es;

        for (long i3 = 0; i3 < nr3; i3++)
        for (long k3 = 0; k3 < ne03; k3++)
        for (long i2 = 0; i2 < nr2; i2++)
        for (long k2 = 0; k2 < ne02; k2++)
        for (long i1 = 0; i1 < nr1; i1++)
        for (long k1 = 0; k1 < ne01; k1++)
        for (long i0 = 0; i0 < nr0; i0++)
        {
            byte* d = dst.Data + (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 + (i1 * ne01 + k1) * nb1 + i0 * ne00 * nb0;
            byte* s = src0.Data + k3 * nb03 + k2 * nb02 + k1 * nb01;
            NativeMemory.Copy(s, d, (nuint)rowBytes);
        }
    }

    // ---- REPEAT_BACK: gradient, sum src0 tiles into dst ---------------------------------

    public static void ForwardRepeatBack(in ComputeParams p, GgmlTensor dst)
    {
        if (p.Ith != 0) return;
        var src0 = dst.Src[0]!;
        long ne0 = dst.Ne[0], ne1 = dst.Ne[1], ne2 = dst.Ne[2], ne3 = dst.Ne[3];
        long ne00 = src0.Ne[0], ne01 = src0.Ne[1], ne02 = src0.Ne[2], ne03 = src0.Ne[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nr0 = ne00 / ne0, nr1 = ne01 / ne1, nr2 = ne02 / ne2, nr3 = ne03 / ne3;

        // zero dst (contiguous)
        NativeMemory.Clear(dst.Data, (nuint)(dst.NElements() * dst.Nb[0]));

        for (long i3 = 0; i3 < nr3; i3++)
        for (long k3 = 0; k3 < ne3; k3++)
        for (long i2 = 0; i2 < nr2; i2++)
        for (long k2 = 0; k2 < ne2; k2++)
        for (long i1 = 0; i1 < nr1; i1++)
        for (long k1 = 0; k1 < ne1; k1++)
        for (long i0 = 0; i0 < nr0; i0++)
        {
            float* d = (float*)(dst.Data + k3 * nb3 + k2 * nb2 + k1 * nb1);
            float* s = (float*)(src0.Data + (i3 * ne3 + k3) * nb03 + (i2 * ne2 + k2) * nb02 + (i1 * ne1 + k1) * nb01 + i0 * ne0 * src0.Nb[0]);
            for (int i = 0; i < (int)ne0; i++) d[i] += s[i];
        }
    }

    // ---- SET: dst = copy(src0) then overwrite a slice from src1 ---------------------------

    public static void ForwardSet(in ComputeParams p, GgmlTensor dst)
    {
        if (p.Ith != 0) return;
        var src0 = dst.Src[0]!;
        var src1 = dst.Src[1]!;
        long nb1 = GetI32(dst, 0);
        long nb2 = GetI32(dst, 1);
        long nb3 = GetI32(dst, 2);
        long offset = GetI32(dst, 3);
        bool inplace = GetI32(dst, 4) != 0;

        if (!inplace)
            NativeMemory.Copy(src0.Data, dst.Data, (nuint)(dst.NElements() * dst.Nb[0]));

        int nc = (int)src1.Ne[0];
        long ne11 = src1.Ne[1], ne12 = src1.Ne[2];
        long nb12 = src1.Nb[2], nb13 = src1.Nb[3];
        for (long i2 = 0; i2 < ne12; i2++)
        for (long i1 = 0; i1 < ne11; i1++)
        {
            byte* s = src1.Data + i2 * nb13 + i1 * nb12;
            byte* d = dst.Data + i2 * nb3 + i1 * nb2 + offset;
            NativeMemory.Copy(s, d, (nuint)(nc * 4));
        }
    }

    // ---- PAD: zero/circular pad in each dim ------------------------------------------------

    public static void ForwardPad(in ComputeParams p, GgmlTensor dst, bool circular)
    {
        var src0 = dst.Src[0]!;
        long ne0 = dst.Ne[0], ne1 = dst.Ne[1], ne2 = dst.Ne[2], ne3 = dst.Ne[3];
        long ne00 = src0.Ne[0], ne01 = src0.Ne[1], ne02 = src0.Ne[2], ne03 = src0.Ne[3];
        long nb00 = src0.Nb[0], nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        int lp0 = GetI32(dst, 0), rp0 = GetI32(dst, 1), lp1 = GetI32(dst, 2), rp1 = GetI32(dst, 3);
        int lp2 = GetI32(dst, 4), rp2 = GetI32(dst, 5), lp3 = GetI32(dst, 6), rp3 = GetI32(dst, 7);

        float* dstData = (float*)dst.Data;
        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = p.Ith; i1 < ne1; i1 += p.Nth)
        for (long i0 = 0; i0 < ne0; i0++)
        {
            long dstIdx = i3 * ne0 * ne1 * ne2 + i2 * ne0 * ne1 + i1 * ne0 + i0;
            if (circular)
            {
                long si0 = WrapAround(i0 - lp0, ne00), si1 = WrapAround(i1 - lp1, ne01), si2 = WrapAround(i2 - lp2, ne02), si3 = WrapAround(i3 - lp3, ne03);
                dstData[dstIdx] = *(float*)(src0.Data + si3 * nb03 + si2 * nb02 + si1 * nb01 + si0 * nb00);
            }
            else
            {
                if (i0 >= lp0 && i0 < ne0 - rp0 && i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3)
                {
                    long si0 = i0 - lp0, si1 = i1 - lp1, si2 = i2 - lp2, si3 = i3 - lp3;
                    dstData[dstIdx] = *(float*)(src0.Data + si3 * nb03 + si2 * nb02 + si1 * nb01 + si0 * nb00);
                }
                else dstData[dstIdx] = 0;
            }
        }
    }

    public static void ForwardPadReflect1D(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        int p0 = GetI32(dst, 0), p1 = GetI32(dst, 1);
        long ne0 = dst.Ne[0], ne1 = dst.Ne[1], ne2 = dst.Ne[2], ne3 = dst.Ne[3];
        long ne00 = src0.Ne[0];
        long nb0 = dst.Nb[0], nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];

        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = p.Ith; i1 < ne1; i1 += p.Nth)
        {
            float* left = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1 + p0 * nb0);
            float* right = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1 + (ne0 - p1 - 1) * nb0);
            float* src = (float*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            NativeMemory.Copy(src, left, (nuint)(ne00 * 4));
            for (int i = 1; i <= p0; i++) left[-i] = left[i];
            for (int i = 1; i <= p1; i++) right[i] = right[-i];
        }
    }

    // ---- ROLL -----------------------------------------------------------------------------

    public static void ForwardRoll(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        float* src = (float*)src0.Data;
        float* d = (float*)dst.Data;
        long ne0 = dst.Ne[0], ne1 = dst.Ne[1], ne2 = dst.Ne[2], ne3 = dst.Ne[3];
        long ne00 = src0.Ne[0], ne01 = src0.Ne[1], ne02 = src0.Ne[2], ne03 = src0.Ne[3];
        long nb0 = dst.Nb[0], nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb00 = src0.Nb[0], nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        int s0 = GetI32(dst, 0), s1 = GetI32(dst, 1), s2 = GetI32(dst, 2), s3 = GetI32(dst, 3);

        long total = ne1 * ne2 * ne3;
        long perThread = (total + p.Nth) / p.Nth;
        long start = p.Ith * perThread;
        long end = Math.Min(start + perThread, total);
        for (long i = start; i < end; i++)
        {
            long i1 = i % ne1;
            long i2 = (i / ne1) % ne2;
            long i3 = i / (ne2 * ne1);
            float* dstRow = d + (i3 * nb3 + i2 * nb2 + i1 * nb1) / sizeof(float);
            long i01 = WrapIndex(i1 - s1, ne01), i02 = WrapIndex(i2 - s2, ne02), i03 = WrapIndex(i3 - s3, ne03);
            float* srcRow = src + (i03 * nb03 + i02 * nb02 + i01 * nb01) / sizeof(float);
            long s = WrapIndex(-s0, ne00);
            long n = ne00 - s;
            NativeMemory.Copy(srcRow + s, dstRow, (nuint)(n * sizeof(float)));
            NativeMemory.Copy(srcRow, dstRow + n, (nuint)(s * sizeof(float)));
        }
    }

    // ---- ARANGE ---------------------------------------------------------------------------

    public static void ForwardArange(in ComputeParams p, GgmlTensor dst)
    {
        float start = GgmlTensor.GetOpParamsF32(dst, 0);
        float stop = GgmlTensor.GetOpParamsF32(dst, 1);
        float step = GgmlTensor.GetOpParamsF32(dst, 2);
        long steps = (long)MathF.Ceiling((stop - start) / step);
        for (long i = p.Ith; i < steps; i += p.Nth)
            ((float*)dst.Data)[i] = start + step * i;
    }

    // ---- FILL -----------------------------------------------------------------------------

    public static void ForwardFill(in ComputeParams p, GgmlTensor dst)
    {
        float c = GgmlTensor.GetOpParamsF32(dst, 0);
        long ne0 = dst.Ne[0], ne1 = dst.Ne[1], ne2 = dst.Ne[2];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        var (ir0, ir1) = p.ThreadRowRange(dst.NRows());
        for (long ir = ir0; ir < ir1; ir++)
        {
            long i3 = ir / (ne2 * ne1);
            long i2 = (ir - i3 * ne2 * ne1) / ne1;
            long i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
            float* row = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            for (int i = 0; i < (int)ne0; i++) row[i] = c;
        }
    }

    // ---- TRI --------------------------------------------------------------------------------

    public static void ForwardTri(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        var ttype = (GgmlTriType)GetI32(dst, 0);
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];

        var (ir0, ir1) = p.ThreadRowRange(src0.NRows());
        for (long ir = ir0; ir < ir1; ir++)
        {
            long i3 = ir / (ne2 * ne1);
            long i2 = (ir - i3 * ne2 * ne1) / ne1;
            long i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
            float* s = (float*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            float* d = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            for (int i0 = 0; i0 < (int)ne0; i0++)
                d[i0] = TriPred(ttype, i0, (int)i1) ? s[i0] : 0f;
        }
    }

    private static bool TriPred(GgmlTriType t, int i, int r) => t switch
    {
        GgmlTriType.LOWER => i < r,
        GgmlTriType.LOWER_DIAG => i <= r,
        GgmlTriType.UPPER => i > r,
        _ => i >= r,
    };

    // ---- ARGSORT ------------------------------------------------------------------------------

    public static void ForwardArgsort(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        var order = (GgmlSortOrder)GetI32(dst, 0);
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1];
        long nb01 = src0.Nb[1], nb1 = dst.Nb[1];

        for (long i1 = p.Ith; i1 < ne1; i1 += p.Nth)
        {
            float* s = (float*)(src0.Data + i1 * nb01);
            int* d = (int*)(dst.Data + i1 * nb1);
            int n = (int)ne0;
            var idx = new int[n];
            for (int j = 0; j < n; j++) idx[j] = j;
            Array.Sort(idx, (a, b) => order == GgmlSortOrder.ASC ? s[a].CompareTo(s[b]) : s[b].CompareTo(s[a]));
            for (int j = 0; j < n; j++) d[j] = idx[j];
        }
    }

    // ---- TOP_K ---------------------------------------------------------------------------------

    public static void ForwardTopK(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1];
        long nb01 = src0.Nb[1], nb1 = dst.Nb[1];
        int topK = (int)ne0;

        for (long i1 = p.Ith; i1 < ne1; i1 += p.Nth)
        {
            float* s = (float*)(src0.Data + i1 * nb01);
            int* d = (int*)(dst.Data + i1 * nb1);
            int n = (int)ne0;
            var idx = new int[n];
            for (int j = 0; j < n; j++) idx[j] = j;
            Array.Sort(idx, (a, b) => s[b].CompareTo(s[a])); // descending by value
            for (int j = 0; j < topK; j++) d[j] = idx[j];
            if (topK > 1) (d[0], d[1]) = (d[1], d[0]); // emphasize order is not important
        }
    }

    // ---- TIMESTEP_EMBEDDING ----------------------------------------------------------------------

    public static void ForwardTimestepEmbedding(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        int dim = GetI32(dst, 0);
        int maxPeriod = GetI32(dst, 1);
        int half = dim / 2;
        long ne00 = src0.Ne[0];
        long nb1 = dst.Nb[1];

        for (long i = 0; i < ne00; i++)
        {
            float* embed = (float*)(dst.Data + i * nb1);
            float timestep = ((float*)src0.Data)[i];
            for (long j = p.Ith; j < half; j += p.Nth)
            {
                float freq = MathF.Exp(-MathF.Log(maxPeriod) * j / half);
                float arg = timestep * freq;
                embed[j] = MathF.Cos(arg);
                embed[j + half] = MathF.Sin(arg);
            }
            if (dim % 2 != 0 && p.Ith == 0) embed[2 * half] = 0f;
        }
    }

    // ---- RMS_NORM_BACK: dz, x -> dx = (x*(-sum_xdz/sum_eps) + dz) * rrms ---------------------------

    public static void ForwardRmsNormBack(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!; // dz
        var src1 = dst.Src[1]!; // x
        float eps = GgmlTensor.GetOpParamsF32(dst, 0);
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb11 = src1.Nb[1], nb12 = src1.Nb[2], nb13 = src1.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        int n = (int)ne0;

        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = p.Ith; i1 < ne1; i1 += p.Nth)
        {
            float* dz = (float*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            float* x = (float*)(src1.Data + i3 * nb13 + i2 * nb12 + i1 * nb11);
            float* dx = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            double sumXx = 0, sumXdz = 0;
            for (int i = 0; i < n; i++) { sumXx += (double)x[i] * x[i]; sumXdz += (double)x[i] * dz[i]; }
            float meanEps = (float)(sumXx) / n + eps;
            float sumEps = (float)(sumXx) + eps * n;
            float rrms = 1f / MathF.Sqrt(meanEps);
            for (int i = 0; i < n; i++)
                dx[i] = (x[i] * (-(float)sumXdz) / sumEps + dz[i]) * rrms;
        }
    }

    private static int GetI32(GgmlTensor t, int i) => GgmlTensor.GetOpParamsI32(t, i);

    private static long WrapIndex(long i, long ne)
    {
        if (i < 0) return i + ne;
        if (i >= ne) return i - ne;
        return i;
    }

    private static long WrapAround(long i, long ne)
    {
        long r = i % ne;
        return r < 0 ? r + ne : r;
    }
}