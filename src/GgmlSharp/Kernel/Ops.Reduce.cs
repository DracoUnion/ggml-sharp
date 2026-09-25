using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace GgmlSharp.Kernel;

/// <summary>
/// Ports of the GGML reduction ops from <c>ops.cpp</c>: SUM, SUM_ROWS, MEAN, ARGMAX,
/// COUNT_EQUAL, CUMSUM. Reductions that fold to a scalar run on thread 0 only (the C code
/// returns early for ith != 0); COUNT_EQUAL is likewise done on thread 0 to avoid the shared
/// wdata/barrier machinery (correct, single-threaded for now).
/// </summary>
public static unsafe class OpsReduce
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float RowSum(float* x, int n)
    {
        int i = 0;
        float sum = 0;
        if (Avx2.IsSupported)
        {
            var acc = Vector256<float>.Zero;
            for (; i <= n - 8; i += 8)
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(Vector256.Create(1f), Avx.LoadVector256(x + i), acc)
                    : Avx.Add(acc, Avx.LoadVector256(x + i));
            sum = HorizontalSum(acc);
        }
        for (; i < n; i++) sum += x[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalSum(Vector256<float> v)
    {
        var x = Sse.Add(v.GetLower(), v.GetUpper());
        x = Sse.Add(x, Sse.MoveHighToLow(x, x));
        x = Sse.Add(x, Sse.Shuffle(x, x, 0x55));
        return x.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Scale(float* y, int n, float s)
    {
        int i = 0;
        if (Avx.IsSupported)
        {
            var sv = Vector256.Create(s);
            for (; i <= n - 8; i += 8)
                Avx.Store(y + i, Avx.Multiply(Avx.LoadVector256(y + i), sv));
        }
        for (; i < n; i++) y[i] *= s;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Mad1(float* y, float* x, int n, float s, float b)
    {
        int i = 0;
        if (Avx.IsSupported)
        {
            var sv = Vector256.Create(s);
            var bv = Vector256.Create(b);
            for (; i <= n - 8; i += 8)
                Avx.Store(y + i, Fma.IsSupported
                    ? Fma.MultiplyAdd(Avx.LoadVector256(x + i), sv, bv)
                    : Avx.Add(Avx.Multiply(Avx.LoadVector256(x + i), sv), bv));
        }
        for (; i < n; i++) y[i] = x[i] * s + b;
    }

    // ---- SUM: reduce all elements to a scalar ---------------------------------

    public static void ForwardSum(in ComputeParams p, GgmlTensor dst)
    {
        if (p.Ith != 0) return;
        var src0 = dst.Src[0]!;
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];

        double sum = 0;
        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = 0; i1 < ne1; i1++)
        {
            byte* row = src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01;
            sum += src0.Type switch
            {
                GgmlType.F32  => RowSum((float*)row, (int)ne0),
                GgmlType.F16  => RowSumF16((ushort*)row, (int)ne0),
                GgmlType.BF16 => RowSumBf16((ushort*)row, (int)ne0),
                _ => throw new NotSupportedException($"sum: type {src0.Type}"),
            };
        }
        switch (dst.Type)
        {
            case GgmlType.F32:  *(float*)dst.Data = (float)sum; break;
            case GgmlType.F16:  *(ushort*)dst.Data = GgmlMath.F32ToF16((float)sum); break;
            case GgmlType.BF16: *(ushort*)dst.Data = GgmlMath.F32ToBf16((float)sum); break;
            default: throw new NotSupportedException($"sum dst: {dst.Type}");
        }
    }

    private static float RowSumF16(ushort* x, int n)
    {
        float s = 0;
        for (int i = 0; i < n; i++) s += GgmlMath.F16ToF32(x[i]);
        return s;
    }

    private static float RowSumBf16(ushort* x, int n)
    {
        float s = 0;
        for (int i = 0; i < n; i++) s += GgmlMath.Bf16ToF32(x[i]);
        return s;
    }

    // ---- SUM_ROWS: reduce each row to a scalar ---------------------------------

    public static void ForwardSumRows(in ComputeParams p, GgmlTensor dst)
    {
        if (p.Ith != 0) return;
        var src0 = dst.Src[0]!;
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];

        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = 0; i1 < ne1; i1++)
        {
            byte* src = src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01;
            byte* d = dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            *(float*)d = RowSum((float*)src, (int)ne0);
        }
    }

    // ---- MEAN: reduce each row and divide by ne00 ------------------------------

    public static void ForwardMean(in ComputeParams p, GgmlTensor dst)
    {
        if (p.Ith != 0) return;
        var src0 = dst.Src[0]!;
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];

        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = 0; i1 < ne1; i1++)
        {
            byte* src = src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01;
            byte* d = dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1;
            *(float*)d = RowSum((float*)src, (int)ne0) / ne0;
        }
    }

    // ---- ARGMAX: per-row index of max -------------------------------------------

    public static void ForwardArgmax(in ComputeParams p, GgmlTensor dst)
    {
        if (p.Ith != 0) return;
        var src0 = dst.Src[0]!;
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1];
        long nb01 = src0.Nb[1], nb0 = dst.Nb[0];

        for (long i1 = 0; i1 < ne1; i1++)
        {
            float* src = (float*)(src0.Data + i1 * nb01);
            int best = 0;
            float bestVal = float.NegativeInfinity;
            for (int i = 0; i < (int)ne0; i++)
            {
                if (src[i] > bestVal) { bestVal = src[i]; best = i; }
            }
            *(int*)(dst.Data + i1 * nb0) = best;
        }
    }

    // ---- COUNT_EQUAL: i32 element-wise equality, scalar i64 out ------------------

    public static void ForwardCountEqual(in ComputeParams p, GgmlTensor dst)
    {
        if (p.Ith != 0) return;
        var src0 = dst.Src[0]!;
        var src1 = dst.Src[1]!;
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2], ne3 = src0.Ne[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb11 = src1.Nb[1], nb12 = src1.Nb[2], nb13 = src1.Nb[3];

        long count = 0;
        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = 0; i2 < ne2; i2++)
        for (long i1 = 0; i1 < ne1; i1++)
        {
            int* a = (int*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            int* b = (int*)(src1.Data + i3 * nb13 + i2 * nb12 + i1 * nb11);
            for (long i0 = 0; i0 < ne0; i0++) count += (a[i0] == b[i0]) ? 1 : 0;
        }
        *(long*)dst.Data = count;
    }

    // ---- CUMSUM: per-row prefix sum -----------------------------------------------

    public static void ForwardCumsum(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        long ne0 = src0.Ne[0], ne1 = src0.Ne[1], ne2 = src0.Ne[2];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];

        var (ir0, ir1) = p.ThreadRowRange(src0.NRows());
        for (long ir = ir0; ir < ir1; ir++)
        {
            long i3 = ir / (ne2 * ne1);
            long i2 = (ir - i3 * ne2 * ne1) / ne1;
            long i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
            float* src = (float*)(src0.Data + i3 * nb03 + i2 * nb02 + i1 * nb01);
            float* d = (float*)(dst.Data + i3 * nb3 + i2 * nb2 + i1 * nb1);
            float s = 0;
            for (int i = 0; i < (int)ne0; i++) { s += src[i]; d[i] = s; }
        }
    }
}