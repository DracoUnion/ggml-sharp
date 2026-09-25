using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace GgmlSharp.Kernel;

/// <summary>
/// Port of <c>binary-ops.cpp</c>: GGML_OP_ADD / SUB / MUL / DIV with GGML's two-level
/// broadcasting. <c>src1</c> may repeat across the higher dims (row broadcast via
/// <c>i%ne1k</c>) and/or within a row (<c>nr0 = ne00/ne10</c>). src1 can be a scalar-per-row
/// (ne10==1), a full row, or a strided sub-row (non-contiguous). Only the non-quantized
/// f32/f16/bf16 codecs are supported, matching the C <c>binary_op</c> dispatch (which aborts
/// otherwise). AVX2 fast path for f32+f32+f32 (bit-exact vs scalar).
/// </summary>
public static unsafe class OpsBinary
{
    private delegate float BinF(float a, float b);
    private delegate Vector256<float> BinV(Vector256<float> a, Vector256<float> b);

    public static void Forward(in ComputeParams p, GgmlTensor dst)
    {
        BinF f = (dst.Op) switch
        {
            GgmlOp.ADD => (a, b) => a + b,
            GgmlOp.SUB => (a, b) => a - b,
            GgmlOp.MUL => (a, b) => a * b,
            GgmlOp.DIV => (a, b) => a / b,
            _ => throw new NotSupportedException($"binary op {dst.Op}"),
        };
        BinV? v = (dst.Op) switch
        {
            GgmlOp.ADD => Avx.Add,
            GgmlOp.SUB => Avx.Subtract,
            GgmlOp.MUL => Avx.Multiply,
            GgmlOp.DIV => Avx.Divide,
            _ => null,
        };

        var src0 = dst.Src[0]!;
        var src1 = dst.Src[1]!;
        var codec0 = Codec.For(src0.Type);
        var codec1 = Codec.For(src1.Type);
        var codecD = Codec.For(dst.Type);

        bool allF32 = src0.Type == GgmlType.F32 && src1.Type == GgmlType.F32 && dst.Type == GgmlType.F32;

        long ne00 = src0.Ne[0], ne01 = src0.Ne[1], ne02 = src0.Ne[2];
        long ne10 = src1.Ne[0], ne11 = src1.Ne[1], ne12 = src1.Ne[2], ne13 = src1.Ne[3];
        long nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb11 = src1.Nb[1], nb12 = src1.Nb[2], nb13 = src1.Nb[3];
        long nb10 = src1.Nb[0];
        long plane = ne02 * ne01;
        bool src1ContigRows = IsContiguousRows(src1);
        long nr0 = ne10 > 0 ? ne00 / ne10 : 0;

        var (ir0, ir1) = p.ThreadRowRange(src0.NRows());

        for (long ir = ir0; ir < ir1; ir++)
        {
            long i03 = ir / plane;
            long i02 = (ir - i03 * plane) / ne01;
            long i01 = ir - i03 * plane - i02 * ne01;
            long i13 = i03 % ne13;
            long i12 = i02 % ne12;
            long i11 = i01 % ne11;

            byte* dstPtr = dst.Data + i03 * nb3 + i02 * nb2 + i01 * nb1;
            byte* src0Ptr = src0.Data + i03 * nb03 + i02 * nb02 + i01 * nb01;
            byte* src1Ptr = src1.Data + i13 * nb13 + i12 * nb12 + i11 * nb11;

            if (src1ContigRows)
            {
                for (long r = 0; r < nr0; r++)
                {
                    long elemOff = r * ne10;
                    if (allF32)
                        RowContigF32((float*)(dstPtr + elemOff * 4), (float*)(src0Ptr + elemOff * 4), (float*)src1Ptr, (int)ne10, f, v);
                    else
                        RowContig(dstPtr + elemOff * codecD.Size, src0Ptr + elemOff * codec0.Size, src1Ptr, (int)ne10, codec0, codec1, codecD, f);
                }
            }
            else
            {
                RowNonContig(dstPtr, src0Ptr, src1Ptr, (int)ne00, (int)ne10, nb10, codec0, codec1, codecD, f);
            }
        }
    }

    private static bool IsContiguousRows(GgmlTensor t)
    {
        var tt = GgmlTypeTraits.Get(t.Type);
        return t.Ne[0] == tt.BlckSize || t.Nb[0] == tt.TypeSize;
    }

    // ---- f32 fast path -------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RowContigF32(float* z, float* x, float* y, int n, BinF f, BinV? v)
    {
        int i = 0;
        if (v != null && Avx2.IsSupported)
            for (; i <= n - 8; i += 8)
                Avx.Store(z + i, v(Avx.LoadVector256(x + i), Avx.LoadVector256(y + i)));
        for (; i < n; i++) z[i] = f(x[i], y[i]);
    }

    // ---- generic (byte-addressed, codec-decoded) paths ------------------------

    private static void RowContig(byte* z, byte* x, byte* y, int n, Codec c0, Codec c1, Codec cd, BinF f)
    {
        for (int i = 0; i < n; i++)
        {
            byte* xp = x + i * c0.Size, yp = y + i * c1.Size, zp = z + i * cd.Size;
            cd.Write(zp, f(c0.Read(xp), c1.Read(yp)));
        }
    }

    private static void RowNonContig(byte* z, byte* x, byte* y, int n, int ne10, long nb10, Codec c0, Codec c1, Codec cd, BinF f)
    {
        for (int i = 0; i < n; i++)
        {
            int i10 = i % ne10;
            byte* xp = x + i * c0.Size;
            byte* yp = y + i10 * nb10;
            byte* zp = z + i * cd.Size;
            cd.Write(zp, f(c0.Read(xp), c1.Read(yp)));
        }
    }

    /// <summary>An element codec (read-to-f32 / write-from-f32) for one GGML type.</summary>
    private readonly struct Codec
    {
        internal delegate float ReadFn(byte* p);
        internal delegate void WriteFn(byte* p, float v);

        public readonly int Size;
        internal readonly ReadFn Read;
        internal readonly WriteFn Write;

        private Codec(int size, ReadFn r, WriteFn w)
        { Size = size; Read = r; Write = w; }

        public static Codec For(GgmlType t) => t switch
        {
            GgmlType.F32  => new(4, ReadF32, WriteF32),
            GgmlType.F16  => new(2, ReadF16, WriteF16),
            GgmlType.BF16 => new(2, ReadBf16, WriteBf16),
            _ => throw new NotSupportedException($"binary op: unsupported codec type {t}"),
        };

        private static float ReadF32(byte* p) => *(float*)p;
        private static void   WriteF32(byte* p, float v) => *(float*)p = v;
        private static float ReadF16(byte* p) => GgmlMath.F16ToF32(*(ushort*)p);
        private static void   WriteF16(byte* p, float v) => *(ushort*)p = GgmlMath.F32ToF16(v);
        private static float ReadBf16(byte* p) => GgmlMath.Bf16ToF32(*(ushort*)p);
        private static void   WriteBf16(byte* p, float v) => *(ushort*)p = GgmlMath.F32ToBf16(v);
    }
}