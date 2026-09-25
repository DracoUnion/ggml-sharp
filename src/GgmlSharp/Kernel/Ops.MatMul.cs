using System.Runtime.InteropServices;

namespace GgmlSharp.Kernel;

/// <summary>
/// Ports of the GGML matmul family from <c>ops.cpp</c>: OUT_PROD (F32) and MUL_MAT for
/// non-quantized src0 (F32/F16/BF16) with F32 src1/dst — the classic weight·activation
/// GEMM. GGML layout: dst[i0,i1,..] = &Sigma;_k src0[k, i0, i2%ne02, i3%ne03] * src1[k, i1, i2, i3].
/// Quantized src0 types and MUL_MAT_ID are the remaining M3 work.
/// </summary>
public static unsafe class OpsMatMul
{
    private static float ReadElem(byte* p, GgmlType t) => t switch
    {
        GgmlType.F32 => *(float*)p,
        GgmlType.F16 => GgmlMath.F16ToF32(*(ushort*)p),
        GgmlType.BF16 => GgmlMath.Bf16ToF32(*(ushort*)p),
        _ => throw new NotSupportedException($"mul_mat src0 type {t}"),
    };

    // ---- MUL_MAT: dst[i0,i1] = src0[:,i0] · src1[:,i1] -------------------------------

    public static void ForwardMulMat(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        var src1 = dst.Src[1]!;
        if (src1.Type != GgmlType.F32 || dst.Type != GgmlType.F32)
            throw new NotSupportedException("mul_mat: only F32 src1/dst supported");
        var toF = GgmlTypeTraits.Get(src0.Type).ToFloat
            ?? throw new NotSupportedException($"mul_mat src0 type {src0.Type} not yet supported");

        long ne0 = dst.Ne[0], ne1 = dst.Ne[1], ne2 = dst.Ne[2], ne3 = dst.Ne[3];
        long ne01 = src0.Ne[1], ne02 = src0.Ne[2], ne03 = src0.Ne[3];
        long nb0 = dst.Nb[0], nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb00 = src0.Nb[0], nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb10 = src1.Nb[0], nb11 = src1.Nb[1], nb12 = src1.Nb[2], nb13 = src1.Nb[3];
        long K = src0.Ne[0];

        var scratch = (float*)NativeMemory.AlignedAlloc((nuint)(K * 4), 64);
        try
        {
            // Parallelize over dst rows (i1,i2,i3); inner over dst cols i0.
            var (ir0, ir1) = p.ThreadRowRange(dst.NRows());
            for (long ir = ir0; ir < ir1; ir++)
            {
                long i3 = ir / (ne2 * ne1);
                long i2 = (ir - i3 * ne2 * ne1) / ne1;
                long i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
                long i2s = i2 % ne02, i3s = i3 % ne03;

                for (long i0 = 0; i0 < ne0; i0++)
                {
                    // Dequantize src0 column (i0, batch) into scratch.
                    toF(src0.Data + i0 * nb01 + i2s * nb02 + i3s * nb03, scratch, K);

                    byte* col1 = src1.Data + i1 * nb11 + i2 * nb12 + i3 * nb13;
                    double acc = 0;
                    for (long k = 0; k < K; k++)
                        acc += (double)scratch[k] * *(float*)(col1 + k * nb10);

                    *(float*)(dst.Data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3) = (float)acc;
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(scratch);
        }
    }

    // ---- OUT_PROD: dst[i0,i1] = src0[i0,:] · src1[i1,:] (outer product summed over k) ----

    public static void ForwardOutProd(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        var src1 = dst.Src[1]!;
        if (dst.Type != GgmlType.F32 || src0.Type != GgmlType.F32 || src1.Type != GgmlType.F32)
            throw new NotSupportedException("out_prod: only F32 supported");

        long ne0 = dst.Ne[0], ne1 = dst.Ne[1], ne2 = dst.Ne[2], ne3 = dst.Ne[3];
        long ne01 = src0.Ne[1], ne02 = src0.Ne[2], ne03 = src0.Ne[3];
        long nb0 = dst.Nb[0], nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb00 = src0.Nb[0], nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];
        long nb10 = src1.Nb[0], nb11 = src1.Nb[1], nb12 = src1.Nb[2], nb13 = src1.Nb[3];
        long dps2 = ne2 / ne02, dps3 = ne3 / ne03;

        // zero dst (thread 0; others no-op — serial caller avoids a barrier race)
        if (p.Ith == 0)
            NativeMemory.Clear(dst.Data, (nuint)(dst.NElements() * dst.Nb[0]));

        var (ir0, ir1) = p.ThreadRowRange(ne1 * ne2 * ne3);
        for (long ir = ir0; ir < ir1; ir++)
        {
            long i3 = ir / (ne2 * ne1);
            long i2 = (ir - i3 * ne2 * ne1) / ne1;
            long i1 = ir - i3 * ne2 * ne1 - i2 * ne1;
            long i02 = i2 / dps2, i03 = i3 / dps3;

            float* d = (float*)(dst.Data + i1 * nb1 + i2 * nb2 + i3 * nb3);
            for (long i0 = 0; i0 < ne0; i0++)
            {
                float acc = 0;
                for (long i01 = 0; i01 < ne01; i01++)
                {
                    float s0 = *(float*)(src0.Data + i01 * nb01 + i02 * nb02 + i03 * nb03 + i0 * nb00);
                    float s1 = *(float*)(src1.Data + i1 * nb10 + i01 * nb11 + i2 * nb12 + i3 * nb13);
                    acc += s0 * s1;
                }
                d[i0 * (dst.Nb[0] / 4)] = acc;
            }
        }
    }
}