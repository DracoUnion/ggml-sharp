using System.Runtime.CompilerServices;

namespace GgmlSharp.Kernel;

/// <summary>Constants mirroring <c>ggml.h</c>.</summary>
public static class GgmlConst
{
    public const int MaxDims     = 4;   // GGML_MAX_DIMS
    public const int MaxSrc      = 10;  // GGML_MAX_SRC
    public const int MaxOpParams = 64;  // GGML_MAX_OP_PARAMS
    public const int MaxName     = 64;  // GGML_MAX_NAME
    public const int MemAlign    = 16;  // GGML_MEM_ALIGN
}

/// <summary>
/// Tensor descriptor mirroring <c>struct ggml_tensor</c>. In C this lives in a linear
/// arena and is indexed by byte strides <c>nb[]</c>; here it is a managed node holding a
/// raw <c>Data</c> pointer (allocated by <see cref="GgmlContext"/>). GGML is column-major:
/// <c>nb[0]</c> is the innermost stride (one element), <c>nb[3]</c> the outermost.
/// </summary>
public sealed unsafe class GgmlTensor
{
    public GgmlType Type;
    public long[] Ne = new long[GgmlConst.MaxDims]; // element counts per dim
    public long[] Nb = new long[GgmlConst.MaxDims]; // byte strides per dim
    public byte* Data;

    public GgmlOp Op;
    public byte[] OpParams = new byte[GgmlConst.MaxOpParams];

    public GgmlTensor?[] Src = new GgmlTensor?[GgmlConst.MaxSrc];
    public GgmlTensor? ViewSrc;
    public long ViewOffs;
    public string? Name;
    public int Flags;
    public void* Extra;

    // ---- shape queries (mirror ggml_nelements / ggml_nrows) ------------------

    public long NElements() => Ne[0] * Ne[1] * Ne[2] * Ne[3];

    public long NRows() => Ne[1] * Ne[2] * Ne[3];

    public long RowSize() => GgmlTypeTraits.Get(Type).TypeSize * Ne[0] / Math.Max(1, GgmlTypeTraits.Get(Type).BlckSize);

    public void SetShape(int nDims, params long[] ne)
    {
        for (int i = 0; i < GgmlConst.MaxDims; i++) Ne[i] = i < nDims && i < ne.Length ? ne[i] : 1;
    }

    public void ComputeStrides()
    {
        var tt = GgmlTypeTraits.Get(Type);
        Nb[0] = tt.TypeSize;
        Nb[1] = Nb[0] * (Ne[0] / Math.Max(1, tt.BlckSize));
        for (int i = 2; i < GgmlConst.MaxDims; i++) Nb[i] = Nb[i - 1] * Ne[i - 1];
    }

    // ---- op_params accessors (mirror ggml_get_op_params_i32 / _f32) -----------

    // ggml treats op_params as a byte array; the int/f32 getters memcpy 4 bytes
    // (little-endian on x86). Read/write through the same byte layout.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetOpParamsI32(GgmlTensor t, int index) =>
        t.OpParams[4 * index] | (t.OpParams[4 * index + 1] << 8) |
        (t.OpParams[4 * index + 2] << 16) | (t.OpParams[4 * index + 3] << 24);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GetOpParamsF32(GgmlTensor t, int index) =>
        BitConverter.Int32BitsToSingle(GetOpParamsI32(t, index));

    public static void SetOpParamsI32(GgmlTensor t, int index, int value)
    {
        t.OpParams[4 * index]     = (byte)(value);
        t.OpParams[4 * index + 1] = (byte)(value >> 8);
        t.OpParams[4 * index + 2] = (byte)(value >> 16);
        t.OpParams[4 * index + 3] = (byte)(value >> 24);
    }

    public static void SetOpParamsF32(GgmlTensor t, int index, float value) =>
        SetOpParamsI32(t, index, BitConverter.SingleToInt32Bits(value));
}