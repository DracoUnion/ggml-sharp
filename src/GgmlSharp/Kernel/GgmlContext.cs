using System.Runtime.InteropServices;

namespace GgmlSharp.Kernel;

/// <summary>
/// A minimal arena allocator mirroring <c>ggml_context</c>: one growable aligned buffer,
/// objects carved off the tail. Unlike the C version it keeps managed <see cref="GgmlTensor"/>
/// nodes (the raw data pointers are still carved from the arena). Freeing the context frees
/// everything allocated in it.
/// </summary>
public unsafe class GgmlContext : IDisposable
{
    private byte* _buffer;
    private long _size;
    private long _offset;
    private readonly List<GgmlTensor?> _tensors = new();

    public GgmlContext(long reserveBytes = 1 << 20)
    {
        Allocate(AvailTo(reserveBytes));
    }

    private void Allocate(long nbytes)
    {
        _buffer = (byte*)NativeMemory.AlignedAlloc((nuint)nbytes, (nuint)GgmlConst.MemAlign);
        _size = nbytes;
        _offset = 0;
    }

    private static long AvailTo(long requested)
    {
        long pad = requested == 0 ? GgmlConst.MemAlign : 0;
        long aligned = (requested + pad + GgmlConst.MemAlign - 1) / GgmlConst.MemAlign * GgmlConst.MemAlign;
        return Math.Max(aligned, GgmlConst.MemAlign);
    }

    private void Ensure(long nbytes)
    {
        if (AvailTo(nbytes) <= _size - _offset) return;
        long oldSize = _size;
        long newSize = Math.Max(_size * 2, _size + nbytes);
        var newBuf = (byte*)NativeMemory.AlignedAlloc((nuint)newSize, (nuint)GgmlConst.MemAlign);
        NativeMemory.Copy(_buffer, newBuf, (nuint)oldSize);
        NativeMemory.AlignedFree(_buffer);
        _buffer = newBuf;
        _size = newSize;
    }

    /// <summary>Allocate <paramref name="nbytes"/> from the arena (aligned up).</summary>
    public byte* AllocBytes(long nbytes)
    {
        Ensure(nbytes);
        nbytes = AvailTo(nbytes);
        byte* p = _buffer + _offset;
        _offset += nbytes;
        return p;
    }

    /// <summary>
    /// Create a tensor with contiguous row-major-over-column data. <paramref name="ne"/> gives
    /// the dims (innermost = ne[0]). Data is zero-filled when <paramref name="nDims"/> &gt; 0 and
    /// <paramref name="noAlloc"/> is false; pass <paramref name="data"/> to alias external memory.
    /// </summary>
    public GgmlTensor NewTensor(GgmlType type, int nDims, long[] ne, GgmlTensor?[]? src = null,
        bool noAlloc = false, byte* data = null)
    {
        var ten = new GgmlTensor { Type = type };
        int dims = Math.Min(nDims, GgmlConst.MaxDims);
        for (int i = 0; i < GgmlConst.MaxDims; i++) ten.Ne[i] = i < dims ? ne[i] : 1; // unused dims = 1
        if (src != null)
            for (int i = 0; i < src.Length && i < GgmlConst.MaxSrc; i++) ten.Src[i] = src[i];

        ten.ComputeStrides();
        if (data != null)
        {
            ten.Data = data;
        }
        else if (noAlloc)
        {
            ten.Data = null;
        }
        else
        {
            long size = ten.RowSize(); // bytes of one row (ne[0])
            for (int i = 1; i < dims; i++) size *= ne[i];
            ten.Data = AllocBytes(Math.Max(1, size));
        }

        _tensors.Add(ten);
        return ten;
    }

    /// <summary>For unary tests: an op node with src0 = <paramref name="src"/> and op_params carrying the unary op enum.</summary>
    public GgmlTensor NewUnaryOp(GgmlType type, long[] ne, GgmlTensor src, GgmlUnaryOp unaryOp)
    {
        var ten = NewTensor(type, ne.Length, ne, new[] { src });
        ten.Op = GgmlOp.UNARY;
        GgmlTensor.SetOpParamsI32(ten, 0, (int)unaryOp);
        return ten;
    }

    /// <summary>A binary op node with two sources (allows different shapes for broadcast).</summary>
    public GgmlTensor NewBinaryOp(GgmlType type, long[] ne, GgmlTensor src0, GgmlTensor src1, GgmlOp op)
    {
        var ten = NewTensor(type, ne.Length, ne, new[] { src0, src1 });
        ten.Op = op;
        return ten;
    }

    /// <summary>Generic op node (mirrors ggml's builder pattern: build leaf-first).</summary>
    public GgmlTensor NewOp(GgmlOp op, GgmlType type, long[] ne, params GgmlTensor[] srcs)
    {
        var ten = NewTensor(type, ne.Length, ne, srcs);
        ten.Op = op;
        return ten;
    }

    public void Dispose()
    {
        if (_buffer != null)
        {
            NativeMemory.AlignedFree(_buffer);
            _buffer = null;
        }
    }
}