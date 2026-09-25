using System.Runtime.InteropServices;
using GgmlSharp.Kernel;

namespace GgmlSharp.Graph;

/// <summary>
/// Graph construction + execution, ported from <c>ggml.c</c>'s builder ops and
/// <c>ggml_graph_compute</c>. Builders allocate a tensor node, wire <c>src</c> and
/// <c>op_params</c>, and append it to the node list. Execute runs the whole graph once per
/// thread (each kernel threads internally via <see cref="ComputeParams.Ith"/>/<c>Nth</c>),
/// mirroring GGML's per-thread graph compute.
/// </summary>
public sealed unsafe class GgmlGraph : IDisposable
{
    private readonly List<GgmlTensor> _nodes = new();

    public GgmlContext Ctx { get; }

    public GgmlGraph(long reserveBytes = 1 << 20) => Ctx = new GgmlContext(reserveBytes);

    public IReadOnlyList<GgmlTensor> Nodes => _nodes;

    // ---- leaves --------------------------------------------------------------------

    public GgmlTensor NewTensor(GgmlType type, params long[] ne)
        => Ctx.NewTensor(type, ne.Length, ne);

    public GgmlTensor NewTensor(GgmlType type, long[] ne, params GgmlTensor[] srcs)
        => Ctx.NewTensor(type, ne.Length, ne, srcs);

    /// <summary>Alias a source tensor's base data (VIEW/RESHAPE are structural; no compute).</summary>
    public GgmlTensor View(GgmlTensor a, long[] ne)
    {
        var t = Ctx.NewOp(GgmlOp.NONE, a.Type, ne, a);
        t.Data = a.Data;
        t.Nb = (long[])a.Nb.Clone();
        t.Op = GgmlOp.NONE;
        _nodes.Add(t);
        return t;
    }

    // ---- op builders (mirror ggml_add/mul/mul_mat/rms_norm/soft_max/silu/rope/scale) --

    public GgmlTensor Add(GgmlTensor a, GgmlTensor b) => Op(GgmlOp.ADD, F32, Shape(a), a, b);
    public GgmlTensor Sub(GgmlTensor a, GgmlTensor b) => Op(GgmlOp.SUB, F32, Shape(a), a, b);
    public GgmlTensor Mul(GgmlTensor a, GgmlTensor b) => Op(GgmlOp.MUL, F32, BroadcastShape(a, b), a, b);
    public GgmlTensor Div(GgmlTensor a, GgmlTensor b) => Op(GgmlOp.DIV, F32, BroadcastShape(a, b), a, b);

    /// <summary>dst = a^T b: a [K,N], b [K,M] -&gt; [N,M] (batches broadcast).</summary>
    public GgmlTensor MulMat(GgmlTensor a, GgmlTensor b)
    {
        long[] ne = { a.Ne[1], b.Ne[1], b.Ne[2], b.Ne[3] };
        return Op(GgmlOp.MUL_MAT, F32, ne, a, b);
    }

    public GgmlTensor RmsNorm(GgmlTensor a, float eps)
    {
        var t = Op(GgmlOp.RMS_NORM, F32, Shape(a), a);
        GgmlTensor.SetOpParamsF32(t, 0, eps);
        return t;
    }

    public GgmlTensor Norm(GgmlTensor a, float eps)
    {
        var t = Op(GgmlOp.NORM, F32, Shape(a), a);
        GgmlTensor.SetOpParamsF32(t, 0, eps);
        return t;
    }

    public GgmlTensor SoftMax(GgmlTensor a, float scale = 1f)
    {
        var t = Op(GgmlOp.SOFT_MAX, F32, Shape(a), a);
        GgmlTensor.SetOpParamsF32(t, 0, scale);
        GgmlTensor.SetOpParamsF32(t, 1, 0f);
        return t;
    }

    public GgmlTensor Silu(GgmlTensor a) => Unary(GgmlUnaryOp.SILU, a);
    public GgmlTensor Relu(GgmlTensor a) => Unary(GgmlUnaryOp.RELU, a);
    public GgmlTensor Gelu(GgmlTensor a) => Unary(GgmlUnaryOp.GELU, a);
    public GgmlTensor Tanh(GgmlTensor a) => Unary(GgmlUnaryOp.TANH, a);
    public GgmlTensor Exp(GgmlTensor a) => Unary(GgmlUnaryOp.EXP, a);

    public GgmlTensor Scale(GgmlTensor a, float s)
    {
        var t = Op(GgmlOp.SCALE, F32, Shape(a), a);
        GgmlTensor.SetOpParamsF32(t, 0, s);
        GgmlTensor.SetOpParamsF32(t, 1, 0f);
        return t;
    }

    public GgmlTensor Rope(GgmlTensor a, int[] positions, int nDims, GgmlRopeType mode, float freqBase = 10000f)
    {
        // src1 = positions (i32), shape [ne2]
        var pos = Ctx.NewTensor(GgmlType.I32, 1, new long[] { a.Ne[2] });
        for (int i = 0; i < positions.Length && i < a.Ne[2]; i++) ((int*)pos.Data)[i] = positions[i];
        var t = Ctx.NewOp(GgmlOp.ROPE, a.Type, Shape(a), a, pos);
        GgmlTensor.SetOpParamsI32(t, 1, nDims);
        GgmlTensor.SetOpParamsI32(t, 2, (int)mode);
        GgmlTensor.SetOpParamsF32(t, 5, freqBase);
        GgmlTensor.SetOpParamsF32(t, 6, 1f);   // freq_scale
        GgmlTensor.SetOpParamsF32(t, 7, 0f);   // ext_factor
        GgmlTensor.SetOpParamsF32(t, 8, 1f);   // attn_factor
        GgmlTensor.SetOpParamsF32(t, 9, 32f);  // beta_fast
        GgmlTensor.SetOpParamsF32(t, 10, 1f);  // beta_slow
        GgmlTensor.SetOpParamsI32(t, 4, (int)a.Ne[2]); // n_ctx_orig
        _nodes.Add(t);
        return t;
    }

    public GgmlTensor Repmat(GgmlTensor a, long[] dstShape)
        => Op(GgmlOp.REPEAT, a.Type, dstShape, a);

    // ---- internals -----------------------------------------------------------------

    private static readonly GgmlType F32 = GgmlType.F32;

    private GgmlTensor Unary(GgmlUnaryOp uop, GgmlTensor a)
    {
        var t = Op(GgmlOp.UNARY, a.Type, Shape(a), a);
        GgmlTensor.SetOpParamsI32(t, 0, (int)uop);
        return t;
    }

    private GgmlTensor Op(GgmlOp op, GgmlType type, long[] ne, params GgmlTensor[] srcs)
    {
        var t = Ctx.NewOp(op, type, ne, srcs);
        _nodes.Add(t);
        return t;
    }

    private static long[] Shape(GgmlTensor t) => (long[])t.Ne.Clone();
    private static long[] BroadcastShape(GgmlTensor a, GgmlTensor b)
    {
        var ne = new long[4];
        for (int i = 0; i < 4; i++) ne[i] = Math.Max(a.Ne[i], b.Ne[i]);
        return ne;
    }

    // ---- execution -----------------------------------------------------------------

    public void Execute(int nth = 1)
    {
        for (int ith = 0; ith < nth; ith++)
        {
            var p = new ComputeParams(ith, nth);
            foreach (var node in _nodes)
                if (node.Op != GgmlOp.NONE)
                    GgmlCompute.Forward(p, node);
        }
    }

    public void Dispose() => Ctx.Dispose();
}