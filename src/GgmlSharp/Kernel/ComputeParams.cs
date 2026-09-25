namespace GgmlSharp.Kernel;

/// <summary>
/// Mirrors <c>struct ggml_compute_params</c>. <c>Ith</c>/<c>Nth</c> drive the row split:
/// each kernel processes rows <c>[dr*ith, min(dr*ith+dr, nr))</c> where <c>dr=(nr+nth-1)/nth</c>
/// (ceil), matching ggml's <c>get_thread_range</c>. The <c>WorkData</c> buffer is used by
/// ops that need a scratch (e.g. quantizing src1 in mul_mat); unary ops ignore it.
/// </summary>
public readonly unsafe struct ComputeParams
{
    public readonly int Ith;
    public readonly int Nth;
    public readonly long WorkSize;
    public readonly byte* WorkData;

    public ComputeParams(int ith = 0, int nth = 1, long workSize = 0, byte* workData = null)
    {
        Ith = ith;
        Nth = nth;
        WorkSize = workSize;
        WorkData = workData;
    }

    /// <summary>The [ir0, ir1) row range for this thread over <paramref name="nrows"/>.</summary>
    public (long ir0, long ir1) ThreadRowRange(long nrows)
    {
        long dr = (nrows + Nth - 1) / Nth;
        long ir0 = dr * Ith;
        long ir1 = Math.Min(ir0 + dr, nrows);
        return (ir0, ir1);
    }
}
