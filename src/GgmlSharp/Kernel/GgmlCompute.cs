namespace GgmlSharp.Kernel;

/// <summary>
/// Top-level forward dispatch mirroring <c>ggml_compute_forward()</c> in <c>ggml-cpu.c</c>:
/// skip NONE/empty tensors, then switch on <c>dst-&gt;Op</c> to the per-op kernel. Milestone 0
/// wires GGML_OP_UNARY; the remaining cases are filled in as their kernels are ported.
/// </summary>
public static unsafe class GgmlCompute
{
    public static void Forward(in ComputeParams p, GgmlTensor dst)
    {
        if (dst.Op == GgmlOp.NONE) return;
        if (dst.NElements() == 0) return;

        switch (dst.Op)
        {
            case GgmlOp.UNARY:          OpsUnary.Forward(p, dst); break;
            case GgmlOp.ADD:
            case GgmlOp.SUB:
            case GgmlOp.MUL:
            case GgmlOp.DIV:            OpsBinary.Forward(p, dst); break;

            case GgmlOp.SUM:            OpsReduce.ForwardSum(p, dst); break;
            case GgmlOp.SUM_ROWS:       OpsReduce.ForwardSumRows(p, dst); break;
            case GgmlOp.MEAN:           OpsReduce.ForwardMean(p, dst); break;
            case GgmlOp.ARGMAX:         OpsReduce.ForwardArgmax(p, dst); break;
            case GgmlOp.COUNT_EQUAL:    OpsReduce.ForwardCountEqual(p, dst); break;
            case GgmlOp.CUMSUM:         OpsReduce.ForwardCumsum(p, dst); break;

            case GgmlOp.NORM:           OpsNorm.ForwardNorm(p, dst); break;
            case GgmlOp.RMS_NORM:       OpsNorm.ForwardRmsNorm(p, dst); break;
            case GgmlOp.L2_NORM:        OpsNorm.ForwardL2Norm(p, dst); break;
            case GgmlOp.GROUP_NORM:     OpsNorm.ForwardGroupNorm(p, dst); break;

            case GgmlOp.SOFT_MAX:       OpsSoftmax.Forward(p, dst); break;
            case GgmlOp.LEAKY_RELU:     OpsSimple.ForwardLeakyRelu(p, dst); break;
            case GgmlOp.SILU_BACK:      OpsSimple.ForwardSiluBack(p, dst); break;
            case GgmlOp.SCALE:          OpsSimple.ForwardScale(p, dst); break;
            case GgmlOp.RMS_NORM_BACK:  OpsRemap.ForwardRmsNormBack(p, dst); break;

            case GgmlOp.DUP:
            case GgmlOp.CPY:
            case GgmlOp.CONT:           OpsRemap.ForwardDup(p, dst); break;
            case GgmlOp.REPEAT:         OpsRemap.ForwardRepeat(p, dst); break;
            case GgmlOp.REPEAT_BACK:    OpsRemap.ForwardRepeatBack(p, dst); break;
            case GgmlOp.SET:            OpsRemap.ForwardSet(p, dst); break;
            case GgmlOp.PAD:            OpsRemap.ForwardPad(p, dst, GgmlTensor.GetOpParamsI32(dst, 8) != 0); break;
            case GgmlOp.PAD_REFLECT_1D: OpsRemap.ForwardPadReflect1D(p, dst); break;
            case GgmlOp.ROLL:           OpsRemap.ForwardRoll(p, dst); break;
            case GgmlOp.ARANGE:         OpsRemap.ForwardArange(p, dst); break;
            case GgmlOp.FILL:           OpsRemap.ForwardFill(p, dst); break;
            case GgmlOp.TRI:            OpsRemap.ForwardTri(p, dst); break;
            case GgmlOp.ARGSORT:        OpsRemap.ForwardArgsort(p, dst); break;
            case GgmlOp.TOP_K:          OpsRemap.ForwardTopK(p, dst); break;
            case GgmlOp.TIMESTEP_EMBEDDING: OpsRemap.ForwardTimestepEmbedding(p, dst); break;
            case GgmlOp.UPSCALE:        OpsUpscale.Forward(p, dst); break;
            case GgmlOp.MUL_MAT:        OpsMatMul.ForwardMulMat(p, dst); break;
            case GgmlOp.OUT_PROD:       OpsMatMul.ForwardOutProd(p, dst); break;
            case GgmlOp.ROPE:           OpsRope.Forward(p, dst, true); break;
            case GgmlOp.GLU:            OpsGlu.Forward(p, dst); break;

            default:
                throw new NotImplementedException($"op {dst.Op} is not yet ported (M3 covers MUL_MAT/OUT_PROD F32)");
        }
    }
}
