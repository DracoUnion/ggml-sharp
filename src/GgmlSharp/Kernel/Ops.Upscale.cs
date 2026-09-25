namespace GgmlSharp.Kernel;

/// <summary>
/// Port of <c>ggml_compute_forward_upscale_f32</c> (ops.cpp). Implements NEAREST,
/// BILINEAR, and antialiased BILINEAR; BICUBIC is not yet ported. scale factors are
/// dst/source per dim; <c>align_corners</c> sets pixel_offset to 0.
/// </summary>
public static unsafe class OpsUpscale
{
    public static void Forward(in ComputeParams p, GgmlTensor dst)
    {
        var src0 = dst.Src[0]!;
        long ne0 = dst.Ne[0], ne1 = dst.Ne[1], ne2 = dst.Ne[2], ne3 = dst.Ne[3];
        long ne00 = src0.Ne[0], ne01 = src0.Ne[1], ne02 = src0.Ne[2], ne03 = src0.Ne[3];
        long nb0 = dst.Nb[0], nb1 = dst.Nb[1], nb2 = dst.Nb[2], nb3 = dst.Nb[3];
        long nb00 = src0.Nb[0], nb01 = src0.Nb[1], nb02 = src0.Nb[2], nb03 = src0.Nb[3];

        float sf0 = (float)ne0 / ne00, sf1 = (float)ne1 / ne01, sf2 = (float)ne2 / ne02, sf3 = (float)ne3 / ne03;
        float pixelOffset = 0.5f;
        int flags = GgmlTensor.GetOpParamsI32(dst, 0);
        var mode = (GgmlScaleMode)(flags & 0xFF);
        if ((flags & GgmlScaleFlag.AlignCorners) != 0)
        {
            pixelOffset = 0f;
            sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
            sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
        }

        switch (mode)
        {
            case GgmlScaleMode.NEAREST:
                Nearest(src0, dst, ne0, ne1, ne2, ne3, sf0, sf1, sf2, sf3, nb0, nb1, nb2, nb3, nb00, nb01, nb02, nb03, p);
                break;
            case GgmlScaleMode.BILINEAR when (flags & GgmlScaleFlag.Antialias) != 0:
                BilinearAntialias(src0, dst, ne0, ne1, ne2, ne3, sf0, sf1, nb0, nb1, nb2, nb3, nb00, nb01, nb02, nb03, pixelOffset, p);
                break;
            case GgmlScaleMode.BILINEAR:
                Bilinear(src0, dst, ne0, ne1, ne2, ne3, sf0, sf1, nb0, nb1, nb2, nb3, nb00, nb01, nb02, nb03, pixelOffset, p);
                break;
            default: throw new NotImplementedException($"upscale mode {mode}");
        }
    }

    private static void Nearest(GgmlTensor src0, GgmlTensor dst,
        long ne0, long ne1, long ne2, long ne3, float sf0, float sf1, float sf2, float sf3,
        long nb0, long nb1, long nb2, long nb3, long nb00, long nb01, long nb02, long nb03,
        in ComputeParams p)
    {
        for (long i3 = 0; i3 < ne3; i3++)
        {
            long i03 = (long)(i3 / sf3);
            for (long i2 = p.Ith; i2 < ne2; i2 += p.Nth)
            {
                long i02 = (long)(i2 / sf2);
                for (long i1 = 0; i1 < ne1; i1++)
                {
                    long i01 = (long)(i1 / sf1);
                    for (long i0 = 0; i0 < ne0; i0++)
                    {
                        long i00 = (long)(i0 / sf0);
                        float v = *(float*)(src0.Data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
                        *(float*)(dst.Data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3) = v;
                    }
                }
            }
        }
    }

    private static void Bilinear(GgmlTensor src0, GgmlTensor dst,
        long ne0, long ne1, long ne2, long ne3, float sf0, float sf1,
        long nb0, long nb1, long nb2, long nb3, long nb00, long nb01, long nb02, long nb03,
        float pixelOffset, in ComputeParams p)
    {
        long ne00 = src0.Ne[0], ne01 = src0.Ne[1];
        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = p.Ith; i2 < ne2; i2 += p.Nth)
        for (long i1 = 0; i1 < ne1; i1++)
        {
            float y = (i1 + pixelOffset) / sf1 - pixelOffset;
            long y0 = Math.Clamp((long)MathF.Floor(y), 0, ne01 - 1);
            long y1 = Math.Clamp(y0 + 1, 0, ne01 - 1);
            float dy = Math.Clamp(y - y0, 0f, 1f);
            for (long i0 = 0; i0 < ne0; i0++)
            {
                float x = (i0 + pixelOffset) / sf0 - pixelOffset;
                long x0 = Math.Clamp((long)MathF.Floor(x), 0, ne00 - 1);
                long x1 = Math.Clamp(x0 + 1, 0, ne00 - 1);
                float dx = Math.Clamp(x - x0, 0f, 1f);
                float a = *(float*)(src0.Data + x0 * nb00 + y0 * nb01 + i2 * nb02 + i3 * nb03);
                float b = *(float*)(src0.Data + x1 * nb00 + y0 * nb01 + i2 * nb02 + i3 * nb03);
                float c = *(float*)(src0.Data + x0 * nb00 + y1 * nb01 + i2 * nb02 + i3 * nb03);
                float dd = *(float*)(src0.Data + x1 * nb00 + y1 * nb01 + i2 * nb02 + i3 * nb03);
                float val = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + dd * dx * dy;
                *(float*)(dst.Data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3) = val;
            }
        }
    }

    private static void BilinearAntialias(GgmlTensor src0, GgmlTensor dst,
        long ne0, long ne1, long ne2, long ne3, float sf0, float sf1,
        long nb0, long nb1, long nb2, long nb3, long nb00, long nb01, long nb02, long nb03,
        float pixelOffset, in ComputeParams p)
    {
        long ne00 = src0.Ne[0], ne01 = src0.Ne[1];
        float support1 = MathF.Max(1f, 1f / sf1), invscale1 = 1f / support1;
        float support0 = MathF.Max(1f, 1f / sf0), invscale0 = 1f / support0;
        for (long i3 = 0; i3 < ne3; i3++)
        for (long i2 = p.Ith; i2 < ne2; i2 += p.Nth)
        for (long i1 = 0; i1 < ne1; i1++)
        {
            float y = (i1 + pixelOffset) / sf1;
            for (long i0 = 0; i0 < ne0; i0++)
            {
                float x = (i0 + pixelOffset) / sf0;
                long xMin = Math.Max((long)(x - support0 + pixelOffset), 0);
                long xMax = Math.Min((long)(x + support0 + pixelOffset), ne00);
                long yMin = Math.Max((long)(y - support1 + pixelOffset), 0);
                long yMax = Math.Min((long)(y + support1 + pixelOffset), ne01);
                float val = 0, totalWeight = 0;
                for (long sy = yMin; sy < yMax; sy++)
                {
                    float wy = Triangle((sy - y + pixelOffset) * invscale1);
                    for (long sx = xMin; sx < xMax; sx++)
                    {
                        float wx = Triangle((sx - x + pixelOffset) * invscale0);
                        float w = wx * wy;
                        if (w <= 0) continue;
                        val += *(float*)(src0.Data + sx * nb00 + sy * nb01 + i2 * nb02 + i3 * nb03) * w;
                        totalWeight += w;
                    }
                }
                if (totalWeight > 0) val /= totalWeight;
                *(float*)(dst.Data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3) = val;
            }
        }
    }

    private static float Triangle(float x) => MathF.Max(1f - MathF.Abs(x), 0f);
}