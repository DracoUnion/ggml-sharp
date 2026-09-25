namespace GgmlSharp.Kernel;

/// <summary>Mirrors <c>enum ggml_tri_type</c>.</summary>
public enum GgmlTriType
{
    LOWER,
    LOWER_DIAG,
    UPPER,
    UPPER_DIAG,
}

/// <summary>Mirrors <c>enum ggml_sort_order</c>.</summary>
public enum GgmlSortOrder
{
    ASC,
    DESC,
}

/// <summary>Mirrors <c>enum ggml_scale_mode</c> and the upscale flag bits.</summary>
public enum GgmlScaleMode
{
    NEAREST = 0,
    BILINEAR = 1,
    BICUBIC = 2,
}

public static class GgmlScaleFlag
{
    public const int AlignCorners = 1 << 16;
    public const int Antialias    = 1 << 17;
}
