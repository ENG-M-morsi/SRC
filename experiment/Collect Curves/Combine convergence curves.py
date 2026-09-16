"""
Convergence Curves Composer — assembles per-model training curve panels
into single multi-panel figures suitable for international journal submission
========================================================================
Takes individual curve images (DHTCUN vs Model at each upscaling factor) and
arranges them side by side in one horizontal row per model, with a bold
sub-caption (a)(b)(c)(d) under each panel, so you get one composite figure
per model instead of separate images.

Output: one composite figure per model group (DAT, ELAN, HAT, OmniSR,
SwinT2 — each with 4 sub-panels), plus one extra composite for
SRFormer + Wavelet (only evaluated at x4).

Publication-quality safeguards built into this script:
    - Figure width/height are computed from a target PRINT width in inches
      and a target DPI, so every composite is guaranteed to meet common
      journal resolution requirements (default: 600 DPI, IEEE double-column
      width of 7.16 in) instead of an arbitrary fixed pixel size.
    - Source panels are only ever downscaled, never upscaled. Upscaling a
      low-resolution source image would fabricate detail that was never
      there and produce a blurry, unpublishable panel; if a source image is
      smaller than the computed target size, the script keeps it at its
      native resolution and prints a clear warning instead of stretching it.
    - Every composite is exported as PDF (vector-wrapped, ideal for LaTeX
      submissions), PNG (lossless raster), and JPEG (quality 100, no chroma
      subsampling, for journals that specifically require .jpg).
    - All on-figure text is English, and sub-caption placement is
      auto-centered under each panel so labels never overlap each other or
      the artwork.

Required packages:
    pip install pillow --break-system-packages
"""

from PIL import Image, ImageDraw, ImageFont
import os
import sys

# =====================================================================
# ============================ CONFIG ================================
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Folder containing the individual curve images (DHTCUN vs each model) ---
CURVES_DIR = os.path.join(SCRIPT_DIR, "curves")

# --- Folder where the composite output figures are saved ---
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_composite")

# --- For each model: list of (sub-panel label, image filename in CURVES_DIR) ---
# >>> Edit the filenames below to match exactly what you have saved <<<
MODEL_GROUPS = {
    "DAT": [
        ("(a) \u00d72", "all_curves_DATx2.jpeg"),
        ("(b) \u00d73", "all_curves_DATx3.jpeg"),
        ("(c) \u00d74", "all_curves_DATx4.jpeg"),
        ("(d) \u00d78", "all_curves_DATx8.jpeg"),
    ],
    "ELAN": [
        ("(a) \u00d72", "all_curves_ELANx2.jpeg"),
        ("(b) \u00d73", "all_curves_ELANx3.jpeg"),
        ("(c) \u00d74", "all_curves_ELANx4.jpeg"),
        ("(d) \u00d78", "all_curves_ELANx8.jpeg"),
    ],
    "HAT": [
        ("(a) \u00d72", "all_curves_HATx2.jpeg"),
        ("(b) \u00d73", "all_curves_HATx3.jpeg"),
        ("(c) \u00d74", "all_curves_HATx4.jpeg"),
        ("(d) \u00d78", "all_curves_HATx8.jpeg"),
    ],
    "OmniSR": [
        ("(a) \u00d72", "all_curves_Omnix2.jpeg"),
        ("(b) \u00d73", "all_curves_Omnix3.jpeg"),
        ("(c) \u00d74", "all_curves_Omnix4.jpeg"),
        ("(d) \u00d78", "all_curves_Omnix8.jpeg"),
    ],
    "SwinT2": [
        ("(a) \u00d72", "all_curves_SwinTx2.jpeg"),
        ("(b) \u00d73", "all_curves_SwinTx3.jpeg"),
        ("(c) \u00d74", "all_curves_SwinTx4.jpeg"),
        ("(d) \u00d78", "all_curves_SwinTx8.jpeg"),
    ],
    # SRFormer and Wavelet were only evaluated at x4 (not part of the
    # multi-scale study), so they get their own 2-panel composite.
    "SRFormer_Wavelet": [
        ("(a) SRFormer, \u00d74", "all_curves_SRFormerx4.jpeg"),
        ("(b) Wavelet, \u00d74", "all_curves_Waveletx4.jpeg"),
    ],
}

# --- Target print size / resolution (drives everything below) ---
# 7.16 in = IEEE/Elsevier standard double-column figure width.
# Use 3.5 in instead if you want a single-column figure.
TARGET_TOTAL_WIDTH_IN = 7.16
# 600 DPI is a safe, commonly required resolution for combination
# figures (line art + text). 300 DPI is the usual *minimum* for
# photographic content, so 600 gives headroom for sharp text/lines.
TARGET_DPI = 600
# The pixel budget above is computed from a 4-panel row; groups with
# fewer panels (e.g. the 2-panel SRFormer/Wavelet composite) reuse the
# same per-panel width, so panel size is consistent across every figure
# in the paper.
BASE_N_PANELS = 4

# --- White gap between sub-panels, in inches (converted to pixels below) ---
PANEL_GAP_IN = 0.06

# --- White margin around the whole composite, in inches ---
MARGIN_IN = 0.06

# --- Sub-caption font size, expressed as a fraction of panel width so it
#     scales automatically with TARGET_DPI / TARGET_TOTAL_WIDTH_IN ---
FONT_SIZE_FRACTION = 0.032

# --- Bold font search order: Linux (this environment) first, then the
#     common Windows/Mac locations, then a bundled DejaVu fallback name
#     so the script never silently falls back to an unreadable tiny font.
BOLD_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "DejaVuSans-Bold.ttf",
]

JPEG_QUALITY = 100  # combined with subsampling=0 below -> no visible artifacts

# =====================================================================
# ========================= END OF CONFIG ==============================
# =====================================================================

# ---- Derived pixel measurements (computed once, from the config above) ----
PANEL_GAP_PX = round(PANEL_GAP_IN * TARGET_DPI)
MARGIN_PX = round(MARGIN_IN * TARGET_DPI)
# Budget = total target width MINUS the two outer margins and the gaps
# between panels, so that margins + panels + gaps together add up to
# exactly TARGET_TOTAL_WIDTH_IN once the composite is assembled — not
# more than it (the previous version forgot to subtract the margins,
# which made the final PDF/PNG page slightly wider than requested).
_budget_px = (TARGET_TOTAL_WIDTH_IN * TARGET_DPI
              - 2 * MARGIN_PX
              - PANEL_GAP_PX * (BASE_N_PANELS - 1))
PANEL_WIDTH_PX = round(_budget_px / BASE_N_PANELS)
FONT_SIZE_PX = max(12, round(PANEL_WIDTH_PX * FONT_SIZE_FRACTION))
LABEL_AREA_PX = round(FONT_SIZE_PX * 2.2)  # vertical space reserved for the caption


def load_bold_font(size):
    """Try each candidate bold font in turn; only fall back to the (tiny,
    unreadable) PIL default font if literally nothing else is available,
    and warn loudly if that happens so it never fails silently."""
    for font_name in BOLD_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    print("   WARNING: no bold TrueType font found on this system — "
          "falling back to PIL's default bitmap font, which will look "
          "small and unpublishable. Install a font such as DejaVu Sans "
          "Bold or Liberation Sans Bold to fix this.")
    return ImageFont.load_default()


def load_and_scale_panel(filename):
    """Load one source panel and scale it to PANEL_WIDTH_PX.

    Only ever downscales. If the source is already narrower than the
    target width, it is kept at its native resolution (never stretched
    up) and a warning is printed, because upscaling would fabricate
    detail that was never in the original image and produce a blurry,
    unpublishable panel.
    """
    img_path = os.path.join(CURVES_DIR, filename)
    if not os.path.exists(img_path):
        print(f"   ERROR: source image not found, skipping this panel: {img_path}")
        return None

    img = Image.open(img_path).convert("RGB")

    if img.width >= PANEL_WIDTH_PX:
        ratio = PANEL_WIDTH_PX / img.width
        new_h = round(img.height * ratio)
        img = img.resize((PANEL_WIDTH_PX, new_h), Image.LANCZOS)
    else:
        print(f"   WARNING: '{filename}' is only {img.width}px wide, below the "
              f"{PANEL_WIDTH_PX}px target for a {TARGET_TOTAL_WIDTH_IN}in-wide "
              f"figure at {TARGET_DPI} DPI. Keeping it at native resolution "
              f"instead of upscaling (upscaling would blur it, not sharpen "
              f"it). Re-export this source curve at a higher resolution for "
              f"a fully consistent, publication-quality composite.")

    return img


def compose_model_group(panels, font):
    loaded = []
    for label, filename in panels:
        img = load_and_scale_panel(filename)
        if img is not None:
            loaded.append((label, img))

    if not loaded:
        return None

    panel_h = max(im.height for _, im in loaded)
    n = len(loaded)
    inner_w = sum(im.width for _, im in loaded) + PANEL_GAP_PX * (n - 1)
    total_w = inner_w + 2 * MARGIN_PX
    total_h = panel_h + LABEL_AREA_PX + 2 * MARGIN_PX

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    x = MARGIN_PX
    for label, img in loaded:
        # Vertically center shorter panels against the tallest one in the row
        y = MARGIN_PX + (panel_h - img.height) // 2
        canvas.paste(img, (x, y))

        # Center the caption under this panel's own width so labels never
        # collide with the neighbouring panel's caption.
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x + (img.width - text_w) / 2
        text_y = MARGIN_PX + panel_h + (LABEL_AREA_PX - FONT_SIZE_PX) / 2
        draw.text((text_x, text_y), label, fill="black", font=font)

        x += img.width + PANEL_GAP_PX

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isdir(CURVES_DIR):
        print(f"ERROR: curves folder not found: {CURVES_DIR}")
        sys.exit(1)

    font = load_bold_font(FONT_SIZE_PX)

    print("=" * 75)
    print("Convergence Curves Composer — publication-ready export")
    print("=" * 75)
    print(f"Target figure width : {TARGET_TOTAL_WIDTH_IN} in")
    print(f"Target resolution   : {TARGET_DPI} DPI")
    print(f"Panel width (px)    : {PANEL_WIDTH_PX}")
    print(f"Caption font size   : {FONT_SIZE_PX} px")
    print()

    any_saved = False
    for model_name, panels in MODEL_GROUPS.items():
        print(f"-> Composing: {model_name}")
        composite = compose_model_group(panels, font)
        if composite is None:
            print(f"   SKIPPED: no valid source panels found for '{model_name}'")
            continue

        base_path = os.path.join(OUTPUT_DIR, f"Fig_convergence_{model_name}")

        # PDF: vector-wrapped raster, ideal for LaTeX \includegraphics submissions
        composite.save(f"{base_path}.pdf", "PDF", resolution=TARGET_DPI)
        # PNG: lossless raster, safest default for review and revision cycles
        composite.save(f"{base_path}.png", "PNG", dpi=(TARGET_DPI, TARGET_DPI))
        # JPEG: quality 100 and subsampling=0 (4:4:4) to avoid the chroma
        # blurring around text/line edges that default JPEG settings cause
        composite.save(f"{base_path}.jpeg", "JPEG", quality=JPEG_QUALITY,
                        subsampling=0, dpi=(TARGET_DPI, TARGET_DPI))

        print(f"   Saved: {base_path}.pdf / .png / .jpeg "
              f"({composite.width}x{composite.height}px)")
        any_saved = True

    print()
    if any_saved:
        print(f"Done. All composite figures are in: {OUTPUT_DIR}")
    else:
        print("No figures were produced — check the file paths in MODEL_GROUPS "
              "and the CURVES_DIR setting above.")


if __name__ == "__main__":
    main()