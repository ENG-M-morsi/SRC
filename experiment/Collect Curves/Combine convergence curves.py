"""
Convergence Curves Composer — يجمّع منحنيات التدريب الفردية لكل موديل
========================================================================
بياخد الصور الفردية (DHTCUN vs Model عند كل مقياس تكبير) ويرصّهم صف واحد
أفقي جنب بعض لكل موديل، مع حرف فرعي (a)(b)(c)(d) تحت كل صورة يوضح المقياس،
عشان يبقى عندك شكل مُجمَّع واحد بدل 4 صور منفصلة لكل موديل.

الناتج: 6 صور مُجمَّعة (DAT, ELAN, HAT, OmniSR, SwinT2 = كل واحد بـ 4 لوحات
فرعية، + صورة سادسة لـ SRFormer + Wavelet مع بعض لأنهم عند x4 بس).

المكتبات المطلوبة:
    pip install pillow --break-system-packages
"""

from PIL import Image, ImageDraw, ImageFont
import os

# =====================================================================
# ============================ CONFIG ================================
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- مجلد الصور الفردية (منحنيات DHTCUN مقابل كل موديل) ---
CURVES_DIR = os.path.join(SCRIPT_DIR, "curves")

# --- مجلد حفظ الصور المُجمَّعة الناتجة ---
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_composite")

# --- لكل موديل: قايمة (تسمية اللوحة الفرعية، اسم ملف الصورة داخل CURVES_DIR) ---
# >>> عدّل أسماء الملفات هنا بالظبط زي ما هي محفوظة عندك <<<
MODEL_GROUPS = {
    "DAT": [
        ("(a) ×2", "all_curves_DATx2.jpeg"),
        ("(b) ×3", "all_curves_DATx3.jpeg"),
        ("(c) ×4", "all_curves_DATx4.jpeg"),
        ("(d) ×8", "all_curves_DATx8.jpeg"),
    ],
    "ELAN": [
        ("(a) ×2", "all_curves_ELANx2.jpeg"),
        ("(b) ×3", "all_curves_ELANx3.jpeg"),
        ("(c) ×4", "all_curves_ELANx4.jpeg"),
        ("(d) ×8", "all_curves_ELANx8.jpeg"),
    ],
    "HAT": [
        ("(a) ×2", "all_curves_HATx2.jpeg"),
        ("(b) ×3", "all_curves_HATx3.jpeg"),
        ("(c) ×4", "all_curves_HATx4.jpeg"),
        ("(d) ×8", "all_curves_HATx8.jpeg"),
    ],
    "OmniSR": [
        ("(a) ×2", "all_curves_Omnix2.jpeg"),
        ("(b) ×3", "all_curves_Omnix3.jpeg"),
        ("(c) ×4", "all_curves_Omnix4.jpeg"),
        ("(d) ×8", "all_curves_Omnix8.jpeg"),
    ],
    "SwinT2": [
        ("(a) ×2", "all_curves_SwinTx2.jpeg"),
        ("(b) ×3", "all_curves_SwinTx3.jpeg"),
        ("(c) ×4", "all_curves_SwinTx4.jpeg"),
        ("(d) ×8", "all_curves_SwinTx8.jpeg"),
    ],
    # SRFormer و Wavelet عندهم بس x4 (مش من ضمن دراسة تعدد المقاييس)
    "SRFormer_Wavelet": [
        ("(a) SRFormer, ×4", "all_curves_SRFormerx4.jpeg"),
        ("(b) Wavelet, ×4",  "all_curves_Waveletx4.jpeg"),
    ],
}

# --- عرض كل لوحة فرعية داخل الصورة المُجمَّعة (بالبكسل) ---
PANEL_WIDTH = 480          # <-- حجم كل منحنى فرعي (الطول بيتحسب تلقائيًا حسب أبعاد الصورة الأصلية)

# --- الفاصل الأبيض بين اللوحات الفرعية ---
PANEL_GAP = 16

# --- الخط تحت كل لوحة فرعية (Bold أسود، زي باقي أشكال البحث) ---
BOLD_FONT_CANDIDATES = [
    "arialbd.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "DejaVuSans-Bold.ttf",
]
FONT_SIZE = 20

JPEG_QUALITY = 95

# =====================================================================
# ========================= END OF CONFIG ==============================
# =====================================================================


def load_bold_font(size):
    for font_name in BOLD_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def compose_model_group(model_name, panels, font):
    label_h = 40  # ارتفاع منطقة كتابة تسمية اللوحة (a)(b)(c)(d)
    loaded = []
    for label, filename in panels:
        img_path = os.path.join(CURVES_DIR, filename)
        img = Image.open(img_path).convert("RGB")
        # نحافظ على نسبة الأبعاد الأصلية للمنحنى وقت تصغيره لعرض موحّد
        ratio = PANEL_WIDTH / img.width
        new_h = int(img.height * ratio)
        img_resized = img.resize((PANEL_WIDTH, new_h), Image.LANCZOS)
        loaded.append((label, img_resized))

    panel_h = max(im.height for _, im in loaded)
    n = len(loaded)
    total_w = PANEL_WIDTH * n + PANEL_GAP * (n - 1)
    total_h = panel_h + label_h

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, img in loaded:
        canvas.paste(img, (x, 0))
        w = draw.textlength(label, font=font)
        draw.text((x + (PANEL_WIDTH - w) / 2, panel_h + 8), label, fill="black", font=font)
        x += PANEL_WIDTH + PANEL_GAP

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font = load_bold_font(FONT_SIZE)

    for model_name, panels in MODEL_GROUPS.items():
        composite = compose_model_group(model_name, panels, font)
        out_path = os.path.join(OUTPUT_DIR, f"Fig_convergence_{model_name}.jpg")
        composite.save(out_path, "JPEG", quality=JPEG_QUALITY, dpi=(300, 300))
        print(f"تم الحفظ: {out_path}")


if __name__ == "__main__":
    main()