"""
Qualitative Comparison Figure Generator (v3 - مُصحَّحة)
=========================================================
تعديلات هذه النسخة عن v2:
  1) المربع الأحمر حول منطقة الاقتصاص بقى أسمك وأوضح (RED_BOX_WIDTH).
  2) بقى فيه فاصل أبيض واضح بين كل صورة مقصوصة والتانية جنبها في الشبكة
     (أفقيًا وعموديًا)، عشان الصور متطغاش على بعض (GRID_GAP).

المكتبات المطلوبة:
    pip install pillow scikit-image numpy --break-system-packages
"""

from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import numpy as np
import os

# =====================================================================
# ============================ CONFIG ================================
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

HR_IMAGE_FILENAME = "img052.png"

# إحداثيات منطقة الاقتصاص (x_left, y_top, x_right, y_bottom)
CROP_BOX = (826, 280, 914,369)

# --- [تعديل 1] سُمك خط المربع الأحمر حول منطقة الاقتصاص ---
#     كان 4 قبل كده. كبّرناها لتبقى واضحة وملحوظة أكتر. زوّد الرقم لو
#     عايزها أسمك، أو قلّله لو حسيتها تقيلة أوي على الصورة
RED_BOX_WIDTH = 6                               # <-- سُمك المربع الأحمر

GROUND_TRUTH_LABEL = "HR"

MODELS = [
    ("HR",        "img052.png"),     
    ("Bicubic",       "img052_bicubic.png"),
    ("DHTCUN",         "img052_DHTCUN.png"),
    ("DAT",             "img052_DAT.png"),
    ("ELAN",           "img052_ELAN.png"),
    ("HAT",             "img052_HAT.png"),
    ("OmniSR",          "img052_OmniSR.png"),
    ("SRFormer",      "img052_SRFormer.png"),
    ("SwinT2",           "img052_SwinT.png"),
    ("Wavelet",          "img052_Wavelet.png"),
]

GRID_COLS = 5
PATCH_DISPLAY_SIZE = 220

# --- [تعديل 2] المسافة الفاصلة (بالبكسل) بين كل صورة والتانية في الشبكة ---
#     دي بتتحط أفقيًا بين الأعمدة وعموديًا بين الصفوف، وبتبقى بيضاء
#     عشان تفصل بصريًا بين كل موديل والتاني وميحصلش تلاصق/طغيان
GRID_GAP = 14                                    # <-- حجم الفاصل الأبيض

AUTO_COMPUTE_METRICS = True
MANUAL_METRICS = {
    "DHTCUN":  (36.49, 0.9439),
}

BOLD_FONT_CANDIDATES = [
    "arialbd.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "DejaVuSans-Bold.ttf",
]
FONT_SIZE_NAME = 18
FONT_SIZE_METRIC = 16

OUTPUT_FILENAME = "Fig_qualitative_img052_x4.jpg"
JPEG_QUALITY = 95

# =====================================================================
# ========================= END OF CONFIG =============================
# =====================================================================


def load_bold_font(size):
    for font_name in BOLD_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def compute_metrics(hr_crop_np, sr_crop_np):
    psnr_val = sk_psnr(hr_crop_np, sr_crop_np, data_range=255)
    ssim_val = sk_ssim(hr_crop_np, sr_crop_np, channel_axis=-1, data_range=255)
    return psnr_val, ssim_val


def draw_labeled_patch(img_crop, label_top, label_bottom, font_name, font_metric):
    label_height = 50
    canvas = Image.new("RGB", (PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE + label_height), "white")
    canvas.paste(img_crop, (0, 0))
    draw = ImageDraw.Draw(canvas)

    w1 = draw.textlength(label_top, font=font_name)
    draw.text(((PATCH_DISPLAY_SIZE - w1) / 2, PATCH_DISPLAY_SIZE + 4),
               label_top, fill="black", font=font_name)

    w2 = draw.textlength(label_bottom, font=font_metric)
    draw.text(((PATCH_DISPLAY_SIZE - w2) / 2, PATCH_DISPLAY_SIZE + 26),
               label_bottom, fill="black", font=font_metric)

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_name = load_bold_font(FONT_SIZE_NAME)
    font_metric = load_bold_font(FONT_SIZE_METRIC)

    hr_path = os.path.join(IMAGES_DIR, HR_IMAGE_FILENAME)

    # ---------- 1) تحميل صورة HR الكاملة ورسم المربع الأحمر (أسمك دلوقتي) ----------
    hr_full = Image.open(hr_path).convert("RGB")
    hr_with_box = hr_full.copy()
    draw = ImageDraw.Draw(hr_with_box)
    draw.rectangle(CROP_BOX, outline="red", width=RED_BOX_WIDTH)

    hr_crop_np = np.array(
        hr_full.crop(CROP_BOX).resize((PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE), Image.LANCZOS)
    )

    # ---------- 2) قصّ نفس المنطقة من كل موديل ----------
    patches = []
    for name, filename in MODELS:
        img_path = os.path.join(IMAGES_DIR, filename)
        img = Image.open(img_path).convert("RGB")
        crop = img.crop(CROP_BOX).resize((PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE), Image.LANCZOS)

        if name == GROUND_TRUTH_LABEL:
            metric_text = "PSNR/SSIM"
        elif AUTO_COMPUTE_METRICS:
            crop_np = np.array(crop)
            psnr_val, ssim_val = compute_metrics(hr_crop_np, crop_np)
            metric_text = f"{psnr_val:.2f}/{ssim_val:.4f}"
        else:
            psnr_val, ssim_val = MANUAL_METRICS.get(name, (0, 0))
            metric_text = f"{psnr_val:.2f}/{ssim_val:.4f}"

        labeled = draw_labeled_patch(crop, name, metric_text, font_name, font_metric)
        patches.append(labeled)

    # ---------- 3) ترصيص شبكة الـ Patches مع فواصل بيضاء بين كل صورة والتانية ----------
    patch_w, patch_h = patches[0].size
    n_rows = (len(patches) + GRID_COLS - 1) // GRID_COLS

    # حجم الشبكة الكلي بيشمل دلوقتي الفواصل (GRID_GAP) بين الأعمدة والصفوف
    grid_w = patch_w * GRID_COLS + GRID_GAP * (GRID_COLS - 1)
    grid_h = patch_h * n_rows + GRID_GAP * (n_rows - 1)
    grid_img = Image.new("RGB", (grid_w, grid_h), "white")

    for idx, patch in enumerate(patches):
        row, col = divmod(idx, GRID_COLS)
        x = col * (patch_w + GRID_GAP)
        y = row * (patch_h + GRID_GAP)
        grid_img.paste(patch, (x, y))

    # ---------- 4) تجهيز صورة HR الكاملة + كابشن تحتها ----------
    hr_display_h = grid_h
    hr_display_w = int(hr_with_box.width * (hr_display_h / hr_with_box.height))
    hr_resized = hr_with_box.resize((hr_display_w, hr_display_h - 40), Image.LANCZOS)

    hr_panel = Image.new("RGB", (hr_display_w, hr_display_h), "white")
    hr_panel.paste(hr_resized, (0, 0))
    d2 = ImageDraw.Draw(hr_panel)
    caption = "Ground Truth HR"
    wc = d2.textlength(caption, font=font_name)
    d2.text(((hr_display_w - wc) / 2, hr_display_h - 34), caption, fill="black", font=font_name)

    # ---------- 5) دمج HR + الشبكة جنب بعض (فاصل أبيض بينهم كمان) ----------
    gap_hr_grid = GRID_GAP + 6   # فاصل شوية أكبر بين لوحة HR وأول عمود في الشبكة
    final_w = hr_display_w + gap_hr_grid + grid_w
    final_h = grid_h
    final_img = Image.new("RGB", (final_w, final_h), "white")
    final_img.paste(hr_panel, (0, 0))
    final_img.paste(grid_img, (hr_display_w + gap_hr_grid, 0))

    # ---------- 6) الحفظ ----------
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_img.save(output_path, "JPEG", quality=JPEG_QUALITY, dpi=(300, 300))
    print(f"تم الحفظ بنجاح: {output_path}")


if __name__ == "__main__":
    main()