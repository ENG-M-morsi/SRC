"""
Qualitative Comparison Figure Generator (v2 - مُصحَّحة)
=========================================================
يعمل شكل مقارنة بصرية بنفس أسلوب الأبحاث: صورة HR كاملة على الشمال مع
مربع أحمر يوضح منطقة الاقتصاص (Crop)، وعلى اليمين شبكة (Grid) من نفس
المنطقة مقصوصة من كل موديل، وتحت كل واحدة اسم الموديل + PSNR/SSIM.

التعديلات في هذه النسخة:
  1) المسارات بقت تُحسب بالنسبة لمكان ملف الكود نفسه (مش مكان تشغيل الأمر)،
     فمجلد images ومجلد outputs بيتلاقوا صح دايمًا مهما شغّلت من أي مكان.
  2) مجلد الصور المدخلة: "images/" بجانب الكود مباشرة.
  3) مجلد الناتج: "outputs/" بجانب الكود (بنفس مستوى مجلد images، مش جواه).
  4) قصّة HR (الأصل) دلوقتي بتاخد تسمية خاصة "PSNR/SSIM" (رمزية، مش أرقام
     محسوبة) عشان توضح إنها المرجع اللي بيتقارن بيه الباقي، بدون أي حساب
     يسبب قسمة على صفر.
  5) الخط تحت كل صورة بقى Bold أسود وأوضح، بخط مناسب للنشر العلمي.

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

# --- مسار مجلد الكود نفسه (تلقائي، متلمسوش) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- مجلد الصور المدخلة: لازم يكون اسمه "images" وموجود بجانب الكود ---
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

# --- مجلد حفظ الناتج: هيتعمل تلقائي بجانب مجلد images (نفس مستواه) ---
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# --- 1) اسم ملف صورة الأصل كاملة الدقة (HR) داخل مجلد images ---
HR_IMAGE_FILENAME = "GOOD_KISS_Ver2_HR.png"              # <-- غيّر اسم الملف بس هنا

# --- 2) إحداثيات منطقة الاقتصاص (Crop Box) على الصورة الأصلية ---
#     الترتيب: (x_left, y_top, x_right, y_bottom) بالبكسل
#     >>> ده المكان اللي تتحكم فيه بمكان وحجم منطقة الاقتصاص <<<
CROP_BOX = (38, 541, 257,639)                 # <-- عدّل الأرقام دي حسب صورتك

# --- 3) اسم رمزي يُستخدم لتمييز "قصّة HR المرجعية" عن باقي الموديلات ---
#     لازم يبقى بالظبط الاسم ده في MODELS لو عايز تضيف عمود HR كأول عمود
GROUND_TRUTH_LABEL = "HR"

# --- 4) قائمة الموديلات: (الاسم اللي هيظهر تحت الصورة، اسم ملف صورته
#     داخل مجلد images فقط -- من غير مسار كامل)
#     رتّبهم بنفس الترتيب اللي عايز يظهروا بيه في الشبكة (صف1 ثم صف2)
#     >>> لو عايز تضيف HR كأول عمود، حطه بالاسم GROUND_TRUTH_LABEL وأي مسار
#         (مش هيتحسب له PSNR/SSIM فعليًا، هيتكتب تحته النص الرمزي بس)
MODELS = [
    ("HR",        "GOOD_KISS_Ver2_HR.png"),
    ("DHTCUN",         "GOOD_KISS_Ver2_DHTCUN.png"),
    ("ELAN",           "GOOD_KISS_Ver2_ELAN.png"),
    ("DAT",            "GOOD_KISS_Ver2_DAT.png"),
    ("HAT",            "GOOD_KISS_Ver2_HAT.png"),
    ("OmniSR",         "GOOD_KISS_Ver2_OmniSR.png"),
    ("SwinT",         "GOOD_KISS_Ver2_SwinT.png"),
    ("Wavelet",         "GOOD_KISS_Ver2_Wavelet.png"),
]

# --- 5) كام عمود في الشبكة (عدد الصور جنب بعض في الصف الواحد) ---
GRID_COLS = 4                                    # <-- عدد الأعمدة (راجع الشرح تحت للسؤال 6)

# --- 6) حجم عرض كل صورة مقصوصة داخل الشبكة (بالبكسل) ---
PATCH_DISPLAY_SIZE = 220                         # <-- حجم كل مربع في الشبكة

# --- 7) هل تحسب PSNR/SSIM أوتوماتيك (True) ولا تكتبهم يدويًا (False)؟ ---
AUTO_COMPUTE_METRICS = True

# لو AUTO_COMPUTE_METRICS = False، اكتب القيم هنا يدويًا (PSNR, SSIM)
MANUAL_METRICS = {
    "DHTCUN":  (36.49, 0.9439),
    # ... كمّل الباقي لو مش هتحسبهم أوتوماتيك
}

# --- 8) الخط تحت كل صورة (Bold وأسود، مناسب للنشر العلمي) ---
# قائمة خطوط Bold شائعة (السكريبت بيجرّب واحد واحد لحد ما يلاقي المتاح)
BOLD_FONT_CANDIDATES = [
    "arialbd.ttf",                       # Arial Bold (ويندوز)
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/timesbd.ttf",       # Times New Roman Bold (شائع في IEEE)
    "DejaVuSans-Bold.ttf",                # لينكس/ماك
]
FONT_SIZE_NAME = 18      # حجم خط اسم الموديل (السطر الأول)
FONT_SIZE_METRIC = 16    # حجم خط PSNR/SSIM (السطر الثاني)

# --- 9) اسم ملف الناتج (هيتحفظ جوه OUTPUT_DIR أوتوماتيك) ---
OUTPUT_FILENAME = "Fig_qualitative_zebra_x4.jpg"
JPEG_QUALITY = 95

# =====================================================================
# ========================= END OF CONFIG =============================
# =====================================================================


def load_bold_font(size):
    """يجرّب كل الخطوط في BOLD_FONT_CANDIDATES لحد ما يلاقي واحد شغال."""
    for font_name in BOLD_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def compute_metrics(hr_crop_np, sr_crop_np):
    """يحسب PSNR و SSIM بين قصّة الأصل وقصّة الناتج."""
    psnr_val = sk_psnr(hr_crop_np, sr_crop_np, data_range=255)
    ssim_val = sk_ssim(hr_crop_np, sr_crop_np, channel_axis=-1, data_range=255)
    return psnr_val, ssim_val


def draw_labeled_patch(img_crop, label_top, label_bottom, font_name, font_metric):
    """يرسم مربع الصورة المقصوصة + سطرين نص Bold أسود تحته."""
    label_height = 50
    canvas = Image.new("RGB", (PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE + label_height), "white")
    canvas.paste(img_crop, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # السطر الأول: اسم الموديل (وسط، Bold أسود)
    w1 = draw.textlength(label_top, font=font_name)
    draw.text(((PATCH_DISPLAY_SIZE - w1) / 2, PATCH_DISPLAY_SIZE + 4),
               label_top, fill="black", font=font_name)

    # السطر الثاني: PSNR/SSIM أو الرمز التوضيحي (وسط، Bold أسود برضه)
    w2 = draw.textlength(label_bottom, font=font_metric)
    draw.text(((PATCH_DISPLAY_SIZE - w2) / 2, PATCH_DISPLAY_SIZE + 26),
               label_bottom, fill="black", font=font_metric)

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_name = load_bold_font(FONT_SIZE_NAME)
    font_metric = load_bold_font(FONT_SIZE_METRIC)

    hr_path = os.path.join(IMAGES_DIR, HR_IMAGE_FILENAME)

    # ---------- 1) تحميل صورة HR الكاملة ورسم المربع الأحمر ----------
    hr_full = Image.open(hr_path).convert("RGB")
    hr_with_box = hr_full.copy()
    draw = ImageDraw.Draw(hr_with_box)
    draw.rectangle(CROP_BOX, outline="red", width=4)

    hr_crop_np = np.array(
        hr_full.crop(CROP_BOX).resize((PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE), Image.LANCZOS)
    )

    # ---------- 2) قصّ نفس المنطقة من كل موديل (بما فيهم HR كحالة خاصة) ----------
    patches = []
    for name, filename in MODELS:
        img_path = os.path.join(IMAGES_DIR, filename)
        img = Image.open(img_path).convert("RGB")
        crop = img.crop(CROP_BOX).resize((PATCH_DISPLAY_SIZE, PATCH_DISPLAY_SIZE), Image.LANCZOS)

        if name == GROUND_TRUTH_LABEL:
            # حالة HR: مفيش حساب PSNR/SSIM فعلي (هيطلع لا نهائي لأنها بتتقارن
            # بنفسها) -- بنكتب بدل الأرقام الرمز التوضيحي "PSNR/SSIM" نفسه
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

    # ---------- 3) ترصيص شبكة الـ Patches (صفوف × أعمدة) ----------
    patch_w, patch_h = patches[0].size
    n_rows = (len(patches) + GRID_COLS - 1) // GRID_COLS
    grid_w = patch_w * GRID_COLS
    grid_h = patch_h * n_rows
    grid_img = Image.new("RGB", (grid_w, grid_h), "white")
    for idx, patch in enumerate(patches):
        row, col = divmod(idx, GRID_COLS)
        grid_img.paste(patch, (col * patch_w, row * patch_h))

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

    # ---------- 5) دمج HR + الشبكة جنب بعض ----------
    gap = 20
    final_w = hr_display_w + gap + grid_w
    final_h = grid_h
    final_img = Image.new("RGB", (final_w, final_h), "white")
    final_img.paste(hr_panel, (0, 0))
    final_img.paste(grid_img, (hr_display_w + gap, 0))

    # ---------- 6) الحفظ ----------
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_img.save(output_path, "JPEG", quality=JPEG_QUALITY, dpi=(300, 300))
    print(f"تم الحفظ بنجاح: {output_path}")


if __name__ == "__main__":
    main()

