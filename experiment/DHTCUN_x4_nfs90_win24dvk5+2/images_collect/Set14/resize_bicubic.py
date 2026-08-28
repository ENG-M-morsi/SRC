import os
from PIL import Image

def resize_image_bicubic(image_path, scale_factor, output_folder=None):
    """
    تكبير الصورة باستخدام bicubic interpolation.
    
    Args:
        image_path (str): مسار الصورة الأصلية.
        scale_factor (int or float): عامل التكبير.
        output_folder (str, optional): مجلد حفظ الصور المكبرة (افتراضي: نفس مجلد الصورة الأصلية).
    
    Returns:
        str: مسار الصورة المحفوظة.
    """
    # فتح الصورة
    img = Image.open(image_path)
    
    # حساب الأبعاد الجديدة
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    
    # تكبير باستخدام bicubic
    resized = img.resize((new_width, new_height), Image.BICUBIC)
    
    # تحديد مسار الحفظ
    if output_folder is None:
        output_folder = os.path.dirname(image_path)
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    base, ext = os.path.splitext(os.path.basename(image_path))
    output_filename = f"{base}_bicubic_x{scale_factor}{ext}"
    output_path = os.path.join(output_folder, output_filename)
    
    # حفظ الصورة (جودة عالية لـ JPEG)
    if ext.lower() in ('.jpg', '.jpeg'):
        resized.save(output_path, quality=95)
    else:
        resized.save(output_path)
    
    print(f"✅ تم حفظ: {output_path}")
    return output_path


def main():
    # مجلد السكربت الحالي
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # البحث عن أول صورة في المجلد (يمكنك تعديل الاسم مباشرة)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
    image_files = [f for f in os.listdir(script_dir) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print("❌ لم يتم العثور على أي صورة في المجلد الحالي.")
        return
    
    # استخدام أول صورة تم العثور عليها (يمكن تغييرها إلى اسم محدد)
    #input_image = "GOOD_KISS_Ver2x4.jpg"
    input_image = image_files[0]
    input_path = os.path.join(script_dir, input_image)
    print(f"📷 سيتم معالجة الصورة: {input_image}")
    
    # عوامل التكبير المطلوبة
    scales = [4]
    #scales = [2, 3, 4, 8]
    
    # تكبير الصورة لكل عامل
    for scale in scales:
        resize_image_bicubic(input_path, scale)
    
    print("\n✅ تم الانتهاء من جميع عمليات التكبير.")


if __name__ == "__main__":
    main()