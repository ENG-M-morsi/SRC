import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_all_columns_from_excel(file_path=None, x_col_index=0, output_base="all_curves"):
    """
    Plot all columns (except X-axis column) from an Excel file.
    Marks maximum point for each curve.
    Saves PDF, high-resolution PNG, and high-quality JPEG.
    
    Parameters:
    - file_path: Path to Excel file (if None, searches for first .xlsx in script folder)
    - x_col_index: Index of the column to use as X-axis (default 0)
    - output_base: Base name for output files (without extension)
    """
    # Search for an Excel file in the same folder if path not provided
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        excel_files = glob.glob(os.path.join(script_dir, "*.xlsx"))
        if not excel_files:
            print("❌ No Excel file found in the script directory.")
            return
        file_path = excel_files[0]
        print(f"✅ Found file: {os.path.basename(file_path)}")

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        xl = pd.ExcelFile(file_path, engine='openpyxl')
        sheet_names = xl.sheet_names
        if not sheet_names:
            print("No sheets found in the Excel file.")
            return
        sheet_name = sheet_names[0]
        print(f"Using sheet: '{sheet_name}'")

        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        print(f"Successfully read: {os.path.basename(file_path)}")
        print(f"Columns found: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    x_col = df.columns[x_col_index]
    x = df[x_col]

    y_columns = [col for i, col in enumerate(df.columns) if i != x_col_index]

    if not y_columns:
        print("No columns available for plotting.")
        return

    y_label = sheet_name

    plt.figure(figsize=(14, 8))

    num_curves = len(y_columns)
    colors = []
    cmaps = [plt.cm.tab10, plt.cm.Set1, plt.cm.Dark2, plt.cm.tab20c, plt.cm.Paired]
    for cmap in cmaps:
        colors.extend([cmap(i) for i in range(cmap.N)])
    if len(colors) < num_curves:
        additional = num_curves - len(colors)
        colors.extend([plt.cm.hsv(i / additional) for i in range(additional)])
    else:
        colors = colors[:num_curves]

    for idx, col in enumerate(y_columns):
        y = df[col]
        if y.isna().all():
            print(f"⚠️ Column '{col}' contains only NaN values, skipped.")
            continue
        
        # رسم المنحنى
        plt.plot(x, y, label=col, color=colors[idx], linewidth=2, marker='.', markersize=3)
        
        # حساب النقطة القصوى
        max_val = y.max()
        max_idx = y.idxmax()  # فهرس السطر
        max_x = x.iloc[max_idx]
        # إضافة نقطة القصوى
        plt.scatter(max_x, max_val, color=colors[idx], s=100, zorder=5)
        plt.annotate(f'{max_val:.3f}', (max_x, max_val), xytext=(5, 5), textcoords='offset points',
                     fontsize=8, color=colors[idx])

    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.title(f"All Curves - {base_name}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=9, framealpha=0.9)
    plt.tight_layout()

    # حفظ بثلاث صيغ
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_pdf = os.path.join(script_dir, f"{output_base}.pdf")
    output_png = os.path.join(script_dir, f"{output_base}.png")
    output_jpeg = os.path.join(script_dir, f"{output_base}.jpeg")

    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_jpeg, dpi=300, bbox_inches='tight', format='jpeg', pil_kwargs={'quality': 95})

    print(f"✅ Plot saved to:")
    print(f"   PDF:  {output_pdf}")
    print(f"   PNG:  {output_png}")
    print(f"   JPEG: {output_jpeg}")

    plt.show()

if __name__ == "__main__":
    plot_all_columns_from_excel(output_base="all_curves")