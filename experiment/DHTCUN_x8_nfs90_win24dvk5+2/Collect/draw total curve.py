import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_all_columns_from_excel(file_path=None, x_col_index=0, output_base="all_curves"):
    """
    Plot all columns (except X-axis column) from an Excel file.
    Saves as PDF and JPEG (high quality).
    """
    # Find Excel file if not specified
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        excel_files = glob.glob(os.path.join(script_dir, "*.xlsx"))
        if not excel_files:
            print("❌ No Excel file found.")
            return
        file_path = excel_files[0]
        print(f"✅ Found file: {os.path.basename(file_path)}")

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        xl = pd.ExcelFile(file_path, engine='openpyxl')
        sheet_names = xl.sheet_names
        if not sheet_names:
            print("No sheets found.")
            return
        sheet_name = sheet_names[0]
        print(f"Using sheet: '{sheet_name}'")
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error: {e}")
        return

    x_col = df.columns[x_col_index]
    x = df[x_col]
    y_columns = [col for i, col in enumerate(df.columns) if i != x_col_index]

    if not y_columns:
        print("No data columns to plot.")
        return

    plt.figure(figsize=(14, 8))

    # Colors
    num_curves = len(y_columns)
    colors = []
    cmaps = [plt.cm.tab10, plt.cm.Set1, plt.cm.Dark2, plt.cm.tab20c, plt.cm.Paired]
    for cmap in cmaps:
        colors.extend([cmap(i) for i in range(cmap.N)])
    colors = colors[:num_curves]

    for idx, col in enumerate(y_columns):
        y = df[col]
        if not y.isna().all():
            plt.plot(x, y, label=col, color=colors[idx], linewidth=2, marker='.', markersize=3)

    plt.xlabel(x_col)
    plt.ylabel(sheet_name)
    plt.title(f"All Curves - {base_name}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_pdf = os.path.join(script_dir, f"{output_base}.pdf")
    output_jpeg = os.path.join(script_dir, f"{output_base}.jpeg")

    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_jpeg, dpi=300, bbox_inches='tight', format='jpeg')

    print(f"✅ Saved: {output_pdf} and {output_jpeg}")
    plt.show()

if __name__ == "__main__":
    plot_all_columns_from_excel(output_base="all_curves")