import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_all_columns_from_excel(file_path=None, x_col_index=0, output_pdf="all_curves.pdf"):
    """
    Plot all columns (except X-axis column) from an Excel file on a single chart.
    Y-axis label is taken from the sheet name.
    
    Parameters:
    - file_path: Path to Excel file (if None, searches for first .xlsx in script folder)
    - x_col_index: Index of the column to use as X-axis (default 0)
    - output_pdf: Output PDF file name
    """
    # Search for an Excel file in the same folder if path not provided
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        excel_files = glob.glob(os.path.join(script_dir, "*.xlsx"))
        if not excel_files:
            print("❌ No Excel file found in the script directory.")
            return
        file_path = excel_files[0]  # Take the first Excel file found
        print(f"✅ Found file: {os.path.basename(file_path)}")

    # Get file name without extension for title
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        # Get sheet names
        xl = pd.ExcelFile(file_path, engine='openpyxl')
        sheet_names = xl.sheet_names
        if not sheet_names:
            print("No sheets found in the Excel file.")
            return
        sheet_name = sheet_names[0]  # Use first sheet
        print(f"Using sheet: '{sheet_name}'")

        # Read the sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        print(f"Successfully read: {os.path.basename(file_path)}")
        print(f"Columns found: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # X-axis column
    x_col = df.columns[x_col_index]
    x = df[x_col]

    # Columns to plot (all except X-axis column)
    y_columns = [col for i, col in enumerate(df.columns) if i != x_col_index]

    if not y_columns:
        print("No columns available for plotting.")
        return

    # Y-axis label is the sheet name
    y_label = sheet_name

    # Setup plot
    plt.figure(figsize=(14, 8))

    # Highly contrasting colors from multiple colormaps
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
        plt.plot(x, y, label=col, color=colors[idx], linewidth=2, marker='.', markersize=3)

    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.title(f"All Curves - {base_name}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=9, framealpha=0.9)
    plt.tight_layout()

    # Save PDF in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_pdf)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")

    # Display plot (optional)
    plt.show()

if __name__ == "__main__":
    plot_all_columns_from_excel(output_pdf="all_curves.pdf")