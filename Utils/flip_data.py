import pandas as pd

def invert_csv_rows(input_path, output_path, has_header=False):
    """
    Reads a CSV from input_path, reverses the row order, and writes the result to output_path.
    If has_header=True, pandas will treat the first row as column headers and preserve them.
    If has_header=False, no header is assumed in the CSV.
    """
    # Read the CSV; if has_header is False, we set header=None
    if has_header:
        df = pd.read_csv(input_path)
    else:
        df = pd.read_csv(input_path, header=None)
    
    # Reverse the DataFrame row order
    df_inverted = df.iloc[::-1].reset_index(drop=True)
    
    # Write to CSV, preserving or omitting headers based on has_header
    df_inverted.to_csv(output_path, header=has_header, index=False)
    print(f"Successfully inverted rows.\nInput: {input_path}\nOutput: {output_path}")

# Example usage:
if __name__ == "__main__":
    # Provide your input and output file paths here:
    input_csv = r"<INSERT FILE PATH>"
    output_csv = r"<INSERT FILE PATH>"

    # If your original CSV has no headers, set has_header=False
    invert_csv_rows(input_csv, output_csv, has_header=False)
