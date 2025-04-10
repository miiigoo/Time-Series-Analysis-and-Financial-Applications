import pandas as pd
import numpy as np

def clean_data(file_path, output_path):
    # Step 1: Load the data
    data = pd.read_csv(file_path, header=None)  # Load without headers

    # Step 2: Extract the first column
    first_column = data.iloc[:, 0]  # Extract all rows of the first column

    # Step 3: Convert to numeric and handle non-numeric values
    # Coerce invalid entries to NaN, then drop them
    cleaned_column = pd.to_numeric(first_column, errors='coerce').dropna()

    # Step 4: Format values to two decimal places
    cleaned_column = cleaned_column.apply(lambda x: f"{x:.2f}")

    # Step 5: Save the cleaned data to a new CSV file
    cleaned_column.to_csv(output_path, index=False, header=False)
    print(f"Cleaned data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    input_file = r"<INSERT FILE PATH>"  # Replace with your input file path
    output_file = r"<INSERT FILE PATH>"  # Specify where to save the cleaned data
    clean_data(input_file, output_file)
