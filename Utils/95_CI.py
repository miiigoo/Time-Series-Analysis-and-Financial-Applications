import numpy as np
import pandas as pd
from scipy import stats

# Path to your CSV file
file_path = r"<INSERT FILE PATH>"  # Replace with your file path

# Read the data from the CSV file
data = pd.read_csv(file_path)

# Extract the first and second columns
rmse_data = data.iloc[:, 0]
hit_rate_data = data.iloc[:, 1]

# Function to calculate 95% CI
def calculate_ci(data, label):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    
    # Get the t critical value for 95% CI with n - 1 degrees of freedom
    t_critical = stats.t.ppf(0.975, n - 1)
    
    # Margin of Error
    margin_of_error = t_critical * (std / np.sqrt(n))
    
    # Print result in the required format
    print(f"{label} CI = {mean:.2f} ± {margin_of_error:.2f}")

# Calculate for RMSE (first column)
calculate_ci(rmse_data, "RMSE")

# Calculate for Hit Rate (second column)
calculate_ci(hit_rate_data, "Hit Rate")
