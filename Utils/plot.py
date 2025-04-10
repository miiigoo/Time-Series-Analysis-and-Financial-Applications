import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
df = pd.read_excel(r"<INSERT FILE PATH>")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Ensure 'Model' is string
df['Model'] = df['Model'].astype(str)

# Create numeric indices for plotting
x = np.arange(len(df))

fig = plt.figure(figsize=(14, 6))

#############################
# Subplot for RMSE
#############################
ax1 = plt.subplot(1, 2, 1)  
rmse_bars = ax1.errorbar(
    x=x, 
    y=df['RMSE'], 
    yerr=df['RMSE_CI'], 
    capsize=5, 
    linestyle='None', 
    marker='s', 
    markersize=8, 
    color='blue'
)
ax1.set_title("RMSE with 95% CI (Walk Forward and Volume SPX Testing)", fontsize=13)
ax1.set_xlabel("Model", fontsize=13)
ax1.set_ylabel("RMSE", fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'], rotation=45)
ax1.grid(True, linestyle='--', alpha=0.3)

#############################
# Subplot for Hit Rate
#############################
ax2 = plt.subplot(1, 2, 2)
hr_bars = ax2.errorbar(
    x=x, 
    y=df['Hit_Rate'], 
    yerr=df['Hit_Rate_CI'], 
    capsize=5, 
    linestyle='None', 
    marker='s', 
    markersize=8, 
    color='green'
)
ax2.set_title("Hit Rate with 95% CI (Walk Forward and Volume SPX Testing)", fontsize=13)
ax2.set_xlabel("Model", fontsize=13)
ax2.set_ylabel("Hit Rate/ %", fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(df['Model'], rotation=45)
ax2.grid(True, linestyle='--', alpha=0.3)

# ----------------------------------------------------------------
# 1) Retrieve the y-limits from each subplot AFTER plotting
# ----------------------------------------------------------------
rmse_ymin, rmse_ymax = ax1.get_ylim()
hr_ymin, hr_ymax = ax2.get_ylim()

# ----------------------------------------------------------------
# 2) Compute a small offset as a fraction of the y-range
# ----------------------------------------------------------------
rmse_range = rmse_ymax - rmse_ymin
hr_range = hr_ymax - hr_ymin

rmse_offset = 0.01 * rmse_range
hr_offset   = 0.015 * hr_range

# ----------------------------------------------------------------
# 3) Place annotations above the error bars + offset
# ----------------------------------------------------------------
for i, row in df.iterrows():
    # For RMSE
    rmse_annotation_y = row['RMSE'] + row['RMSE_CI'] + rmse_offset
    ax1.text(
        x[i], 
        rmse_annotation_y, 
        f"{row['RMSE']:.4f}", 
        fontsize=9, 
        ha='center', 
        va='bottom'
    )
    
    # For Hit Rate
    hr_annotation_y = row['Hit_Rate'] + row['Hit_Rate_CI'] + hr_offset
    ax2.text(
        x[i], 
        hr_annotation_y, 
        f"{row['Hit_Rate']:.2f}", 
        fontsize=9, 
        ha='center', 
        va='bottom'
    )

# ----------------------------------------------------------------
# 4) Manually adjust y-limits so all annotations fit
# ----------------------------------------------------------------
# Make sure the upper y-limit is large enough to accommodate text
rmse_new_ymax = max(rmse_ymax, (df['RMSE'] + df['RMSE_CI']).max() + rmse_offset * 2)
ax1.set_ylim(rmse_ymin, rmse_new_ymax)

hr_new_ymax = max(hr_ymax, (df['Hit_Rate'] + df['Hit_Rate_CI']).max() + hr_offset * 2)
ax2.set_ylim(hr_ymin, hr_new_ymax)

# ----------------------------------------------------------------
# 5) Adjust subplot spacing manually
# ----------------------------------------------------------------
plt.subplots_adjust(
    left=0.07,
    right=0.98,
    wspace=0.15,  # space between RMSE and Hit Rate plots
    bottom=0.15  # give more space at bottom for rotated labels
)

# Show the plot
plt.show()
