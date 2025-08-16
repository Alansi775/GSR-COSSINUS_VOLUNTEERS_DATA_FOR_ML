import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv("cleaned_GSR_Data-4.csv")

# Define stage colors
stage_colors = {
    "Calibration (20s)": "blue",
    "Normal - Watch Video (4 min)": "green",
    "Remove 5 Easy Pieces (3 min)": "orange",
    "Remove 12 Pieces with Noise (1 min)": "red",
    "Relaxation with Music (1 min)": "purple"
}

# Create the plot
plt.figure(figsize=(14, 6))
for stage, color in stage_colors.items():
    stage_df = df[df["Stage"] == stage]
    plt.plot(stage_df["Time"], stage_df["Resistance (Ohms)"], label=stage, color=color)

# Add labels and title
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Resistance (0-1)")
plt.title("GSR Resistance over Time by Stage")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()