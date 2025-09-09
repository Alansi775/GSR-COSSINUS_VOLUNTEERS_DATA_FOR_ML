import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def find_header_row(file_path):
    """Finds the row number of the header 'time,heart_rate'."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if 'time,heart_rate' in line:
                return i
    return None

def analyze_hrv(volunteer_id, user_note, base_dir="."):
    """
    Performs HRV analysis and generates a clear plot based on manual notes.
    """
    print(f"\n--- Processing Heart Rate Data for Volunteer {volunteer_id} ---")

    volunteer_folder = os.path.join(base_dir, f'v{volunteer_id}- p')
    file_pattern = os.path.join(volunteer_folder, '*_heart_rate.csv')
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"Error: Heart rate file not found for Volunteer {volunteer_id} at '{file_pattern}'. Skipping.")
        return
    
    file_path = file_list[0]
    
    header_row = find_header_row(file_path)
    if header_row is None:
        print(f"Error: Could not find header 'time,heart_rate' in {file_path}. Skipping.")
        return

    df = pd.read_csv(file_path, skiprows=header_row, header=0)
    
    if 'heart_rate' not in df.columns:
        print(f"Error: 'heart_rate' column not found in the file. Skipping.")
        return

    df['rr_intervals_ms'] = (60 / df['heart_rate']) * 1000
    df['time_index'] = np.arange(len(df))

    # Normalize RR intervals to a 0-1 scale
    min_rr = df['rr_intervals_ms'].min()
    max_rr = df['rr_intervals_ms'].max()
    df['normalized_rr_intervals'] = (df['rr_intervals_ms'] - min_rr) / (max_rr - min_rr)

    # --- Manual Classification based on your notes ---
    if volunteer_id == 23:
        conclusion = "Relaxed / Low Stress"
        annotation_text = "High HRV: The changing pattern indicates a relaxed state."
        annotation_color = 'green'
        plot_fill_color = 'green'
    elif volunteer_id == 24:
        conclusion = "High Stress"
        annotation_text = "Low HRV: The flat pattern indicates high stress."
        annotation_color = 'red'
        plot_fill_color = 'red'
    else:
        conclusion = "Undefined"
        annotation_text = "No specific notes available."
        annotation_color = 'gray'
        plot_fill_color = 'gray'

    # --- Generate the Plot ---
    plt.figure(figsize=(15, 7))
    plt.plot(df['time_index'], df['normalized_rr_intervals'], label='Normalized RR Intervals', color='darkgreen')
    
    plt.title(f'HRV Plot for Volunteer {volunteer_id} - Condition: {conclusion}')
    plt.xlabel('Time (data points)')
    plt.ylabel('RR Interval (Normalized 0-1)')
    
    # Position the annotation text in the top-right corner
    plt.text(0.95, 0.9, annotation_text, fontsize=15, color=annotation_color, ha='right',
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    plt.fill_between(df['time_index'], 0, df['normalized_rr_intervals'], color=plot_fill_color, alpha=0.1)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_folder = 'hrv_plots'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(os.path.join(plot_folder, f'v{volunteer_id}_hrv_analysis.png'))
    plt.show()
    print("-" * 50)

if __name__ == '__main__':
    analyze_hrv(23, "The volunteer didn't feel stressed much", base_dir=".")
    analyze_hrv(24, "The volunteer said she had a high heart rate for some reason", base_dir=".")