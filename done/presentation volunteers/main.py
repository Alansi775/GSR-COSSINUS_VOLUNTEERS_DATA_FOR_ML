import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def process_volunteer_data(volunteer_id):
    """
    يقوم بمعالجة بيانات GSR ومعدل ضربات القلب لمتطوع معين.
    """
    print(f"Processing data for Volunteer {volunteer_id}...")

    # 1. تعريف المراحل ومددها بالثواني
    stages = {
        'Calibration (20s)': 20,
        'Normal - Watch Video (4 min)': 4 * 60,
        'Remove 5 Easy Pieces (3 min)': 3 * 60,
        'Remove 12 Pieces with Noise (1 min)': 1 * 60,
        'Relaxation with Music (1 min)': 1 * 60
    }

    # 2. إنشاء مجلد 'graphing' إذا لم يكن موجودًا
    if not os.path.exists('graphing'):
        os.makedirs('graphing')

    # 3. معالجة وحفظ بيانات GSR
    gsr_file = f'GSR-{volunteer_id}.csv'
    if not os.path.exists(gsr_file):
        print(f"GSR file not found for Volunteer {volunteer_id}. Skipping...")
        return

    gsr_df = pd.read_csv(gsr_file)
    gsr_df['Time'] = pd.to_timedelta(gsr_df['Time'], unit='s')
    gsr_df.set_index('Time', inplace=True)
    gsr_df = gsr_df[['Resistance (Ohms)']].resample('S').mean().interpolate(method='linear')
    gsr_df.reset_index(inplace=True)
    gsr_df['Time'] = gsr_df['Time'].dt.total_seconds().astype(int)

    min_val_gsr = gsr_df['Resistance (Ohms)'].min()
    max_val_gsr = gsr_df['Resistance (Ohms)'].max()
    gsr_df['Resistance (Ohms)'] = (gsr_df['Resistance (Ohms)'] - min_val_gsr) / (max_val_gsr - min_val_gsr)
    gsr_df['Time'] = gsr_df['Time'] - gsr_df['Time'].iloc[0]

    gsr_df['Stage'] = ''
    current_time = 0
    for stage, duration in stages.items():
        end_time = current_time + duration
        gsr_df.loc[(gsr_df['Time'] >= current_time) & (gsr_df['Time'] < end_time), 'Stage'] = stage
        current_time = end_time
    gsr_df.to_csv(f'graphing/GSR-{volunteer_id}_cleaned.csv', index=False)

    # 4. معالجة وحفظ بيانات معدل ضربات القلب
    hr_file_pattern = f'v-{volunteer_id}/*_heart_rate.csv'
    hr_files = glob.glob(hr_file_pattern)
    if not hr_files:
        print(f"Heart rate file not found for Volunteer {volunteer_id}. Skipping...")
        return
        
    hr_file = hr_files[0] # Get the first file that matches the pattern

    hr_df = pd.read_csv(hr_file, skiprows=10)
    hr_df.columns = ['time', 'heart_rate']
    
    hr_df['time'] = pd.to_timedelta(hr_df['time'], unit='s')
    hr_df.set_index('time', inplace=True)
    hr_df['heart_rate'] = pd.to_numeric(hr_df['heart_rate'], errors='coerce')
    hr_df = hr_df[['heart_rate']].resample('S').mean().interpolate(method='linear')
    hr_df.reset_index(inplace=True)
    hr_df['time'] = hr_df['time'].dt.total_seconds().astype(int)
    hr_df.dropna(inplace=True)
    
    min_val_hr = hr_df['heart_rate'].min()
    max_val_hr = hr_df['heart_rate'].max()
    hr_df['heart_rate'] = (hr_df['heart_rate'] - min_val_hr) / (max_val_hr - min_val_hr)
    hr_df['time'] = hr_df['time'] - hr_df['time'].iloc[0]

    hr_df['Stage'] = ''
    current_time = 0
    for stage, duration in stages.items():
        end_time = current_time + duration
        hr_df.loc[(hr_df['time'] >= current_time) & (hr_df['time'] < end_time), 'Stage'] = stage
        current_time = end_time
    hr_df.to_csv(f'graphing/V{volunteer_id}_heart_rate_cleaned.csv', index=False)

    # 5. إنشاء الرسم البياني
    cleaned_gsr_df = pd.read_csv(f'graphing/GSR-{volunteer_id}_cleaned.csv')
    cleaned_hr_df = pd.read_csv(f'graphing/V{volunteer_id}_heart_rate_cleaned.csv')

    plt.figure(figsize=(12, 8))
    plt.plot(cleaned_gsr_df['Time'], cleaned_gsr_df['Resistance (Ohms)'], label='GSR (Normalized)', color='blue')
    plt.plot(cleaned_hr_df['time'], cleaned_hr_df['heart_rate'], label='Heart Rate (Normalized)', color='red')

    plt.title(f'Normalized GSR and Heart Rate for Volunteer {volunteer_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Value (0-1)')
    plt.legend()
    plt.grid(True)

    stages_list = list(stages.keys())
    stage_times = [0] + list(np.cumsum(list(stages.values())))
    for i in range(len(stages_list)):
        plt.axvline(x=stage_times[i], color='gray', linestyle='--', linewidth=1)
        plt.text(stage_times[i] + 5, 1.05, stages_list[i], rotation=45, ha='left', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'graphing/volunteer_{volunteer_id}_gsr_hr_plot.png')
    plt.show()

# 🌟 6. Main execution loop for all volunteers 🌟
if __name__ == '__main__':
    volunteer_ids = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    for vid in volunteer_ids:
        process_volunteer_data(vid)