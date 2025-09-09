import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_gsr_data(volunteer_id):
    """
    يقوم بمعالجة بيانات GSR لمتطوع معين باستخدام المراحل المحددة.
    """
    print(f"Processing GSR Data for Volunteer {volunteer_id}...")

    # 1. تعريف المراحل ومددها بالثواني (المراحل الأصلية)
    stages = {
        'Calibration (20s)': 20,
        'Normal - Watch Video (4 min)': 4 * 60,
        'Remove 5 Easy Pieces (3 min)': 3 * 60,
        'Remove 12 Pieces with Noise (1 min)': 1 * 60,
        'Relaxation with Music (1 min)': 1 * 60
    }

    # 2. إنشاء مجلد 'gsr_data_graphs' إذا لم يكن موجودًا
    if not os.path.exists('gsr_data_graphs'):
        os.makedirs('gsr_data_graphs')

    # 3. معالجة وحفظ بيانات GSR
    gsr_file = f'GSR_Data-{volunteer_id}.csv'
    if not os.path.exists(gsr_file):
        print(f"File not found: {gsr_file}. Skipping...")
        return

    gsr_df = pd.read_csv(gsr_file)
    
    # 🌟 استخدام اسم العمود الصحيح 'Resistance (Ohms)' 🌟
    gsr_data_column = 'Resistance (Ohms)'

    # التأكد من وجود عمود 'Time'
    if 'Time' not in gsr_df.columns:
        print(f"Error: 'Time' column not found in {gsr_file}. Skipping.")
        return
    
    # تحويل عمود 'Time' إلى نوع زمني
    gsr_df['Time'] = pd.to_timedelta(gsr_df['Time'], unit='s')
    gsr_df.set_index('Time', inplace=True)
    
    # إعادة أخذ العينات والاستقراء فقط على عمود البيانات
    gsr_df = gsr_df[[gsr_data_column]].resample('S').mean().interpolate(method='linear')
    gsr_df.reset_index(inplace=True)
    gsr_df['Time'] = gsr_df['Time'].dt.total_seconds().astype(int)

    # التطبيع
    min_val_gsr = gsr_df[gsr_data_column].min()
    max_val_gsr = gsr_df[gsr_data_column].max()
    gsr_df[gsr_data_column] = (gsr_df[gsr_data_column] - min_val_gsr) / (max_val_gsr - min_val_gsr)
    gsr_df['Time'] = gsr_df['Time'] - gsr_df['Time'].iloc[0]

    # تعيين المراحل
    gsr_df['Stage'] = ''
    current_time = 0
    for stage, duration in stages.items():
        end_time = current_time + duration
        gsr_df.loc[(gsr_df['Time'] >= current_time) & (gsr_df['Time'] < end_time), 'Stage'] = stage
        current_time = end_time
    gsr_df.to_csv(f'gsr_data_graphs/GSR_Data-{volunteer_id}_cleaned.csv', index=False)

    # 4. إنشاء الرسم البياني
    cleaned_gsr_df = pd.read_csv(f'gsr_data_graphs/GSR_Data-{volunteer_id}_cleaned.csv')

    plt.figure(figsize=(12, 8))
    plt.plot(cleaned_gsr_df['Time'], cleaned_gsr_df[gsr_data_column], label='GSR (Normalized)', color='blue')

    plt.title(f'Normalized GSR Data for Volunteer {volunteer_id}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Value (0-1)')
    plt.legend()
    plt.grid(True)

    # إضافة خطوط المراحل
    stages_list = list(stages.keys())
    stage_times = [0] + list(np.cumsum(list(stages.values())))
    for i in range(len(stages_list)):
        plt.axvline(x=stage_times[i], color='gray', linestyle='--', linewidth=1)
        plt.text(stage_times[i] + 5, 1.05, stages_list[i], rotation=45, ha='left', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'gsr_data_graphs/GSR_Data-{volunteer_id}_plot.png')
    plt.show()

# 5. تشغيل الكود لكل المتطوعين من 1 إلى 9
if __name__ == '__main__':
    volunteer_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for vid in volunteer_ids:
        process_gsr_data(vid)