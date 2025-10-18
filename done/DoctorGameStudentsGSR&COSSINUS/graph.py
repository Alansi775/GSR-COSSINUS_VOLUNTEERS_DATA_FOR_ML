import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# 1. تعريف مسارات الملفات والبيانات الثابتة
GSR_FILE = 'GSR_34.csv' 
HR_FILE = 'Cossinus_34/29R7.1ELE_2025-10-14_16-34-28_heart_rate.csv' 
SUBJECT_ID = 34
ACTIVITY = "Doctor Game"

# تعريف المراحل الزمنية بالثواني
STAGES_DURATIONS = {
    "Calibration": 20,
    "Normal": 240,  # 4 minutes
    "Stress": 180,  # 3 minutes
    "Relaxation": 60,   # 1 minute
}

# حساب نقاط بداية المراحل
stage_start_points = {}
cumulative_time = 0
for stage, duration in STAGES_DURATIONS.items():
    stage_start_points[stage] = cumulative_time
    cumulative_time += duration
TOTAL_TIME = cumulative_time # 500 ثانية (8 دقائق و 20 ثانية)

# 2. دالة تنظيف وتجهيز ملف GSR
def process_gsr(file_path):
    # GSR مقاس كل ثانية، لذا نحافظ على الوقت مُقرباً
    df_gsr = pd.read_csv(file_path)
    df_gsr.rename(columns={'Time (s)': 'Time_sec', 'Resistance (Ω)': 'Resistance_Ohm'}, inplace=True)
    return df_gsr

# 3. دالة تنظيف وتجهيز ملف Heart Rate (تم التعديل: إزالة تقريب الوقت)
def process_hr(file_path):
    df_hr = pd.read_csv(file_path)
    df_hr.rename(columns={'time': 'Time_sec_HR', 'heart_rate': 'HeartRate_BPM'}, inplace=True)
    
    # توحيد نقطة البداية (إزالة الـ Offset لكن لا نقرب الوقت)
    if df_hr['Time_sec_HR'].min() > 0:
        time_offset = df_hr['Time_sec_HR'].min()
        df_hr['Time_sec_HR'] = df_hr['Time_sec_HR'] - time_offset
    
    # === حساب NN Interval (ms) ===
    df_hr['NN_Interval_ms'] = 60000 / df_hr['HeartRate_BPM']
    
    # لإزالة تكرار النقاط في نفس الثانية (للتخفيف فقط، لكن نحافظ على دقة الوقت)
    # merged_hr = df_hr.drop_duplicates(subset=['Time_sec_HR'], keep='first').copy()
    
    return df_hr


# 4. دالة إضافة عمود المرحلة
def add_stage_column(df, time_column='Time_sec'):
    def get_stage(time_sec):
        current_time = 0
        for stage, duration in STAGES_DURATIONS.items():
            current_time += duration
            if time_sec < current_time:
                return stage
        return "Finished"
        
    df['Stage'] = df[time_column].apply(get_stage)
    return df

# 5. تنفيذ العمليات على البيانات
df_gsr = process_gsr(GSR_FILE)
df_hr = process_hr(HR_FILE)

# دمج البيانات: لن ندمج كل شيء في جدول واحد الآن، بل سنرسم كل مقياس على حدة
# سنبقي جدول GSR للـ Normalization/Smoothing
merged_df_gsr = add_stage_column(df_gsr)
merged_df_hr = add_stage_column(df_hr, time_column='Time_sec_HR')


# 6. تطبيق التطبيع (Normalization) والتنعيم (Smoothing)

scaler = MinMaxScaler()

# 1. GSR: التطبيع والتنعيم
merged_df_gsr['GSR_Normalized'] = scaler.fit_transform(merged_df_gsr[['Resistance_Ohm']])

# نافذة المتوسط المتحرك (لـ GSR فقط)
SMOOTHING_WINDOW = 15 
merged_df_gsr['GSR_Smoothed'] = merged_df_gsr['GSR_Normalized'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()


# 7. الرسم البياني الموحد (استخدام محاور Y مزدوجة ورسم NN Interval كنقاط)
plt.figure(figsize=(15, 7))

# المحور الأول (اليسار) لـ GSR
ax1 = plt.gca() 
ax1.set_ylabel('GSR Resistance (Normalized 0 to 1)', color='#ef4444', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#ef4444')

# رسم بيانات GSR المنعمة (خط ناعم)
ax1.plot(merged_df_gsr['Time_sec'], merged_df_gsr['GSR_Smoothed'], 
        label='GSR Resistance (Normalized & Smoothed)', color='#ef4444', linewidth=2.5) 
ax1.grid(True, linestyle=':', alpha=0.6)

# المحور الثاني (اليمين) لـ NN Interval
ax2 = ax1.twinx()  # إنشاء محور Y ثانٍ يشارك المحور X مع المحور الأول
ax2.set_ylabel('NN Interval (ms)', color='#3b82f6', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#3b82f6')

# رسم بيانات NN Interval (خط حاد - بدون تنعيم)
# ملاحظة: هذا سيحاكي شكل NN Interval series، حيث كل نقطة هي قياس
ax2.plot(merged_df_hr['Time_sec_HR'], merged_df_hr['NN_Interval_ms'], 
        label='NN Interval (ms)', color='#3b82f6', linewidth=1, linestyle='-', alpha=0.9)
# يمكن إضافة نقاط دائرية صغيرة لجعلها تبدو كأنها نقاط فردية (optional)
# ax2.scatter(merged_df_hr['Time_sec_HR'], merged_df_hr['NN_Interval_ms'], s=5, color='#3b82f6')


# إضافة خطوط عمودية (Vertical Lines) لفصل المراحل
stage_colors = {
    "Calibration": '#6b7280', 
    "Normal": '#10b981',      
    "Stress": '#f97316',      
    "Relaxation": '#9333ea'   
}

text_y_position = 1.05 

for i, (stage, duration) in enumerate(STAGES_DURATIONS.items()):
    start_point = stage_start_points[stage]
    
    if i > 0:
        ax1.axvline(x=start_point, color=stage_colors[stage], linestyle='--', 
                   alpha=0.7, linewidth=1.5) 

    mid_point = start_point + duration / 2
    if duration > 0:
        plt.text(mid_point, text_y_position, 
                 stage, 
                 horizontalalignment='center', 
                 verticalalignment='center', 
                 fontsize=10, 
                 color=stage_colors[stage], 
                 fontweight='bold',
                 transform=ax1.get_xaxis_transform())


# إعداد القراف (العناوين والمحاور)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left')

plt.title(f'Biometric Response (GSR & NN Interval Series) for Subject {SUBJECT_ID} during {ACTIVITY} Task', 
          fontsize=16, pad=30)
ax1.set_xlabel('Session Time (Seconds)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# عرض الرسم البياني
plt.show()

# 8. عرض البيانات الموحدة والمعالجة (اختياري)
print("\n--- GSR Data Head ---")
print(merged_df_gsr[['Time_sec', 'Stage', 'GSR_Smoothed']].head())
print("\n--- HR/NN Data Head ---")
print(merged_df_hr[['Time_sec_HR', 'Stage', 'NN_Interval_ms']].head())