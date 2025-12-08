# before you run this code from the termianl run this  source venv_biometrics/bin/activate from mackbook@Mackbook-MacBook-Air ~ % 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import glob
from pathlib import Path
import re 

# =========================================================
# 1. الثوابت والمتغيرات
# =========================================================
BASE_DIR = '.' # مجلد العمل الحالي
CLEANED_DIR = os.path.join(BASE_DIR, 'cleaned_data')

# تعريف المراحل الزمنية (بالثواني)
STAGES_DURATIONS = {
    "Calibration": (1, 19), 
    "Normal": (20, 259),    
    "Stress": (260, 439),   
    "Relaxation": (440, 499), 
}
TOTAL_SESSION_TIME_SEC = 499 

# =========================================================
# 2. الدوال المساعدة لمعالجة البيانات
# =========================================================

def get_stage(time_sec):
    """تحدد المرحلة بناءً على الوقت بالثانية (Stage)."""
    if pd.isna(time_sec):
        return None
    for stage, (start, end) in STAGES_DURATIONS.items():
        if start <= time_sec <= end:
            return stage
    return None

def find_cossinus_files(volunteer_dir):
    """تجد جميع ملفات Heart Rate داخل مجلد المتطوع."""
    # البحث المتكرر (recursive) عن الملفات التي تنتهي بـ heart_rate.csv
    hr_files = glob.glob(os.path.join(volunteer_dir, '**', '*heart_rate.csv'), recursive=True)
    return sorted(hr_files) 

def process_gsr(file_path):
    """ينظف ملف GSR ويوحد عمود الوقت والمرحلة."""
    print(f"  -> معالجة GSR: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    df.rename(columns={'Time (s)': 'Time_sec', 'Resistance (Ω)': 'Resistance_Ohm'}, inplace=True)
    
    # 1. إصلاح الوقت وتوحيده (التقريب لأقرب ثانية صحيحة)
    df['Time_sec'] = df['Time_sec'].round(0).astype(int)
    
    # 2. إزالة التكرارات الناتجة عن التقريب (الحفاظ على أول قيمة)
    df.drop_duplicates(subset=['Time_sec'], keep='first', inplace=True)
    
    # 3. قص البيانات حتى نهاية الجلسة
    df = df[df['Time_sec'] <= TOTAL_SESSION_TIME_SEC].copy()
    
    # 4. إصلاح عمود Stage (الاستبدال بناءً على جدولنا لضمان التوحيد)
    df['Stage'] = df['Time_sec'].apply(get_stage)
    
    return df[['Time_sec', 'Resistance_Ohm', 'Stage']]

def process_cossinus_hr(file_path, stage_df):
    """ينظف ملف Heart Rate ويوحده زمنياً مع بيانات Stage. (تم التعديل النهائي)"""
    print(f"  -> معالجة Cossinus HR: {os.path.basename(file_path)}")
    
    # 🛑 الحل النهائي: تخطي 10 صفوف من الميتاداتا لجميع ملفات Heart Rate.
    skip_rows = 10 

    try:
        # قراءة الملف مع تخطي الصفوف العشرة الأولى. 
        # السطر الحادي عشر (الإندكس 10 بعد التخطي) سيصبح رأس الأعمدة.
        df = pd.read_csv(file_path, skiprows=skip_rows, header=0)
        print(f"  -> ملاحظة: تم تخطي أول {skip_rows} صفوف من الميتاداتا في ملف HR.")
    except Exception as e:
        print(f"--- ❌ خطأ في قراءة ملف HR بعد التخطي: {e}")
        return None 

    # البحث عن أسماء الأعمدة الفعلية (Case-Insensitive)
    # بناءً على البيانات التي أرسلتها: time و heart_rate
    time_col_name = next((col for col in df.columns if col.lower() == 'time'), None)
    hr_col_name = next((col for col in df.columns if col.lower() == 'heart_rate'), None)
    
    # يجب أن يتم العثور عليهما الآن
    if not time_col_name or not hr_col_name:
        # إذا لم يتم العثور عليهما، فربما نحتاج لتخطي عدد آخر من الصفوف
        print(f"--- ❌ خطأ: لا يوجد عمود Time ('{time_col_name}') أو HeartRate ('{hr_col_name}') بعد التخطي (10).")
        return None

    # إعادة تسمية الأعمدة للتنسيق الموحد في الكود
    df.rename(columns={time_col_name: 'Time_sec_raw', hr_col_name: 'HeartRate_BPM'}, inplace=True)
    
    # 1. توحيد نقطة البداية (إزالة الإزاحة الزمنية لتبدأ من الصفر تقريباً)
    if df['Time_sec_raw'].min() > 0:
        time_offset = df['Time_sec_raw'].min()
        df['Time_sec_raw'] = df['Time_sec_raw'] - time_offset
    
    # 2. تقريب الوقت لأقرب ثانية لغرض الدمج
    df['Time_sec'] = df['Time_sec_raw'].round(0).astype(int)

    # 3. إزالة التكرارات (الحفاظ على أول قيمة في كل ثانية)
    df.drop_duplicates(subset=['Time_sec'], keep='first', inplace=True)
    
    # 4. قص البيانات وتوحيدها مع Stage
    df = df[df['Time_sec'] <= TOTAL_SESSION_TIME_SEC].copy()
    
    # دمج معلومات Stage النظيفة من GSR/STAGE_DF
    df_merged = pd.merge(df[['Time_sec', 'HeartRate_BPM']], 
                         stage_df[['Time_sec', 'Stage']], 
                         on='Time_sec', 
                         how='left')
                         
    return df_merged[['Time_sec', 'HeartRate_BPM', 'Stage']]


def fill_missing_values(df, column_name):
    """تملأ الفراغات (NaN) في عمود باستخدام interpolation خطي."""
    if column_name in df.columns and df[column_name].isna().any():
        # interpolation خطي لملء الفراغات
        df[column_name] = df[column_name].interpolate(method='linear', limit_direction='both')
    return df

def normalize_data(df, column_name):
    """تطبق التطبيع (Normalization) على عمود بين 0 و 1."""
    data_clean = df[[column_name]].dropna()
    
    # إذا كان العمود لا يحتوي على أي بيانات صالحة (NaN فقط)، نملؤه بـ NaN
    if data_clean.empty:
        df[f'{column_name}_Normalized'] = np.nan
        return df

    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(data_clean)
    
    # إعادة بناء السلسلة لضمان الإندكس الصحيح
    normalized_series = pd.Series(
        normalized_array.flatten(), 
        index=data_clean.index,
        name=f'{column_name}_Normalized'
    )
    
    df[f'{column_name}_Normalized'] = normalized_series
    return df

# =========================================================
# 3. الدالة الرئيسية لتنظيف ورسم متطوع واحد
# =========================================================

def process_and_plot_volunteer(volunteer_id):
    """تنفذ عمليات التنظيف والدمج والرسم لمتطوع واحد."""
    print(f"\n==============================================")
    print(f"بدء معالجة المتطوع رقم {volunteer_id}")
    print(f"==============================================")
    
    volunteer_dir = os.path.join(BASE_DIR, str(volunteer_id))
    
    # 1. البحث عن ملفات GSR
    gsr_files = glob.glob(os.path.join(volunteer_dir, 'GSR_*.csv'))
    
    if not gsr_files:
        print(f"--- ⚠️ تحذير: لم يتم العثور على ملفات GSR للمتطوع {volunteer_id}، سيتم تخطي المعالجة.")
        return
        
    gsr_data_list = [process_gsr(f) for f in gsr_files]
    df_gsr_raw = pd.concat(gsr_data_list, ignore_index=True)
    df_gsr_raw.sort_values(by='Time_sec', inplace=True)
    df_gsr_raw.drop_duplicates(subset=['Time_sec'], keep='first', inplace=True)

    # 2. تحديد جنس المتطوع (للرسم فقط)
    is_female = any('_f.csv' in os.path.basename(f).lower() for f in gsr_files)
    gender_label = 'Girl' if is_female else 'Boy'
    
    # 3. البحث ومعالجة ملفات Cossinus (Heart Rate)
    hr_files = find_cossinus_files(volunteer_dir)
    
    df_hr_raw = None
    if hr_files:
        hr_data_list = [process_cossinus_hr(f, df_gsr_raw[['Time_sec', 'Stage']]) for f in hr_files]
        # إزالة أي None ناتج عن فشل القراءة
        hr_data_list = [df for df in hr_data_list if df is not None]
        
        if hr_data_list:
            df_hr_raw = pd.concat(hr_data_list, ignore_index=True)
            df_hr_raw.sort_values(by='Time_sec', inplace=True)
            df_hr_raw.drop_duplicates(subset=['Time_sec'], keep='first', inplace=True)
            df_hr_raw['Stage'] = df_hr_raw['Time_sec'].apply(get_stage)
        else:
            print(f"--- ⚠️ تحذير: فشلت معالجة جميع ملفات HR للمتطوع {volunteer_id}.")
            
    # 4. الدمج النهائي
    if df_hr_raw is not None:
        # الدمج عند وجود بيانات HR
        merged_df = df_gsr_raw.merge(df_hr_raw[['Time_sec', 'HeartRate_BPM']], on='Time_sec', how='outer')
    else:
        # إذا لم يكن هناك بيانات HR، استخدم بيانات GSR فقط وأضف عمود HR فارغاً
        merged_df = df_gsr_raw.copy() 
        merged_df['HeartRate_BPM'] = np.nan 

    merged_df['Stage'] = merged_df['Time_sec'].apply(get_stage)
    merged_df.sort_values(by='Time_sec', inplace=True)
    
    # 5. ملء الفراغات في Heart Rate (interpolation خطي)
    merged_df = fill_missing_values(merged_df, 'HeartRate_BPM')
    
    # 6. التطبيع (Normalization)
    merged_df = normalize_data(merged_df, 'Resistance_Ohm')
    if 'HeartRate_BPM' in merged_df.columns:
        merged_df = normalize_data(merged_df, 'HeartRate_BPM')
        
    # 7. حفظ البيانات النظيفة
    cleaned_file_name = f'cleaned_data_{volunteer_id}.csv'
    cleaned_file_path = os.path.join(CLEANED_DIR, cleaned_file_name)
    merged_df.to_csv(cleaned_file_path, index=False)
    print(f"  -> تم حفظ البيانات النظيفة في: {cleaned_file_path}")
    
    # 8. الرسم البياني
    multi_file_check = len(gsr_files) > 1 or len(hr_files) > 1 
    multi_file_note = " (Data Merged from Multiple Files)" if multi_file_check else ""
    
    plot_combined_data(merged_df, volunteer_id, gender_label, multi_file_note)


def plot_combined_data(df, volunteer_id, gender_label, multi_file_note):
    """تنشئ الرسم البياني."""
    
    plt.figure(figsize=(15, 7))
    ax = plt.gca() 

    # GSR (أحمر)
    ax.plot(df['Time_sec'], df['Resistance_Ohm_Normalized'], 
            label='GSR Resistance (Normalized)', color='#ef4444', linewidth=1.5) 
    
    # Heart Rate (أزرق)
    if 'HeartRate_BPM_Normalized' in df.columns and df['HeartRate_BPM_Normalized'].notna().any():
        ax.plot(df['Time_sec'], df['HeartRate_BPM_Normalized'], 
                label='Heart Rate (Normalized)', color='#3b82f6', linewidth=1.5) 
    else:
         print(f"  -> ملاحظة: لم يتم رسم Heart Rate للمتطوع {volunteer_id} (بيانات مفقودة).")

    # إضافة خطوط عمودية وعناوين المراحل
    stage_colors = {
        "Calibration": '#6b7280', 
        "Normal": '#10b981',      
        "Stress": '#f97316',      
        "Relaxation": '#9333ea'   
    }

    text_y_position = 1.05
    
    for stage, (start_sec, end_sec) in STAGES_DURATIONS.items():
        if start_sec > 1:
            ax.axvline(x=start_sec, color=stage_colors[stage], linestyle='--', 
                       alpha=0.7, linewidth=1.5) 

        duration = end_sec - start_sec + 1
        mid_point = start_sec + duration / 2 - 1 
        
        plt.text(mid_point, text_y_position, 
                 stage, 
                 horizontalalignment='center', 
                 verticalalignment='center', 
                 fontsize=10, 
                 color=stage_colors[stage], 
                 fontweight='bold',
                 transform=ax.get_xaxis_transform())

    # إعداد القراف
    title = f'Normalized Biometric Data (GSR & HR) for V.{volunteer_id} ({gender_label}) during Doctor Game Task{multi_file_note}'
    plt.title(title, fontsize=16, pad=30)
    plt.xlabel('Session Time (Seconds)', fontsize=12)
    plt.ylabel('Normalized Value (0 to 1)', fontsize=12)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # حفظ الرسم
    plot_file_name = f'plot_{volunteer_id}.png'
    plot_file_path = os.path.join(CLEANED_DIR, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()
    print(f"  -> تم حفظ الرسم البياني في: {plot_file_path}")


# =========================================================
# 4. تنفيذ العملية على جميع المتطوعين
# =========================================================

def main():
    # إنشاء مجلد البيانات النظيفة إذا لم يكن موجوداً
    Path(CLEANED_DIR).mkdir(parents=True, exist_ok=True)
    print(f"تم إنشاء مجلد البيانات النظيفة في: {CLEANED_DIR}")

    # العثور على جميع مجلدات المتطوعين التي تحتوي على أرقام
    volunteer_folders = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and d.isdigit()]
    volunteer_ids = sorted([int(v) for v in volunteer_folders])
    
    for vid in volunteer_ids:
        try:
            process_and_plot_volunteer(vid)
        except Exception as e:
            print(f"--- ❌ خطأ فادح غير متوقع في معالجة المتطوع {vid}: {e}")
            print(f"تم تخطي المتطوع {vid}.")

if __name__ == '__main__':
    main()