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

def normalize_stage_name(stage_str):
    """تطبيع أسماء المراحل من أشكال مختلفة إلى الشكل الموحد."""
    if pd.isna(stage_str):
        return None
    stage_lower = str(stage_str).lower().strip()
    if 'calib' in stage_lower:
        return 'Calibration'
    elif 'normal' in stage_lower:
        return 'Normal'
    elif 'stress' in stage_lower:
        return 'Stress'
    elif 'relax' in stage_lower:
        return 'Relaxation'
    return stage_str

def has_video(volunteer_dir):
    """تتحقق من وجود فيديو للمتطوع (webm أو mp4)."""
    video_extensions = ['*.webm', '*.mp4', '*.avi', '*.mov']
    for ext in video_extensions:
        videos = glob.glob(os.path.join(volunteer_dir, ext))
        if videos:
            return True
    return False

def find_cossinus_files(volunteer_dir):
    """تجد جميع ملفات Heart Rate داخل مجلد المتطوع."""
    # البحث المتكرر (recursive) عن الملفات التي تنتهي بـ heart_rate.csv
    hr_files = glob.glob(os.path.join(volunteer_dir, '**', '*heart_rate.csv'), recursive=True)
    return sorted(hr_files)

def find_rr_interval_files(volunteer_dir):
    """تجد جميع ملفات RR Intervals المباشرة من الجهاز داخل مجلد المتطوع."""
    # البحث المتكرر (recursive) عن الملفات التي تنتهي بـ rr_int.csv
    rr_files = glob.glob(os.path.join(volunteer_dir, '**', '*rr_int.csv'), recursive=True)
    return sorted(rr_files) 

def process_gsr(file_path):
    """ينظف ملف GSR ويوحد عمود الوقت والمرحلة."""
    print(f"  -> معالجة GSR: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    # التعامل مع أسماء الأعمدة المختلفة (قديم وجديد)
    # للملفات القديمة: Time (s), Resistance (Ω), Stage
    # للملفات الجديدة: Time (s), Resistance (Ω), Conductance (µS), Stage
    
    # إذا كان الملف الجديد يحتوي على Conductance، استخدمها مباشرة
    if 'Conductance (µS)' in df.columns:
        df.rename(columns={'Time (s)': 'Time_sec', 'Conductance (µS)': 'Conductance_microS'}, inplace=True)
    else:
        # وإلا تحويل من Resistance إلى Conductance
        df.rename(columns={'Time (s)': 'Time_sec', 'Resistance (Ω)': 'Conductance_microS'}, inplace=True)
    
    # 1. إصلاح الوقت وتوحيده (التقريب لأقرب ثانية صحيحة)
    df['Time_sec'] = df['Time_sec'].round(0).astype(int)
    
    # 2. إزالة التكرارات الناتجة عن التقريب (الحفاظ على أول قيمة)
    df.drop_duplicates(subset=['Time_sec'], keep='first', inplace=True)
    
    # 3. قص البيانات حتى نهاية الجلسة
    df = df[df['Time_sec'] <= TOTAL_SESSION_TIME_SEC].copy()
    
    # 4. إصلاح عمود Stage (الاستبدال بناءً على جدولنا لضمان التوحيد)
    df['Stage'] = df['Time_sec'].apply(get_stage)
    
    return df[['Time_sec', 'Conductance_microS', 'Stage']]

def process_cossinus_hr(file_path, stage_df):
    """ينظف ملف Heart Rate ويوحده زمنياً مع بيانات Stage. (تم التعديل النهائي)"""
    print(f"  -> معالجة Cossinus HR: {os.path.basename(file_path)}")
    
    # 🛑 الحل النهائي: تخطي 11 سطر من الميتاداتا (الـ header + 10 metadata)
    skip_rows = 11 

    try:
        # قراءة الملف مع تخطي أول 11 سطر (10 metadata + 1 header)
        # السطر 12 (الإندكس 11 بعد التخطي) سيصبح رأس الأعمدة الفعلي
        df = pd.read_csv(file_path, skiprows=skip_rows, header=0)
        print(f"  -> ملاحظة: تم تخطي أول {skip_rows} سطر من الميتاداتا في ملف HR.")
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
    
    # 5. حساب RR Intervals من BPM (Ground Truth)
    df = calculate_rr_intervals(df)
    
    # دمج معلومات Stage النظيفة من GSR/STAGE_DF
    df_merged = pd.merge(df[['Time_sec', 'HeartRate_BPM', 'RR_Interval_ms']], 
                         stage_df[['Time_sec', 'Stage']], 
                         on='Time_sec', 
                         how='left')
                         
    return df_merged[['Time_sec', 'HeartRate_BPM', 'RR_Interval_ms', 'Stage']]


def smooth_heart_rate(df, column_name='HeartRate_BPM', window_size=3):
    """تطبق smoothing على Heart Rate باستخدام Savitzky-Golay filter للحفاظ على التفاصيل المهمة."""
    if column_name in df.columns and df[column_name].notna().any():
        try:
            from scipy.signal import savgol_filter
            # استخدام Savitzky-Golay filter - أفضل من moving average لأنه يحافظ على القمم
            valid_data = df[column_name].dropna()
            if len(valid_data) >= window_size:
                df.loc[valid_data.index, column_name] = savgol_filter(
                    valid_data.values, 
                    window_length=window_size, 
                    polyorder=2
                )
        except ImportError:
            # إذا لم تكن scipy موجودة، استخدم moving average خفيف
            df[column_name] = df[column_name].rolling(
                window=3, 
                center=True, 
                min_periods=1
            ).mean()
    return df


def fill_missing_values(df, column_name):
    """تملأ الفراغات (NaN) في عمود باستخدام interpolation خطي فقط بين النقاط الموجودة."""
    if column_name in df.columns and df[column_name].isna().any():
        # forward fill ثم backward fill للفراغات بين البيانات الموجودة فقط
        # لا نملأ الفراغات في البداية أو النهاية
        df[column_name] = df[column_name].interpolate(method='linear', limit_area='inside')
    return df

def calculate_rr_intervals(df):
    """حساب RR Intervals من Heart Rate (Ground Truth - peek to peek في الملليثانية)."""
    if 'HeartRate_BPM' not in df.columns:
        return df
    
    # الصيغة: RR_ms = 60000 / BPM
    df['RR_Interval_ms'] = df['HeartRate_BPM'].apply(
        lambda x: 60000 / x if x > 0 else np.nan
    )
    
    # تطبيق حدود بيولوجية (300-2000 ملليثانية = 30-200 نبضة في الدقيقة)
    df.loc[df['RR_Interval_ms'] < 300, 'RR_Interval_ms'] = np.nan
    df.loc[df['RR_Interval_ms'] > 2000, 'RR_Interval_ms'] = np.nan
    
    # طباعة النطاق للتصحيح
    valid_rr = df['RR_Interval_ms'].dropna()
    if len(valid_rr) > 0:
        print(f"  -> نطاق RR Intervals: {valid_rr.min():.1f} - {valid_rr.max():.1f} ملليثانية")
    
    return df


def process_rr_intervals_file(file_path, stage_df):
    """تقرأ وتنظف ملف RR Intervals المباشر من الجهاز."""
    print(f"  -> معالجة RR Intervals: {os.path.basename(file_path)}")
    
    try:
        # قراءة الملف مع تخطي أول 11 سطر (10 metadata + 1 header)
        df = pd.read_csv(file_path, skiprows=11, header=0)
    except Exception as e:
        print(f"--- ❌ خطأ في قراءة ملف RR Intervals: {e}")
        return None
    
    # البحث عن أسماء الأعمدة
    time_col_name = next((col for col in df.columns if col.lower() == 'time'), None)
    rr_col_name = next((col for col in df.columns if col.lower() == 'rr_int'), None)
    
    if not time_col_name or not rr_col_name:
        print(f"--- ❌ خطأ: لا يوجد عمود Time أو rr_int")
        return None
    
    # إعادة تسمية الأعمدة
    df.rename(columns={time_col_name: 'Time_sec_raw', rr_col_name: 'RR_Interval_ms'}, inplace=True)
    
    # 1. توحيد نقطة البداية
    if df['Time_sec_raw'].min() > 0:
        time_offset = df['Time_sec_raw'].min()
        df['Time_sec_raw'] = df['Time_sec_raw'] - time_offset
    
    # 2. تقريب الوقت لأقرب ثانية
    df['Time_sec'] = df['Time_sec_raw'].round(0).astype(int)
    
    # 3. تطبيق حدود بيولوجية صارمة (300-2000 ms = 30-200 bpm)
    # القيم الأقل من 300 أو أكثر من 2000 تكون artifacts
    df.loc[df['RR_Interval_ms'] < 300, 'RR_Interval_ms'] = np.nan
    df.loc[df['RR_Interval_ms'] > 2000, 'RR_Interval_ms'] = np.nan
    
    # 4. إزالة التكرارات (الحفاظ على أول قيمة صحيحة في كل ثانية)
    df = df[df['RR_Interval_ms'].notna()].copy()
    df.drop_duplicates(subset=['Time_sec'], keep='first', inplace=True)
    
    # 5. قص البيانات
    df = df[df['Time_sec'] <= TOTAL_SESSION_TIME_SEC].copy()
    
    # 6. دمج معلومات Stage
    df_merged = pd.merge(df[['Time_sec', 'RR_Interval_ms']], 
                         stage_df[['Time_sec', 'Stage']], 
                         on='Time_sec', 
                         how='left')
    
    # طباعة النطاق
    valid_rr = df_merged['RR_Interval_ms'].dropna()
    if len(valid_rr) > 0:
        print(f"  -> نطاق RR Intervals (بعد التنظيف): {valid_rr.min():.1f} - {valid_rr.max():.1f} ملليثانية ({len(valid_rr)} قيمة صحيحة)")
    
    return df_merged[['Time_sec', 'RR_Interval_ms', 'Stage']]

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
    
    # 1. البحث عن ملفات GSR (دعم أسماء مختلفة: GSR_*.csv و V*.csv)
    gsr_files = glob.glob(os.path.join(volunteer_dir, 'GSR_*.csv'))
    if not gsr_files:
        # البحث عن الملفات الجديدة بنمط V*.csv
        gsr_files = glob.glob(os.path.join(volunteer_dir, f'{volunteer_id}.csv'))
    if not gsr_files:
        gsr_files = glob.glob(os.path.join(volunteer_dir, 'V*.csv'))
    
    if not gsr_files:
        print(f"--- ⚠️ تحذير: لم يتم العثور على ملفات GSR للمتطوع {volunteer_id}، سيتم تخطي المعالجة.")
        return
        
    gsr_data_list = [process_gsr(f) for f in gsr_files]
    df_gsr_raw = pd.concat(gsr_data_list, ignore_index=True)
    df_gsr_raw.sort_values(by='Time_sec', inplace=True)
    df_gsr_raw.drop_duplicates(subset=['Time_sec'], keep='first', inplace=True)

    # 2. تحديد جنس المتطوع (للرسم فقط)
    is_female = any('_f' in str(volunteer_id).lower() for _ in [1]) or any('_f.csv' in os.path.basename(f).lower() for f in gsr_files)
    gender_label = 'Girl' if is_female else 'Boy'
    
    # 2.5. كشف الفيديو
    video_available = has_video(volunteer_dir)
    video_note = ' [Video Available]' if video_available else ''
    
    # 3. البحث ومعالجة ملفات RR Intervals (الأولوية للملفات المباشرة من الجهاز)
    rr_files = find_rr_interval_files(volunteer_dir)
    
    df_rr_raw = None
    if rr_files:
        print(f"  -> وجدت {len(rr_files)} ملف(ات) RR Intervals")
        rr_data_list = [process_rr_intervals_file(f, df_gsr_raw[['Time_sec', 'Stage']]) for f in rr_files]
        # إزالة أي None ناتج عن فشل القراءة
        rr_data_list = [df for df in rr_data_list if df is not None]
        
        if rr_data_list:
            df_rr_raw = pd.concat(rr_data_list, ignore_index=True)
            df_rr_raw.sort_values(by='Time_sec', inplace=True)
            df_rr_raw.drop_duplicates(subset=['Time_sec'], keep='first', inplace=True)
        else:
            print(f"--- ⚠️ تحذير: فشلت معالجة ملفات RR Intervals للمتطوع {volunteer_id}.")
    else:
        # البحث عن ملفات Heart Rate كبديل (إذا لم توجد ملفات RR)
        hr_files = find_cossinus_files(volunteer_dir)
        
        if hr_files:
            print(f"  -> وجدت {len(hr_files)} ملف(ات) Heart Rate بدلاً من RR Intervals")
            hr_data_list = [process_cossinus_hr(f, df_gsr_raw[['Time_sec', 'Stage']]) for f in hr_files]
            hr_data_list = [df for df in hr_data_list if df is not None]
            
            if hr_data_list:
                df_rr_raw = pd.concat(hr_data_list, ignore_index=True)
                df_rr_raw = df_rr_raw[['Time_sec', 'RR_Interval_ms', 'Stage']].copy()
                df_rr_raw.sort_values(by='Time_sec', inplace=True)
                df_rr_raw.drop_duplicates(subset=['Time_sec'], keep='first', inplace=True)
            else:
                print(f"--- ⚠️ تحذير: فشلت معالجة جميع ملفات HR للمتطوع {volunteer_id}.")
            
    # 4. الدمج النهائي
    if df_rr_raw is not None:
        # الدمج عند وجود بيانات RR
        merged_df = df_gsr_raw.merge(df_rr_raw[['Time_sec', 'RR_Interval_ms']], on='Time_sec', how='outer')
    else:
        # إذا لم يكن هناك بيانات RR، استخدم بيانات GSR فقط وأضف عمود RR فارغ
        merged_df = df_gsr_raw.copy() 
        merged_df['RR_Interval_ms'] = np.nan
        # إضافة عمود BPM أيضاً (حتى لو كان فارغاً)
        merged_df['HeartRate_BPM'] = np.nan

    merged_df['Stage'] = merged_df['Time_sec'].apply(get_stage)
    merged_df.sort_values(by='Time_sec', inplace=True)
    
    # 5. تحسين Heart Rate (smoothing لتقليل الـ noise) - فقط إذا كان موجوداً
    if 'HeartRate_BPM' in merged_df.columns:
        merged_df = smooth_heart_rate(merged_df, 'HeartRate_BPM', window_size=5)
    
    # 6. ملء الفراغات في RR Intervals (interpolation خطي)
    merged_df = fill_missing_values(merged_df, 'RR_Interval_ms')
    
    # 7. التطبيع (Normalization)
    merged_df = normalize_data(merged_df, 'Conductance_microS')
    if 'RR_Interval_ms' in merged_df.columns:
        merged_df = normalize_data(merged_df, 'RR_Interval_ms')
        
    # 8. حفظ البيانات النظيفة
    cleaned_file_name = f'cleaned_data_{volunteer_id}.csv'
    cleaned_file_path = os.path.join(CLEANED_DIR, cleaned_file_name)
    
    # تحديد الأعمدة للحفظ
    cols_to_save = ['Time_sec', 'Conductance_microS', 'Stage', 'RR_Interval_ms']
    if 'Conductance_microS_Normalized' in merged_df.columns:
        cols_to_save.append('Conductance_microS_Normalized')
    if 'RR_Interval_ms_Normalized' in merged_df.columns:
        cols_to_save.append('RR_Interval_ms_Normalized')
    if 'HeartRate_BPM' in merged_df.columns:
        cols_to_save.insert(3, 'HeartRate_BPM')
    
    merged_df[cols_to_save].to_csv(cleaned_file_path, index=False)
    print(f"  -> تم حفظ البيانات النظيفة في: {cleaned_file_path}")
    
    # 9. التحقق من اكتمالية البيانات قبل الرسم
    min_gsr_data = 100  # حد أدنى من نقاط GSR
    min_rr_data = 50    # حد أدنى من نقاط RR
    
    gsr_count = merged_df['Conductance_microS'].notna().sum()
    rr_count = merged_df['RR_Interval_ms'].notna().sum()
    
    if gsr_count < min_gsr_data:
        print(f"--- ⚠️ تحذير: بيانات GSR ناقصة جداً ({gsr_count} نقطة فقط)، سيتم تخطي الرسم لتجنب القرافات المشوهة.")
        return
    
    if rr_count < min_rr_data:
        print(f"--- ⚠️ تحذير: بيانات RR ناقصة ({rr_count} نقطة فقط)، سيتم رسم GSR فقط.")
    
    # 10. الرسم البياني
    multi_file_check = len(gsr_files) > 1 or len(rr_files) > 1 
    multi_file_note = " (Data Merged from Multiple Files)" if multi_file_check else ""
    
    plot_combined_data(merged_df, volunteer_id, gender_label, multi_file_note, video_note)
    
    # 11. رسم منفصل للـ Heart Rate (NN Intervals) إذا كانت البيانات كافية
    if rr_count >= min_rr_data:
        plot_heart_rate_nn_interval(merged_df, volunteer_id, gender_label, multi_file_note, video_note)
    else:
        print(f"  -> لم يتم رسم NN Intervals لأن بيانات RR ناقصة.")


def plot_combined_data(df, volunteer_id, gender_label, multi_file_note, video_note=''):
    """تنشئ الرسم البياني."""
    
    # Debug: اطبع الأعمدة الموجودة
    print(f"\n  -> DEBUG: الأعمدة الموجودة في الـ DataFrame:")
    print(f"  -> {list(df.columns)}")
    print(f"  -> RR_Interval_ms values: {df['RR_Interval_ms'].notna().sum()} non-null من {len(df)}")
    if 'RR_Interval_ms_Normalized' in df.columns:
        print(f"  -> RR_Interval_ms_Normalized values: {df['RR_Interval_ms_Normalized'].notna().sum()} non-null")
    
    plt.figure(figsize=(15, 7))
    ax = plt.gca() 

    # GSR (أحمر)
    ax.plot(df['Time_sec'], df['Conductance_microS_Normalized'], 
            label='GSR Conductance (Normalized)', color='#ef4444', linewidth=1.5) 
    
    # Heart Rate (أزرق) - RR Intervals (Ground Truth - peek to peek)
    if 'RR_Interval_ms_Normalized' in df.columns and df['RR_Interval_ms_Normalized'].notna().any():
        # رسم الخط الأزرق - RR Intervals
        ax.plot(df['Time_sec'], df['RR_Interval_ms_Normalized'], 
                label='Heart Rate (RR Intervals - ms)', color='#3b82f6', linewidth=2, alpha=0.8)
        
        # إضافة تظليل خفيف تحت الخط
        ax.fill_between(df['Time_sec'], df['RR_Interval_ms_Normalized'], 
                         alpha=0.15, color='#3b82f6')
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
    title = f'Normalized Biometric Data (GSR Conductance & Heart Rate) for V.{volunteer_id} ({gender_label}) during Doctor Game Task{multi_file_note}{video_note}'
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


def plot_heart_rate_nn_interval(df, volunteer_id, gender_label, multi_file_note, video_note=''):
    """تنشئ رسم بياني منفصل للـ Heart Rate RR Intervals بالدقائق (مثل الورقة العلمية)."""
    
    # نستخدم الثواني (للمواءمة مع الرسم الأول)
    df_plot = df.copy()
    # تصفية البيانات (إزالة القيم الفارغة)
    df_plot = df_plot[df_plot['RR_Interval_ms'].notna()].copy()
    
    if len(df_plot) == 0:
        print(f"  -> ملاحظة: لا توجد بيانات RR Interval صالحة للمتطوع {volunteer_id}، سيتم تخطي رسم الـ Heart Rate.")
        return
    
    plt.figure(figsize=(15, 7))
    ax = plt.gca()
    
    # رسم RR Intervals بخط أزرق سميك
    ax.plot(df_plot['Time_sec'], df_plot['RR_Interval_ms'], 
            label='NN Interval (RR)', color='#3b82f6', linewidth=2.5, marker='o', markersize=3, alpha=0.8)
    
    # إضافة تظليل خفيف تحت الخط
    ax.fill_between(df_plot['Time_sec'], df_plot['RR_Interval_ms'], 
                     alpha=0.15, color='#3b82f6')
    
    # تحويل نطاق المراحل من الثواني لدقائق وإضافة خطوط عمودية
    stage_colors = {
        "Calibration": '#6b7280', 
        "Normal": '#10b981',      
        "Stress": '#f97316',      
        "Relaxation": '#9333ea'   
    }
    
    text_y_position = df_plot['RR_Interval_ms'].max() * 1.05
    
    for stage, (start_sec, end_sec) in STAGES_DURATIONS.items():
        # نستخدم القيم بالثواني مباشرة للتوافق مع المحور
        if start_sec > 1:
            ax.axvline(x=start_sec, color=stage_colors[stage], linestyle='--', 
                       alpha=0.7, linewidth=2)

        # إضافة اسم المرحلة عند منتصف المقطع بالثواني
        mid_point = (start_sec + end_sec) / 2
        plt.text(mid_point, text_y_position, 
                 stage, 
                 horizontalalignment='center', 
                 verticalalignment='bottom', 
                 fontsize=11, 
                 color=stage_colors[stage], 
                 fontweight='bold')
    
    # إعداد القراف
    title = f'Heart Rate Variability (NN Intervals) for V.{volunteer_id} ({gender_label}) during Doctor Game Task{multi_file_note}{video_note}'
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Session Time (Seconds)', fontsize=12)
    plt.ylabel('NN Interval (Milliseconds)', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # حفظ الرسم
    plot_file_name = f'plot_{volunteer_id}_nn_interval.png'
    plot_file_path = os.path.join(CLEANED_DIR, plot_file_name)
    plt.savefig(plot_file_path, dpi=100)
    plt.close()
    print(f"  -> تم حفظ رسم Heart Rate (NN Intervals) في: {plot_file_path}")


# =========================================================
# 4. تنفيذ العملية على جميع المتطوعين
# =========================================================

def main():
    # إنشاء مجلد البيانات النظيفة إذا لم يكن موجوداً
    Path(CLEANED_DIR).mkdir(parents=True, exist_ok=True)
    print(f"تم إنشاء مجلد البيانات النظيفة في: {CLEANED_DIR}")

    # العثور على جميع مجلدات المتطوعين (أرقام أو Vxxxx)
    volunteer_folders = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and (d.isdigit() or d.upper().startswith('V'))]
    
    # تحويل إلى أرقام وترتيب
    volunteer_ids = []
    for v in volunteer_folders:
        if v.isdigit():
            volunteer_ids.append((int(v), v))
        elif v.upper().startswith('V'):
            try:
                volunteer_ids.append((int(v[1:]), v))  # استخراج الرقم من V###
            except ValueError:
                pass
    
    volunteer_ids = sorted(volunteer_ids, key=lambda x: x[0])
    
    for _, vid in volunteer_ids:
        try:
            process_and_plot_volunteer(vid)
        except Exception as e:
            print(f"--- ❌ خطأ فادح غير متوقع في معالجة المتطوع {vid}: {e}")
            import traceback
            traceback.print_exc()
            print(f"تم تخطي المتطوع {vid}.")

if __name__ == '__main__':
    main()