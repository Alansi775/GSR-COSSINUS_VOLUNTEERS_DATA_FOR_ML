import os
import pandas as pd
import glob

def prepare_and_label_dataset():
    """
    يجمع جميع ملفات GSR النظيفة، ويضيف عمود تصنيف، ويحفظ البيانات في ملف واحد.
    """
    # المسار إلى مجلد البيانات النظيفة
    data_path = 'gsr_data_graphs'
    
    # استخدام glob للعثور على كل ملفات CSV في المجلد
    all_files = glob.glob(os.path.join(data_path, 'GSR_Data-*.csv'))
    
    if not all_files:
        print("Error: No GSR data files found in the specified directory.")
        return

    # دمج كل الملفات في DataFrame واحد
    combined_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

    # إضافة عمود التصنيف بناءً على المرحلة
    def categorize_stage(stage):
        if 'Relaxation' in str(stage) or 'Calibration' in str(stage):
            return 'Relaxed'
        elif 'Normal' in str(stage):
            return 'Normal'
        elif 'Remove' in str(stage):
            return 'Stress'
        return 'Unknown' # في حالة وجود مراحل غير متوقعة

    combined_df['Label'] = combined_df['Stage'].apply(categorize_stage)

    # إزالة أي صفوف لم يتم تصنيفها بشكل صحيح
    combined_df = combined_df[combined_df['Label'] != 'Unknown']

    # حفظ مجموعة البيانات الموحدة والجاهزة للتدريب
    output_filename = 'combined_labeled_gsr_dataset.csv'
    combined_df.to_csv(output_filename, index=False)

    print(f"Dataset successfully prepared and saved as '{output_filename}'.")
    print(f"Total number of data points: {len(combined_df)}")
    print(f"Distribution of labels:\n{combined_df['Label'].value_counts()}")

if __name__ == '__main__':
    prepare_and_label_dataset()