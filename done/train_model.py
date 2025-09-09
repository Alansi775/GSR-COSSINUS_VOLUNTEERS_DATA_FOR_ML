import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def build_and_train_model():
    """
    يقوم بتحميل البيانات المصنفة، وتدريب نموذج مصنف شجرة القرارات، ثم يحفظ النموذج.
    """
    # 1. تحميل مجموعة البيانات الموحدة
    df = pd.read_csv('combined_labeled_gsr_dataset.csv')

    # 2. تحديد المتغيرات (الميزات والملصقات)
    X = df[['Resistance (Ohms)']]  # الميزة (المدخل)
    y = df['Label']               # الملصق (المخرج المستهدف)

    # 3. تقسيم البيانات إلى مجموعات تدريب واختبار (80%/20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. بناء النموذج
    model = DecisionTreeClassifier(random_state=42)
    
    # 5. تدريب النموذج
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 6. التنبؤ وتقييم الأداء
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Overall Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 7. حفظ النموذج المدرب
    joblib.dump(model, 'gsr_stress_model.pkl')
    print("\nModel saved as 'gsr_stress_model.pkl'. It's ready for deployment!")

if __name__ == '__main__':
    build_and_train_model()