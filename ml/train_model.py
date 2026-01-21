import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os

def train():
    # ---------------------------------------------------------
    # 1. НАСТРОЙКА ПУТЕЙ (Самая важная часть)
    # ---------------------------------------------------------
    # Определяем, где лежит этот файл (ml/train_model.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Получаем корень проекта (поднимаемся на уровень выше из ml/)
    project_root = os.path.dirname(current_dir)
    
    # Путь к данным: project_root/data/credit_risk_dataset.csv
    data_path = os.path.join(project_root, "data", "credit_risk_dataset.csv")
    
    # Путь для сохранения модели: project_root/app/models/credit_model.pkl
    model_dir = os.path.join(project_root, "app", "models")
    model_path = os.path.join(model_dir, "credit_model.pkl")

    print(f"Путь к данным: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл не найден по пути: {data_path}")

    # ---------------------------------------------------------
    # 2. ЗАГРУЗКА И ОБУЧЕНИЕ
    # ---------------------------------------------------------
    df = pd.read_csv(data_path)
    print(f"Загружено {len(df)} строк данных.")

    # Разделяем фичи и таргет
    X = df.drop('default', axis=1)
    y = df['default']

    # Делим на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Обучение RandomForest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 3. ОЦЕНКА КАЧЕСТВА
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Результаты ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # ---------------------------------------------------------
    # 4. СОХРАНЕНИЕ
    # ---------------------------------------------------------
    # Создаем папку app/models, если её нет
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"\nМодель успешно сохранена в: {model_path}")

if __name__ == "__main__":
    train()
