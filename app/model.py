# Класс для загрузки и использования ML модели
import joblib
import pandas as pd
import os

class CreditScoringModel:
    model = None

    def load_model(self):
        """Загружает модель при старте приложения"""
        # Путь к модели относительно этого файла
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "credit_model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
            
        self.model = joblib.load(model_path)
        print("ML модель успешно загружена")

    def predict(self, data: dict):
        """
        Принимает словарь данных, возвращает вероятность дефолта.
        """
        if not self.model:
            raise RuntimeError("Модель не загружена")
            
        # Важно: порядок колонок должен совпадать с обучением!
        # Мы создаем DataFrame из одного ряда
        df = pd.DataFrame([data])
        
        # Список колонок, как было при обучении (важен порядок)
        feature_order = [
            'age', 'income', 'years_employed', 'credit_limit', 
            'credit_utilization', 'delinquencies_2y', 'loan_amount'
        ]
        
        # Гарантируем правильный порядок
        df = df[feature_order]
        
        # Получаем вероятность класса 1 (дефолт)
        probability = self.model.predict_proba(df)[0][1]
        return probability

# Создаем глобальный экземпляр, чтобы импортировать его в main.py
scoring_service = CreditScoringModel()
