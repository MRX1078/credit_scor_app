# Генератор синтетических данных
import pandas as pd
import numpy as np
import os

# Фиксируем seed для воспроизводимости
np.random.seed(42)

def generate_credit_data(n_samples: int = 5000, save_path: str = None) -> pd.DataFrame:
    """
    Генерирует синтетический датасет для банковского скоринга.
    """
    
    # 1. ГЕНЕРАЦИЯ ФИЧ (FEATURES)
    
    # Возраст: от 18 до 70
    age = np.random.randint(18, 70, n_samples)
    
    # Годовой доход: Нормальное распределение, смещенное к 60k, минимум 10k
    income = np.random.normal(60000, 15000, n_samples)
    income = np.maximum(income, 10000) # Обрезаем отрицательные или слишком низкие значения
    
    # Стаж работы (лет): зависти от возраста (не может быть стаж 20 лет, если возраст 18)
    years_employed = np.random.randint(0, 20, n_samples)
    years_employed = np.minimum(years_employed, age - 18)
    
    # Кредитный лимит (по всем картам)
    credit_limit = np.random.normal(20000, 10000, n_samples)
    credit_limit = np.maximum(credit_limit, 1000)

    # Утилизация кредитного лимита (от 0 до 1+). 
    # Бета-распределение здесь подходит лучше, так как большинство держится в рамках, но есть выбросы
    credit_utilization = np.random.beta(2, 5, n_samples) 
    
    # Количество просрочек за последние 2 года (Распределение Пуассона - редкие события)
    delinquencies_2y = np.random.poisson(0.3, n_samples)
    
    # Запрашиваемая сумма кредита
    loan_amount = np.random.normal(15000, 5000, n_samples)
    loan_amount = np.maximum(loan_amount, 1000)

    # Сборка DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income.astype(int),
        'years_employed': years_employed,
        'credit_limit': credit_limit.astype(int),
        'credit_utilization': credit_utilization.round(2),
        'delinquencies_2y': delinquencies_2y,
        'loan_amount': loan_amount.astype(int)
    })

    # 2. ЛОГИКА ГЕНЕРАЦИИ ТАРГЕТА (TARGET)
    
    # Рассчитываем "Риск-балл" (Latent variable). Чем выше, тем хуже.
    # Коэффициенты подобраны эмпирически для логики.
    
    # Высокая утилизация кредитки сильно повышает риск
    term_utilization = df['credit_utilization'] * 5.0 
    
    # Каждая просрочка критически повышает риск
    term_delinq = df['delinquencies_2y'] * 2.5
    
    # Высокий доход снижает риск (отрицательный вес)
    term_income = - (df['income'] / 20000)
    
    # Большая сумма займа относительно дохода повышает риск (аналог DTI)
    term_dti = (df['loan_amount'] / (df['income'] + 1)) * 10
    
    # Возраст и стаж немного снижают риск
    term_stability = - (df['age'] / 100) - (df['years_employed'] / 5)

    # Базовый риск + шум (чтобы модель не выучила идеальную формулу)
    noise = np.random.normal(0, 1.5, n_samples)
    
    risk_score = -2 + term_utilization + term_delinq + term_income + term_dti + term_stability + noise

    # Превращаем скор в вероятность через сигмоиду: 1 / (1 + e^-x)
    probability_of_default = 1 / (1 + np.exp(-risk_score))
    
    # Генерируем класс (0 или 1) на основе вероятности
    df['default'] = np.random.binomial(1, probability_of_default)

    # Вывод статистики для проверки баланса классов
    default_rate = df['default'].mean()
    print(f"Dataset generated with {n_samples} samples.")
    print(f"Default Rate: {default_rate:.2%} (Target usually 10-20% for balanced tasks)")

    if save_path:
        # Убедимся, что папка существует (если path включает папки)
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
        
    return df

if __name__ == "__main__":
    # Тестовый запуск: сохраняем в текущую директорию или в ml/data
    generate_credit_data(save_path="credit_risk_dataset.csv")
