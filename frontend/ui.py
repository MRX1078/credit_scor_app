import streamlit as st
import requests
import json
import os


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Конфигурация страницы
st.set_page_config(
    page_title="Bank Scoring System",
    page_icon="🏦",
    layout="centered"
)

# Заголовок и описание
st.title("🏦 Кредитный Скоринг")
st.write("Введите данные клиента для оценки вероятности дефолта.")

# Создаем форму для ввода данных
with st.form("application_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Возраст", min_value=18, max_value=100, value=30)
        income = st.number_input("Годовой доход ($)", min_value=0, value=50000, step=1000)
        years_employed = st.number_input("Стаж работы (лет)", min_value=0, value=5)
        credit_limit = st.number_input("Общий кредитный лимит ($)", min_value=0, value=20000)

    with col2:
        loan_amount = st.number_input("Запрашиваемая сумма ($)", min_value=0, value=15000)
        delinquencies = st.number_input("Кол-во просрочек (2 года)", min_value=0, value=0)
        # Слайдер для утилизации (от 0% до 100% и выше)
        utilization_percent = st.slider("Использование кредиток (%)", 0, 150, 30)
        # Преобразуем проценты в коэффициент (30% -> 0.3)
        credit_utilization = utilization_percent / 100.0

    submitted = st.form_submit_button("Рассчитать риск")

# Логика обработки нажатия
if submitted:
    # Формируем JSON, который ждет наш FastAPI
    client_data = {
        "age": age,
        "income": income,
        "years_employed": years_employed,
        "credit_limit": credit_limit,
        "credit_utilization": credit_utilization,
        "delinquencies_2y": delinquencies,
        "loan_amount": loan_amount
    }

    try:
        # Отправляем запрос на локальный API
        # ВАЖНО: Убедись, что FastAPI запущен на этом порту
        response = requests.post(f"{API_URL}/predict", json=client_data)
        
        if response.status_code == 200:
            result = response.json()
            prob = result['default_probability']
            decision = result['decision']
            
            st.markdown("---")
            st.subheader("Результат скоринга")
            
            # Визуализация результата
            if decision == "Approved":
                st.success(f"✅ Кредит ОДОБРЕН")
            else:
                st.error(f"❌ Кредит ОТКЛОНЕН")
            
            st.write(f"Вероятность дефолта: **{prob:.2%}**")
            
            # Прогресс-бар риска
            st.progress(prob, text="Уровень риска")
            
            # Дополнительная инфо
            if prob > 0.35:
                st.warning("⚠️ Риск слишком высок ( > 35%)")
            
        else:
            st.error(f"Ошибка сервера: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Не удалось подключиться к API. Убедитесь, что backend запущен.")
