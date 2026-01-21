import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_read_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Credit Scoring API is online. Use /predict to score clients."}

def test_prediction_approved():
    """Проверка сценария хорошего клиента"""
    good_client = {
        "age": 45,
        "income": 80000,
        "years_employed": 10,
        "credit_limit": 50000,
        "credit_utilization": 0.1,
        "delinquencies_2y": 0,
        "loan_amount": 10000
    }
    
    # Использование 'with' гарантирует запуск startup-событий (загрузку модели)
    with TestClient(app) as client:
        response = client.post("/predict", json=good_client)
        
        # Если упадет, выведет текст ошибки
        assert response.status_code == 200, f"Ошибка сервера: {response.text}"
        
        data = response.json()
        assert data["decision"] == "Approved"
        assert data["risk_class"] == 0

def test_prediction_rejected():
    """Проверка сценария плохого клиента"""
    bad_client = {
        "age": 20,
        "income": 15000,
        "years_employed": 0,
        "credit_limit": 1000,
        "credit_utilization": 0.95,
        "delinquencies_2y": 4,
        "loan_amount": 20000
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=bad_client)
        assert response.status_code == 200, f"Ошибка сервера: {response.text}"
        
        data = response.json()
        assert data["decision"] == "Rejected"
        assert data["risk_class"] == 1

def test_validation_error():
    invalid_client = {
        "age": 150,
        "income": -500
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_client)
        assert response.status_code == 422
