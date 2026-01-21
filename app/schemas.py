# Pydantic модели для валидации входных данных
from pydantic import BaseModel, Field

class LoanApplication(BaseModel):
    # Описываем поля, которые использовались при обучении
    age: int = Field(..., ge=18, le=100, description="Возраст клиента")
    income: int = Field(..., ge=0, description="Годовой доход")
    years_employed: int = Field(..., ge=0, description="Стаж работы в годах")
    credit_limit: int = Field(..., ge=0, description="Общий кредитный лимит")
    credit_utilization: float = Field(..., ge=0.0, le=5.0, description="Использование кредитки (0.3 = 30%)")
    delinquencies_2y: int = Field(..., ge=0, description="Количество просрочек за 2 года")
    loan_amount: int = Field(..., ge=0, description="Запрашиваемая сумма кредита")

class PredictionResponse(BaseModel):
    default_probability: float
    risk_class: int
    decision: str  # "Approved" или "Rejected"
