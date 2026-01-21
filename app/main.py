from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import LoanApplication, PredictionResponse
from app.model import scoring_service

# Lifespan - современная замена для on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код перед запуском (Load ML model)
    try:
        scoring_service.load_model()
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
    
    yield  # Приложение работает здесь
    
    # Код после выключения (Clean up) - нам пока не нужен

app = FastAPI(title="Bank Credit Scoring API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is online. Use /predict to score clients."}

@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(application: LoanApplication):
    # .model_dump() - современная замена .dict() (убирает Warning)
    data = application.model_dump()
    
    try:
        # Получаем вероятность дефолта
        prob = scoring_service.predict(data)
        
        threshold = 0.35
        risk_class = 1 if prob > threshold else 0
        decision = "Rejected" if risk_class == 1 else "Approved"
        
        return {
            "default_probability": round(prob, 4),
            "risk_class": risk_class,
            "decision": decision
        }
    except Exception as e:
        # Если ошибка, мы увидим её в ответе
        raise HTTPException(status_code=500, detail=str(e))
