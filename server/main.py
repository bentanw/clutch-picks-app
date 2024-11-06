from fastapi import FastAPI, HTTPException
from models.ufc_model import UFCEloEngine

app = FastAPI()
model = UFCEloEngine()

@app.get("/")
async def root():
    return {"message": "Welcome to UFC Predictor API"}

@app.get("/ufc/predict")
async def ufc_fighter_prediction(fighter_1: str, fighter_2: str):
    try:
        result = model.predict(fighter_1, fighter_2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))