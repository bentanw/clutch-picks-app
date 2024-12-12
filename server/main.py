from fastapi import FastAPI, HTTPException
from models.ufc_model import UFCEloEngine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = UFCEloEngine()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow all origins (or specify ["http://localhost:5173"] for specific frontend)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


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


@app.get("/ufc/fighters")
def get_fighters():
    list_of_fighters = model.get_fighters()  # Assuming this returns a list of names
    fighters_with_details = [{"name": fighter} for fighter in list_of_fighters]
    return fighters_with_details