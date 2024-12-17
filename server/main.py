from fastapi import FastAPI, HTTPException
from models.ufc_model import UFCEloEngine, get_fighter_info
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

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
        # Predict the winner
        winner = model.predict(fighter_1, fighter_2)
        loser = fighter_2 if winner == fighter_1 else fighter_1

        # Function to fetch match history for a fighter
        def get_match_history(fighter_name):
            fighter_matches_1 = model.ufcfights[
                (model.ufcfights["fighter_1"] == fighter_name)
            ][
                [
                    "event",
                    "fighter_2",
                    "result",
                    "fighter_1_elo_start",
                    "fighter_2_elo_start",
                    "fighter_1_elo_end",
                ]
            ].rename(
                columns={
                    "fighter_2": "opponent",
                    "fighter_1_elo_start": "own_elo",
                    "fighter_2_elo_start": "opponent_elo",
                    "fighter_1_elo_end": "own_elo_after",
                }
            )

            fighter_matches_2 = model.ufcfights[
                (model.ufcfights["fighter_2"] == fighter_name)
            ][
                [
                    "event",
                    "fighter_1",
                    "result",
                    "fighter_1_elo_start",
                    "fighter_2_elo_start",
                    "fighter_2_elo_end",
                ]
            ].rename(
                columns={
                    "fighter_1": "opponent",
                    "fighter_2_elo_start": "own_elo",
                    "fighter_1_elo_start": "opponent_elo",
                    "fighter_2_elo_end": "own_elo_after",
                }
            )
            fighter_matches_2["result"] = fighter_matches_2["result"].str.replace(
                "win", "loss"
            )

            return (
                pd.concat([fighter_matches_1, fighter_matches_2])
                .reset_index(drop=True)
                .to_dict(orient="records")
            )

        # Get ratings
        winner_rating = get_fighter_info(winner, model.elo_ratings, model.ufcfights)
        loser_rating = get_fighter_info(loser, model.elo_ratings, model.ufcfights)

        # Fetch match history
        winner_history = get_match_history(winner)
        loser_history = get_match_history(loser)

        return {
            "winner": winner,
            "loser": loser,
            "message": f"{winner} is predicted to win against {loser}.",
            "winner_rating": winner_rating,
            "loser_rating": loser_rating,
            "winner_history": winner_history,
            "loser_history": loser_history,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ufc/fighters")
def get_fighters():
    list_of_fighters = model.get_fighters()  # Assuming this returns a list of names
    fighters_with_details = [{"name": fighter} for fighter in list_of_fighters]
    return fighters_with_details
