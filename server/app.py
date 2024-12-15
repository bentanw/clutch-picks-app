from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


# Function to load and predict MVP for a given year
def predict_mvp(year):
     # Map year to dataset
    data_file = f"data/{year}.csv"
    mvp_data = pd.read_csv("data/mvp_seasons.csv")  # Historical MVP data
    season_data = pd.read_csv(data_file)

    # Print the columns to debug
    print("Columns in season data:", season_data.columns)

    # Preprocess the data
    season_data = season_data[season_data['MP'] >= 25]  # Filter players with minimal minutes played
    season_data.fillna(0, inplace=True)
    mvp_data.fillna(0, inplace=True)

    # Extract features and scale
    features = mvp_data[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS/48']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Target variable
    target = mvp_data['MVP']

    # Train the model
    model = LogisticRegression()
    model.fit(features_scaled, target)

    # Scale current season data
    current_season_features = season_data[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'WS/48']]
    current_season_scaled = scaler.transform(current_season_features)

    # Predict probabilities
    probabilities = model.predict_proba(current_season_scaled)[:, 1]
    season_data['MVP_Probability'] = probabilities

    # Get top 10 players
    top_10_players = season_data[['Player', 'MVP_Probability', 'PTS', 'TRB', 'AST', 'WS/48']].sort_values(by='MVP_Probability', ascending=False).head(10)

    return top_10_players

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def results():
    year = request.form.get("year")
    
    # Get the top 10 predicted MVPs for the selected year
    top_10 = predict_mvp(year)

    # Dictionary mapping each season to its actual MVP
    actual_mvp_dict = {
    "05-06": "Steve Nash",
    "09-10": "LeBron James",
    "16-17": "Russell Westbrook",
    "23-24": "Giannis Antetokounmpo",
    "24-25": "TBD",  # Fcurrent season
}
    # Map year to actual MVP using the dictionary
    actual_mvp = actual_mvp_dict.get(year, "TBD")

    # Add additional statistics to the top 10 data
    top_10 = top_10[['Player', 'MVP_Probability', 'PTS', 'TRB', 'AST', 'WS/48']]  # Keep only relevant columns

    # Render the results page with the top 10 players and the actual MVP
    return render_template("results.html", year=year, top_10=top_10, actual_mvp=actual_mvp)

if __name__ == "__main__":
    app.run(debug=True)
