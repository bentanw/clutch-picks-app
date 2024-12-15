from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Global variables to hold data and model
def getWinner(teamA, teamB):
    #clean the data
    df_2024_2025 = pd.read_csv("../data/nbateams/nbateams2024-2025.csv")  
    df_2024_2025 = df_2024_2025.dropna(axis=1, how='all')

    df_2023_2024 = pd.read_csv("../data/nbateams/nbateams2023-2024.csv")  
    df_2023_2024  = df_2023_2024.dropna(axis=1, how='all')

    df_2022_2023 = pd.read_csv("../data/nbateams/nbateams2022-2023.csv")  
    df_2022_2023  = df_2022_2023.dropna(axis=1, how='all')

    # Ensure there are no NaNs in either dataset by dropping rows with NaNs
    df_2024_2025 = df_2024_2025[['Team', 'W', 'L', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA']].dropna()
    df_2023_2024 = df_2023_2024[['Team', 'W', 'L', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA']].dropna()
    df_2022_2023 = df_2022_2023[['Team', 'W', 'L', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA']].dropna()

    # Combine both datasets into a single DataFrame
    df = pd.concat([df_2024_2025, df_2023_2024, df_2022_2023], ignore_index=True)
    print(df.head)
    # Define a function to create pairwise feature differences between two teams
    def create_features(team_a, team_b, data):
        team_a_data = data[data['Team'] == team_a].iloc[0]
        team_b_data = data[data['Team'] == team_b].iloc[0]
        # Calculate the difference in statistics (drop non-numeric columns)
        feature_diff = team_a_data.drop(['Team', 'W', 'L']) - team_b_data.drop(['Team', 'W', 'L'])
        return feature_diff

    # Generate training data
    X = []
    y = []

    # Generate pairwise combinations for training
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            team_a = df.iloc[i]
            team_b = df.iloc[j]
            # Create feature differences
            features = create_features(team_a['Team'], team_b['Team'], df)
            X.append(features.values)
            # Label as 1 if team A has more wins than team B, otherwise 0
            y.append(1 if team_a['W'] > team_b['W'] else 0)

    # Convert X to a DataFrame and apply infer_objects to downcast types without NaN handling
    X = pd.DataFrame(X)
    X = X.infer_objects(copy=False)  # Apply downcasting
    y = pd.Series(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Predict the winner between two teams
    team_a = teamA
    team_b = teamB
    new_features = create_features(team_a, team_b, df).infer_objects(copy=False)  # Downcast without NaN handling
    prediction = model.predict([new_features])[0]

    return (team_a if prediction == 1 else team_b) , accuracy

teams = [
    "Boston Celtics", "Cleveland Cavaliers", "Philadelphia 76ers", "Memphis Grizzlies", "Milwaukee Bucks", 
    "Denver Nuggets", "New York Knicks", "Sacramento Kings", "Phoenix Suns", "New Orleans Pelicans", 
    "Golden State Warriors", "Toronto Raptors", "Chicago Bulls", "Oklahoma City Thunder", "Brooklyn Nets", 
    "Los Angeles Lakers", "Los Angeles Clippers", "Atlanta Hawks", "Dallas Mavericks", "Minnesota Timberwolves", 
    "Miami Heat", "Utah Jazz", "Washington Wizards", "Orlando Magic", "Indiana Pacers", 
    "Portland Trail Blazers", "Charlotte Hornets", "Houston Rockets", "Detroit Pistons", "San Antonio Spurs"
]


@app.route('/')
def home():
    return render_template('teamvsteam.html', teams=teams)


@app.route('/predict', methods=['POST','GET'])

def predict():
    team_a = request.form.get('team_a')
    team_b = request.form.get('team_b')

    if team_a == team_b:
        return render_template(
            'teamvsteam.html', 
            teams=teams, 
            error="Teams cannot play against themselves",
            form_data=request.form
        )
    
    winner,accuracy = getWinner(team_a, team_b)
    return render_template(
        'teamvsteam.html', 
        teams=teams, 
        winner=winner, 
        form_data=request.form,
        accuracy=str(round(accuracy*100)) + "%"
    )


if __name__ == '__main__':
    app.run(debug=True)
