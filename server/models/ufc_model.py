import pandas as pd
import numpy as np
from IPython.display import display
import os
import pickle
from typing import Dict, List
from tqdm import tqdm
import matplotlib.pyplot as plt


# Function to calculate the expected score
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


# Function to update Elo ratings
def update_elo(winner_elo, loser_elo, k_factor):
    expected_win = expected_score(winner_elo, loser_elo)
    new_winner_elo = winner_elo + k_factor * (1 - expected_win)
    new_loser_elo = loser_elo + k_factor * (0 - (1 - expected_win))
    return round(new_winner_elo, 2), round(new_loser_elo, 2)


def get_fighter_info(fighter_name, all_ratings, ufcfights, initial_elo=1000):
    # Check if the fighter exists in the Elo ratings dictionary
    if fighter_name in all_ratings:
        elo = all_ratings[fighter_name]
    else:
        elo = initial_elo

    # Find all matches where the fighter appeared as either fighter_1 or fighter_2
    fighter_matches_1 = ufcfights[(ufcfights["fighter_1"] == fighter_name)][
        [
            "event",
            "fighter_2",
            "result",
            "event_id",
            "fighter_1_elo_start",
            "fighter_2_elo_start",
            "fighter_1_elo_end",
        ]
    ].rename(
        columns={
            "fighter_2": "opponent",
            "fighter_2_elo_start": "opponent elo",
            "fighter_1_elo_start": "own elo",
            "fighter_1_elo_end": "own elo after",
        }
    )

    fighter_matches_2 = ufcfights[(ufcfights["fighter_2"] == fighter_name)][
        [
            "event",
            "fighter_1",
            "result",
            "event_id",
            "fighter_1_elo_start",
            "fighter_2_elo_start",
            "fighter_2_elo_end",
        ]
    ].rename(
        columns={
            "fighter_1": "opponent",
            "fighter_1_elo_start": "opponent elo",
            "fighter_2_elo_start": "own elo",
            "fighter_2_elo_end": "own elo after",
        }
    )
    fighter_matches_2["result"] = fighter_matches_2["result"].str.replace("win", "loss")

    return elo[-1]


class UFCEloEngine:
    def __init__(self):
        self.model_path = os.path.join(
            os.path.dirname(__file__), "../models/ufc_elo_model.pkl"
        )
        self.training_data = os.path.join(
            os.path.dirname(__file__), "../data/ufcfights10_26_24.csv"
        )
        self.initial_elo = 1000
        self.k_factor = 200

        # Try to load existing model
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            # Initialize new model
            ufcfights_not_sorted = pd.read_csv(self.training_data, index_col=0)
            self.ufcfights = ufcfights_not_sorted.reset_index()
            self.ufcfights = self.ufcfights.sort_index(ascending=False)

            # Create unique event IDs
            unique_events = (
                self.ufcfights[["event"]].drop_duplicates().reset_index(drop=True)
            )
            unique_events["event_id"] = range(1, len(unique_events) + 1)
            self.ufcfights = self.ufcfights.merge(unique_events, on="event")

            # Drop unnecessary columns
            self.ufcfights.drop(columns=["method", "round", "time"], inplace=True)

            self.elo_ratings: Dict[str, List[float]] = {}

    def save_model(self):
        """Save the trained model to disk"""
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_data = {"ufcfights": self.ufcfights, "elo_ratings": self.elo_ratings}
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self):
        """Load the trained model from disk"""
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
            self.ufcfights = model_data["ufcfights"]
            self.elo_ratings = model_data["elo_ratings"]

    def train(self):
        """Train the model and save it"""
        # Create unique match IDs
        self.ufcfights["cc_match"] = np.arange(1, len(self.ufcfights) + 1)

        # Add columns for Elo ratings
        self.ufcfights["fighter_1_elo_start"] = 0
        self.ufcfights["fighter_2_elo_start"] = 0
        self.ufcfights["fighter_1_elo_end"] = 0
        self.ufcfights["fighter_2_elo_end"] = 0

        # Calculate Elo ratings for each match
        for index, row in self.ufcfights.iterrows():
            fighter_1, fighter_2 = row["fighter_1"], row["fighter_2"]

            if fighter_1 not in self.elo_ratings:
                self.elo_ratings[fighter_1] = [self.initial_elo]
            if fighter_2 not in self.elo_ratings:
                self.elo_ratings[fighter_2] = [self.initial_elo]

            fighter_1_elo_start, fighter_2_elo_start = (
                self.elo_ratings[fighter_1][-1],
                self.elo_ratings[fighter_2][-1],
            )

            self.ufcfights.at[index, "fighter_1_elo_start"] = fighter_1_elo_start
            self.ufcfights.at[index, "fighter_2_elo_start"] = fighter_2_elo_start

            if row["result"] == "win":
                new_fighter1_elo, new_fighter2_elo = update_elo(
                    fighter_1_elo_start, fighter_2_elo_start, self.k_factor
                )
            elif row["result"] == "draw":
                new_fighter1_elo, new_fighter2_elo = update_elo(
                    fighter_1_elo_start, fighter_2_elo_start, self.k_factor / 2
                )
            else:
                new_fighter1_elo, new_fighter2_elo = (
                    fighter_1_elo_start,
                    fighter_2_elo_start,
                )

            self.ufcfights.at[index, "fighter_1_elo_end"] = new_fighter1_elo
            self.ufcfights.at[index, "fighter_2_elo_end"] = new_fighter2_elo

            self.elo_ratings[fighter_1].append(new_fighter1_elo)
            self.elo_ratings[fighter_2].append(new_fighter2_elo)

        # Save the trained model
        self.save_model()

    def predict(self, first_fighter: str, second_fighter: str):
        fighter1_elo_rating = get_fighter_info(
            first_fighter, self.elo_ratings, self.ufcfights
        )
        fighter2_elo_rating = get_fighter_info(
            second_fighter, self.elo_ratings, self.ufcfights
        )

        if fighter1_elo_rating > fighter2_elo_rating:
            return first_fighter
        else:
            return second_fighter

    def get_fighters(self):
        return list(self.elo_ratings.keys())


def run_accuracy_tests(
    model: UFCEloEngine, test_data_path: str, runs: int = 50, sample_size: int = 100
):
    accuracies = []

    for _ in tqdm(range(runs)):
        # Reload the model for each run
        model = UFCEloEngine()
        model.train()

        # Sample the test data
        test_data = pd.read_csv(test_data_path, index_col=0).sample(
            sample_size, random_state=np.random.randint(0, 10000)
        )

        # Run the accuracy test and collect accuracy
        correct_predictions = 0
        total_matches = 0

        for index, row in test_data.iterrows():
            fighter_1 = row["fighter_1"]
            fighter_2 = row["fighter_2"]
            actual_result = row["result"]

            predicted_winner = model.predict(fighter_1, fighter_2)

            if actual_result == "win":
                actual_winner = fighter_1
            elif actual_result == "loss":
                actual_winner = fighter_2
            else:
                continue

            if predicted_winner == actual_winner:
                correct_predictions += 1
            total_matches += 1

        accuracy = correct_predictions / total_matches * 100
        accuracies.append(accuracy)

    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, runs + 1), accuracies, marker="o", linestyle="--")
    plt.title("Elo Model Accuracy Across Multiple Runs")
    plt.xlabel("Run Number")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.show()

    # Print summary statistics
    print(f"Average Accuracy: {np.mean(accuracies):.2f}%")
    print(f"Minimum Accuracy: {np.min(accuracies):.2f}%")
    print(f"Maximum Accuracy: {np.max(accuracies):.2f}%")


if __name__ == "__main__":
    # Instantiate and train the model
    model = UFCEloEngine()
    model.train()

    # Define test data path
    test_data_path = model.training_data

    # Run the accuracy tests multiple times
    run_accuracy_tests(model, test_data_path, runs=50, sample_size=100)
