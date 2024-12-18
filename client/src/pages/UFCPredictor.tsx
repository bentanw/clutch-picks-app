import { useQuery } from "@tanstack/react-query";
import React, { useState } from "react";

import { fetchFighters } from "../service/fightersQuery";
import { Fighter } from "../types/fighters";

interface UFCPrediction {
  winner: string;
  message: string;
}

const UFCPredictor: React.FC = () => {
  const {
    data: fighters,
    isLoading,
    error,
  } = useQuery<Fighter[]>({
    queryKey: ["fighters"],
    queryFn: fetchFighters,
  });

  const [fighterLeft, setFighterLeft] = useState<Fighter | null>(null);
  const [fighterRight, setFighterRight] = useState<Fighter | null>(null);

  const [prediction, setPrediction] = useState<UFCPrediction | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handlePredict = async () => {
    if (fighterLeft && fighterRight) {
      try {
        // Clear previous prediction
        setPrediction(null);

        const response = await fetch(
          `http://localhost:8000/ufc/predict?fighter_1=${encodeURIComponent(
            fighterLeft.name
          )}&fighter_2=${encodeURIComponent(fighterRight.name)}`
        );
        if (!response.ok) {
          throw new Error("Failed to fetch prediction");
        }
        const data = (await response.json()) as UFCPrediction;
        setPrediction(data);
        setIsModalOpen(true);
      } catch (error) {
        alert(`Error: ${(error as Error).message}`);
      }
    } else {
      alert("Please select both fighters.");
    }
  };

  if (isLoading)
    return (
      <div className="min-h-screen w-screen flex flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800 text-white font-sans">
        Loading...
      </div>
    );
  if (error instanceof Error)
    return (
      <div className="min-h-screen w-screen flex flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800 text-white font-sans">
        Error: {error.message}
      </div>
    );

  return (
    <div className="min-h-screen w-screen flex flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800 text-white font-sans">
      <h1 className="text-3xl font-bold mb-8 uppercase tracking-wider">
        UFC Fight Predictor
      </h1>
      <div className="flex flex-col sm:flex-row items-center space-y-6 sm:space-y-0 sm:space-x-10 max-w-4xl mx-auto">
        {/* Fighter Left Dropdown */}
        <div className="flex flex-col items-center bg-gray-700 p-6 rounded-lg shadow-lg w-full sm:w-1/3">
          <h2 className="text-xl font-semibold mb-4 text-center">
            Fighter Left
          </h2>
          <select
            className="bg-gray-800 text-white rounded p-2 w-full"
            value={fighterLeft?.name || ""}
            onChange={(e) =>
              setFighterLeft(
                fighters?.find((f) => f.name === e.target.value) || null
              )
            }
          >
            <option value="" disabled>
              Select Fighter
            </option>
            {fighters?.map((fighter) => (
              <option key={fighter.name} value={fighter.name}>
                {fighter.name}
              </option>
            ))}
          </select>

          {fighterLeft && (
            <div className="text-center mt-4">
              <p>Selected: {fighterLeft.name}</p>
            </div>
          )}
        </div>

        {/* Predict Button */}
        <div>
          <button
            onClick={handlePredict}
            className="bg-red-600 hover:bg-red-700 transition-colors text-white font-bold py-2 px-6 rounded shadow"
          >
            Predict
          </button>
        </div>

        {/* Fighter Right Dropdown */}
        <div className="flex flex-col items-center bg-gray-700 p-6 rounded-lg shadow-lg w-full sm:w-1/3">
          <h2 className="text-xl font-semibold mb-4 text-center">
            Fighter Right
          </h2>
          <select
            className="bg-gray-800 text-white rounded p-2 w-full"
            value={fighterRight?.name || ""}
            onChange={(e) =>
              setFighterRight(
                fighters?.find((f) => f.name === e.target.value) || null
              )
            }
          >
            <option value="" disabled>
              Select Fighter
            </option>
            {fighters?.map((fighter) => (
              <option key={fighter.name} value={fighter.name}>
                {fighter.name}
              </option>
            ))}
          </select>

          {fighterRight && (
            <div className="text-center mt-4">
              <p>Selected: {fighterRight.name}</p>
            </div>
          )}
        </div>
      </div>

      {isModalOpen && prediction && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-4xl shadow-lg">
            <h2 className="text-2xl font-bold text-center mb-6">
              Prediction Result
            </h2>
            <div className="grid grid-cols-2 gap-4">
              {/* Fighter Left Info */}
              <div className="flex flex-col items-center bg-violet-950 p-4 rounded-lg">
                <h3 className="text-xl font-semibold mb-2">
                  {prediction.winner}
                </h3>
                <p className="text-sm text-gray-300">
                  Current Rating: {prediction.winner_rating}
                </p>
                <div className="mt-4 w-full">
                  <h4 className="font-semibold">Recent Fights</h4>
                  <ul className="text-sm list-disc list-inside h-40 overflow-y-auto bg-gray-900 p-2 rounded">
                    {prediction.winner_history
                      .slice()
                      .reverse()
                      .map((fight, index) => (
                        <li key={index}>
                          {fight.event} vs {fight.opponent} ({fight.result})
                        </li>
                      ))}
                  </ul>
                </div>
              </div>

              {/* Fighter Right Info */}
              <div className="flex flex-col items-center bg-blue-800 p-4 rounded-lg">
                <h3 className="text-xl font-semibold mb-2">
                  {prediction.loser}
                </h3>
                <p className="text-sm text-gray-300">
                  Current Rating: {prediction.loser_rating}
                </p>
                <div className="mt-4 w-full">
                  <h4 className="font-semibold">Recent Fights</h4>
                  <ul className="text-sm list-disc list-inside h-40 overflow-y-auto bg-gray-900 p-2 rounded">
                    {prediction.loser_history
                      .slice()
                      .reverse()
                      .map((fight, index) => (
                        <li key={index}>
                          {fight.event} vs {fight.opponent} ({fight.result})
                        </li>
                      ))}
                  </ul>
                </div>
              </div>
            </div>

            {/* Winner Announcement */}
            <div className="mt-6 text-center">
              <p className="text-lg font-bold text-green-600">
                Winner: {prediction.winner}
              </p>
              <p className="text-gray-700">{prediction.message}</p>
            </div>

            {/* Close Button */}
            <div className="text-center mt-4">
              <button
                onClick={() => setIsModalOpen(false)}
                className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UFCPredictor;
