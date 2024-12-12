import axios from "axios";

import { Fighter } from "../types/fighters";

export const fetchFighters = async (): Promise<Fighter[]> => {
  const response = await axios.get<Fighter[]>(
    "http://localhost:8000/ufc/fighters"
  );
  return response.data;
};
