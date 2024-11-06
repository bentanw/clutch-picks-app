# sports-betting-nn

## Set up backend

### Create virtual environment
- Windows: ```virtualenv env```
- Linux/MacOS: ```python -m venv env```

### Activate virtual environment
- Windows: ```.\env\Scripts\activate```
- Linux/MacOS: ```source env/bin/activate```
- ```pip3 install -r requirements.txt```

### Start backend server
- local: ```fastapi dev main.py```
- production:  ```uvicorn main:app --reload```

### Test
- go to ```localhost:8000/docs```
- test the api such as "first_fighter: Ilia Topuria", "second_fighter: Max Holloway"