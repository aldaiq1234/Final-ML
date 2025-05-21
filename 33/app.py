from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('model.pkl')

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class InputData(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_male: int
    Embarked_C: int
    Embarked_Q: int
    Embarked_S: int

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(data: InputData):
    X = np.array([[data.Pclass, data.Age, data.SibSp, data.Parch, data.Fare, data.Sex_male, data.Embarked_C, data.Embarked_Q, data.Embarked_S]])
    proba = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    return {"survived": bool(pred), "probability": round(proba, 4)}
