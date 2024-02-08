# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("insurance_prediction_model")

# Create input/output pydantic models
input_model = create_model("insurance_prediction_model_input", **{'age': 21, 'sex': 'male', 'bmi': 23.75, 'children': 2, 'smoker': 'no', 'region': 'northwest'})
output_model = create_model("insurance_prediction_model_output", prediction=3077.0955)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
