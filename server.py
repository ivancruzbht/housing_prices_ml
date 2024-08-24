import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import inference, load_model
from data_pipeline import load_pipeline
import traceback
from utils import load_config

app = FastAPI()
config = load_config("config.yaml")
model = load_model(config)
pipeline = load_pipeline(config)


# Housing price data model
class HousingData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


@app.post("/predict")
async def predict(data: HousingData):
    try:
        input_data = [[
            data.longitude,
            data.latitude,
            data.housing_median_age,
            data.total_rooms,
            data.total_bedrooms,
            data.population,
            data.households,
            data.median_income,
            0.0,            # TODO: Placeholder for median_house_value, refactor to remove
            data.ocean_proximity,
        ]]

        columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",
        "ocean_proximity"
        ]

        X = pipeline.transform(pd.DataFrame(input_data, columns=columns))
        prediction = inference(config, X, model)
        return {"prediction": prediction.item()}
    
    except Exception as e:
        tb_str = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e} - {tb_str}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)