from fastapi import APIRouter
from app.schemas.prediction_schema import FraudPredictionInput
from scripts.prediction import predict

router = APIRouter()


@router.post("/predict")
def prediction(data:FraudPredictionInput):
    result=predict(data.model_dump())
    prediction= "froud" if result["prediction"] == 1 else "no froud"
    return {
        "Prediction": prediction,
        "Propensity": result["propensity"],
        "Model Statistics": result["model_statistics"]
    } 
