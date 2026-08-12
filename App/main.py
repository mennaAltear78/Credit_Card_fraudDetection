from fastapi import FastAPI
from app.routes.prediction import router
from scripts import prediction as prediction_script

app = FastAPI()


@app.on_event("startup")
def startup_event():
	# Initialize logging and load model once at startup (prevents reload loops).
	try:
		prediction_script.init_model_and_logging()
	except Exception:
		# Avoid raising during startup to allow error visibility in logs; re-raise if desired.
		raise


app.include_router(router)