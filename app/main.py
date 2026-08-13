from fastapi import FastAPI
from app.routes.prediction import router
from scripts import prediction as prediction_script
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    #    "https://credit-fraud-frontend.vercel.app",
	   "https://credit-fraud-frontend-qzp2.vercel.app/"
    ],    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def startup_event():
	# Initialize logging and load model once at startup (prevents reload loops).
	try:
		prediction_script.init_model_and_logging()
	except Exception:
		# Avoid raising during startup to allow error visibility in logs; re-raise if desired.
		raise


app.include_router(router)