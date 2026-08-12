from fastapi import FastAPI
from App.routes.prediction import router

app = FastAPI()

app.include_router(router)