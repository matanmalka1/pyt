#!/usr/bin/env python3
"""
app.py — FastAPI web interface for PlantVillage training & inference.

Run:
    cd src && uvicorn api.app:app --reload --port 8000
Then open: http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import OUTPUT_DIR, STATIC_DIR
from .routes import router

app = FastAPI(title="PlantVillage AI")
app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)),  name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)),  name="outputs")
app.include_router(router)
