from random import random
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

#cors
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:8000",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple post call in fastapi inputs string and [string] and returns [flaot]
class Item(BaseModel):
    query: str
    data: list[str]

@app.post("/", response_model=list[float])
async def create_item(item: Item):
    if not item.query or not item.data:
        raise HTTPException(status_code=400, detail="Invalid input")
    result = [random() for value in item.data]
    return JSONResponse(content=dict(cosine_sims=result,query_enc=[ord(x) for x in item.query]))