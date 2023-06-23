#python 3.9.12
#python -m uvicorn app12:app --reload

from fastapi import FastAPI, File, UploadFile, Request
from proses import *
from fastapi.templating import Jinja2Templates


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/hello')
async def hello_world():
    return "Model klasifikasi Motif Batik"

@app.post("/prediksi")
async def predict_image(file: UploadFile = File(...)):
    conf, label = proses(file)
    hasil = label + " ("+str(f"{conf*100:.2f}") + "%)"
    
    return {"Text":hasil}
    




