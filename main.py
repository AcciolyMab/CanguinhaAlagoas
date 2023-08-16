# main.py
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from geolocation_router import router as geolocation_router
from utils import consultarProduto
from optimization_router import optimization_router

app = FastAPI()  # Cria a instância do FastAPI

app.include_router(geolocation_router)
app.include_router(optimization_router)

class Location(BaseModel):
    latitude: float
    longitude: float


templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

global response_list
response_list = []
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/consultarProduto/", response_class=HTMLResponse)
async def consultar_produto(request: Request):
    global response_list  # Refer to the global variable
    form_data = await request.form()

    formData = json.loads(form_data.get("formData"))
    dias = int(form_data.get("dias"))
    raio = int(form_data.get("raio"))
    address = form_data.get("address")
    latitude = float(form_data.get("latitude"))
    longitude = float(form_data.get("longitude"))

    # Clear the existing response list
    response_list.clear()

    for item in formData:
        item_description = item['description']
        item_values = item['gtins']

        for item_code in item_values:
            response = consultarProduto(item_code, latitude, longitude, raio, dias)
            if response is not None:
                for item in response.get('conteudo', []):
                    if item.get('estabelecimento', {}).get('nomeFantasia'):
                        latitude_formatada = "{:.7f}".format(item['estabelecimento']['endereco']['latitude'])
                        longitude_formatada = "{:.7f}".format(item['estabelecimento']['endereco']['longitude'])
                        item['estabelecimento']['endereco']['latitude'] = latitude_formatada
                        item['estabelecimento']['endereco']['longitude'] = longitude_formatada
                        item['item_description'] = item_description  # Adiciona a descrição do item aos resultados
                        response_list.append(item)

    return templates.TemplateResponse("results.html", {"request": request, "data": response_list})


