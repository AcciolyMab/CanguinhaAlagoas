# main.py
import json
<<<<<<< HEAD
=======

>>>>>>> 2090620 (Primeiro Commit)
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
<<<<<<< HEAD
from geolocation_router import router as geolocation_router
from utils import consultarProduto
from optimization_router import optimization_router

app = FastAPI()  # Cria a instância do FastAPI

app.include_router(geolocation_router)
app.include_router(optimization_router)
=======

from geolocation_router import router as geolocation_router

app = FastAPI()

>>>>>>> 2090620 (Primeiro Commit)

class Location(BaseModel):
    latitude: float
    longitude: float


<<<<<<< HEAD
=======
app.include_router(geolocation_router)

>>>>>>> 2090620 (Primeiro Commit)
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

<<<<<<< HEAD
global response_list
response_list = []
=======

>>>>>>> 2090620 (Primeiro Commit)
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

<<<<<<< HEAD
@app.post("/consultarProduto/", response_class=HTMLResponse)
async def consultar_produto(request: Request):
    global response_list  # Refer to the global variable
=======

@app.post("/consultarProduto/", response_class=HTMLResponse)
async def consultar_produto(request: Request):
>>>>>>> 2090620 (Primeiro Commit)
    form_data = await request.form()

    formData = json.loads(form_data.get("formData"))
    dias = int(form_data.get("dias"))
    raio = int(form_data.get("raio"))
    address = form_data.get("address")
    latitude = float(form_data.get("latitude"))
    longitude = float(form_data.get("longitude"))

<<<<<<< HEAD
    # Clear the existing response list
    response_list.clear()
=======
    response_list = []
>>>>>>> 2090620 (Primeiro Commit)

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
<<<<<<< HEAD
=======


@app.get("/consultarProduto/", response_class=HTMLResponse)
def consultarProduto(gtin_code: str, latitude: float, longitude: float, raio: int, dias: int):
    url = 'http://api.sefaz.al.gov.br/sfz-economiza-alagoas-api/api/public/produto/pesquisa'

    data = {
        "produto": {
            "gtin": gtin_code
        },
        "estabelecimento": {
            "geolocalizacao": {
                "latitude": latitude,
                "longitude": longitude,
                "raio": raio
            }
        },
        "dias": dias,
        "pagina": 1,
        "registrosPorPagina": 50
    }
    headers = {
        "Content-Type": "application/json",
        "AppToken": "ad909a7a6f0d6a130941ae2a9706eec58c0bb65d"
    }

    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None
>>>>>>> 2090620 (Primeiro Commit)
