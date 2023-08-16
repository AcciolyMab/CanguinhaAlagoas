# utils.py
import requests
import json

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
