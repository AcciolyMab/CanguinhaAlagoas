# Description: Router for geolocation endpoints (Google Maps API) geolocation_router.py
import requests
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/geolocation", response_class=JSONResponse)
async def get_geolocation(address: str):
    GOOGLE_MAPS_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": "AIzaSyCMIgk-1yvr5wqGJrDbjPDAa7BmN6OB6Sk"}

    # Realizar a requisição à Google Maps API
    res = requests.get(GOOGLE_MAPS_API_URL, params=params)

    if res.status_code != 200:
        raise HTTPException(
            status_code=res.status_code, detail="Google Maps API request failed"
        )

    data = res.json()

    # Caso o status retornado pela Google Maps API seja 'OK', recuperar a latitude e a longitude
    if data["status"] == "OK":
        latitude = data["results"][0]["geometry"]["location"]["lat"]
        longitude = data["results"][0]["geometry"]["location"]["lng"]
    else:
        raise HTTPException(
            status_code=400, detail="Failed to get geolocation from Google Maps API"
        )

    return {"latitude": latitude, "longitude": longitude}
