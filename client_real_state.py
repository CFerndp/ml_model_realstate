"""
Cliente sencillo para la API de precios de vivienda.
Server corriendo: 
    uvicorn app.main:app --reload
"""

import requests
from pprint import pprint

BASE_URL = "http://127.0.0.1:8000"


def check_health():
    """Comprueba el estado de la API."""
    url = f"{BASE_URL}/health"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def predict_price(base_data: dict):
    """
    Llama al endpoint /predict_price.

    base_data debe tener las claves:
        med_inc, house_age, ave_rooms, ave_bedrooms,
        population, ave_occup, latitude, longitude
    """
    url = f"{BASE_URL}/predict_price"
    resp = requests.post(url, json=base_data)
    resp.raise_for_status()
    return resp.json()


def feature_curve(feature_name: str, base_data: dict,
                  min_value: float, max_value: float,
                  num_points: int = 20):
    """
    Llama al endpoint /feature_curve para obtener la curva
    de precios al variar una característica.

    Parámetros:
        feature_name: nombre de la variable a variar
                      (med_inc, house_age, ave_rooms, ave_bedrooms,
                       population, ave_occup, latitude, longitude)
        base_data: valores base del resto de variables
        min_value, max_value: rango de la variable a estudiar
        num_points: número de puntos en la curva
    """
    url = f"{BASE_URL}/feature_curve"
    payload = {
        "feature_name": feature_name,
        "base": base_data,
        "min_value": min_value,
        "max_value": max_value,
        "num_points": num_points,
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def main():
    print("== Health check ==")
    try:
        health = check_health()
        pprint(health)
    except Exception as e:
        print("Error al conectar con la API:", e)
        return

    base_input = {
        "med_inc": 4.0,       # ingresos medios (decenas de miles de $)
        "house_age": 20.0,    # edad media
        "ave_rooms": 5.0,     # habitaciones medias
        "ave_bedrooms": 1.0,  # dormitorios medios
        "population": 1000.0, # población
        "ave_occup": 3.0,     # ocupantes medios
        "latitude": 34.0,     # latitud
        "longitude": -118.0,  # longitud
    }

    print("\n== Predicción puntual ==")
    pred = predict_price(base_input)
    print(f"Precio estimado: {pred['predicted_price_formatted']}")
    print("Detalles:")
    pprint(pred["details"])

    print("\n== Curva de precios vs med_inc ==")
    curve = feature_curve(
        feature_name="med_inc",
        base_data=base_input,
        min_value=1.0,
        max_value=10.0,
        num_points=15,
    )

    x_vals = curve["x_values"]
    prices = curve["prices"]

    for x, p in zip(x_vals[:5], prices[:5]):  
        print(f"med_inc={x:.2f} -> precio ≈ {p/1000:.1f} mil $")


if __name__ == "__main__":
    main()
