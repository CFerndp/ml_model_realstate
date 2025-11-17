from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# --------- Carga del modelo ---------
BASE_DIR = Path(__file__).resolve().parent
artifact = joblib.load(BASE_DIR / "housing_model.joblib")
model = artifact["model"]
feature_names = artifact["feature_names"]
metrics = artifact["metrics"]

# Orden de características tal y como las usamos en el modelo
FEATURE_ORDER = [
    "med_inc",
    "house_age",
    "ave_rooms",
    "ave_bedrooms",
    "population",
    "ave_occup",
    "latitude",
    "longitude",
]

# --------- Definición de la app ---------
app = FastAPI(
    title="Housing Price API",
    description="API de ejemplo para predecir precios de vivienda (dataset California Housing)",
    version="1.0.0",
)

# Static & templates (para el dashboard)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# --------- Esquemas Pydantic ---------
class HousingInput(BaseModel):
    med_inc: float = Field(
        ...,
        description="Ingresos medios del distrito (en decenas de miles de dólares)",
    )
    house_age: float = Field(
        ...,
        description="Edad media de las casas en el distrito",
    )
    ave_rooms: float = Field(
        ...,
        description="Número medio de habitaciones por vivienda",
    )
    ave_bedrooms: float = Field(
        ...,
        description="Número medio de dormitorios por vivienda",
    )
    population: float = Field(
        ...,
        description="Población del distrito",
    )
    ave_occup: float = Field(
        ...,
        description="Número medio de ocupantes por vivienda",
    )
    latitude: float = Field(
        ...,
        description="Latitud del distrito",
    )
    longitude: float = Field(
        ...,
        description="Longitud del distrito",
    )


class HousingPrediction(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    details: Dict[str, float]


class FeatureCurveRequest(BaseModel):
    feature_name: str = Field(
        ...,
        description=f"Nombre de la característica a variar. Debe ser una de: {FEATURE_ORDER}",
    )
    base: HousingInput
    min_value: float
    max_value: float
    num_points: int = Field(20, ge=2, le=200)


class FeatureCurveResponse(BaseModel):
    feature_name: str
    x_values: List[float]
    prices: List[float]


# --------- Rutas API ---------
@app.get("/health")
def health_check():
    return {"status": "ok", "model_metrics": metrics}


@app.post("/predict_price", response_model=HousingPrediction)
def predict_price(input_data: HousingInput):
    # Orden de features según FEATURE_ORDER
    X = np.array(
        [
            [
                input_data.med_inc,
                input_data.house_age,
                input_data.ave_rooms,
                input_data.ave_bedrooms,
                input_data.population,
                input_data.ave_occup,
                input_data.latitude,
                input_data.longitude,
            ]
        ]
    )

    # Predicción en unidades de 100k dólares
    y_100k = float(model.predict(X)[0])
    price_dollars = y_100k * 100_000.0

    return HousingPrediction(
        predicted_price=price_dollars,
        predicted_price_formatted=f"{price_dollars:,.0f} $",
        details={"y_100k": y_100k},
    )


@app.post("/feature_curve", response_model=FeatureCurveResponse)
def feature_curve(req: FeatureCurveRequest):
    """
    Devuelve cómo varía el precio estimado al cambiar una característica
    (feature_name) en un rango [min_value, max_value], fijando el resto
    con los valores base indicados.
    """
    if req.feature_name not in FEATURE_ORDER:
        raise HTTPException(
            status_code=400,
            detail=f"feature_name debe ser uno de: {FEATURE_ORDER}",
        )

    # Aseguramos orden min <= max
    min_v, max_v = sorted([req.min_value, req.max_value])

    if req.num_points < 2:
        raise HTTPException(
            status_code=400, detail="num_points debe ser mayor o igual que 2"
        )

    # Vector base en el orden correcto
    base_vals = [
        req.base.med_inc,
        req.base.house_age,
        req.base.ave_rooms,
        req.base.ave_bedrooms,
        req.base.population,
        req.base.ave_occup,
        req.base.latitude,
        req.base.longitude,
    ]

    feat_index = FEATURE_ORDER.index(req.feature_name)

    # Generamos los puntos en el eje X (valores de la feature)
    step = (max_v - min_v) / (req.num_points - 1)
    x_values = [min_v + i * step for i in range(req.num_points)]

    prices: List[float] = []

    for val in x_values:
        vals = base_vals.copy()
        vals[feat_index] = val
        X = np.array([vals])
        y_100k = float(model.predict(X)[0])
        price_dollars = y_100k * 100_000.0
        prices.append(price_dollars)

    return FeatureCurveResponse(
        feature_name=req.feature_name,
        x_values=x_values,
        prices=prices,
    )


# --------- Dashboard web ---------
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

