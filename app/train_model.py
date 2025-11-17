from pathlib import Path

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_and_save_model():
    # Carga del dataset de viviendas de California
    data = fetch_california_housing()
    X = data.data
    y = data.target  # MedHouseVal (en unidades de 100.000 dólares)
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: escalado + RandomForest (modelo sencillo y robusto)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE (en unidades de 100k $): {mae:.3f}")
    print(f"R²: {r2:.3f}")

    artifact = {
        "model": pipe,
        "feature_names": feature_names,
        "target_unit": "100k_dollars",
        "metrics": {"mae_100k": mae, "r2": r2},
    }

    artifact_path = Path(__file__).resolve().parent / "housing_model.joblib"
    joblib.dump(artifact, artifact_path)

    print(f"Modelo guardado en {artifact_path}")


if __name__ == "__main__":
    train_and_save_model()
