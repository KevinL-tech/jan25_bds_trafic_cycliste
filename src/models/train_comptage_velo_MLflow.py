
import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Argument parser pour MLproject
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.5)
parser.add_argument("--max_iter", type=int, default=500)
args = parser.parse_args()

params = {
    "learning_rate": args.learning_rate,
    "max_iter": args.max_iter,
    "random_state": 42
}

# Authentification pour MLflow (env déjà injecté via GitHub Actions)
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["MLFLOW_TRACKING_PASSWORD"]

# Configuration du tracking
mlflow.set_tracking_uri("https://dagshub.com/KevinL-tech/jan25_bds_trafic_cycliste.mlflow")
mlflow.set_experiment("modele_comptage_velo")

# Chargement des données
df = pd.read_csv('data/processed/lieu-compteur-one-hot-encoded.csv', index_col=0)
y = np.log1p(df["Comptage horaire"])
X = df.drop(columns=["Comptage horaire"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = HistGradientBoostingRegressor(**params)

# Tracking MLflow
with mlflow.start_run(run_name="HGB_notebook_style") as run:
    mlflow.log_params(params)
    mlflow.set_tags({
        "modele": "HistGradientBoosting",
        "données": "lieu-compteur-one-hot-encoded CSV",
        "objectif": "comptage horaire vélo par compteurs"
    })

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Signature pour éviter le warning
    example_input = X_test.iloc[:1]
    mlflow.sklearn.log_model(model, "model", input_example=example_input)

    print(f"Modèle entraîné avec RMSE={rmse:.4f} et R²={r2:.4f}")

    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "comptage_velo_horaire")