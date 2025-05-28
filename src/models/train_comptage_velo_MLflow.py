
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv

# Charger les identifiants DagsHub depuis le fichier .env

os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

# MLflow tracking sur DagsHub
mlflow.set_tracking_uri("https://dagshub.com/KevinL-tech/jan25_bds_trafic_cycliste.mlflow")
mlflow.set_experiment("modele_comptage_velo")

# Charger les données encodées
df = pd.read_csv('data/processed/lieu-compteur-one-hot-encoded.csv', index_col=0)

# Transformation log1p pour lisser la cible
y = np.log1p(df["Comptage horaire"])
X = df.drop(columns=["Comptage horaire"])

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Paramètres du modèle
params = {"learning_rate": 0.5, "max_iter": 500, "random_state" : 42}
model = HistGradientBoostingRegressor(**params)

# Dossier de sauvegarde
MODELS_DIR = Path("models")
model_filename = MODELS_DIR / f"hgb_regressor_{params['learning_rate']}_{params['max_iter']}.pkl"

# Tracking MLflow
with mlflow.start_run(run_name="HGB_notebook_style"):
    mlflow.log_params(params)
    mlflow.set_tags({
        "modele": "HistGradientBoosting",
        "données": "lieu-compteur-one-hot-encoded CSV",
        "objectif": "comptage horaire vélo par compteurs"
    })

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_final = np.expm1(y_pred)
    y_test_final = np.expm1(y_test)
    y_pred_final = np.maximum(y_pred_final, 0)
    y_pred_final = np.round(y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_test_final, y_pred_final))
    r2 = r2_score(y_test_final, y_pred_final)


    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(model, "model")

    print(f"Modèle entraîné avec RMSE={rmse:.4f} et R²={r2:.4f}")
