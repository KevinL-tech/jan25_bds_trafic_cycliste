from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from prometheus_client import Summary
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

app = FastAPI(title="Traffic Cycliste Prediction API",
              description="API for predicting bicycle traffic in Paris",
              version="1.0.0")

# JWT Authentication Configuration
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"  # In production, use a secure random key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# User database
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

# Simple user database - in production, use a real database
users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": pwd_context.hash("secret"),
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderland",
        "email": "alice@example.com",
        "hashed_password": pwd_context.hash("password"),
        "disabled": False,
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Prometheus instrumentation setup
Instrumentator().instrument(app).expose(app)
inference_time_summary = Summary('inference_time_seconds', 'Time taken for inference')

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "hgb_regressor_0.5_500.pkl"
ENCODER_PATH = MODELS_DIR / "one_hot_encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    print(f"Error loading model and/or one hot encoder: {e}")
    raise e

class ModelInputParameters(BaseModel):
    jour: int
    mois: int
    annee: int
    heure: int
    jour_semaine: int
    jour_ferie: bool
    vacances_scolaires: bool
    site_comptage: str

class PredictionResponse(BaseModel):
    prediction: float
    input_parameters: ModelInputParameters

@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
def index():
    return {"message": "Welcome to the Traffic Cycliste Prediction API"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.post("/predict", response_model=PredictionResponse)
def predict(input_parameters: ModelInputParameters, current_user: User = Depends(get_current_active_user)):
    """Endpoint for making predictions (requires authentication)"""
    try:
        # Transform input into format expected by model
        x = pd.DataFrame([{
            "Nom du site de comptage": input_parameters.site_comptage,
            "Jour": input_parameters.jour,
            "Mois": input_parameters.mois,
            "Année": input_parameters.annee,
            "Heure": input_parameters.heure,
            "Jour_semaine": input_parameters.jour_semaine,
            "Jour férié": int(input_parameters.jour_ferie),
            "Vacances scolaires": int(input_parameters.vacances_scolaires),
        }])

        # One Hot Encoding
        X_p = encoder.transform(x)

        # Prediction
        with inference_time_summary.time():
            y_p = np.expm1(model.predict(X_p))

        return PredictionResponse(
            prediction=y_p,
            input_parameters=input_parameters
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
