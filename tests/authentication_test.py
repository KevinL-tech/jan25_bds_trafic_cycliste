import requests

API_URL = "http://localhost:8000"

# Test data
VALID_USER = {"username": "johndoe", "password": "secret"}
INVALID_USER = {"username": "johndoe", "password": "wrongpassword"}
NONEXISTENT_USER = {"username": "nonexistent", "password": "password"}

# Test data for prediction endpoint
MODEL_INPUT_PARAMS = {
    "jour": 24,
    "mois": 12,
    "annee": 2025,
    "heure": 8,
    "jour_semaine": 2,
    "jour_ferie": False,
    "vacances_scolaires": True,
    "site_comptage": "Pont des Invalides"
}

def get_token(username, password):
    """Helper function to get a token"""
    response = requests.post(
        f"{API_URL}/login",
        data={"username": username, "password": password}
    )
    if response.status_code == 200:
        return response.json().get("access_token")
    return None

def test_login_success():
    """Test successful login with valid credentials"""
    response = requests.post(
        f"{API_URL}/login",
        data={"username": VALID_USER["username"], "password": VALID_USER["password"]}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_invalid_password():
    """Test failed login with an invalid password"""
    response = requests.post(
        f"{API_URL}/login",
        data={"username": INVALID_USER["username"], "password": INVALID_USER["password"]}
    )
    assert response.status_code == 401
    assert "detail" in response.json()

def test_login_nonexistent_user():
    """Test failed login with nonexistent user"""
    response = requests.post(
        f"{API_URL}/login",
        data={"username": NONEXISTENT_USER["username"], "password": NONEXISTENT_USER["password"]}
    )
    assert response.status_code == 401
    assert "detail" in response.json()

def test_predict_with_token():
    """Test accessing /predict endpoint with valid token"""
    token = get_token(VALID_USER["username"], VALID_USER["password"])
    assert token is not None
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{API_URL}/predict",
        json=MODEL_INPUT_PARAMS,
        headers=headers
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["input_parameters"] == MODEL_INPUT_PARAMS

def test_predict_without_token():
    """Test accessing /predict endpoint without a token"""
    response = requests.post(
        f"{API_URL}/predict",
        json=MODEL_INPUT_PARAMS
    )
    assert response.status_code == 401
    assert "detail" in response.json()

def test_predict_with_invalid_token():
    """Test accessing /predict endpoint with an invalid token"""
    headers = {"Authorization": "Bearer invalidtoken"}
    response = requests.post(
        f"{API_URL}/predict",
        json=MODEL_INPUT_PARAMS,
        headers=headers
    )
    assert response.status_code == 401
    assert "detail" in response.json()

def test_users_me_with_token():
    """Test accessing /users/me endpoint with valid token"""
    token = get_token(VALID_USER["username"], VALID_USER["password"])
    assert token is not None
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        f"{API_URL}/users/me",
        headers=headers
    )
    assert response.status_code == 200
    assert response.json()["username"] == VALID_USER["username"]
    assert "email" in response.json()
    assert "full_name" in response.json()

def test_users_me_without_token():
    """Test accessing /users/me endpoint without token"""
    response = requests.get(f"{API_URL}/users/me")
    assert response.status_code == 401
    assert "detail" in response.json()