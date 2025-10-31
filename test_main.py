from fastapi.testclient import TestClient
from main import app # Importas tu app de FastAPI

client = TestClient(app)

def test_login_exitoso():
    """
    Prueba que un usuario con credenciales correctas puede iniciar sesión.
    """
    response = client.post("/login", json={
        "email": "efran@gmail.com",
        "password": "12345678"
    })
    assert response.status_code == 200
    json_response = response.json()
    assert "access_token" in json_response
    assert json_response["token_type"] == "bearer"

def test_login_fallido():
    """
    Prueba que un usuario con credenciales incorrectas no puede iniciar sesión.
    """
    response = client.post("/login", json={
        "email": "efran131@example.com",
        "password": "password_incorrecto"
    })
    assert response.status_code == 400
    assert response.json() == {"detail": "Credenciales incorrectas: Correo o contraseña inválidos."}