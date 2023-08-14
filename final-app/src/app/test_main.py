from fastapi.testclient import TestClient
from datetime import datetime as dt

try:
    from main import app
except:
    from app.main import app

client = TestClient(app)

now = dt.now()
date = now.strftime("%Y-%m-%d")
hour = now.hour

def test_predict_daily():
    response = client.post(f"/prediction/daily/{date}/5")
    assert response.status_code == 200
    assert isinstance(response.json()['Results'], list), 'Results wrong type!'
    assert isinstance(response.json()['Results'][0], float), 'Value of result is wrong type!'

def test_predict_hourly():
    response = client.post(f"/prediction/hourly/{date}/{hour}/24")
    assert response.status_code == 200
    assert isinstance(response.json()['Results'], list), 'Results wrong type!'
    assert isinstance(response.json()['Results'][0], float), 'Value of result is wrong type!'