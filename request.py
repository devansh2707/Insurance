import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age':20, 'sex':1, 'bmi':62.3,'children':1,'smoker':0,'region':2})

print(r.json())