import requests
import json
# Define the input data as a JSON object
data = {
    "headdirection": "left",
    "facebundles": "A",
    "goods": "yes",
    "wrapping": "A",
    "haircolor": "brown",
    "samplescollected": "no",
    "ageatdeath": "adult",
    "depth": 1.5,
    "length": 10.2
}
# Send a POST request to the API with the Content-Type header set to 'application/json'
headers = {'Content-type': 'application/json'}
response = requests.post(
    "http://http://127.0.0.1:5000/predict", headers=headers, json=data)
# Print the response
print(json.loads(response.text))
