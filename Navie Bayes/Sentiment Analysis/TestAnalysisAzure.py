import urllib.request
import json

data = {
        "Inputs": {
                "awesome movie":
                [
                    {
                            'text_column': "",   
                    }
                ],
        },
    "GlobalParameters":  {
    }
}

body = str.encode(json.dumps(data))

url = 'https://ussouthcentral.services.azureml.net/workspaces/6d2357915b024bd682107b7c3f624afa/services/bd88bd696d064b759b5fc49d9efd4a80/execute?api-version=2.0&format=swagger'
api_key = 'v2Tjw7tio7eegwJne7Bw3M7LMn4uhXsyb2VqnIBGeBM1gRcHhP7kgveUdl4EH9Ex6mNhtv373W7FAJ+bWkBhXQ==' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))