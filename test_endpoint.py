import urllib.request
import json
 
# Your primary key (replace with secondary if you want)
api_key = '5p9ggpJc34NY3FH65JmJgIOiaWsBpre2cBH4r38EnmHsCKt0iuAmJQQJ99BFAAAAAAAAAAAAINFRAZML3sA3'
 
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")
 
# Example input data - adjust this according to your model's expected input format
data = {
        "text": "This is a test sentence to evaluate the DeBERTa endpoint."
    }
 
# Encode the data to JSON bytes
body = str.encode(json.dumps(data))
 
# Your endpoint URL
url = 'http://194.171.191.227:30526/api/v1/endpoint/deberta-endpoint/score'
 
# Set the request headers including authorization with your API key
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': 'Bearer ' + api_key
}
 
# Prepare the request
req = urllib.request.Request(url, body, headers)
 
try:
    # Send the request and read the response
    response = urllib.request.urlopen(req)
    result = response.read()
    print("Response from DeBERTa endpoint:")
    print(result.decode("utf-8"))
 
except urllib.error.HTTPError as error:
    # If there's an HTTP error, print status and details
    print("The request failed with status code:", error.code)
    print("Headers:", error.info())
    print("Error response:", error.read().decode("utf8", 'ignore'))