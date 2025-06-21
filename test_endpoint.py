# import os
# import json
# import ssl
# import urllib.request

# def allowSelfSignedHttps(allowed):
#     # bypass the server certificate verification on client side
#     if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
#         ssl._create_default_https_context = ssl._create_unverified_context


# allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.


# # Define the endpoint URL
# url = "http://194.171.191.227:30526/api/v1/endpoint/deberta-endpoint/score"
# api_key = "5p9ggpJc34NY3FH65JmJgIOiaWsBpre2cBH4r38EnmHsCKt0iuAmJQQJ99BFAAAAAAAAAAAAINFRAZML3sA3"


# # Define the headers
# headers = {
#     'Authorization': ('Bearer ' + api_key),
#     'Content-Type': 'application/json',
#     # 'Accept': 'application/json',
#     "azureml-model-deployment": "blue",
# }

# # Define your input data (replace with your actual input)
# data = {
#     "text": "I am so excited about this new project, it's going to be amazing!"
# }
# # data = json.dumps(data)
# body = str.encode(json.dumps(data))

# req = urllib.request.Request(
#     url,
#     body,
#     headers,
# )

# try:
#     response = urllib.request.urlopen(req)

#     result = response.read()
#     print(result)
# except urllib.error.HTTPError as error:
#     print("The request failed with status code: " + str(error.code))

#     # Print the headers - they include the request ID and the timestamp,
#     # which are useful for debugging the failure
#     print(error.info())
#     print(error.read().decode("utf8", 'ignore'))





import os
import json
import ssl
import urllib.request
import urllib.error
import socket
import time
from typing import Dict, Any, Optional

def allowSelfSignedHttps(allowed: bool) -> None:
    """
    Bypass server certificate verification on client side.
    
    Args:
        allowed: Whether to allow self-signed certificates
    """
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

def test_connectivity(host: str, port: int, timeout: int = 10) -> bool:
    """
    Test basic TCP connectivity to the endpoint.
    
    Args:
        host: Target hostname or IP
        port: Target port
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        socket.setdefaulttimeout(timeout)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Connectivity test failed: {e}")
        return False

def make_api_request(url: str, data: Dict[str, Any], headers: Dict[str, str], 
                    timeout: int = 30, max_retries: int = 3) -> Optional[bytes]:
    """
    Make API request with robust error handling and retry logic.
    
    Args:
        url: API endpoint URL
        data: Request payload
        headers: HTTP headers
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Response data or None if all attempts fail
    """
    body = str.encode(json.dumps(data))
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")
            
            req = urllib.request.Request(url, body, headers)
            
            # Set custom timeout
            response = urllib.request.urlopen(req, timeout=timeout)
            result = response.read()
            
            print("Request successful!")
            return result
            
        except urllib.error.HTTPError as error:
            print(f"HTTP Error {error.code}: {error.reason}")
            print("Response headers:", error.info())
            try:
                error_body = error.read().decode("utf8", 'ignore')
                print("Error response:", error_body)
            except:
                pass
            
            # Don't retry on client errors (4xx)
            if 400 <= error.code < 500:
                break
                
        except urllib.error.URLError as error:
            print(f"URL Error: {error.reason}")
            
        except socket.timeout:
            print("Request timed out")
            
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    print("All retry attempts failed")
    return None

def main():
    """Main execution function with comprehensive error handling."""
    
    # Configuration
    url = "http://194.171.191.227:30526/api/v1/endpoint/deberta-endpoint/score"
    api_key = "5p9ggpJc34NY3FH65JmJgIOiaWsBpre2cBH4r38EnmHsCKt0iuAmJQQJ99BFAAAAAAAAAAAAINFRAZML3sA3"
    
    # Parse URL for connectivity test
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
    
    print(f"Testing connectivity to {host}:{port}")
    
    # Test basic connectivity first
    if not test_connectivity(host, port, timeout=10):
        print(f"❌ Cannot establish TCP connection to {host}:{port}")
        print("\nPossible solutions:")
        print("1. Check if the server is running and accessible")
        print("2. Verify firewall settings")
        print("3. Check if you're behind a proxy or VPN")
        print("4. Confirm the URL and port are correct")
        return
    
    print(f"✅ TCP connection to {host}:{port} successful")
    
    # Enable self-signed certificates
    allowSelfSignedHttps(True)
    
    # Prepare headers
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'User-Agent': 'Python-urllib/3.12',
        "azureml-model-deployment": "blue",
    }
    
    # Prepare request data
    data = {
        "text": "I am so excited about this new project, it's going to be amazing!"
    }
    
    print("\nSending API request...")
    result = make_api_request(url, data, headers, timeout=30, max_retries=3)
    
    if result:
        try:
            # Try to parse as JSON for pretty printing
            json_result = json.loads(result.decode('utf-8'))
            print("\n✅ Response received:")
            print(json.dumps(json_result, indent=2))
        except json.JSONDecodeError:
            print("\n✅ Response received (raw):")
            print(result.decode('utf-8', errors='ignore'))
    else:
        print("\n❌ Request failed after all retry attempts")

if __name__ == "__main__":
    main()