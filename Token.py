import requests

# Replace these placeholders with your actual values
api_key = 'da360b6b-874a-40ca-8d7a-d94687c27f9a'
api_secret = 'q89ty6fekg'
redirect_uri = 'https://127.0.0.1'
auth_code = 'the_code_you_received_from_redirect'

token_url = 'https://api.upstox.com/v2/login/authorization/token'
payload = {
    'code': auth_code,
    'client_id': api_key,
    'client_secret': api_secret,
    'redirect_uri': redirect_uri,
    'grant_type': 'authorization_code'
}

response = requests.post(token_url, data=payload)

if response.status_code == 200:
    token_data = response.json()
    access_token = token_data.get('access_token')
    print(f"Your access token is: {access_token}")
else:
    print(f"Failed to get access token. Response: {response.text}")
