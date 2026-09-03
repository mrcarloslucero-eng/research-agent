import requests
import os

api_key = os.environ.get('OPENWEATHER_API_KEY')

print(f"Key your script will use: {api_key}")
print()

response = requests.get(
    "https://api.openweathermap.org/data/2.5/weather",
    params={'q': 'McKinney', 'appid': api_key, 'units': 'imperial'}
)

if response.status_code == 200:
    data = response.json()
    print(f"✅ This key WORKS")
    print(f"   City: {data['name']}")
    print(f"   Temp: {data['main']['temp']}°F")
else:
    print(f"❌ This key DOES NOT work")
    print(f"   Error: {response.json().get('message')}")