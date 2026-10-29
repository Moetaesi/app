from datetime import datetime
import json
import requests

LAT = 51.0447
LON = -114.0719
API_VARIABLES = ["temperature_2m", "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "precipitation", "cloud_cover"]

def cache_weather():
    url = "https://open-meteo.com"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(API_VARIABLES),
        "timezone": "America/Edmonton", 
        "forecast_days": 16,
    }
    try:
        # GitHub Actions servers use clean GitHub IP pools that Open-Meteo does not block
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if "hourly" in data:
                with open("cached_weather.json", "w") as f:
                    json.dump(data["hourly"], f)
                print("✅ Weather cache successfully updated!")
                return
        print(f"❌ Failed to fetch data: {response.status_code}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == "__main__":
    cache_weather()
