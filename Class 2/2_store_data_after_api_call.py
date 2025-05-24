import requests
import json
import csv

# --- Step 1: API Call ---
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 23.8103,     # Dhaka latitude
    "longitude": 90.4125,    # Dhaka longitude
    "current_weather": True
}

response = requests.get(url, params=params)

# --- Step 2: Check and Extract Data ---
if response.status_code == 200:
    data = response.json()
    weather = data["current_weather"]

    print(f" Weather in Dhaka:")
    print(f"Temperature: {weather['temperature']}°C")
    print(f"Windspeed: {weather['windspeed']} km/h")
    print(f"Time: {weather['time']}")

    # --- Step 3a: Save to JSON ---
    with open("weather_dhaka.json", "w") as json_file:
        json.dump(weather, json_file, indent=4)
        print(" Saved to weather_dhaka.json")

    # --- Step 3b: Save to CSV ---
    with open("weather_dhaka.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=weather.keys())
        writer.writeheader()
        writer.writerow(weather)
        print(" Saved to weather_dhaka.csv")

else:
    print(" Error fetching weather data:", response.status_code)