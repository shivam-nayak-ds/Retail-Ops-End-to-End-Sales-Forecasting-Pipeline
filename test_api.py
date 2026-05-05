import requests
import json

# FastAPI URL
URL = "http://127.0.0.1:8000/predict"

# Sample data mimicking the RetailSalesData schema
sample_data = [
    {
        "Store": 1,
        "DayOfWeek": 5,
        "Date": "2015-07-31",
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 1,
        "StoreType": "c",
        "Assortment": "a",
        "CompetitionDistance": 1270.0,
        "sales_lag_1": 5000.0,
        "sales_lag_7": 4800.0,
        "sales_lag_30": 4500.0,
        "sales_roll_mean_7": 4700.0
    },
    {
        "Store": 1,
        "DayOfWeek": 7,
        "Date": "2015-08-02",
        "Open": 0,  # Store Closed
        "Promo": 0,
        "StateHoliday": "0",
        "SchoolHoliday": 0,
        "StoreType": "c",
        "Assortment": "a",
        "CompetitionDistance": 1270.0,
        "sales_lag_1": 0.0,
        "sales_lag_7": 0.0,
        "sales_lag_30": 0.0,
        "sales_roll_mean_7": 0.0
    }
]

try:
    print(f"Sending request to {URL}...")
    response = requests.post(URL, json=sample_data)
    
    if response.status_code == 200:
        result = response.json()
        print("\n[SUCCESS] Prediction Success!")
        print(f"Predicted Sales: {result['predictions'][0]:.2f}")
    else:
        print(f"\n[ERROR] Failed with status code: {response.status_code}")
        print(f"Error: {response.text}")

except Exception as e:
    print(f"\n[CONNECTION ERROR] {str(e)}")
    print("Ensure the FastAPI server is running (python app_fastapi.py)")
