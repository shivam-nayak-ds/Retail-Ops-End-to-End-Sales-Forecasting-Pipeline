import requests
import pandas as pd
import time
import random
import sys

# API Configuration
API_URL = "http://127.0.0.1:8000/forecast"

def run_simulation(sample_count: int = 20):
    """
    Simulates production traffic by sending requests to the forecasting API.
    Injects synthetic drift to test monitoring capabilities.
    """
    print(f"Starting production traffic simulation. Target: {sample_count} requests.")
    
    try:
        # Load ingestion artifacts for sampling
        test_df = pd.read_parquet("artifacts/data_ingestion/test.parquet")
        samples = test_df.sample(sample_count).to_dict(orient="records")
        
        for i, sample in enumerate(samples):
            # Clean data: Replace NaN with 0.0 for JSON compliance
            sample = {k: (0.0 if pd.isna(v) else v) for k, v in sample.items()}

            # Synthetic drift injection: Scaling competition distance
            sample['CompetitionDistance'] = sample['CompetitionDistance'] * random.uniform(1.1, 1.5)
            
            print(f"[{i+1}/{sample_count}] Dispatching request for Store ID: {sample['Store']}")
            
            try:
                # Send single object, not a list
                response = requests.post(API_URL, json=sample, timeout=10)
                
                if response.status_code == 200:
                    print(f"Request successful. Status: {response.json().get('status')}")
                else:
                    print(f"Request failed. Status Code: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as req_e:
                print(f"Network error during request: {str(req_e)}")
            
            time.sleep(0.5) 

        print("Simulation cycle completed successfully.")

    except FileNotFoundError:
        print("Error: Test data artifact not found. Ensure the ingestion pipeline has been executed.")
    except Exception as e:
        print(f"Unexpected error during simulation: {str(e)}")

if __name__ == "__main__":
    run_simulation()
