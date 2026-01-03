import requests
import pandas as pd
import os
import time
import json

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/insurance.csv')
API_URL = "http://127.0.0.1:8000/predict"

def verify_live_api():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Sample 5 records to send to the live API
    samples = df.sample(5).to_dict('records')
    
    print(f"\nSending {len(samples)} requests to {API_URL}...\n")
    
    for i, record in enumerate(samples):
        # Construct payload matching API schema
        payload = {
            "Age": int(record['Age']),
            "Diabetes": int(record['Diabetes']),
            "BloodPressureProblems": int(record['BloodPressureProblems']),
            "AnyTransplants": int(record['AnyTransplants']),
            "AnyChronicDiseases": int(record['AnyChronicDiseases']),
            "Height": int(record['Height']),
            "Weight": int(record['Weight']),
            "KnownAllergies": int(record['KnownAllergies']),
            "HistoryOfCancerInFamily": int(record['HistoryOfCancerInFamily']),
            "NumberOfMajorSurgeries": int(record['NumberOfMajorSurgeries'])
        }
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            duration = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                predicted = result['premium_price']
                actual = record['PremiumPrice']
                error = abs(predicted - actual)
                error_pct = (error / actual) * 100
                
                status_icon = "jq " if error_pct < 30 else "⚠️"
                
                print(f"[{i+1}] {status_icon} Actual: ${actual:,.2f} | Predicted: ${predicted:,.2f} | Diff: {error_pct:.1f}%")
            else:
                print(f"[{i+1}] ❌ Status: {response.status_code} | Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"[{i+1}] Connection Error! Is uvicorn running?")
            return

    print("\n✅ Live requests sent. Check your uvicorn terminal for access logs.")

if __name__ == "__main__":
    verify_live_api()
