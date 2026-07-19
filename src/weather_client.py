import requests, pandas as pd, numpy as np


def fetch_climate(lat, lon, days=30):
    try:
        url = 'https://api.open-meteo.com/v1/forecast'
        params = {
            'latitude': lat, 'longitude': lon,
            'past_days': days, 'timezone': 'auto',
            'hourly': 'temperature_2m,relative_humidity_2m,precipitation'
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()['hourly']
        df = pd.DataFrame({
            'time': pd.to_datetime(data['time']),
            'temperature_2m': data['temperature_2m'],
            'relative_humidity_2m': data['relative_humidity_2m'],
            'precipitation': data['precipitation']
        })
        df.set_index('time', inplace=True)
        df = df.resample('D').mean()
        result = df.tail(30).values.astype(np.float32)
        print(f"Climate data fetched. Shape: {result.shape}")
        return result
    except Exception as e:
        print(f"Weather API error: {e}")
        return None


if __name__ == '__main__':
    arr = fetch_climate(17.36, 78.48)
    if arr is not None:
        print(f"Shape: {arr.shape}")
        print(arr[:3])