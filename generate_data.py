# generate_data.py
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime

def generate_simulated_data(
    output_dir='data',
    start_year=2000,
    end_year=datetime.now().year + 1, # Generate historical up to current year, forecasts for current + 1 year
    entries_per_season_per_region=5,
):
    """
    Generates simulated agricultural yield, weather, and market price data with more realistic Nigerian ranges.
    """
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    regions = [
        'Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 'Borno',
        'Cross River', 'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 'Gombe', 'Imo',
        'Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos',
        'Nasarawa', 'Niger', 'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau', 'Rivers',
        'Sokoto', 'Taraba', 'Yobe', 'Zamfara'
    ]
    crop_types = ['Maize', 'Cassava', 'Rice', 'Yam', 'Sorghum', 'Millet', 'Groundnut', 'Beans']
    seasons = ['Wet', 'Dry']

    # yield ranges (tons/hectare) - based on average to good farm practices in Nigeria
    crop_yield_ranges = {
        'Maize': (2.5, 6.0),
        'Cassava': (15.0, 30.0), 
        'Rice': (2.0, 5.0),     
        'Yam': (10.0, 25.0),
        'Sorghum': (1.5, 4.0),
        'Millet': (1.0, 3.0),
        'Groundnut': (1.0, 2.5),
        'Beans': (1.0, 2.0)
    }

    # Prices generally increase over time due to inflation and market dynamics
    crop_price_ranges = {
        'Maize': (250, 600),   
        'Cassava': (180, 350),  
        'Rice': (800, 1500),
        'Yam': (250, 500),
        'Sorghum': (200, 500),
        'Millet': (200, 450),
        'Groundnut': (400, 900),
        'Beans': (500, 1200)
    }

    # Nigeria's inflation has been high (20-25% recently)
    average_annual_inflation_rate = 0.20
    season_weather_params = {
        'Wet': {'rainfall': (600, 1000), 'temp': (22, 28), 'pest_prob': 0.15},
        'Dry': {'rainfall': (50, 300), 'temp': (28, 35), 'pest_prob': 0.05}
    }

    # --- Simulated data lists ---
    yield_data = []
    weather_forecast_data = []
    market_price_data = []

    print(f"Generating data for {len(regions)} regions and {len(crop_types)} crop types from {start_year} to {end_year} (for forecasts)...")

    # Generate data up to end_year for historical, and end_year for forecasts
    for year in range(start_year, end_year + 1):
        for region in regions:
            for season in seasons:
                # Get base weather for the season
                weather_params = season_weather_params[season]
                avg_rainfall_range = weather_params['rainfall']
                avg_temp_range = weather_params['temp']
                base_pest_prob = weather_params['pest_prob']

                for _ in range(entries_per_season_per_region):
                    # Simulate weather and inputs
                    rainfall_mm = random.uniform(avg_rainfall_range[0], avg_rainfall_range[1])
                    avg_temp_c = random.uniform(avg_temp_range[0], avg_temp_range[1])
                    pest_outbreak_flag = 1 if random.random() < base_pest_prob else 0
                    fertilizer_used_kg = max(50, random.gauss(150, 50))

                    # --- Generate yield data for each crop type (for historical data) ---
                    if year <= datetime.now().year:
                        for crop_type in crop_types:
                            min_yield, max_yield = crop_yield_ranges.get(crop_type, (1.0, 3.0))

                            # Simulate yield, with weather and pest impacts
                            sim_yield = random.uniform(min_yield, max_yield)

                            # Significant rainfall deviation
                            if rainfall_mm < avg_rainfall_range[0] * 0.7 or rainfall_mm > avg_rainfall_range[1] * 1.3:
                                sim_yield *= random.uniform(0.7, 0.9)

                            # Extreme temperature deviation
                            if avg_temp_c < (avg_temp_range[0] + 2) or avg_temp_c > (avg_temp_range[1] - 2):
                                sim_yield *= random.uniform(0.8, 0.95)

                            # Pest outbreak effect
                            if pest_outbreak_flag:
                                sim_yield *= random.uniform(0.6, 0.85)

                            # Ensure yield is not negative or ridiculously low
                            sim_yield = max(0.5, sim_yield) 

                            yield_data.append({
                                'Year': year,
                                'Region': region,
                                'Crop_Type': crop_type,
                                'Season': season,
                                'Rainfall_mm': rainfall_mm,
                                'Avg_Temp_C': avg_temp_c,
                                'Pest_Outbreak_Flag': pest_outbreak_flag,
                                'Fertilizer_Used_kg_per_hectare': fertilizer_used_kg,
                                'Yield_tons_per_hectare': sim_yield
                            })

                    # --- Generate Market Price Data (for historical data) ---
                    if year <= datetime.now().year:
                        for crop_type in crop_types:
                            min_price, max_price = crop_price_ranges.get(crop_type, (100, 200))

                            # Base price for the crop type, adding some random variation
                            price_per_kg = random.uniform(min_price, max_price)

                            # Apply general inflation over years
                            inflation_factor = (1 + average_annual_inflation_rate)**(year - start_year)
                            price_per_kg *= inflation_factor

                            # Seasonal adjustments (often prices higher in dry season/off-season)
                            if season == 'Wet':
                                price_per_kg *= random.uniform(0.9, 1.0)
                            elif season == 'Dry':
                                price_per_kg *= random.uniform(1.0, 1.1)

                            price_per_kg = max(50, price_per_kg)

                            market_price_data.append({
                                'Year': year,
                                'Region': region,
                                'Crop_Type': crop_type,
                                'Season': season,
                                'Price_per_kg': price_per_kg
                            })

                # --- Generate Weather Forecast Data (for current year and future) ---
                if year >= datetime.now().year:
                    # Simulate forecast variability (e.g., forecast might slightly differ from actual)
                    forecast_rainfall = random.uniform(avg_rainfall_range[0] * 0.9, avg_rainfall_range[1] * 1.1)
                    forecast_temp = random.uniform(avg_temp_range[0] * 0.95, avg_temp_range[1] * 1.05)

                    # Determine drought/flood risk for forecast
                    drought_risk = 0
                    flood_risk = 0
                    # Define thresholds relative to average ranges
                    if season == 'Wet':
                        if forecast_rainfall < avg_rainfall_range[0] * 0.8:
                            drought_risk = 1
                        if forecast_rainfall > avg_rainfall_range[1] * 1.2:
                            flood_risk = 1
                    else: # Dry season
                        if forecast_rainfall < avg_rainfall_range[0] * 0.5:
                            drought_risk = 1
                        if forecast_rainfall > avg_rainfall_range[1] * 1.5:
                            flood_risk = 1

                    weather_forecast_data.append({
                        'Year': year,
                        'Region': region,
                        'Season': season,
                        'Predicted_Rainfall_mm': forecast_rainfall,
                        'Predicted_Avg_Temp_C': forecast_temp,
                        'Drought_Risk_Flag': drought_risk,
                        'Flood_Risk_Flag': flood_risk
                    })

    yield_df = pd.DataFrame(yield_data)
    forecast_df = pd.DataFrame(weather_forecast_data).drop_duplicates(subset=['Year', 'Region', 'Season']).reset_index(drop=True)
    price_df = pd.DataFrame(market_price_data).drop_duplicates(subset=['Year', 'Region', 'Crop_Type', 'Season']).reset_index(drop=True)

    # Save to CSV
    yield_output_path = os.path.join(output_dir, 'simulated_yield_data.csv')
    forecast_output_path = os.path.join(output_dir, 'simulated_weather_forecast.csv')
    price_output_path = os.path.join(output_dir, 'simulated_market_price_data.csv')

    yield_df.to_csv(yield_output_path, index=False)
    forecast_df.to_csv(forecast_output_path, index=False)
    price_df.to_csv(price_output_path, index=False)

    print(f"Simulated yield data saved to {yield_output_path} ({len(yield_df)} rows)")
    print(f"Simulated weather forecast data saved to {forecast_output_path} ({len(forecast_df)} rows)")
    print(f"Simulated market price data saved to {price_output_path} ({len(price_df)} rows)")

if __name__ == "__main__":
    generate_simulated_data()