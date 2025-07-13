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
    entries_per_season_per_region=5, # Number of distinct data points per year/season/region
    base_yield_tons=2.0, # Baseline yield per hectare
    yield_variability=0.5, # How much yield can vary
    base_fertilizer_kg=100, # Base fertilizer used
    fertilizer_variability=30
):
    """
    Generates simulated agricultural yield, weather, and market price data.
    """
    # Set random seeds for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Configuration for realistic ranges ---
    regions = [
        'Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 'Borno',
        'Cross River', 'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 'Gombe', 'Imo',
        'Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos',
        'Nasarawa', 'Niger', 'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau', 'Rivers',
        'Sokoto', 'Taraba', 'Yobe', 'Zamfara'
    ]
    crop_types = ['Maize', 'Cassava', 'Rice', 'Yam', 'Sorghum', 'Millet', 'Groundnut', 'Beans']
    seasons = ['Wet', 'Dry']

    # --- Simulated data lists ---
    yield_data = []
    weather_forecast_data = []
    market_price_data = []

    print(f"Generating data for {len(regions)} regions and {len(crop_types)} crop types from {start_year} to {end_year} (for forecasts)...")

    # Generate data up to end_year for historical, and end_year for forecasts
    for year in range(start_year, end_year + 1):
        for region in regions:
            for season in seasons:
                # Base weather conditions for season
                if season == 'Wet':
                    avg_rainfall_range = (600, 1000) # mm
                    avg_temp_range = (22, 28) # °C
                    base_pest_prob = 0.15 # Higher chance of pests in wet
                else: # Dry season
                    avg_rainfall_range = (50, 300) # mm
                    avg_temp_range = (28, 35) # °C
                    base_pest_prob = 0.05 # Lower chance of pests in dry

                for _ in range(entries_per_season_per_region):
                    # Simulate slight variations around base values
                    rainfall_mm = random.uniform(avg_rainfall_range[0], avg_rainfall_range[1])
                    avg_temp_c = random.uniform(avg_temp_range[0], avg_temp_range[1])
                    pest_outbreak_flag = 1 if random.random() < base_pest_prob else 0
                    fertilizer_used_kg = max(0, random.gauss(base_fertilizer_kg, fertilizer_variability))

                    # Simulate yield based on factors (simple linear model for simulation)
                    # More rain, optimal temp, more fertilizer = higher yield
                    # Pest outbreak = lower yield
                    sim_yield = (
                        base_yield_tons
                        + (rainfall_mm / 1000) * 0.5  # Positive effect of rainfall
                        - ((avg_temp_c - 28)**2) * 0.01 # Optimal temp around 28, quadratic penalty for deviation
                        + (fertilizer_used_kg / 100) * 0.3 # Positive effect of fertilizer
                        - (pest_outbreak_flag * 0.4) # Negative effect of pests
                        + random.uniform(-yield_variability, yield_variability) # Random noise
                    )
                    sim_yield = max(0.5, sim_yield) # Ensure yield is not too low

                    # --- Generate yield data for each crop type (for historical data) ---
                    # Only add yield data for historical years (up to current year)
                    if year <= datetime.now().year:
                        for crop_type in crop_types:
                            yield_data.append({
                                'Year': year,
                                'Region': region,
                                'Crop_Type': crop_type,
                                'Season': season,
                                'Rainfall_mm': rainfall_mm,
                                'Avg_Temp_C': avg_temp_c,
                                'Pest_Outbreak_Flag': pest_outbreak_flag,
                                'Fertilizer_Used_kg_per_hectare': fertilizer_used_kg,
                                'Yield_tons_per_hectare': sim_yield # Common yield for all crops given conditions
                            })

                    # --- Generate Market Price Data (for historical data) ---
                    # Only add market price data for historical years (up to current year)
                    if year <= datetime.now().year:
                        for crop_type in crop_types:
                            base_price = 150 # NGN/kg
                            if crop_type == 'Rice': base_price *= 1.2
                            if crop_type == 'Yam': base_price *= 0.9
                            if crop_type == 'Maize': base_price *= 0.8
                            if season == 'Wet': base_price *= 0.95 # Prices might be slightly lower post-harvest wet season
                            if season == 'Dry': base_price *= 1.05 # Prices might be slightly higher due to scarcity
                            price_per_kg = max(50, random.gauss(base_price, 20) + (year - start_year) * 2) # Slight inflation over years

                            market_price_data.append({
                                'Year': year,
                                'Region': region, # Market prices can vary by region too
                                'Crop_Type': crop_type,
                                'Season': season,
                                'Price_per_kg': price_per_kg
                            })

                # --- Generate Weather Forecast Data (for current year and future) ---
                # This ensures we have forecast data for years that the app allows for prediction
                if year >= datetime.now().year: # Generate forecasts for current year and future (up to end_year)
                    # Simulate forecast variability (e.g., forecast might slightly differ from actual)
                    forecast_rainfall = random.uniform(rainfall_mm * 0.9, rainfall_mm * 1.1)
                    forecast_temp = random.uniform(avg_temp_c * 0.95, avg_temp_c * 1.05)

                    # Determine drought/flood risk for forecast
                    drought_risk = 0
                    flood_risk = 0
                    if season == 'Wet':
                        if forecast_rainfall < avg_rainfall_range[0] * 0.8: # Significantly below expected wet season rain
                            drought_risk = 1
                        if forecast_rainfall > avg_rainfall_range[1] * 1.2: # Significantly above expected wet season rain
                            flood_risk = 1
                    else: # Dry season
                        if forecast_rainfall < avg_rainfall_range[0] * 0.5: # Extremely low for dry season
                            drought_risk = 1
                        if forecast_rainfall > avg_rainfall_range[1] * 1.5: # Unusually high for dry season
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

    # Convert to DataFrames
    yield_df = pd.DataFrame(yield_data)
    # Drop duplicates for forecast and price as they are per-region/season/year, not per 'entry'
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