# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px # Import Plotly Express

# --- ADDED: 3MTT Logo ---
logo_path = "3mtt_logo.jpg"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200) # Adjust width as needed
else:
    st.sidebar.warning("Logo image not found. Please place '3mtt_logo.png' in the project folder.")
# --- END ADDED LOGO ---

# --- Set Streamlit Page Configuration ---
st.set_page_config(
    page_title="🌾 AgriSense: Nigeria's Predictive Farm Intelligence Dashboard ",
    page_icon="🌾",
    layout="wide" # Use wide layout to make better use of screen space
)

# --- Configuration Paths ---
DATA_DIR = 'data'
MODELS_DIR = 'models'

YIELD_MODEL_FILE = 'yield_prediction_model.joblib'
YIELD_FEATURES_ORDER_FILE = 'yield_model_features_order.joblib'
PRICE_MODEL_FILE = 'price_prediction_model.joblib' # NEW
PRICE_FEATURES_ORDER_FILE = 'price_model_features_order.joblib' # NEW

WEATHER_FORECAST_FILE = 'simulated_weather_forecast.csv'
YIELD_DATA_FILE = 'simulated_yield_data.csv' # For historical context/EDA
MARKET_PRICE_DATA_FILE = 'simulated_market_price_data.csv' # For future use/EDA

# --- Load Assets (Models, Feature Orders, Data) ---
@st.cache_resource # Cache the models to avoid reloading on every rerun
def load_model_assets():
    try:
        yield_model = joblib.load(os.path.join(MODELS_DIR, YIELD_MODEL_FILE))
        yield_features_order = joblib.load(os.path.join(MODELS_DIR, YIELD_FEATURES_ORDER_FILE))
        price_model = joblib.load(os.path.join(MODELS_DIR, PRICE_MODEL_FILE)) # NEW
        price_features_order = joblib.load(os.path.join(MODELS_DIR, PRICE_FEATURES_ORDER_FILE)) # NEW
        return yield_model, yield_features_order, price_model, price_features_order
    except FileNotFoundError:
        st.error(f"Error: Model or feature order files not found in '{MODELS_DIR}'. Please run train_models.py first.")
        st.stop() # Stop the app if crucial assets are missing
    except Exception as e:
        st.error(f"An error occurred loading model assets: {e}")
        st.stop()

@st.cache_data # Cache static dataframes to avoid reloading on every rerun
def load_all_data():
    try:
        forecast_df = pd.read_csv(os.path.join(DATA_DIR, WEATHER_FORECAST_FILE))
        historical_yield_df = pd.read_csv(os.path.join(DATA_DIR, YIELD_DATA_FILE))
        historical_price_df = pd.read_csv(os.path.join(DATA_DIR, MARKET_PRICE_DATA_FILE))

        # Merge for combined EDA if needed, similar to train_models.py
        combined_historical_df = pd.merge(historical_yield_df, historical_price_df, on=['Region', 'Crop_Type', 'Year', 'Season'], how='outer')
        combined_historical_df.dropna(subset=['Yield_tons_per_hectare', 'Price_per_kg'], inplace=True)
        combined_historical_df['Year'] = combined_historical_df['Year'].astype(int) # Ensure Year is integer

        return forecast_df, historical_yield_df, historical_price_df, combined_historical_df
    except FileNotFoundError as e:
        st.error(f"Error: Data file not found ({e}). Please run generate_data.py first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        st.stop()

# Load all assets at the start
yield_model, yield_model_features_order, price_model, price_model_features_order = load_model_assets()
forecast_df, historical_yield_df, historical_price_df, combined_historical_df = load_all_data()

# --- App Title and Description ---
st.title("🌾 AgriSense: Nigeria's Predictive Farm Intelligence Dashboard") # More inviting title
st.markdown("""
Welcome to **AgriSense**, a powerful tool designed to help Nigerian farmers and agricultural stakeholders make smarter, data-driven decisions!

**Here's what AgriSense can do for you:**
* **Predict Crop Yield:** Estimate how much your crops might produce under various conditions.
* **Predict Market Price:** Get an idea of the potential market price for your harvest.
* **Assess Weather Risks:** Get early warnings for potential droughts or floods in your area.
* **Understand Historical Trends:** Explore how crops and prices have behaved over time in different regions.

---
""") # Use a separator to visually break up sections

# --- Helper function for making a prediction (used internally) ---
def make_predictions_and_get_risks(
    year, region, crop_type, season, rainfall, temperature, pest_outbreak_flag, fertilizer,
    yield_model, yield_features_order, price_model, price_features_order,
    forecast_data, historical_yield_data, historical_price_data
):
    """
    Prepares input, makes yield and price predictions, and checks weather risks.
    Returns predicted yield, predicted price, forecast details, risk flags, and historical comparisons.
    """
    # Prepare common input for both models
    input_data_common = {
        'Year': [year],
        'Rainfall_mm': [rainfall],
        'Avg_Temp_C': [temperature],
        'Pest_Outbreak_Flag': [1 if pest_outbreak_flag else 0],
        'Fertilizer_Used_kg_per_hectare': [fertilizer],
    }

    # One-Hot Encode categorical features for prediction, matching training
    temp_categorical_df = pd.DataFrame({
        'Region': [region],
        'Crop_Type': [crop_type],
        'Season': [season]
    })
    temp_encoded = pd.get_dummies(temp_categorical_df, columns=['Region', 'Crop_Type', 'Season'], drop_first=True)

    # --- Yield Prediction ---
    input_df_yield = pd.DataFrame(input_data_common)
    for col in temp_encoded.columns:
        input_df_yield[col] = temp_encoded[col].iloc[0]
    for feature in yield_features_order: # Ensure all yield model features are present
        if feature not in input_df_yield.columns:
            input_df_yield[feature] = 0
    input_for_yield_prediction = input_df_yield[yield_features_order]
    predicted_yield = yield_model.predict(input_for_yield_prediction)[0]

    # --- Price Prediction ---
    input_df_price = pd.DataFrame(input_data_common)
    for col in temp_encoded.columns:
        input_df_price[col] = temp_encoded[col].iloc[0]
    for feature in price_features_order: # Ensure all price model features are present
        if feature not in input_df_price.columns:
            input_df_price[feature] = 0
    input_for_price_prediction = input_df_price[price_features_order]
    predicted_price = price_model.predict(input_for_price_prediction)[0]


    # --- Historical Comparisons ---
    historical_avg_yield = historical_yield_data[
        (historical_yield_data['Region'] == region) &
        (historical_yield_data['Crop_Type'] == crop_type) &
        (historical_yield_data['Season'] == season)
    ]['Yield_tons_per_hectare'].mean()

    historical_avg_price = historical_price_data[
        (historical_price_data['Region'] == region) &
        (historical_price_data['Crop_Type'] == crop_type) &
        (historical_price_data['Season'] == season)
    ]['Price_per_kg'].mean()

    # --- Risk Alert Logic ---
    relevant_forecast = forecast_data[
        (forecast_data['Region'] == region) &
        (forecast_data['Year'] == year) &
        (forecast_data['Season'] == season)
    ]

    forecast_info = {}
    if not relevant_forecast.empty:
        forecast_info['rainfall'] = relevant_forecast['Predicted_Rainfall_mm'].iloc[0]
        forecast_info['temp'] = relevant_forecast['Predicted_Avg_Temp_C'].iloc[0]
        forecast_info['drought_risk'] = relevant_forecast['Drought_Risk_Flag'].iloc[0]
        forecast_info['flood_risk'] = relevant_forecast['Flood_Risk_Flag'].iloc[0]
    else:
        forecast_info['status'] = "No specific weather forecast found for this year/season/region."

    return predicted_yield, predicted_price, forecast_info, historical_avg_yield, historical_avg_price

# --- Sidebar for Inputs (Single Scenario) ---
st.sidebar.header("Step 1: Get Your Crop Insights")
st.sidebar.write("Adjust the settings below to see how different factors might affect your crop yield and market price. Think about your farm's conditions or what you're planning for the future.")

# Get unique values for dropdowns from historical data (ensure consistency)
unique_regions = sorted(historical_yield_df['Region'].unique().tolist())
unique_crop_types = sorted(historical_yield_df['Crop_Type'].unique().tolist())
unique_seasons = sorted(historical_yield_df['Season'].unique().tolist())

# Input fields
selected_region = st.sidebar.selectbox("1. Select Your Farm's Region", unique_regions)
selected_crop_type = st.sidebar.selectbox("2. Choose Your Crop Type", unique_crop_types)
selected_season = st.sidebar.selectbox("3. Select the Growing Season", unique_seasons, help="Is it the Wet (Rainy) or Dry season?")

current_year = datetime.now().year
selected_year = st.sidebar.number_input("4. Select the Year for Prediction", min_value=2015, max_value=current_year + 1, value=current_year, help="You can look at past years or predict for the upcoming year.")

rainfall_mm = st.sidebar.slider("5. Average Rainfall (mm) Expected", min_value=50, max_value=1200, value=500, help="How much rain do you expect? (e.g., 500mm for moderate rain, 1000mm for heavy rain)")
avg_temp_c = st.sidebar.slider("6. Average Temperature (°C) Expected", min_value=15, max_value=40, value=28, help="What's the typical temperature during this period? (e.g., 28°C for warm, 35°C for hot)")
pest_outbreak = st.sidebar.checkbox("7. Do you expect a Pest Outbreak?", help="Check this box if you anticipate pests affecting your crops.")
fertilizer_used_kg = st.sidebar.slider("8. Fertilizer Used (kg per hectare)", min_value=0, max_value=300, value=100, help="How much fertilizer do you plan to use per hectare?")

# NEW: Input for profitability analysis
st.sidebar.markdown("---")
st.sidebar.subheader("For Profitability Analysis:")
land_area_hectares = st.sidebar.number_input("Land Area (hectares)", min_value=0.1, max_value=100.0, value=1.0, help="Enter the size of your farm area in hectares.")
estimated_costs_per_hectare = st.sidebar.number_input("Estimated Costs (NGN per hectare)", min_value=0, max_value=500000, value=100000, step=10000, help="Estimate your total costs (labor, seeds, etc.) per hectare.")


# Prediction button
predict_button = st.sidebar.button("💡 Get My Prediction & Risk Alerts!") # More action-oriented button text

# --- Main Content Area for Results ---
st.header("Step 2: Your Personalized Agricultural Insights")

if predict_button:
    predicted_yield, predicted_price, forecast_info, historical_avg_yield, historical_avg_price = make_predictions_and_get_risks(
        selected_year, selected_region, selected_crop_type, selected_season,
        rainfall_mm, avg_temp_c, pest_outbreak, fertilizer_used_kg,
        yield_model, yield_model_features_order, price_model, price_model_features_order,
        forecast_df, historical_yield_df, historical_price_df
    )

    col1_pred, col2_pred = st.columns(2)

    with col1_pred:
        st.subheader("🌾 Your Crop Yield Prediction:")
        st.success(f"**Predicted Harvest Yield:** {predicted_yield:.2f} tons per hectare")
        st.write("This is an estimate of how much crop you might harvest per land area (about 2.5 acres).")

        # Display historical yield comparison
        if not np.isnan(historical_avg_yield):
            st.info(f"**💡 For context:** Historically, {selected_crop_type} in {selected_region} during {selected_season} season has yielded an average of **{historical_avg_yield:.2f} tons per hectare**.")
            if predicted_yield > historical_avg_yield:
                st.markdown(f"📈 Good news! Your predicted yield is **{((predicted_yield - historical_avg_yield) / historical_avg_yield * 100):.1f}% higher** than the historical average for this area. This suggests very favorable conditions!")
            elif predicted_yield < historical_avg_yield:
                st.markdown(f"📉 Heads up! Your predicted yield is **{((historical_avg_yield - predicted_yield) / historical_avg_yield * 100):.1f}% lower** than the historical average. This might indicate potential challenges. Consider double-checking your inputs or planning mitigation strategies.")
            else:
                st.markdown("👍 Your predicted yield is similar to the historical average for this area.")
        else:
            st.info(f"🤔 No historical yield data available for {selected_crop_type} in {selected_region} during {selected_season} season to compare.")

    with col2_pred:
        st.subheader("💰 Your Market Price Prediction:")
        st.success(f"**Predicted Market Price:** {predicted_price:.2f} NGN per kg")
        st.write("This is an estimate of what you might sell your crop for per kilogram.")

        # Display historical price comparison
        if not np.isnan(historical_avg_price):
            st.info(f"**💡 For context:** Historically, {selected_crop_type} in {selected_region} during {selected_season} season has fetched an average of **{historical_avg_price:.2f} NGN per kg**.")
            if predicted_price > historical_avg_price:
                st.markdown(f"📈 Excellent! Your predicted price is **{((predicted_price - historical_avg_price) / historical_avg_price * 100):.1f}% higher** than the historical average. Great market potential!")
            elif predicted_price < historical_avg_price:
                st.markdown(f"📉 Be aware! Your predicted price is **{((historical_avg_price - predicted_price) / historical_avg_price * 100):.1f}% lower** than the historical average. Consider market timing or alternative buyers.")
            else:
                st.markdown("👍 Your predicted price is similar to the historical average.")
        else:
            st.info(f"🤔 No historical price data available for {selected_crop_type} in {selected_region} during {selected_season} season to compare.")


    st.markdown("---")

    # NEW: Profitability Analysis
    st.subheader("📊 Potential Profitability Outlook:")
    total_yield_tons = predicted_yield * land_area_hectares
    total_yield_kg = total_yield_tons * 1000 # Convert tons to kg for price calculation
    potential_revenue = total_yield_kg * predicted_price
    total_costs = estimated_costs_per_hectare * land_area_hectares
    potential_profit = potential_revenue - total_costs

    st.write(f"Based on your inputs and predictions for **{land_area_hectares:.1f} hectares**:")
    st.metric(label="Estimated Total Yield", value=f"{total_yield_tons:.2f} tons")
    st.metric(label="Potential Revenue", value=f"₦ {potential_revenue:,.2f}")
    st.metric(label="Estimated Total Costs", value=f"₦ {total_costs:,.2f}")

    if potential_profit > 0:
        st.success(f"**Potential Profit:** ₦ {potential_profit:,.2f} 🎉")
        st.write("This indicates a potentially profitable season! Keep monitoring conditions.")
    else:
        st.error(f"**Potential Loss:** ₦ {abs(potential_profit):,.2f} 😥")
        st.write("This suggests a potential loss. Consider re-evaluating your inputs, exploring cost-saving measures, or adjusting crop/season choices.")

    st.markdown("---")

    st.subheader("🌦️ Weather Risk Alerts & Forecast:")
    if 'status' in forecast_info:
        st.info(f"*(No specific weather forecast available for {selected_region} in {selected_year} {selected_season} season at the moment. Please note, forecasts are only available for the current and upcoming year for specific regions.)*")
    else:
        st.write(f"Here's the weather outlook for **{selected_region}** in **{selected_year} {selected_season} Season** based on our forecast:")
        st.write(f"- Predicted Rainfall: **{forecast_info['rainfall']:.1f} mm**")
        st.write(f"- Predicted Temperature: **{forecast_info['temp']:.1f} °C**")
        if forecast_info['drought_risk'] == 1:
            st.warning("⚠️ **Drought Risk Detected!** The predicted rainfall is lower than ideal. **Action:** Consider planning for irrigation or selecting drought-resistant crop varieties.")
        if forecast_info['flood_risk'] == 1:
            st.warning("🌊 **Flood Risk Detected!** The predicted rainfall is higher than usual. **Action:** Prepare for potential flooding, ensure good drainage, or select flood-tolerant crop varieties.")
        if forecast_info['drought_risk'] == 0 and forecast_info['flood_risk'] == 0:
            st.info("✅ **Favorable Weather Expected!** No immediate drought or flood risk detected based on the forecast. Ideal conditions for your crops.")

    st.markdown("---")

# --- Display Interactive EDA Plots (Now always on full dataset) ---
st.header("Step 3: Explore Historical Data & Trends")
st.write("""
Dive into past data to understand what has influenced crop yield and market prices in Nigeria.
These charts are **interactive**! You can:
* **Hover** your mouse over any point or bar to see exact numbers.
* **Click and drag** on the chart to zoom in on a specific area.
* **Double-click** to zoom out.
* Use the **toolbar** (top-right corner of each chart) for more options like panning, selecting, or downloading the image.
""")

cols_eda = st.columns(2) # Display plots in two columns

# Plot 1: Distribution of Harvest Yield (Histogram)
with cols_eda[0]:
    fig1 = px.histogram(combined_historical_df, x='Yield_tons_per_hectare',
                        title='How Crop Yields Vary Across Nigeria',
                        labels={'Yield_tons_per_hectare': 'Crop Yield (tons per hectare)', 'count': 'Number of Farms/Records'},
                        marginal="rug") # Add rug plot for individual data points
    fig1.update_layout(title_font_size=20) # Make title slightly larger
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("*(This chart shows how common different yield levels have been historically. Taller bars mean more farms achieved that yield.)*")

# Plot 2: Yield vs. Rainfall by Season (Scatter Plot)
with cols_eda[1]:
    fig2 = px.scatter(combined_historical_df, x='Rainfall_mm', y='Yield_tons_per_hectare',
                      color='Season', title='Impact of Rainfall on Crop Yield by Season',
                      labels={'Rainfall_mm': 'Rainfall (mm)', 'Yield_tons_per_hectare': 'Crop Yield (tons/hectare)'},
                      hover_data=['Region', 'Crop_Type', 'Year', 'Fertilizer_Used_kg_per_hectare'])
    fig2.update_layout(title_font_size=20)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("*(See how rainfall affects yield in different seasons. Do higher rainfall lead to higher yields, or is there an optimal range?)*")

# Plot 3: Average Yield by Region (Bar Chart)
with cols_eda[0]:
    # Calculate average yield by region for plotting
    avg_yield_region = combined_historical_df.groupby('Region')['Yield_tons_per_hectare'].mean().sort_values(ascending=False).reset_index()
    fig3 = px.bar(avg_yield_region, x='Region', y='Yield_tons_per_hectare',
                  title='Average Crop Yields Across Nigerian Regions',
                  labels={'Yield_tons_per_hectare': 'Average Yield (tons/hectare)'})
    fig3.update_layout(xaxis_tickangle=-45, title_font_size=20)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("*(Compare average yields in different states. Which regions have historically performed best?)*")

# Plot 4: Average Market Price by Crop Type (Bar Chart)
with cols_eda[1]:
    # Calculate average price by crop type for plotting
    avg_price_crop = combined_historical_df.groupby('Crop_Type')['Price_per_kg'].mean().sort_values(ascending=False).reset_index()
    fig4 = px.bar(avg_price_crop, x='Crop_Type', y='Price_per_kg',
                  title='Average Market Prices for Different Crops (NGN/kg)',
                  labels={'Price_per_kg': 'Average Price (NGN/kg)'})
    fig4.update_layout(title_font_size=20)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("*(Understand which crops typically fetch higher prices in the market.)*")

# NEW: Plot 5: Historical Yield Trend Over Years
st.markdown("---")
st.subheader("Historical Trends Over Time:")
trend_cols = st.columns(2)
with trend_cols[0]:
    selected_trend_crop_yield = st.selectbox("Select Crop for Yield Trend", unique_crop_types, key="yield_trend_crop")
    trend_yield_df = combined_historical_df[combined_historical_df['Crop_Type'] == selected_trend_crop_yield]
    if not trend_yield_df.empty:
        avg_yield_year = trend_yield_df.groupby('Year')['Yield_tons_per_hectare'].mean().reset_index()
        fig_yield_trend = px.line(avg_yield_year, x='Year', y='Yield_tons_per_hectare',
                                  title=f'Average Yield Trend for {selected_trend_crop_yield} Over Years',
                                  labels={'Yield_tons_per_hectare': 'Average Yield (tons/hectare)'})
        st.plotly_chart(fig_yield_trend, use_container_width=True)
        st.markdown("*(See how the average yield for your selected crop has changed each year.)*")
    else:
        st.info(f"No data to show yield trend for {selected_trend_crop_yield} with current filters.")

# NEW: Plot 6: Historical Price Trend Over Years
with trend_cols[1]:
    selected_trend_crop_price = st.selectbox("Select Crop for Price Trend", unique_crop_types, key="price_trend_crop")
    trend_price_df = combined_historical_df[combined_historical_df['Crop_Type'] == selected_trend_crop_price]
    if not trend_price_df.empty:
        avg_price_year = trend_price_df.groupby('Year')['Price_per_kg'].mean().reset_index()
        fig_price_trend = px.line(avg_price_year, x='Year', y='Price_per_kg',
                                  title=f'Average Market Price Trend for {selected_trend_crop_price} Over Years',
                                  labels={'Price_per_kg': 'Average Price (NGN/kg)'})
        st.plotly_chart(fig_price_trend, use_container_width=True)
        st.markdown("*(See how the average market price for your selected crop has changed each year.)*")
    else:
        st.info(f"No data to show price trend for {selected_trend_crop_price} with current filters.")

# NEW: Top Performing Regions/Crops Summary (Now uses combined_historical_df directly)
st.markdown("---")
st.subheader("🏆 Top Performers (Historical Averages):")
top_cols = st.columns(2)

with top_cols[0]:
    st.markdown("##### Top 3 Regions by Average Yield:")
    top_yield_regions = combined_historical_df.groupby('Region')['Yield_tons_per_hectare'].mean().nlargest(3).reset_index()
    for i, row in top_yield_regions.iterrows():
        st.write(f"**{i+1}. {row['Region']}:** {row['Yield_tons_per_hectare']:.2f} tons/hectare")

with top_cols[1]:
    st.markdown("##### Top 3 Crops by Average Market Price:")
    top_price_crops = combined_historical_df.groupby('Crop_Type')['Price_per_kg'].mean().nlargest(3).reset_index()
    for i, row in top_price_crops.iterrows():
        st.write(f"**{i+1}. {row['Crop_Type']}:** ₦ {row['Price_per_kg']:.2f} /kg")

# --- NEW: About AgriSense Section ---
st.markdown("---")
st.header("About AgriSense")
with st.expander("Learn more about this dashboard"):
    st.markdown("""
    AgriSense is developed to empower Nigerian farmers and agricultural stakeholders with predictive insights.

    **How it Works:**
    * **Data Simulation:** For this demonstration, the dashboard uses **simulated data** that mimics real-world agricultural conditions, weather patterns, and market prices in Nigeria. This data covers various regions, crop types, and seasons from 2000 to the present.
    * **Machine Learning Models:** We use **Random Forest Regressor** models, a powerful type of machine learning algorithm, to predict crop yield and market prices. These models learn complex relationships from historical data to make accurate forecasts.
    * **Weather Forecasts:** The risk alerts are based on simulated future weather forecasts for rainfall and temperature, identifying potential drought or flood conditions.
    * **Interactive Visuals:** Powered by Plotly, the charts allow you to explore historical trends, distributions, and relationships within the data.

    **Our Vision:**
    To provide accessible, data-driven tools that help optimize farming practices, mitigate risks, and improve profitability for the agricultural sector in Nigeria.

    **Disclaimer:**
    This version of AgriSense uses simulated data for demonstration purposes. While the models are trained to be robust, real-world conditions can vary. Always combine these insights with your local knowledge and expert advice.
    """)

# --- NEW: Feedback Mechanism ---
st.markdown("---")
st.header("Share Your Feedback")
st.write("Your input helps us improve AgriSense! Please share your thoughts, suggestions, or any issues you encountered.")

feedback_text = st.text_area("Your Feedback:")
if st.button("Submit Feedback"):
    if feedback_text:
        st.success("Thank you for your feedback! Your comments have been recorded. (In a real application, this would be saved to a database or sent to a team.)")
        # In a real application, you would save `feedback_text` here.
        # Example: save_feedback_to_database(feedback_text)
    else:
        st.warning("Please enter some feedback before submitting.")

# --- NEW: Real Data Integration (Conceptual Explanation) ---
st.markdown("---")
st.header("Future Vision: Real Data Integration")
st.write("""
Currently, AgriSense operates on **simulated data**. This allows us to demonstrate the power of predictive analytics without needing access to large, real-time datasets.

**For a real-world application, integrating actual data would be the next crucial step.** This would involve:
* **Weather APIs:** Connecting to services like AccuWeather, OpenWeatherMap, or national meteorological agencies for real-time and historical weather data.
* **Agricultural Databases:** Sourcing data from government agricultural ministries, research institutions, or farmer cooperatives for actual yield records, soil data, and pest reports.
* **Market Data Providers:** Integrating with commodity exchanges or local market data aggregators for real-time price information.
* **Data Pipelines:** Building automated systems to regularly collect, clean, and update these diverse real-world datasets.

This transition would transform AgriSense from a powerful demonstration tool into an indispensable, real-time decision-support system for the agricultural community.
""")
st.markdown("---")
st.caption("© 2025 Mubarak Lawal. All rights reserved.")