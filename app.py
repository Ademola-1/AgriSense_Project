import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gspread
import uuid
from google.oauth2.service_account import Credentials
from datetime import datetime
import plotly.express as px

# --- Logo ---
logo_path = "AgriSense_logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200)
else:
    st.sidebar.warning("Logo image not found. Please place 'AgriSense_logo.png' in the project folder.")

# --- Set Streamlit Page Configuration ---
st.set_page_config(
    page_title="🌾 AgriSense: Nigeria's Predictive Farm Intelligence Dashboard",
    page_icon="🌾",
    layout="wide"
)

# --- Configuration Constants ---
GOOGLE_SHEET_NAME = 'AgriSense Feedback'  # Name of your Google Sheet
WORKSHEET_NAME = 'Sheet1'  # Name of the worksheet/tab within the sheet
DATA_DIR = 'data'
MODELS_DIR = 'models'

# --- File Names ---
YIELD_MODEL_FILE = 'yield_prediction_model.joblib'
YIELD_FEATURES_ORDER_FILE = 'yield_model_features_order.joblib'
PRICE_MODEL_FILE = 'price_prediction_model.joblib'
PRICE_FEATURES_ORDER_FILE = 'price_model_features_order.joblib'

WEATHER_FORECAST_FILE = 'simulated_weather_forecast.csv'
YIELD_DATA_FILE = 'simulated_yield_data.csv'
MARKET_PRICE_DATA_FILE = 'simulated_market_price_data.csv'

# --- Anonymous ID Generator ---
def get_anonymous_id():
    """Generates and stores an anonymous session ID."""
    if 'anon_id' not in st.session_state:
        st.session_state.anon_id = str(uuid.uuid4())[:8]
    return st.session_state.anon_id

# --- Initialize Google Sheets Connection ---
@st.cache_resource(ttl=3600)
def init_google_sheets():
    """Initializes and returns a gspread worksheet object for Google Sheets."""
    try:
        # Check if service account secrets are available
        if 'gcp_service_account' not in st.secrets:
            st.warning("Google Sheets connection not configured in `secrets.toml` or Streamlit Cloud settings. Feedback feature will be disabled.")
            return None

        # Define the scopes for Google Sheets and Drive API access
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        # Load credentials from Streamlit secrets
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        gc = gspread.authorize(creds)

        try:
            # Open the Google Sheet and select the worksheet
            spreadsheet = gc.open(GOOGLE_SHEET_NAME)
            worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
            
            # Define expected headers for the worksheet
            expected_headers = [
                "Timestamp", "Name", "Email", "Feedback", 
                "Rating", "Region", "Crop Type", "Anonymous ID"
            ]
            
            # Get current headers and check if they match expected
            current_headers = worksheet.row_values(1)
            if current_headers != expected_headers:
                if len(current_headers) == 0: 
                    worksheet.append_row(expected_headers)
                else:  
                    st.error("Mismatch in Google Sheet headers. Please ensure the first row of your sheet contains: " + ", ".join(expected_headers))
                    return None
                    
            return worksheet

        except gspread.WorksheetNotFound:
            # If worksheet doesn't exist, create it and add headers
            st.info(f"Worksheet '{WORKSHEET_NAME}' not found. Creating it in Google Sheet '{GOOGLE_SHEET_NAME}'.")
            worksheet = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows=100, cols=20)
            worksheet.append_row(expected_headers)
            return worksheet
            
    except Exception as e:
        # Catch any other exceptions during connection
        st.error(f"Failed to connect to Google Sheets: {e}. Please check your `secrets.toml` configuration and sheet permissions.")
        return None

# --- Load Assets (Models, Feature Orders, Data) ---
@st.cache_resource(show_spinner="Loading predictive models...")
def load_model_assets():
    """Loads pre-trained models and their feature orders."""
    try:
        yield_model = joblib.load(os.path.join(MODELS_DIR, YIELD_MODEL_FILE))
        yield_features_order = joblib.load(os.path.join(MODELS_DIR, YIELD_FEATURES_ORDER_FILE))
        price_model = joblib.load(os.path.join(MODELS_DIR, PRICE_MODEL_FILE))
        price_features_order = joblib.load(os.path.join(MODELS_DIR, PRICE_FEATURES_ORDER_FILE))
        return yield_model, yield_features_order, price_model, price_features_order
    except FileNotFoundError:
        st.error(f"Error: One or more model files not found in '{MODELS_DIR}'. Please ensure 'train_models.py' has been run successfully.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading model assets: {e}")
        st.stop()

@st.cache_data(show_spinner="Loading historical and forecast data...")
def load_all_data():
    """Loads all necessary dataframes (weather forecast, historical yield, historical price)."""
    try:
        forecast_df = pd.read_csv(os.path.join(DATA_DIR, WEATHER_FORECAST_FILE))
        historical_yield_df = pd.read_csv(os.path.join(DATA_DIR, YIELD_DATA_FILE))
        historical_price_df = pd.read_csv(os.path.join(DATA_DIR, MARKET_PRICE_DATA_FILE))

        #Convert price from ₦/ton to ₦/kg if needed
        if historical_price_df['Price_per_kg'].max() > 1000:
            historical_price_df['Price_per_kg'] = historical_price_df['Price_per_kg'] / 1000

        # Merge historical data for combined analysis
        combined_historical_df = pd.merge(
            historical_yield_df, 
            historical_price_df, 
            on=['Region', 'Crop_Type', 'Year', 'Season'], 
            how='outer'
        )
        combined_historical_df.dropna(subset=['Yield_tons_per_hectare', 'Price_per_kg'], inplace=True)
        combined_historical_df['Year'] = combined_historical_df['Year'].astype(int)
        
        return forecast_df, historical_yield_df, historical_price_df, combined_historical_df
    except FileNotFoundError as e:
        st.error(f"Error: One or more data files not found ({e}). Please ensure 'generate_data.py' has been run successfully.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        st.stop()


# --- Helper Function for Predictions and Risk Analysis ---
def make_predictions_and_get_risks(
    year, region, crop_type, season, rainfall, temperature, pest_outbreak_flag, fertilizer,
    yield_model, yield_features_order, price_model, price_features_order,
    forecast_data, historical_yield_data, historical_price_data
):
    """
    Makes yield and price predictions, and retrieves weather risk information.
    """
    input_data_common = {
        'Year': [year],
        'Rainfall_mm': [rainfall],
        'Avg_Temp_C': [temperature],
        'Pest_Outbreak_Flag': [1 if pest_outbreak_flag else 0],
        'Fertilizer_Used_kg_per_hectare': [fertilizer],
    }

    temp_categorical_df = pd.DataFrame({
        'Region': [region],
        'Crop_Type': [crop_type],
        'Season': [season]
    })
    temp_encoded = pd.get_dummies(temp_categorical_df, columns=['Region', 'Crop_Type', 'Season'], drop_first=True)

    # Prepare input for Yield Prediction
    input_df_yield = pd.DataFrame(input_data_common)
    for col in temp_encoded.columns:
        input_df_yield[col] = temp_encoded[col].iloc[0]
    for feature in yield_features_order:
        if feature not in input_df_yield.columns:
            input_df_yield[feature] = 0
    input_for_yield_prediction = input_df_yield[yield_features_order]
    predicted_yield = yield_model.predict(input_for_yield_prediction)[0]

    # Prepare input for Price Prediction
    input_df_price = pd.DataFrame(input_data_common)
    for col in temp_encoded.columns:
        input_df_price[col] = temp_encoded[col].iloc[0]
    for feature in price_features_order:
        if feature not in input_df_price.columns:
            input_df_price[feature] = 0
    input_for_price_prediction = input_df_price[price_features_order]
    predicted_price = price_model.predict(input_for_price_prediction)[0]

    # Calculate Historical Averages for comparison
    historical_avg_yield = historical_yield_data[
        (historical_yield_data['Region'] == region) &
        (historical_yield_data['Crop_Type'] == crop_type) &
        (historical_yield_data['Season'] == season)
    ]['Yield_tons_per_hectare'].mean()

    # Filter data for historical price average
    filtered_price_data = historical_price_data[
        (historical_price_data['Region'] == region) &
        (historical_price_data['Crop_Type'] == crop_type) &
        (historical_price_data['Season'] == season)
    ]

    historical_avg_price = filtered_price_data['Price_per_kg'].mean()

    recent_historical_avg_price = np.nan
    # Removed: current_year = datetime.now().year # No longer needed here

    recent_years_data = filtered_price_data[filtered_price_data['Year'] >= (year - 4)]

    if not recent_years_data.empty:
        recent_historical_avg_price = recent_years_data['Price_per_kg'].mean()

    # Retrieve Weather Risk Information
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

    # Return the new recent_historical_avg_price as well
    return predicted_yield, predicted_price, forecast_info, historical_avg_yield, historical_avg_price, recent_historical_avg_price

# --- Initialize App Assets ---
worksheet = init_google_sheets()
yield_model, yield_model_features_order, price_model, price_model_features_order = load_model_assets()
forecast_df, historical_yield_df, historical_price_df, combined_historical_df = load_all_data()

# Get unique values for dropdowns (using sorted lists for consistent order)
unique_regions = sorted(historical_yield_df['Region'].unique().tolist())
unique_crop_types = sorted(historical_yield_df['Crop_Type'].unique().tolist())
unique_seasons = sorted(historical_yield_df['Season'].unique().tolist())

# --- Sidebar for User Inputs ---
st.sidebar.header("Farm & Crop Selection")

# Initialize session state for selectbox defaults
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = unique_regions[0] if unique_regions else "N/A"
if 'selected_crop_type' not in st.session_state:
    st.session_state.selected_crop_type = unique_crop_types[0] if unique_crop_types else "N/A"

st.session_state.selected_region = st.sidebar.selectbox("1. Select Your Farm's Region", unique_regions)
st.session_state.selected_crop_type = st.sidebar.selectbox("2. Choose Your Crop Type", unique_crop_types)
selected_season = st.sidebar.selectbox("3. Select the Growing Season", unique_seasons, help="Is it the Wet (Rainy) or Dry season?")

current_year = datetime.now().year
selected_year = st.sidebar.number_input("4. Select the Year for Prediction", min_value=2015, max_value=current_year + 1, value=current_year)

rainfall_mm = st.sidebar.slider("5. Average Rainfall (mm) Expected", min_value=50, max_value=1200, value=500)
avg_temp_c = st.sidebar.slider("6. Average Temperature (°C) Expected", min_value=15, max_value=40, value=28)
pest_outbreak = st.sidebar.checkbox("7. Do you expect a Pest Outbreak?")
fertilizer_used_kg = st.sidebar.slider("8. Fertilizer Used (kg per hectare)", min_value=0, max_value=300, value=100)

# Profitability Analysis Inputs
st.sidebar.markdown("---")
st.sidebar.subheader("For Profitability Analysis:")
land_area_hectares = st.sidebar.number_input("Land Area (hectare))", min_value=0.1, max_value=100.0, value=1.0)
estimated_costs_per_hectare = st.sidebar.number_input(
    "Estimated Costs (NGN per (hectare))", 
    min_value=100000,
    max_value=2000000, 
    value=450000,      
    step=50000         
)
predict_button = st.sidebar.button("💡 Get My Prediction & Risk Alerts!")

# --- Main Content Area ---
st.title("🌾 AgriSense: Nigeria's Predictive Farm Intelligence Dashboard")
st.markdown("""
Welcome to **AgriSense**, a powerful tool designed to help Nigerian farmers and agricultural stakeholders make smarter, data-driven decisions!
""")

# --- Prediction Results Section ---
st.header("Prediction & Profitability Outlook")

if predict_button:
    predicted_yield, predicted_price, forecast_info, historical_avg_yield, historical_avg_price, recent_historical_avg_price = make_predictions_and_get_risks(
        selected_year, st.session_state.selected_region, st.session_state.selected_crop_type, selected_season,
        rainfall_mm, avg_temp_c, pest_outbreak, fertilizer_used_kg,
        yield_model, yield_model_features_order, price_model, price_model_features_order,
        forecast_df, historical_yield_df, historical_price_df
    )

    col1_pred, col2_pred = st.columns(2)

    with col1_pred:
        st.subheader("🌾 Your Crop Yield Prediction:")
        st.success(f"**Predicted Harvest Yield:** {predicted_yield:.2f} tons per hectare")
        
        if not pd.isna(historical_avg_yield):
            st.info(f"**💡 For context:** Historically, {st.session_state.selected_crop_type} in {st.session_state.selected_region} during {selected_season} season has yielded an average of **{historical_avg_yield:.2f} tons per hectare**.")
            if predicted_yield > historical_avg_yield:
                st.markdown(f"📈 Good news! Your predicted yield is **{((predicted_yield - historical_avg_yield) / historical_avg_yield * 100):.1f}% higher** than the historical average.")
            elif predicted_yield < historical_avg_yield:
                st.markdown(f"📉 Heads up! Your predicted yield is **{((historical_avg_yield - predicted_yield) / historical_avg_yield * 100):.1f}% lower** than the historical average.")
            else:
                st.markdown("👍 Your predicted yield is similar to the historical average for this area.")
        else:
            st.info(f"🤔 No historical yield data available for {st.session_state.selected_crop_type} in {st.session_state.selected_region} during {selected_season} season to compare.")

    
    with col2_pred:
        st.subheader("💰 Your Market Price Prediction:")
        st.success(f"**Predicted Market Price:** {predicted_price:.2f} NGN per kg")

        if not pd.isna(historical_avg_price):
            st.info(f"**💡 For context (All Historical Data):** Historically, {st.session_state.selected_crop_type} in {st.session_state.selected_region} during {selected_season} season has fetched an average of **{historical_avg_price:.2f} NGN per kg**.")
            if not pd.isna(recent_historical_avg_price):
                st.info(f"**💡 For context (Last 5 Years):** The average price in recent years was **{recent_historical_avg_price:.2f} NGN per kg**.")
                comparison_price = recent_historical_avg_price
                comparison_label = "recent average"
            else:
                comparison_price = historical_avg_price
                comparison_label = "overall historical average"
                st.info("No data available for the last 5 years to provide a recent average context.")
            if predicted_price > comparison_price:
                st.markdown(f"📈 Excellent! Your predicted price is **{((predicted_price - comparison_price) / comparison_price * 100):.1f}% higher** than the {comparison_label}.")
            elif predicted_price < comparison_price:
                st.markdown(f"📉 Be aware! Your predicted price is **{((comparison_price - predicted_price) / comparison_price * 100):.1f}% lower** than the {comparison_label}.")
            else:
                st.markdown(f"👍 Your predicted price is similar to the {comparison_label}.")
        else:
            st.info(f"🤔 No historical price data available for {st.session_state.selected_crop_type} in {st.session_state.selected_region} during {selected_season} season to compare.")
    st.markdown("---")
    st.subheader("📊 Potential Profitability Outlook:")
    
    total_yield_tons = predicted_yield * land_area_hectares
    total_yield_kg = total_yield_tons * 1000
    potential_revenue = total_yield_kg * predicted_price
    total_costs = estimated_costs_per_hectare * land_area_hectares
    potential_profit = potential_revenue - total_costs

    st.write(f"Based on your inputs for **{land_area_hectares:.1f} hectares**:")
    st.metric(label="Estimated Total Yield", value=f"{total_yield_tons:.2f} tons")
    st.metric(label="Potential Revenue", value=f"₦ {potential_revenue:,.2f}")
    st.metric(label="Estimated Total Costs", value=f"₦ {total_costs:,.2f}")

    if potential_profit > 0:
        st.success(f"**Potential Profit:** ₦ {potential_profit:,.2f} 🎉")
    else:
        st.error(f"**Potential Loss:** ₦ {abs(potential_profit):,.2f} 😥")

    st.markdown("---")
    st.subheader("🌦️ Weather Risk Alerts & Forecast:")
    
    if 'status' in forecast_info:
        st.info(forecast_info['status'])
    else:
        st.write(f"Weather outlook for **{st.session_state.selected_region}** in **{selected_year} {selected_season} Season**:")
        st.write(f"- Predicted Rainfall: **{forecast_info['rainfall']:.1f} mm**")
        st.write(f"- Predicted Temperature: **{forecast_info['temp']:.1f} °C**")
        
        if forecast_info['drought_risk'] == 1:
            st.warning("⚠️ **Drought Risk Detected!** Consider planning for irrigation or drought-resistant crops.")
        if forecast_info['flood_risk'] == 1:
            st.warning("🌊 **Flood Risk Detected!** Prepare for potential flooding and ensure good drainage.")
        if forecast_info['drought_risk'] == 0 and forecast_info['flood_risk'] == 0:
            st.info("✅ **Favorable Weather Expected!** No immediate drought or flood risk detected.")

# --- Data Visualization Section ---
st.header("Historical Data & Trends")
st.write("Interactive charts showing historical trends and relationships:")

# Row 1: Distribution plots
cols_dist = st.columns(2)
with cols_dist[0]:
    st.write("**Distribution of Crop Yields**")
    fig_yield_dist = px.histogram(combined_historical_df, x='Yield_tons_per_hectare',
                                  title='Yields Across Nigeria', marginal="box", nbins=30)
    st.plotly_chart(fig_yield_dist, use_container_width=True)

with cols_dist[1]:
    st.write("**Distribution of Market Prices**")
    fig_price_dist = px.histogram(combined_historical_df, x='Price_per_kg',
                                  title='Prices Across Nigeria', marginal="box", nbins=30)
    st.plotly_chart(fig_price_dist, use_container_width=True)

# Row 2: Average by Category
cols_avg = st.columns(2)
with cols_avg[0]:
    avg_yield_region = combined_historical_df.groupby('Region')['Yield_tons_per_hectare'].mean().sort_values(ascending=False).reset_index()
    fig_avg_yield_region = px.bar(avg_yield_region, x='Region', y='Yield_tons_per_hectare',
                                  title='Average Crop Yields by Region')
    st.plotly_chart(fig_avg_yield_region, use_container_width=True)

with cols_avg[1]:
    avg_price_crop = combined_historical_df.groupby('Crop_Type')['Price_per_kg'].mean().sort_values(ascending=False).reset_index()
    fig_avg_price_crop = px.bar(avg_price_crop, x='Crop_Type', y='Price_per_kg',
                                  title='Average Market Prices by Crop Type')
    st.plotly_chart(fig_avg_price_crop, use_container_width=True)

# Row 3: Time Series Trends 
st.markdown("---")
st.subheader("Historical Trends Over Time:")
trend_cols = st.columns(2)

with trend_cols[0]:
    selected_trend_crop_yield = st.selectbox("Select Crop for Yield Trend", unique_crop_types, key="yield_trend_crop")
    trend_yield_df = combined_historical_df[combined_historical_df['Crop_Type'] == selected_trend_crop_yield]
    if not trend_yield_df.empty:
        avg_yield_year = trend_yield_df.groupby('Year')['Yield_tons_per_hectare'].mean().reset_index()
        fig_yield_trend = px.line(avg_yield_year, x='Year', y='Yield_tons_per_hectare',
                                    title=f'Average Yield Trend for {selected_trend_crop_yield}',
                                    labels={'Yield_tons_per_hectare': 'Average Yield (tons/hectare)'})
        st.plotly_chart(fig_yield_trend, use_container_width=True)
        st.markdown("*(See how the average yield for your selected crop has changed each year.)*")
    else:
        st.info(f"No data to show yield trend for {selected_trend_crop_yield} with current filters.")

with trend_cols[1]:
    selected_trend_crop_price = st.selectbox("Select Crop for Price Trend", unique_crop_types, key="price_trend_crop")
    trend_price_df = combined_historical_df[combined_historical_df['Crop_Type'] == selected_trend_crop_price]
    if not trend_price_df.empty:
        avg_price_year = trend_price_df.groupby('Year')['Price_per_kg'].mean().reset_index()
        fig_price_trend = px.line(avg_price_year, x='Year', y='Price_per_kg',
                                    title=f'Average Market Price Trend for {selected_trend_crop_price}',
                                    labels={'Price_per_kg': 'Average Price (NGN/kg)'})
        st.plotly_chart(fig_price_trend, use_container_width=True)
        st.markdown("*(See how the average market price for your selected crop has changed each year.)*")
    else:
        st.info(f"No data to show price trend for {selected_trend_crop_price} with current filters.")

# --- Additional Charts ---
st.markdown("---")
st.subheader("More Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("**Yield Distribution**")
    fig = px.histogram(
        combined_historical_df,
        x='Yield_tons_per_hectare',
        nbins=20,
        marginal='box'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("**Price Distribution**")
    fig = px.histogram(
        combined_historical_df,
        x='Price_per_kg',
        nbins=20,
        marginal='box'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Top Performers Historical Averages ---
st.markdown("---")
st.header("🏆 Top Performers (Historical Averages)")
st.write("Discover the historically best-performing regions for yield and crop types for market price across Nigeria.")

top_n = st.slider("Show Top N Performers", min_value=1, max_value=10, value=3, key="top_n_slider")
col_top_yield, col_top_price = st.columns(2)

with col_top_yield:
    st.subheader(f"🥇 Top {top_n} Regions by Average Yield")
    # Calculate average yield per region
    avg_yield_by_region = combined_historical_df.groupby('Region')['Yield_tons_per_hectare'].mean().sort_values(ascending=False)
    
    if not avg_yield_by_region.empty:
        df_top_yield = avg_yield_by_region.head(top_n).reset_index()
        for index, row in df_top_yield.iterrows():
            st.write(f"**{index + 1}. {row['Region']}:** {row['Yield_tons_per_hectare']:.2f} tons/hectare")
    else:
        st.info("No data available to determine top performing regions by yield.")

with col_top_price:
    st.subheader(f"💰 Top {top_n} Crop Types by Average Price")
    # Calculate average price per crop type
    avg_price_by_crop = combined_historical_df.groupby('Crop_Type')['Price_per_kg'].mean().sort_values(ascending=False)
    
    if not avg_price_by_crop.empty:
        df_top_price = avg_price_by_crop.head(top_n).reset_index()
        for index, row in df_top_price.iterrows():
            st.write(f"**{index + 1}. {row['Crop_Type']}:** ₦{row['Price_per_kg']:.2f} per kg")
    else:
        st.info("No data available to determine top performing crop types by price.")

st.markdown("*(Based on all historical data)*")


# --- Feedback Form ---
if 'anon_id' not in st.session_state:
    st.session_state.anon_id = str(uuid.uuid4())[:8]

st.markdown("---")
st.header("We Value Your Feedback")

# Privacy notice
st.markdown("""
<div style="font-size:12px; color:#666; margin-bottom:16px;">
Note: We use an anonymous session identifier to improve our service. 
No personally identifiable tracking is used.
</div>
""", unsafe_allow_html=True)

with st.form("feedback_form", clear_on_submit=True):
    st.markdown("### Share Your Experience")
    
    col_feedback_input_1, col_feedback_input_2 = st.columns(2)
    with col_feedback_input_1:
        user_name = st.text_input("Your Name (optional)", key="feedback_name")
    with col_feedback_input_2:
        user_email = st.text_input("Your Email (optional)", 
                                    help="Only if you'd like us to follow up",
                                    key="feedback_email")
    
    feedback_text = st.text_area("Your Feedback*", 
                                  help="Please share your thoughts, suggestions, or report any issues",
                                  key="feedback_text")
    rating = st.slider("How would you rate your experience?*", 1, 5, 3, key="feedback_rating")
    
    privacy_agree = st.checkbox("I understand and agree to anonymous usage data collection", value=False, key="privacy_agree")
    
    submitted = st.form_submit_button("Submit Feedback")
    
    if submitted:
        validation_passed = True
        
        if not privacy_agree:
            st.warning("Please agree to the privacy notice before submitting feedback.")
            validation_passed = False
            
        if not feedback_text.strip():
            st.warning("Please provide your feedback before submitting.")
            validation_passed = False
            
        if worksheet is None:
            st.error("Feedback submission is currently unavailable. Please ensure Google Sheets is configured correctly.")
            validation_passed = False
            
        if validation_passed:
            try:
                feedback_data = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_name.strip() or "Anonymous",
                    user_email.strip() or "Not provided",
                    feedback_text.strip(),
                    rating,
                    st.session_state.get('selected_region', 'N/A'), 
                    st.session_state.get('selected_crop_type', 'N/A'),
                    st.session_state.anon_id 
                ]
                
                worksheet.append_row(feedback_data)
                st.success("✅ Thank you for your feedback! Your insights are valuable.")
                
            except Exception as e:
                st.error(f"Failed to submit feedback: {str(e)}")
                print(f"Error submitting feedback: {e}") 


# --- Real Data Integration Section ---
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

# --- About Section ---
st.markdown("---")
st.header("About AgriSense")
with st.expander("Learn more about this dashboard"):
    st.markdown("""
    AgriSense is developed to empower Nigerian farmers and agricultural stakeholders with predictive insights.

    **How it Works:**
    * Uses simulated data that mimics real-world agricultural conditions
    * Employs Random Forest Regressor models for predictions
    * Provides weather risk alerts based on simulated forecasts
    * Offers interactive visualizations for data exploration

    **Our Vision:**
    To provide accessible, data-driven tools that help optimize farming practices and improve profitability.

    **Disclaimer:**
    This version uses simulated data for demonstration purposes. Always combine these insights with local knowledge and expert advice.
    """)

st.markdown("---")
st.caption("© 2025 Mubarak Lawal. All rights reserved.")