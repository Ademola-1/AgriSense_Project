# train_models.py
import pandas as pd
import numpy as np
import os
import joblib # To save/load models
import random # Import random for random.seed()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Set random seeds for reproducibility ---
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
# --- End random seeds ---

print("--- Starting Model Training Process ---")

# --- Configuration ---
DATA_DIR = 'data'
MODELS_DIR = 'models'
PLOTS_DIR = 'plots' # Define PLOTS_DIR explicitly (though no static plots saved here anymore)
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models folder exists
os.makedirs(PLOTS_DIR, exist_ok=True) # Create a folder for plots if you want to save them separately

YIELD_DATA_FILE = 'simulated_yield_data.csv'
PRICE_DATA_FILE = 'simulated_market_price_data.csv'

# --- Load Data ---
try:
    yield_df = pd.read_csv(os.path.join(DATA_DIR, YIELD_DATA_FILE))
    price_df = pd.read_csv(os.path.join(DATA_DIR, PRICE_DATA_FILE))
    print("Data loaded successfully: yield_df, price_df")
except FileNotFoundError as e:
    print(f"Error: Required data file not found: {e}")
    print("Please ensure 'generate_data.py' has been run successfully and CSVs are in the 'data/' folder.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# --- Combine relevant data ---
# Merge yield and price data for comprehensive analysis (and potential features)
combined_df = pd.merge(yield_df, price_df, on=['Region', 'Crop_Type', 'Year', 'Season'], how='outer')
combined_df['Year'] = combined_df['Year'].astype(int)

print("\nData Loaded and Combined. First 5 rows of combined_df:")
print(combined_df.head())

print("\nMissing values before cleaning:")
print(combined_df.isnull().sum())

# Drop rows where either Yield or Price is missing, as both models need this
combined_df.dropna(subset=['Yield_tons_per_hectare', 'Price_per_kg'], inplace=True)

print("\nMissing values after cleaning:")
print(combined_df.isnull().sum())

# --- Feature Engineering & Encoding for Model Training ---
categorical_features_for_ohe = ['Region', 'Crop_Type', 'Season']
df_encoded = pd.get_dummies(combined_df, columns=categorical_features_for_ohe, drop_first=True)

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)

df_encoded['Year'].fillna(df_encoded['Year'].mean(), inplace=True)


print("\nFirst 5 rows of encoded data (for model training):")
print(df_encoded.head())
print("\nColumns after encoding (for model training):")
print(df_encoded.columns.tolist())

# --- Define features for Yield Model ---
# Yield model will use all features except Price_per_kg
yield_model_features_order = [col for col in df_encoded.columns if col not in ['Yield_tons_per_hectare', 'Price_per_kg']]
print("\nFeatures that will be used for the YIELD model (and their order):")
print(yield_model_features_order)

# --- Prepare Features (X) and Target (y) for Yield Model Training ---
y_yield = df_encoded['Yield_tons_per_hectare']
X_yield = df_encoded[yield_model_features_order]

print(f"\nFeatures (X_yield) shape: {X_yield.shape}")
print(f"Target (y_yield) shape: {y_yield.shape}")
print("\nFirst 5 rows of features for yield model training:")
print(X_yield.head())

# --- Train-Test Split for Yield Model ---
X_train_yield, X_test_yield, y_train_yield, y_test_yield = train_test_split(
    X_yield, y_yield, test_size=0.2, random_state=random_seed
)
print(f"\nYield Training set size: {len(X_train_yield)} samples")
print(f"Yield Testing set size: {len(X_test_yield)} samples")

# --- Model Training (Random Forest Regressor for Yield Prediction) ---
print("\nTraining RandomForestRegressor model for Yield Prediction...")
yield_model = RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1)
yield_model.fit(X_train_yield, y_train_yield)
print("Yield model training complete.")

# --- Model Evaluation (Yield) ---
print("\nEvaluating Yield Model Performance...")
y_pred_yield = yield_model.predict(X_test_yield)

mae_yield = mean_absolute_error(y_test_yield, y_pred_yield)
r2_yield = r2_score(y_test_yield, y_pred_yield)

print(f"Mean Absolute Error (MAE) for Yield Prediction: {mae_yield:.2f} tons/hectare")
print(f"R-squared (R2 Score) for Yield Prediction: {r2_yield:.2f}")

# --- Save Yield Model and Features Order ---
print("\nSaving trained yield model and feature order...")
yield_model_filename = 'yield_prediction_model.joblib'
yield_features_order_filename = 'yield_model_features_order.joblib'

joblib.dump(yield_model, os.path.join(MODELS_DIR, yield_model_filename))
joblib.dump(yield_model_features_order, os.path.join(MODELS_DIR, yield_features_order_filename))

print(f"Yield Model saved to: {os.path.join(MODELS_DIR, yield_model_filename)}")
print(f"Yield Model Features Order saved to: {os.path.join(MODELS_DIR, yield_features_order_filename)}")


# --- NEW: Define features for Price Model ---
# Price model will use all features except Yield_tons_per_hectare
price_model_features_order = [col for col in df_encoded.columns if col not in ['Price_per_kg', 'Yield_tons_per_hectare']]
print("\nFeatures that will be used for the PRICE model (and their order):")
print(price_model_features_order)

# --- NEW: Prepare Features (X) and Target (y) for Price Model Training ---
y_price = df_encoded['Price_per_kg']
X_price = df_encoded[price_model_features_order]

print(f"\nFeatures (X_price) shape: {X_price.shape}")
print(f"Target (y_price) shape: {y_price.shape}")
print("\nFirst 5 rows of features for price model training:")
print(X_price.head())

# --- NEW: Train-Test Split for Price Model ---
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
    X_price, y_price, test_size=0.2, random_state=random_seed
)
print(f"\nPrice Training set size: {len(X_train_price)} samples")
print(f"Price Testing set size: {len(X_test_price)} samples")

# --- NEW: Model Training (Random Forest Regressor for Price Prediction) ---
print("\nTraining RandomForestRegressor model for Price Prediction...")
price_model = RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1)
price_model.fit(X_train_price, y_train_price)
print("Price model training complete.")

# --- NEW: Model Evaluation (Price) ---
print("\nEvaluating Price Model Performance...")
y_pred_price = price_model.predict(X_test_price)

mae_price = mean_absolute_error(y_test_price, y_pred_price)
r2_price = r2_score(y_test_price, y_pred_price)

print(f"Mean Absolute Error (MAE) for Price Prediction: {mae_price:.2f} NGN/kg")
print(f"R-squared (R2 Score) for Price Prediction: {r2_price:.2f}")

# --- NEW: Save Price Model and Features Order ---
print("\nSaving trained price model and feature order...")
price_model_filename = 'price_prediction_model.joblib'
price_features_order_filename = 'price_model_features_order.joblib'

joblib.dump(price_model, os.path.join(MODELS_DIR, price_model_filename))
joblib.dump(price_model_features_order, os.path.join(MODELS_DIR, price_features_order_filename))

print(f"Price Model saved to: {os.path.join(MODELS_DIR, price_model_filename)}")
print(f"Price Model Features Order saved to: {os.path.join(MODELS_DIR, price_features_order_filename)}")


print("\n--- All Model Training Processes Complete ---")