=============================================================
# AgriSense: Smart Agriculture Dashboard 🌿
=============================================================

Project Overview
-------------------------------------------------------------
AgriSense is an intuitive Streamlit-based web application designed to empower Nigerian farmers and agricultural stakeholders with data-driven insights. It provides essential predictive capabilities for crop yield and market prices, offers timely weather risk alerts, and allows users to explore crucial historical agricultural trends. This dashboard serves as a robust prototype, vividly demonstrating the transformative potential of machine learning in the agricultural sector.

---

Features
-------------------------------------------------------------
AgriSense offers a suite of functionalities crafted to support informed decision-making:

* **Personalized Crop Predictions:** Input specific farm conditions like region, crop type, season, rainfall, temperature, pest presence, and fertilizer use to receive estimated crop yield (tons per hectare) and potential market price (NGN per kg).
* **Profitability Analysis:** Beyond just predictions, the dashboard calculates potential revenue and profit based on predicted yield, market price, your specified land area, and estimated operational costs.
* **Proactive Weather Risk Alerts:** Get early warnings for potential drought or flood conditions specific to your chosen region and season, helping you plan mitigation strategies.
* **Interactive Historical Data Exploration:** Dive into past agricultural trends with dynamic Plotly charts. Explore distributions of yields, the impact of rainfall on different crops, and historical price movements.
* **Top Performer Summaries:** Quickly identify historically high-yielding regions and crops that have commanded the best market prices.
* **Comprehensive "About AgriSense" Section:** Understand the core mechanics of the dashboard, including its reliance on simulated data for this prototype and the underlying Random Forest machine learning models.
* **User Feedback Mechanism:** A simple interface allows users to provide comments and suggestions, fostering continuous improvement.

---

Disclaimer: Important Information About Data Usage
-------------------------------------------------------------
**PLEASE NOTE:** This version of AgriSense utilizes **simulated data** for demonstration and educational purposes. While the underlying machine learning models are rigorously trained and robust, the predictions and historical figures presented are not based on real-time, verified agricultural data from Nigeria.

For actual farming decisions, it is crucial to:
* Always consult local agricultural experts and extension workers.
* Refer to current, real-time market prices from official sources.
* Obtain up-to-date weather forecasts from national meteorological agencies.

---

Project Structure
-------------------------------------------------------------
The AgriSense project is organized into a clear and manageable directory structure:

AgriSense_Project/
├── data/
│   ├── simulated_market_price_data.csv
│   ├── simulated_weather_forecast.csv
│   └── simulated_yield_data.csv
├── models/
│   ├── price_model_features_order.joblib
│   ├── price_prediction_model.joblib
│   ├── yield_model_features_order.joblib
│   └── yield_prediction_model.joblib
├── 3mtt_logo.png
├── app.py
├── generate_data.py
├── train_models.py
└── README.md


* `data/`: Contains the CSV files for simulated historical yield, weather forecasts, and market prices.
* `models/`: Stores the saved machine learning models (`.joblib` files) and their corresponding feature order lists, ensuring consistent predictions.
* `3mtt_logo.png`: The visual logo displayed in the Streamlit application's sidebar.
* `app.py`: The core Streamlit script that powers the interactive dashboard.
* `generate_data.py`: A Python script responsible for creating the simulated agricultural and weather datasets.
* `train_models.py`: A Python script to train the crop yield and market price prediction models and save them for use in `app.py`.

---

Getting Started
-------------------------------------------------------------

### Prerequisites
* **Python 3.8+**
* **Anaconda** (highly recommended for environment management) or `pip`

### Installation

1.  **Clone the repository** (if using Git) or download the project files directly to your local machine:
    ```bash
    # If using Git
    git clone <your-repository-url-here>
    cd AgriSense_Project
    ```
    *(If you've downloaded the files manually, ensure all contents are extracted into a single folder named `AgriSense_Project`.)*

2.  **Create and activate a dedicated Python environment** (recommended for dependency management):
    ```bash
    # Using Conda (recommended)
    conda create -n agrisense_env python=3.9
    conda activate agrisense_env

    # Or using Python's venv
    python -m venv agrisense_env
    source agrisense_env/bin/activate # On Windows: .\agrisense_env\Scripts\activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install streamlit pandas numpy scikit-learn plotly joblib
    ```
    *(For best practice, you can generate a `requirements.txt` file using `pip freeze > requirements.txt` after installing, and then `pip install -r requirements.txt`.)*

---

### How to Run the Application

Execute these scripts in the specified order from your `AgriSense_Project` directory with your `agrisense_env` activated:

1.  **Generate Simulated Data:**
    This step creates the necessary `.csv` data files in the `data/` directory.
    ```bash
    python generate_data.py
    ```

2.  **Train the Machine Learning Models:**
    This script trains both the yield and price prediction models and saves their `.joblib` files, along with feature orders, into the `models/` directory.
    ```bash
    python train_models.py
    ```

3.  **Launch the Streamlit Dashboard:**
    This command will start the Streamlit server and automatically open the AgriSense dashboard in your default web browser.
    ```bash
    streamlit run app.py
    ```

---

Future Vision: Real Data Integration
-------------------------------------------------------------
While AgriSense currently operates on simulated data, its ultimate potential lies in leveraging real-world information. The next significant phase for this project involves transitioning to actual data sources, which would include:

* **Weather APIs:** Integrating with services like Nigeria's NIMET or international weather providers for real-time and historical climate data.
* **Agricultural Databases:** Sourcing verified yield records, soil data, and farming practice information from government bodies, research institutions, or farmer associations.
* **Market Data Providers:** Connecting to actual commodity exchanges or local market data aggregators for live price feeds.
* **Robust Data Pipelines:** Developing automated systems to continuously collect, clean, transform, and update these diverse real-world datasets.

This transition would empower AgriSense to provide genuinely accurate, actionable, and impactful insights for the Nigerian agricultural community.

---

Contributing
-------------------------------------------------------------
We welcome contributions to AgriSense! Feel free to fork this repository, suggest improvements, or submit pull requests.

---

Contact
-------------------------------------------------------------
* **LinkedIn:** [Mubarak Lawal](https://www.linkedin.com/in/mubarak-lawal/) (Clickable link)
* **Email:** lawalademola71@gmail.com