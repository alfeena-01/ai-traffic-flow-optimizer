🚦 AI Traffic Flow Optimizer

An end‑to‑end machine learning project that predicts traffic flow and visualizes real‑time conditions on an interactive dashboard. Built with Python, Pandas, Scikit‑Learn, Streamlit, and Pydeck.

📂 Project Structure

AI-Traffic-Flow-Optimizer/
│
├── data/                     # Raw and processed datasets
│   ├── traffic_sensor_data.csv
│   ├── weather_conditions.csv
│   └── sensor_locations.json
│
├── models/                   # Trained ML models
│   └── traffic_model.pkl
│
├── apps/                     # Streamlit apps
│   └── dashboard.py
│
├── notebooks/                # Jupyter notebooks for EDA & training
│   └── traffic_modeling.ipynb
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation


📊 Dataset Acquisition

Traffic Sensor Data  
Download from Kaggle:

Example: Jakarta Traffic Sensor Dataset  
Save as data/traffic_sensor_data.csv.

Weather Data  
Download from Kaggle or OpenWeather API:
Save as data/weather_conditions.csv.

Sensor Locations  
JSON file with sensor metadata and coordinates:

{
  "dataset_info": {...},
  "sensors": [
    {
      "sensor_id": "SEN-001",
      "location_name": "Jl. Sudirman - Bundaran HI",
      "coordinates": {"latitude": -6.337762, "longitude": 106.853379},
      "sensor_type": "Camera",
      "status": "Active"
    }
  ]
}

⚙️ Environment Setup

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

requirements.txt