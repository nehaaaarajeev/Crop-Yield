# 🌾 Crop Yield Intelligence Dashboard

An end-to-end Machine Learning pipeline and interactive Streamlit dashboard for analysing crop yield success/failure across Indian agricultural seasons.

---

## 📁 Project Structure

```
crop_yield_app/
├── app.py                    # Streamlit dashboard
├── analysis.py               # Step-by-step ML pipeline (run in terminal)
├── Crop_Yield.csv            # Dataset
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Quick Start

### 1. Clone / Download the repository

```bash
git clone https://github.com/<your-username>/crop-yield-dashboard.git
cd crop-yield-dashboard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the step-by-step analysis script

```bash
python analysis.py
```

This will:
- Perform all 10 steps of the ML pipeline
- Print evaluation tables in the terminal
- Save `confusion_matrices.png`, `feature_importance.png`, and `label_encoding_mapping.csv`

### 5. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (ensure `Crop_Yield.csv` is included).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud will install `requirements.txt` automatically.

---

## 📊 Dashboard Tabs

| Tab | Contents |
|-----|----------|
| 🏠 Overview | KPI cards, comparative bar charts by crop/season/soil, socio-economic pie charts, farmer experience trend |
| 📊 EDA | Box plots, pH histograms, correlation heatmap, rainfall scatter |
| 🤖 Model Performance | Accuracy / precision / recall comparison, confusion matrices |
| 🌟 Feature Importance | Per-model importance bars + side-by-side comparison |

---

## 🧠 ML Models

| Model | Type |
|-------|------|
| Decision Tree | Single tree, interpretable |
| Random Forest | Bagging ensemble (100 trees) |
| Gradient Boosted Trees | Boosting ensemble (100 trees) |

All models use `random_state=42` and an 80 / 20 stratified train-test split.

---

## 📌 Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| Crop Type | Categorical | Wheat, Rice, Maize, Cotton, Sugarcane |
| Season | Categorical | Kharif, Rabi, Zaid |
| Soil Type | Categorical | Loamy, Clay, Sandy, Silty, Black |
| Farm Size | Numeric | Acres |
| Soil Ph | Numeric | pH value |
| Soil Moisture (%) | Numeric | % moisture |
| Rainfall (mm) | Numeric | mm per season |
| Avg Temperature (°C) | Numeric | °C |
| Irrigation Type | Categorical | Flood, Drip, Rainfed, Sprinkler |
| Fertilizer Used (kg) | Numeric | kg per acre |
| Pesticide Used (kg) | Numeric | kg per acre |
| Seed Quality | Categorical | Hybrid, Certified, Local |
| Farming Practice | Categorical | Conventional, Organic, Mixed |
| Farmer Experience (years) | Numeric | Years of experience |
| Access to Credit | Binary (0/1) | Credit access |
| Govt. Subsidy Received | Binary (0/1) | Subsidy received |
| Expected Yield (kg per acre) | Numeric | Target yield |
| Actual Yield (kg per acre) | Numeric | Realised yield |
| **Yield Success** | **Binary (0/1)** | **Target label** |

---

## 🎨 Colour Conventions

| Category | Colour |
|----------|--------|
| Success | 🟢 Green `#16A34A` |
| Failure | 🔴 Red `#DC2626` |
| Kharif Season | 🟢 Green |
| Rabi Season | 🔵 Blue |
| Zaid Season | 🟠 Orange |
| Loamy Soil | 🟤 Brown |
| Clay Soil | 🔴 Dark Red |
| Sandy Soil | 🟡 Yellow |
| Silty Soil | ⚫ Grey |
| Black Soil | ⬛ Black |

---

## 📄 License

MIT License — free to use and modify.
