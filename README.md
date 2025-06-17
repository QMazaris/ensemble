# Ensemble Inspection Platform (Prebuilt Docker Edition)

Welcome to the Ensemble Inspection Platform! This system uses an ensemble of models to analyze weld inspection data, view results, and manage datasets—all through a clean browser-based interface.

This version runs entirely via **prebuilt Docker images** — no manual builds or setups needed.

---

## 🚀 Quick Start

### 1. Clone this repository

```bash
git clone -b Run-only https://github.com/QMazaris/ensemble.git
cd ensemble
```

### 2. Start the system
```bash
docker compose up -d
```

### 3. Open in your browser
```bash
Frontend (Streamlit UI): http://localhost:8501
Backend (FastAPI): http://localhost:8000/docs
```
### How to actually use the site
1. Upload data to the **Data Management** tab and scroll through all of the data to ensure its good. 
2. Name and save the data at the bottom. Errors in the csv are  OK.
3. In the **Preprocessing Config** tab, select the correct dataset, target column, any base models and exclude columns. You must click save under each for the backend to update. 
4. use the **Pipeline Settings** to configure it and press **Run Pipeline**. You will see baloons when it is done. 

#### Notes:
Optomize hyperparameters can take a long time