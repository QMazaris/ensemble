# CSV File Elimination Summary

This document summarizes the changes made to eliminate CSV file dependencies for serving data to the frontend. All data is now stored in memory on the backend and served via API endpoints.

## Key Changes Made

### 1. Backend API Updates (`backend/api/main.py`)

**Added:**
- Import of `data_service` for in-memory data storage
- New endpoints for serving data from memory:
  - `/results/metrics` - Serves model metrics from data_service
  - `/results/threshold-sweep` - Serves threshold sweep data with filtering
  - `/results/model-comparison` - Serves model comparison data
  - `/results/predictions` - Serves model predictions data
  - `/data/load` - Loads training data into memory cache
  - `/data/clear` - Clears all cached data
  - `/data/save-csv-backup` - Force saves CSV backup files for debugging

**Updated:**
- `/data/info` endpoint now caches training data info in memory
- Removed dependency on `pipeline_results` global variable
- All endpoints now use `data_service` for data retrieval

### 2. Frontend Utils Updates (`frontend/streamlit/utils.py`)

**Added:**
- `BACKEND_API_URL` configuration for API communication
- `fetch_data_from_api()` function for API requests
- API-first data loading with fallback hierarchy:
  1. Backend API (primary)
  2. Data service (fallback)
  3. CSV files (last resort for backward compatibility)

**Updated:**
- `load_metrics_data()` - Now tries API first, then data service, then files
- `load_predictions_data()` - Same API-first approach

### 3. Frontend Tabs Updates (`frontend/streamlit/tabs.py`)

**Updated:**
- `render_downloads_tab()` - Now gets predictions from API/data service instead of CSV file
- `render_model_metrics_cheat_tab()` - Uses API/data service instead of reading CSV directly
- Updated documentation to reflect in-memory data source

### 4. Data Service Updates (`shared/data_service.py`)

**Updated:**
- `save_to_files()` method now has optional `save_csv_backup` parameter (default: False)
- CSV files are only saved when explicitly requested for backup/debugging
- JSON files are always saved for efficiency

### 5. Backend Helper Updates

**`backend/helpers/export_metrics_for_streamlit.py`:**
- Updated `save_to_files()` call to not save CSV files by default

**`backend/helpers/reporting.py`:**
- Updated `save_all_model_probabilities_from_structure()` to have optional `save_csv_backup` parameter
- CSV files only saved when explicitly requested

**`backend/run.py`:**
- Updated function call to not save CSV backup by default

### 6. Dependencies

**Added to `requirements.txt`:**
- `requests>=2.28.0` for frontend API communication

## Data Flow Architecture

### Before (CSV-dependent):
```
Pipeline → CSV Files → Frontend reads CSV files
```

### After (Memory-based with API):
```
Pipeline → Data Service (in-memory) → API Endpoints → Frontend
                ↓ (optional backup)
            JSON/CSV files
```

## Benefits

1. **Performance**: No file I/O for data serving, faster response times
2. **Reliability**: No file system dependencies, reduced race conditions
3. **Scalability**: In-memory data can be easily shared across multiple frontend instances
4. **Maintainability**: Single source of truth for data, cleaner architecture
5. **Flexibility**: API endpoints allow for easy filtering and data transformation

## Backward Compatibility

- CSV files can still be generated for backup/debugging by calling `/data/save-csv-backup` endpoint
- Frontend still has fallback to read CSV files if API is unavailable
- Existing file-based workflows continue to work

## Testing

- Data service functionality verified with comprehensive tests
- API endpoints return proper error messages when data is unavailable
- Frontend gracefully handles API failures with appropriate fallbacks

## Usage

1. **Normal Operation**: Run pipeline → Data stored in memory → Frontend uses API
2. **Debugging**: Call `/data/save-csv-backup` to generate CSV files for inspection
3. **Development**: Frontend automatically falls back to CSV files if API is unavailable

All CSV file dependencies for frontend data serving have been successfully eliminated while maintaining backward compatibility and adding robust error handling. 