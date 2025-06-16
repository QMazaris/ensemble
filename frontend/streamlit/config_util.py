import streamlit as st
from pathlib import Path
import requests
import os

# Backend API URL - use environment variable for Docker compatibility
BACKEND_API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def on_config_change(section: str, key: str, state_key: str):
    """Generic callback for updating config and pushing it to the backend API."""
    new_value = st.session_state[state_key]
    
    # Ensure config structure exists
    if 'config_settings' not in st.session_state:
        st.session_state['config_settings'] = {}
    if section not in st.session_state['config_settings']:
        st.session_state['config_settings'][section] = {}
    
    st.session_state['config_settings'][section][key] = new_value

def update_config_direct(section: str, key: str, value):
    """Update config directly with a value (for multiselects and other direct updates)."""
    if 'config_settings' not in st.session_state:
        st.session_state['config_settings'] = {}
    if section not in st.session_state['config_settings']:
        st.session_state['config_settings'][section] = {}
    st.session_state['config_settings'][section][key] = value
    
    # Optionally send immediate update to backend for critical changes
    try:
        config_update = {section: {key: value}}
        requests.post(
            f"{BACKEND_API_URL}/config/update",
            json=config_update,
            timeout=2.0
        )
    except:
        # Silently fail - main config save will handle this
        pass

def on_dataset_change():
    """Callback for dataset selection changes - requires path manipulation."""
    if 'config_settings' not in st.session_state:
        st.session_state['config_settings'] = {}
    if 'data' not in st.session_state['config_settings']:
        st.session_state['config_settings']['data'] = {}
    
    selected_dataset_name = st.session_state['dataset_selection']
    data_dir = Path("/data")
    selected_dataset_path = data_dir / selected_dataset_name
    st.session_state['config_settings']['data']['path'] = selected_dataset_path.as_posix()

def sync_config_to_backend():
    """Explicitly sync all config changes to backend."""
    try:
        config = st.session_state.get('config_settings', {})
        response = requests.post(
            f"{BACKEND_API_URL}/config/update",
            json=config,
            timeout=10
        )
        return response.status_code == 200
    except Exception:
        return False