import streamlit as st
from pathlib import Path

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

def on_dataset_change():
    """Callback for dataset selection changes - requires path manipulation."""
    if 'config_settings' not in st.session_state:
        st.session_state['config_settings'] = {}
    if 'data' not in st.session_state['config_settings']:
        st.session_state['config_settings']['data'] = {}
    
    selected_dataset_name = st.session_state['dataset_selection']
    data_dir = Path("data")
    selected_dataset_path = data_dir / selected_dataset_name
    st.session_state['config_settings']['data']['path'] = selected_dataset_path.as_posix()