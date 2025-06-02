def get_fresh_config():
    """Get fresh config from session state to avoid stale references."""
    return st.session_state.get('config_settings', {})

def update_config_in_session(section: str, updates: dict):
    """
    Update a section of the config stored in Streamlit session state.
    
    Args:
        section (str): The section of the config to update (e.g. 'data', 'model_params').
        updates (dict): The key-value pairs to update within that section.
    """
    config = st.session_state.config_settings

    if section not in config:
        st.error(f"❌ Config section '{section}' does not exist.")
        return

    # Update the local config
    config[section].update(updates)
    st.session_state.config_settings = config


def render_config_selectbox(
    label: str,
    section: str,
    key: str,
    options: list,
    help: str = "",
    notification_container=None,
    multi: bool = False,  # New parameter to switch between selectbox and multiselect
) -> str | list:
    """
    Render a selectbox or multiselect bound to a config key, and auto-update if changed.

    Args:
        label (str): The label shown above the widget.
        section (str): The config section (e.g., "data").
        key (str): The config key to update (e.g., "target_column").
        options (list): List of available options.
        help (str): Optional help tooltip.
        notification_container: Optional st.empty() placeholder for update messages.
        multi (bool): If True, renders a multiselect instead of a selectbox.

    Returns:
        The selected value(s): str for selectbox, list[str] for multiselect.
    """
    fresh_cfg = get_fresh_config()
    current_value = fresh_cfg.get(section, {}).get(key, [] if multi else "")

    # Determine default selection
    try:
        if multi:
            default_value = current_value if isinstance(current_value, list) else []
        else:
            default_index = options.index(current_value) if current_value in options else 0
    except ValueError:
        default_index = 0

    # Render appropriate widget
    if multi:
        selected = st.multiselect(
            label,
            options,
            default=default_value,
            key=f"{section}_{key}_multiselect",
            help=help
        )
    else:
        selected = st.selectbox(
            label,
            options,
            index=default_index,
            key=f"{section}_{key}_selectbox",
            help=help
        )

    # Compare and update if changed
    if selected != current_value:
        if notification_container is None:
            notification_container = st.empty()
        update_config_in_session(section, {key: selected})
        # instant_update_config(section, {key: selected})

    return selected


def render_config_input(
    label: str,
    section: str,
    key: str,
    help: str = "",
    placeholder: str = "",
    notification_container=None,
) -> str:
    fresh_cfg = get_fresh_config()
    current_value = fresh_cfg.get(section, {}).get(key, "")

    text_value = st.text_input(
        label,
        value=current_value,
        key=f"{section}_{key}_input",
        help=help,
        placeholder=placeholder
    )

    if text_value != current_value:
        if notification_container is None:
            notification_container = st.empty()
        update_config_in_session(section, {key: text_value})
        # instant_update_config(section, {key: text_value})

    return text_value