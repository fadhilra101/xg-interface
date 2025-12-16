"""
Custom shot management utilities for handling multiple custom shots.
"""

import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional


def initialize_custom_shots_session():
    """
    Initialize session state for custom shots management.
    """
    if 'custom_shots_data' not in st.session_state:
        st.session_state.custom_shots_data = []
    if 'custom_shots_counter' not in st.session_state:
        st.session_state.custom_shots_counter = 0


def add_custom_shot(shot_data: Dict, xg_value: float, shot_name: str = None) -> None:
    """
    Add a new custom shot to the session storage.
    
    Args:
        shot_data: Dictionary containing shot features
        xg_value: Predicted xG value
        shot_name: Optional custom name for the shot
    """
    # Ensure session state is initialized
    initialize_custom_shots_session()
    
    st.session_state.custom_shots_counter += 1
    
    if shot_name is None:
        shot_name = f"Shot {st.session_state.custom_shots_counter}"
    
    # Create shot record with metadata
    shot_record = {
        'shot_id': st.session_state.custom_shots_counter,
        'shot_name': shot_name,
        'xg_value': xg_value,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **shot_data  # Include all shot features
    }
    
    st.session_state.custom_shots_data.append(shot_record)
    
    # Force session state update
    st.session_state.custom_shots_data = st.session_state.custom_shots_data


def get_custom_shots_dataframe() -> pd.DataFrame:
    """
    Convert custom shots data to pandas DataFrame.
    
    Returns:
        DataFrame containing all custom shots
    """
    # Ensure session state is initialized
    initialize_custom_shots_session()
    
    if 'custom_shots_data' not in st.session_state or not st.session_state.custom_shots_data:
        return pd.DataFrame()
    
    return pd.DataFrame(st.session_state.custom_shots_data)


def remove_custom_shot(shot_id: int) -> bool:
    """
    Remove a custom shot by ID.
    
    Args:
        shot_id: ID of the shot to remove
        
    Returns:
        True if shot was removed, False if not found
    """
    original_length = len(st.session_state.custom_shots_data)
    st.session_state.custom_shots_data = [
        shot for shot in st.session_state.custom_shots_data 
        if shot['shot_id'] != shot_id
    ]
    return len(st.session_state.custom_shots_data) < original_length


def clear_all_custom_shots() -> None:
    """
    Clear all custom shots from session storage.
    """
    st.session_state.custom_shots_data = []
    st.session_state.custom_shots_counter = 0


def get_custom_shots_count() -> int:
    """
    Get the number of stored custom shots.
    
    Returns:
        Number of custom shots
    """
    # Ensure session state is initialized
    initialize_custom_shots_session()
    
    if 'custom_shots_data' not in st.session_state:
        return 0
    
    return len(st.session_state.custom_shots_data)


def prepare_custom_shots_for_download() -> str:
    """
    Prepare custom shots data for CSV download.
    
    Returns:
        CSV string of custom shots data
    """
    df = get_custom_shots_dataframe()
    if df.empty:
        return ""
    
    # Reorder columns for better readability
    column_order = [
        'shot_id', 'shot_name', 'xg_value', 'created_at',
        'start_x', 'start_y', 'minute', 'second', 'period',
        'play_pattern', 'position', 'shot_technique', 'shot_body_part',
        'shot_type', 'type_before', 'shot_open_goal', 'shot_one_on_one',
        'shot_aerial_won', 'shot_first_time', 'shot_key_pass', 'under_pressure'
    ]
    
    # Only include columns that exist in the DataFrame
    available_columns = [col for col in column_order if col in df.columns]
    df_ordered = df[available_columns]
    
    return df_ordered.to_csv(index=False)


def get_custom_shots_summary() -> Dict:
    """
    Get summary statistics of custom shots.
    
    Returns:
        Dictionary containing summary statistics
    """
    df = get_custom_shots_dataframe()
    if df.empty:
        return {
            'total_shots': 0,
            'avg_xg': 0,
            'min_xg': 0,
            'max_xg': 0,
            'total_expected_goals': 0
        }
    
    return {
        'total_shots': len(df),
        'avg_xg': df['xg_value'].mean(),
        'min_xg': df['xg_value'].min(),
        'max_xg': df['xg_value'].max(),
        'total_expected_goals': df['xg_value'].sum()
    }


def validate_shot_name(shot_name: str) -> bool:
    """
    Validate if shot name is unique and valid.
    
    Args:
        shot_name: Name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not shot_name or len(shot_name.strip()) == 0:
        return False
    
    # Ensure session state exists
    initialize_custom_shots_session()
    
    existing_names = [shot['shot_name'] for shot in st.session_state.custom_shots_data]
    
    return shot_name not in existing_names
