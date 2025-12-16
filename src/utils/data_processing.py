"""
Data processing utilities for the xG prediction application.
"""

import numpy as np
import pandas as pd
from .constants import GOAL_COORDS


def calculate_distance_and_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates distance and angle from shot coordinates to the center of the goal.
    
    Args:
        df: DataFrame containing shot data with 'start_x' and 'start_y' columns
        
    Returns:
        DataFrame with added 'distance_to_goal' and 'angle_to_goal' columns
    """
    df = df.copy()
    df['distance_to_goal'] = np.sqrt(
        (df['start_x'] - GOAL_COORDS[0])**2 + (df['start_y'] - GOAL_COORDS[1])**2
    )
    # Angle in radians, then converted to degrees for easier interpretation if needed
    df['angle_to_goal'] = np.arctan2(
        np.abs(df['start_y'] - GOAL_COORDS[1]), 
        df['start_x'] - GOAL_COORDS[0]
    )
    return df


def bin_shot_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bins the 'shot_type' column to reduce cardinality.
    Keep Open Play (87) and Free Kick (62), group others into -1.
    
    Args:
        df: DataFrame containing shot data with 'shot_type' column
        
    Returns:
        DataFrame with binned 'shot_type' column
    """
    df = df.copy()
    df['shot_type'] = df['shot_type'].apply(lambda x: x if x in [87, 62] else -1)
    return df


def bin_shot_body_part(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bins the 'shot_body_part' column.
    Keep Right Foot (40), Left Foot (38), Head (37), group others into -1.
    
    Args:
        df: DataFrame containing shot data with 'shot_body_part' column
        
    Returns:
        DataFrame with binned 'shot_body_part' column
    """
    df = df.copy()
    df['shot_body_part'] = df['shot_body_part'].apply(lambda x: x if x in [40, 38, 37] else -1)
    return df


def preprocess_shot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to shot data.
    
    Args:
        df: Raw shot data DataFrame
        
    Returns:
        Preprocessed DataFrame ready for model prediction
    """
    processed_df = df.copy()
    
    # Add missing columns with default values if they don't exist
    if 'period' not in processed_df.columns:
        # Default to first half (1) for custom shots
        processed_df['period'] = 1
    
    if 'shot_first_time' not in processed_df.columns:
        # Default to not first time (0)
        processed_df['shot_first_time'] = 0
    
    if 'shot_key_pass' not in processed_df.columns:
        # Default to not from key pass (0)
        processed_df['shot_key_pass'] = 0
    
    # Apply existing preprocessing
    processed_df = calculate_distance_and_angle(processed_df)
    processed_df = bin_shot_type(processed_df)
    processed_df = bin_shot_body_part(processed_df)
    
    return processed_df


def validate_columns(df: pd.DataFrame, required_columns: list) -> list:
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        List of missing column names (empty if all columns present)
    """
    return [col for col in required_columns if col not in df.columns]
