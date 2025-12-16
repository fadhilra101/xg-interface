"""
Constants and mappings used throughout the xG prediction application.
"""

import os

# Model file path (absolute, relative to this src/utils/ location)
# This resolves to: <project_root>/xg_interface/xg_model.joblib on Streamlit
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_FILE = os.path.join(BASE_DIR, 'xg_model.joblib')

# Goal coordinates for distance and angle calculations
# Keep horizontal coordinates for model compatibility
GOAL_COORDS = (120, 40)

# Mappings for UI (User-friendly Label: StatsBomb ID)
PLAY_PATTERN_MAP = {
    'Regular Play': 1,
    'From Corner': 2,
    'From Free Kick': 3,
    'From Throw-in': 4,
    'From Counter': 6,
    'From Goal Kick': 7,
    'From Kick Off': 9
}

POSITION_MAP = {
    'Striker': 23,
    'Secondary Striker': 25,
    'Center Attacking Midfield': 19,
    'Right Wing': 17,
    'Left Wing': 21,
    'Right Midfield': 12,
    'Left Midfield': 16
}

SHOT_TECHNIQUE_MAP = {
    'Normal': 93,
    'Volley': 95,
    'Half Volley': 91,
    'Diving Header': 90,
    'Lob': 92,
    'Overhead Kick': 94,
    'Backheel': 89
}

SHOT_BODY_PART_MAP = {
    'Right Foot': 40,
    'Left Foot': 38,
    'Head': 37,
    'Other': -1
}

SHOT_TYPE_MAP = {
    'Open Play': 87,
    'Free Kick': 62,
    'Penalty': 88,
    'From Corner': 61,
    'Other': -1
}

TYPE_BEFORE_MAP = {
    'Pass': 30,
    'Carry': 43,
    'Dribble': 14,
    'Ball Recovery': 2,
    'Interception': 10,
    'Miscontrol': 38
}

# Required columns for dataset validation
REQUIRED_COLUMNS = [
    'minute', 'second', 'period', 'play_pattern', 'position', 'shot_technique',
    'shot_body_part', 'shot_type', 'shot_open_goal', 'shot_one_on_one',
    'shot_aerial_won', 'shot_first_time', 'shot_key_pass', 'under_pressure', 
    'start_x', 'start_y', 'type_before'
]

# Feature names for model input
FEATURE_NAMES = [
    'minute', 'second', 'period', 'play_pattern', 'position', 'shot_technique',
    'shot_body_part', 'shot_type', 'shot_open_goal', 'shot_one_on_one',
    'shot_aerial_won', 'shot_first_time', 'shot_key_pass', 'under_pressure', 
    'start_x', 'start_y', 'type_before', 'distance_to_goal', 'angle_to_goal'
]

# Categorical features for preprocessing
CATEGORICAL_FEATURES = [
    'period', 'play_pattern', 'position', 'shot_technique', 'shot_body_part', 'shot_type', 'type_before'
]
