"""
Configuration settings for the xG prediction application.
"""

# Application settings
APP_TITLE = "⚽ Expected Goals (xG) Prediction Interface"
APP_LAYOUT = "wide"

# Model settings
MODEL_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

# Visualization settings
SHOT_MAP_FIGSIZE = (12, 8)
SINGLE_SHOT_FIGSIZE = (10, 7)
PREVIEW_FIGSIZE = (8, 6)

# Pitch colors
DARK_PITCH_COLOR = '#22312b'
DARK_LINE_COLOR = '#c7d5cc'
LIGHT_PITCH_COLOR = '#f4f4f4'
LIGHT_LINE_COLOR = 'grey'

# Default values for custom shot simulation
DEFAULT_MINUTE = 45
DEFAULT_SECOND = 30
DEFAULT_START_X = 108
DEFAULT_START_Y = 40

# File upload settings
ALLOWED_FILE_TYPES = ["csv"]
MAX_FILE_SIZE_MB = 200
