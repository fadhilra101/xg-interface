"""
UI helpers to enforce a compact, neat, and consistent Streamlit UI across pages.
Applies a small, modern visual polish aligned with basic UI/UX principles:
- clear visual hierarchy with tighter spacing
- consistent typography and controls
- compact buttons and inputs
- reduced visual noise
"""

import streamlit as st


def apply_compact_ui():
    """Inject compact CSS once per session for a tidy, professional UI."""
    if st.session_state.get("_compact_ui_applied"):
        return

    css = """
    <style>
    /* Base */
    html, body { font-family: Inter, Segoe UI, Arial, sans-serif; }
    .block-container { padding-top: 0.75rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem; }

    /* Headings */
    h1 { font-size: 1.8rem; margin: 0.25rem 0 0.75rem 0; }
    h2 { font-size: 1.4rem; margin: 0.25rem 0 0.5rem 0; }
    h3 { font-size: 1.1rem; margin: 0.25rem 0 0.4rem 0; }
    h4 { font-size: 1.0rem; margin: 0.25rem 0 0.35rem 0; }

    /* Text spacing */
    [data-testid="stMarkdownContainer"] p { margin: 0.25rem 0 0.5rem 0; }
    .st-emotion-cache-1jicfl2 p { margin: 0.25rem 0 0.5rem 0; } /* fallback */

    /* Buttons */
    .stButton>button, .stDownloadButton>button { 
        padding: 0.45rem 0.8rem; border-radius: 8px; font-weight: 600; 
        line-height: 1.1; min-height: 34px;
    }
    .stButton>button small { display: none; }

    /* Inputs */
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
        padding: 0.4rem 0.6rem; min-height: 34px; border-radius: 6px;
    }
    [data-testid="stSelectbox"] div[data-baseweb] {
        min-height: 36px;
    }
    [data-testid="stSlider"] { padding-top: 0.2rem; padding-bottom: 0.2rem; }

    /* Labels */
    [data-testid="stWidgetLabel"] { font-size: 0.92rem; margin-bottom: 0.25rem; }

    /* Metrics */
    [data-testid="stMetric"] { padding: 0.35rem 0.5rem; border-radius: 8px; background: rgba(255,255,255,0.04); }
    [data-testid="stMetricValue"] { font-size: 1.25rem; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Expander */
    details>summary { font-weight: 600; }

    /* Divider */
    hr { margin: 0.75rem 0; opacity: 0.25; }

    /* Sidebar spacing */
    section[data-testid="stSidebar"] .block-container { padding-top: 0.5rem; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.session_state["_compact_ui_applied"] = True
