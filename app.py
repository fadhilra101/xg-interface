"""
SPDX-License-Identifier: MIT

xG Prediction Interface - Main Application

A Streamlit application for predicting Expected Goals (xG) from shot data.
Supports both batch prediction from uploaded datasets and individual shot simulation.
"""

import streamlit as st
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.model_manager import create_dummy_model_if_not_exists, load_model, safe_load_model, show_model_missing_during_work
from src.pages.dataset_prediction import render_dataset_prediction_page
from src.pages.custom_shot import render_custom_shot_page
from src.utils.language import LANGUAGES, TRANSLATIONS, get_translation
from src.utils.ui import apply_compact_ui


def initialize_session_state():
    """Initialize session state variables."""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'  # Default to English


def render_language_switcher():
    """Render the language switcher in the sidebar."""
    st.sidebar.markdown("---")
    
    # Get current language for display
    current_lang = st.session_state.language
    t_lang = get_translation("language", current_lang)
    
    # Language selector
    selected_display = None
    for display, code in LANGUAGES.items():
        if code == current_lang:
            selected_display = display
            break
    
    new_language_display = st.sidebar.selectbox(
        f"🌐 {t_lang}",
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(selected_display) if selected_display else 0
    )
    
    # Update language if changed
    new_language_code = LANGUAGES[new_language_display]
    if new_language_code != st.session_state.language:
        st.session_state.language = new_language_code
        st.rerun()


def render_author_info():
    """Render author information and copyright in the sidebar."""
    lang = st.session_state.language
    
    st.sidebar.markdown("---")
    
    # Author info section
    if lang == "id":
        st.sidebar.markdown("""
        ### 👨‍💻 Tentang Pembuat
        
        **Dibuat oleh:**  
        🎓 **Fadhil Raihan Akbar**  
        🏛️ **UIN Syarif Hidayatullah Jakarta**  
        📚 **Program Studi Sistem Informasi**  
        
        📝 **Tugas Akhir Strata 1**  
        *PENERAPAN LIGHT GRADIENT BOOSTING MACHINE (LIGHTGBM) UNTUK PREDIKSI NILAI EXPECTED GOALS (xG) DALAM ANALISIS SEPAK BOLA*
        
        **Kontak & Portfolio:**  
        🔗 [GitHub](https://github.com/fadhilra101)  
        📱 [Instagram](https://www.instagram.com/fadhilra_)
        
        ---
        💡 *Aplikasi ini dikembangkan sebagai bagian dari penelitian tugas akhir untuk membantu analisis performa dalam sepak bola menggunakan teknologi machine learning.*
        """)
    else:
        st.sidebar.markdown("""
        ### 👨‍💻 About the Creator
        
        **Created by:**  
        🎓 **Fadhil Raihan Akbar**  
        🏛️ **UIN Syarif Hidayatullah Jakarta**  
        📚 **Information Systems Study Program**  
        
        📝 **Undergraduate Thesis Project**  
        *APPLICATION OF LIGHT GRADIENT BOOSTING MACHINE (LIGHTGBM) FOR EXPECTED GOALS (xG) VALUE PREDICTION IN FOOTBALL ANALYSIS*
        
        **Contact & Portfolio:**  
        🔗 [GitHub](https://github.com/fadhilra101)  
        📱 [Instagram](https://www.instagram.com/fadhilra_)
        
        ---
        💡 *This application was developed as part of undergraduate thesis research to assist football performance analysis using machine learning technology.*
        """)
    
    # Copyright notice
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 20px;'>
    © 2025 Fadhil Raihan Akbar<br>
    All Rights Reserved
    </div>
    """, unsafe_allow_html=True)


def render_model_missing_sidebar(lang):
    """Render helpful sidebar information when model is missing."""
    st.sidebar.markdown("---")
    
    # Model status
    st.sidebar.error("🚫 " + get_translation("model_required", lang))
    
    # Quick help
    with st.sidebar.expander("🆘 " + get_translation("quick_help", lang) if "quick_help" in TRANSLATIONS[lang] else "🆘 Quick Help"):
        st.markdown(f"""
        **{get_translation("steps_to_fix", lang) if "steps_to_fix" in TRANSLATIONS[lang] else "Steps to Fix:"}**
        
        1. 📁 {get_translation("locate_model", lang) if "locate_model" in TRANSLATIONS[lang] else "Locate your model file"}
        2. 📋 {get_translation("copy_to_root", lang) if "copy_to_root" in TRANSLATIONS[lang] else "Copy to root directory"}
        3. 🔄 {get_translation("restart_app", lang) if "restart_app" in TRANSLATIONS[lang] else "Restart application"}
        """)
    
    # Features available after model setup
    with st.sidebar.expander("⚽ " + get_translation("available_features", lang) if "available_features" in TRANSLATIONS[lang] else "⚽ Available Features"):
        st.markdown(f"""
        **{get_translation("after_model_setup", lang) if "after_model_setup" in TRANSLATIONS[lang] else "After model setup:"}**
        
        • 📊 {get_translation("predict_from_dataset", lang)}
        • 🎯 {get_translation("simulate_custom_shot", lang)}
        • 📈 {get_translation("advanced_visualizations", lang) if "advanced_visualizations" in TRANSLATIONS[lang] else "Advanced visualizations"}
        """)


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Get current language
    lang = st.session_state.language
    
    # Configure Streamlit page
    st.set_page_config(
        layout="wide", 
        page_title="xG Prediction App",
        page_icon="⚽"
    )
    
    # Apply compact UI across the app
    apply_compact_ui()
    
    # App title with version
    st.title(f"{get_translation('app_title', lang)} v0.0.1")
    
    # Sidebar navigation title
    st.sidebar.title(get_translation("navigation", lang))
    
    # Check if model exists first
    model_available = create_dummy_model_if_not_exists(lang)
    
    # Check if we're in the middle of work and model went missing
    if 'model_missing_mid_work' in st.session_state and st.session_state.model_missing_mid_work:
        render_language_switcher()
        render_author_info()
        show_model_missing_during_work(lang)
        return
    
    if not model_available:
        # Model not found - show language switcher and help
        render_language_switcher()
        render_author_info()
        render_model_missing_sidebar(lang)
        return
    
    # Initialize model with safe loading
    model = safe_load_model()
    
    if model is None:
        # Model failed to load - might have been deleted during work
        if 'custom_shot_data' in st.session_state:
            # User has work in progress in custom shots
            st.session_state.model_missing_mid_work = True
            st.rerun()
        else:
            # No work in progress, show regular error
            st.error(get_translation("model_load_error", lang))
            render_language_switcher()
            render_author_info()
            st.sidebar.markdown("---")
            st.sidebar.error(get_translation("model_load_error", lang))
        return
    
    # Page selection (when model is available) - Custom Shot as main page
    page_options = [
        get_translation("simulate_custom_shot", lang),
        get_translation("predict_from_dataset", lang)
    ]
    
    page = st.sidebar.radio(
        get_translation("navigation", lang), 
        page_options,
        label_visibility="collapsed"
    )
    
    # Language switcher and author info at bottom
    render_language_switcher()
    render_author_info()
    
    # Route to appropriate page - Custom Shot first
    if page == get_translation("simulate_custom_shot", lang):
        render_custom_shot_page(model, lang)
    elif page == get_translation("predict_from_dataset", lang):
        render_dataset_prediction_page(model, lang)


if __name__ == "__main__":
    main()
