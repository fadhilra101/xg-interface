"""
Custom shot simulation page for the xG prediction application.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ..utils.data_processing import preprocess_shot_data
from ..utils.visualization import get_visualization_options, create_visualization_by_type, create_download_filename, get_plotly_chart_config
from ..utils.language import get_translation, get_language_options
from ..utils.custom_shot_manager import (
    initialize_custom_shots_session, add_custom_shot, get_custom_shots_dataframe,
    get_custom_shots_count, prepare_custom_shots_for_download, get_custom_shots_summary,
    clear_all_custom_shots, remove_custom_shot, validate_shot_name
)
from ..models.model_manager import predict_xg


def create_interactive_pitch_simple(current_x=108, current_y=40):
    """
    Create a simple interactive vertical pitch using Plotly.
    Takes horizontal coordinates but displays as vertical pitch.
    
    Args:
        current_x: Current x coordinate (horizontal system)
        current_y: Current y coordinate (horizontal system)
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Transform coordinates for vertical display
    display_x = current_y  # horizontal Y becomes vertical X
    display_y = current_x  # horizontal X becomes vertical Y (NO inversion)
    
    # Vertical pitch outline (80x120)
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=80, y1=120,
        line=dict(color="white", width=3),
        fillcolor="rgba(0,0,0,0)",
        layer='below'
    )
    
    # Center circle (adjusted for vertical)
    fig.add_shape(
        type="circle",
        x0=30, y0=50, x1=50, y1=70,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)",
        layer='below'
    )
    
    # Center line (horizontal for vertical pitch)
    fig.add_shape(
        type="line",
        x0=0, y0=60, x1=80, y1=60,
        line=dict(color="white", width=2),
        layer='below'
    )
    
    # Center spot
    fig.add_shape(
        type="circle",
        x0=39, y0=59, x1=41, y1=61,
        line=dict(color="white", width=1),
        fillcolor="white",
        layer='below'
    )
    
    # Top penalty area (attacking goal)
    fig.add_shape(
        type="rect",
        x0=22, y0=102, x1=58, y1=120,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)",
        layer='below'
    )
    
    # Bottom penalty area  
    fig.add_shape(
        type="rect",
        x0=22, y0=0, x1=58, y1=18,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)",
        layer='below'
    )
    
    # Top 6-yard box
    fig.add_shape(
        type="rect",
        x0=30, y0=114, x1=50, y1=120,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)",
        layer='below'
    )
    
    # Bottom 6-yard box
    fig.add_shape(
        type="rect",
        x0=30, y0=0, x1=50, y1=6,
        line=dict(color="white", width=2),
        fillcolor="rgba(0,0,0,0)",
        layer='below'
    )
    
    # Add current shot location marker (using transformed coordinates)
    fig.add_trace(go.Scatter(
        x=[display_x],
        y=[display_y],
        mode='markers',
        marker=dict(size=15, color='red', symbol='circle', 
                   line=dict(width=2, color='white')),
        name='Shot Location',
        hovertemplate=f"<b>Shot Location</b><br>x: {current_x:.1f}, y: {current_y:.1f}<extra></extra>"
    ))
    
    # Update layout for vertical pitch with unified theme
    fig.update_layout(
        plot_bgcolor='#22312b',
        paper_bgcolor='#1b2622',
        font=dict(color='white'),
        showlegend=False,
        width=400,  # Narrower for vertical
        height=600,  # Taller for vertical
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(range=[-5, 85], showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(range=[-5, 125], showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1)
    
    return fig


def render_custom_shot_page(model, lang="en"):
    """
    Render the custom shot simulation page.
    
    Args:
        model: Trained model pipeline
        lang: Language code ('en' or 'id')
    """
    # Initialize custom shots session
    initialize_custom_shots_session()
    
    st.header(get_translation("simulate_header", lang))
    st.markdown(get_translation("simulate_desc", lang))

    # Get localized options
    options = get_language_options(lang)

    # Initialize shot coordinates in session state (horizontal coordinates)
    if 'shot_x' not in st.session_state:
        st.session_state.shot_x = 108  # Near goal in horizontal coordinates
    if 'shot_y' not in st.session_state:
        st.session_state.shot_y = 40  # Center width in horizontal coordinates

    col1, col2 = st.columns((1, 1))

    with col1:
        st.subheader(get_translation("shot_characteristics", lang))
        
        # Play pattern selection
        play_pattern_label = st.selectbox(
            get_translation("play_pattern", lang), 
            options=list(options["play_pattern"].keys())
        )
        
        # Position selection
        position_label = st.selectbox(
            get_translation("player_position_select", lang), 
            options=list(options["position"].keys())
        )
        
        # Shot technique selection
        shot_tech_label = st.selectbox(
            get_translation("shot_technique_select", lang), 
            options=list(options["shot_technique"].keys())
        )
        
        # Body part selection
        body_part_label = st.selectbox(
            get_translation("body_part", lang), 
            options=list(options["shot_body_part"].keys())
        )
        
        # Period selection (moved from Additional Features)
        period_options = {
            get_translation("first_half", lang): 1,
            get_translation("second_half", lang): 2
        }
        period_label = st.selectbox(
            get_translation("period", lang), 
            options=list(period_options.keys())
        )
        period = period_options[period_label]

    with col2:
        # Shot type selection
        shot_type_label = st.selectbox(
            get_translation("shot_type", lang), 
            options=list(options["shot_type"].keys())
        )
        
        # Event before shot selection
        type_before_label = st.selectbox(
            get_translation("event_before", lang), 
            options=list(options["type_before"].keys())
        )
        
        st.subheader(get_translation("shot_context_section", lang))
        # Context checkboxes
        open_goal = st.checkbox(get_translation("open_goal", lang))
        one_on_one = st.checkbox(get_translation("one_on_one", lang))
        aerial_won = st.checkbox(get_translation("aerial_won", lang))
        under_pressure = st.checkbox(get_translation("under_pressure", lang))
        
        # Additional shot characteristics (moved from Additional Features)
        shot_first_time = st.checkbox(get_translation("shot_first_time", lang), value=False)
        shot_key_pass = st.checkbox(get_translation("shot_key_pass", lang), value=False)

        st.subheader(get_translation("time_of_shot", lang))
        
        # Time input with both sliders and number inputs
        col_time1, col_time2 = st.columns(2)
        
        with col_time1:
            minute_slider = st.slider(get_translation("minute", lang), 0, 120, 45, key="minute_slider")
            minute_input = st.number_input(
                get_translation("minute", lang),
                min_value=0,
                max_value=120,
                value=minute_slider,
                key="minute_input",
                label_visibility="collapsed"
            )
            # Use the input value if it's different from slider, otherwise use slider
            minute = minute_input if minute_input != minute_slider else minute_slider
            
        with col_time2:
            second_slider = st.slider(get_translation("second", lang), 0, 59, 30, key="second_slider")
            second_input = st.number_input(
                get_translation("second", lang),
                min_value=0,
                max_value=59,
                value=second_slider,
                key="second_input",
                label_visibility="collapsed"
            )
            # Use the input value if it's different from slider, otherwise use slider
            second = second_input if second_input != second_slider else second_slider

    # Shot location section with enhanced UI
    st.subheader("🎯 " + get_translation("shot_location_xy", lang))
    
    # Current position status
    distance_to_goal = 120 - st.session_state.shot_x
    if 36 <= st.session_state.shot_y <= 44:
        position_quality = "🎯 Excellent angle"
    elif 30 <= st.session_state.shot_y <= 50:
        position_quality = "👍 Good angle"
    else:
        position_quality = "⚠️ Wide angle"
    
    st.markdown(f"**📍 Current Position Analysis:** Distance to goal: **{distance_to_goal}m** • {position_quality}")
    
    # Instructions for coordinate selection
    st.info("🔧 Use the controls below to set your shot coordinates. The pitch will update in real-time!")
    
    # Create coordinate selection layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Display pitch visualization with current coordinates
        fig = create_interactive_pitch_simple(st.session_state.shot_x, st.session_state.shot_y)
        st.plotly_chart(fig, use_container_width=True, key="pitch_display", config=get_plotly_chart_config(hide_modebar=True))
    
    with col_right:
        # Professional coordinate input panel
        st.markdown("### 🎯 " + get_translation('current_coordinates', lang))
        
        # Current coordinates display with styling
        coord_col1, coord_col2 = st.columns(2)
        with coord_col1:
            st.metric(
                label="📍 X Position", 
                value=f"{st.session_state.shot_x}",
                help="0 = Own goal, 120 = Attacking goal"
            )
        with coord_col2:
            st.metric(
                label="📍 Y Position", 
                value=f"{st.session_state.shot_y}",
                help="0 = Left side, 80 = Right side"
            )
        
        st.markdown("---")
        st.markdown("### ⚙️ Manual Input")
        
        # Enhanced sliders with better styling and visual feedback
        st.markdown("**📏 Distance from Goal (X-Axis)**")
        new_x = st.slider(
            label="X Coordinate",
            min_value=0, max_value=120, 
            value=st.session_state.shot_x, 
            step=1,
            key="slider_x",
            help="🥅 0 = Own goal line • 120 = Opponent goal line",
            label_visibility="collapsed"
        )
        
        # Visual indicator for X position with distance info
        distance_to_goal = 120 - new_x
        if new_x <= 30:
            x_zone = "🔵 Defensive Third"
            zone_color = "blue"
        elif new_x <= 90:
            x_zone = "🟡 Middle Third"
            zone_color = "orange"
        else:
            x_zone = "🔴 Attacking Third"
            zone_color = "red"
        
        st.markdown(f"<span style='color: {zone_color}; font-weight: bold;'>{x_zone}</span> • Distance to goal: **{distance_to_goal}m**", unsafe_allow_html=True)
        
        st.markdown("**⬅️➡️ Side Position (Y-Axis)**")
        new_y = st.slider(
            label="Y Coordinate",
            min_value=0, max_value=80, 
            value=st.session_state.shot_y, 
            step=1,
            key="slider_y",
            help="⬅️ 0 = Left touchline • 80 = Right touchline",
            label_visibility="collapsed"
        )
        
        # Visual indicator for Y position with angle info
        if new_y <= 25:
            y_zone = "⬅️ Left Wing"
            angle_desc = "Wide angle"
        elif new_y <= 55:
            y_zone = "🎯 Central"
            angle_desc = "Good angle"
        else:
            y_zone = "➡️ Right Wing"
            angle_desc = "Wide angle"
        
        st.markdown(f"**{y_zone}** • {angle_desc}")
        
        # Quick position summary
        st.markdown("---")
        st.markdown("**📊 Position Summary**")
        
        # Calculate shot difficulty
        if new_x >= 105 and 30 <= new_y <= 50:
            difficulty = "🟢 Easy"
        elif new_x >= 90 and 20 <= new_y <= 60:
            difficulty = "🟡 Medium"
        elif new_x >= 75:
            difficulty = "🟠 Hard"
        else:
            difficulty = "🔴 Very Hard"
        
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.metric("Distance", f"{distance_to_goal}m", delta=None)
        with summary_col2:
            st.markdown(f"**Difficulty:** {difficulty}")
        
        st.caption(f"Position: {new_x}, {new_y} • Angle: {angle_desc}")
        
        # Visual indicator for Y position
        if new_y <= 25:
            y_zone = "⬅️ Left Wing"
        elif new_y <= 55:
            y_zone = "🎯 Central"
        else:
            y_zone = "➡️ Right Wing"
        
        # Auto-update when sliders change
        if new_x != st.session_state.shot_x or new_y != st.session_state.shot_y:
            st.session_state.shot_x = new_x
            st.session_state.shot_y = new_y
            st.rerun()
    
    # Enhanced pitch area selection
    st.markdown("---")
    st.markdown("### 🗺️ " + ("Quick Area Selection" if lang == "en" else "Pilihan Cepat Area"))
    st.caption("Click any area below to instantly set coordinates")
    
    # Organize areas by tactical zones
    attacking_areas = [
        ("area_penalty", "Penalty Area ⚽", 108, 40, "Area Penalti ⚽"),
        ("area_six_yard", "Six-yard Box 🥅", 116, 40, "Kotak 6 Yard 🥅"),
        ("area_far_post", "Far Post 🎯", 115, 30, "Tiang Jauh 🎯"),
        ("area_near_post", "Near Post 🎯", 115, 50, "Tiang Dekat 🎯"),
    ]
    
    midfield_areas = [
        ("area_outside_box", "Outside Box 📦", 85, 40, "Luar Kotak Penalti 📦"),
        ("area_center", "Center Circle ⭕", 60, 40, "Lingkaran Tengah ⭕"),
    ]
    
    wing_areas = [
        ("area_left_wing", "Left Wing ⬅️", 100, 15, "Sayap Kiri ⬅️"),
        ("area_right_wing", "Right Wing ➡️", 100, 65, "Sayap Kanan ➡️"),
        ("area_left_flank", "Left Flank ⬅️", 90, 10, "Sisi Kiri ⬅️"),
        ("area_right_flank", "Right Flank ➡️", 90, 70, "Sisi Kanan ➡️")
    ]
    
    # Attacking third
    st.markdown("**🔴 Attacking Third**")
    cols = st.columns(4)
    for i, (area_id, name_en, x, y, name_id) in enumerate(attacking_areas):
        with cols[i]:
            area_name = name_id if lang == "id" else name_en
            if st.button(area_name, key=area_id, help=f"Coordinates: ({x}, {y})", use_container_width=True):
                st.session_state.shot_x = x
                st.session_state.shot_y = y
                st.success(f"✅ Position set to ({x}, {y})")
                st.rerun()
    
    # Middle third
    st.markdown("**🟡 Middle Third**")
    cols = st.columns(2)
    for i, (area_id, name_en, x, y, name_id) in enumerate(midfield_areas):
        with cols[i]:
            area_name = name_id if lang == "id" else name_en
            if st.button(area_name, key=area_id, help=f"Coordinates: ({x}, {y})", use_container_width=True):
                st.session_state.shot_x = x
                st.session_state.shot_y = y
                st.success(f"✅ Position set to ({x}, {y})")
                st.rerun()
    
    # Wing areas
    st.markdown("**⬅️➡️ Wing Areas**")
    cols = st.columns(3)
    for i, (area_id, name_en, x, y, name_id) in enumerate(wing_areas):
        with cols[i % 3]:
            area_name = name_id if lang == "id" else name_en
            if st.button(area_name, key=area_id, help=f"Coordinates: ({x}, {y})", use_container_width=True):
                st.session_state.shot_x = x
                st.session_state.shot_y = y
                st.success(f"✅ Position set to ({x}, {y})")
                st.rerun()

    # Add custom preset coordinates
    st.markdown("---")
    st.markdown("**🎖️ Famous Shot Locations**")
    
    famous_shots = [
        ("penalty_spot", "Penalty Spot 🎯", 108, 40, "Titik Penalti 🎯"),
        ("edge_of_box", "Edge of Box 📦", 102, 40, "Pinggir Kotak 📦"),
        ("top_corner", "Top Corner 📐", 108, 35, "Sudut Atas 📐"),
        ("bottom_corner", "Bottom Corner 📐", 108, 45, "Sudut Bawah 📐"),
    ]
    
    cols = st.columns(4)
    for i, (shot_id, name_en, x, y, name_id) in enumerate(famous_shots):
        with cols[i]:
            shot_name = name_id if lang == "id" else name_en
            if st.button(shot_name, key=shot_id, help=f"Classic position: ({x}, {y})", use_container_width=True):
                st.session_state.shot_x = x
                st.session_state.shot_y = y
                st.success(f"🌟 Classic position set!")
                st.rerun()
    
    # Random coordinate generator for testing
    st.markdown("---")
    random_col1, random_col2 = st.columns(2)
    
    with random_col1:
        if st.button("🎲 Random Attacking Position", help="Generate random coordinates in attacking third", use_container_width=True):
            import random
            random_x = random.randint(90, 118)
            random_y = random.randint(10, 70)
            st.session_state.shot_x = random_x
            st.session_state.shot_y = random_y
            st.success(f"🎲 Random position: ({random_x}, {random_y})")
            st.rerun()
    
    with random_col2:
        if st.button("🏠 Reset to Default", help="Reset to penalty spot", use_container_width=True):
            st.session_state.shot_x = 108
            st.session_state.shot_y = 40
            st.success("↩️ Reset to penalty spot!")
            st.rerun()

    # Use session state coordinates for calculation
    start_x = st.session_state.shot_x
    start_y = st.session_state.shot_y

    # Prepare shot data outside of button click to ensure it's available for all operations
    shot_data = {
        'minute': minute,
        'second': second,
        'period': period,
        'play_pattern': options["play_pattern"][play_pattern_label],
        'position': options["position"][position_label],
        'shot_technique': options["shot_technique"][shot_tech_label],
        'shot_body_part': options["shot_body_part"][body_part_label],
        'shot_type': options["shot_type"][shot_type_label],
        'shot_open_goal': int(open_goal),
        'shot_one_on_one': int(one_on_one),
        'shot_aerial_won': int(aerial_won),
        'shot_first_time': int(shot_first_time),
        'shot_key_pass': int(shot_key_pass),
        'under_pressure': int(under_pressure),
        'start_x': start_x,
        'start_y': start_y,
        'type_before': options["type_before"][type_before_label],
    }

    # Shot name input (optional) - placed before the calculate button
    st.markdown("---")
    st.subheader(get_translation("enter_shot_name", lang) + " (Optional)")
    col_name, col_help = st.columns([3, 1])
    with col_name:
        # Handle resetting shot name after successful addition
        default_name = f"Shot {get_custom_shots_count() + 1}"
        if hasattr(st.session_state, 'reset_shot_name') and st.session_state.reset_shot_name:
            # Reset the flag and use new default name
            st.session_state.reset_shot_name = False
            if 'shot_name_input' in st.session_state:
                del st.session_state.shot_name_input
        
        shot_name = st.text_input(
            get_translation("enter_shot_name", lang),
            value=default_name,
            key="shot_name_input",
            help="Akan menggunakan nama default jika dikosongkan" if lang == "id" else "Will use default name if left empty"
        )
    with col_help:
        st.write("")  # Spacing
        st.info("💡 " + ("Semua tembakan akan otomatis disimpan ke koleksi" if lang == "id" else "All shots will be automatically saved to collection"))

    if st.button(get_translation("calculate_xg", lang), use_container_width=True):
        # Store shot data in session state for persistence
        st.session_state.current_shot_data = shot_data
        st.session_state.custom_shot_data = shot_data  # For work preservation check
        
        # Convert to DataFrame and preprocess
        shot_df = pd.DataFrame([shot_data])
        shot_df = preprocess_shot_data(shot_df)
        
        # Predict with safe handling and timing
        try:
            predictions, prediction_time = predict_xg(model, shot_df)
            
            if predictions is None:
                # Model went missing during prediction
                st.session_state.model_missing_mid_work = True
                st.rerun()
                return
            
            if predictions is not None:
                xg_value = predictions[0]
                st.session_state.current_xg_value = xg_value
                st.session_state.current_shot_df = shot_df

                # Display results with model performance timing
                col_xg, col_perf = st.columns([2, 1])
                with col_xg:
                    st.metric(
                        label=get_translation("predicted_xg", lang), 
                        value=f"{xg_value:.3f}"
                    )
                with col_perf:
                    st.metric(
                        label=f"⚡ {get_translation('model_performance', lang)}", 
                        value=f"{prediction_time:.4f}s"
                    )

                # (Removed) Single-shot visualizations. We now only show visualizations
                # based on the entire custom shots dataset below in the collection section.
                
                # Automatically add to collection - use default name if empty
                final_shot_name = shot_name.strip() if shot_name and shot_name.strip() else f"Shot {get_custom_shots_count() + 1}"
                
                if validate_shot_name(final_shot_name):
                    add_custom_shot(shot_data, xg_value, final_shot_name)
                    st.success(f"✅ {get_translation('shot_added_success', lang)} - {final_shot_name}")
                    # Set flag to reset input name on next run
                    st.session_state.reset_shot_name = True
                else:
                    # If default name conflicts, find next available number
                    counter = get_custom_shots_count() + 1
                    while not validate_shot_name(f"Shot {counter}"):
                        counter += 1
                    final_shot_name = f"Shot {counter}"
                    add_custom_shot(shot_data, xg_value, final_shot_name)
                    st.success(f"✅ {get_translation('shot_added_success', lang)} - {final_shot_name}")
                    # Set flag to reset input name on next run
                    st.session_state.reset_shot_name = True
                
        except Exception as e:
            st.error(f"{get_translation('prediction_error_custom', lang)} {e}")
    
    # Remove custom PNG download section and rely on Plotly modebar
    # (Previously provided manual PNG download buttons here.)
    
    # Custom Shots Collection Section
    st.markdown("---")
    st.header(get_translation("custom_shots_collection", lang))
    
    # Force refresh session state
    initialize_custom_shots_session()
    shots_count = get_custom_shots_count()
    
    if shots_count == 0:
        st.info(get_translation("no_custom_shots", lang))
    else:
        # Display summary statistics
        summary = get_custom_shots_summary()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(get_translation("shots_count", lang), summary['total_shots'])
        with col2:
            st.metric(get_translation("average_xg", lang), f"{summary['avg_xg']:.3f}")
        with col3:
            st.metric(get_translation("total_expected_goals", lang), f"{summary['total_expected_goals']:.3f}")
        with col4:
            st.metric("Min/Max xG", f"{summary['min_xg']:.3f}/{summary['max_xg']:.3f}")
        
        # Display custom shots table
        df_custom = get_custom_shots_dataframe()
        
        # Show table with shot management
        st.subheader("Shot Details")
        for _, shot in df_custom.iterrows():
            with st.expander(f"🎯 {shot['shot_name']} - xG: {shot['xg_value']:.3f}"):
                col_info, col_remove = st.columns([4, 1])
                
                with col_info:
                    st.write(f"**Created:** {shot['created_at']}")
                    st.write(f"**Location:** X={shot['start_x']}, Y={shot['start_y']}")
                    st.write(f"**Time:** {shot['minute']}:{shot['second']:02d} - Period {shot['period']}")
                    st.write(f"**xG Value:** {shot['xg_value']:.3f}")
                
                with col_remove:
                    if st.button(f"🗑️ {get_translation('remove_shot', lang)}", key=f"remove_{shot['shot_id']}"):
                        remove_custom_shot(shot['shot_id'])
                        st.rerun()
        
        # Visualization and download options for the entire simulation dataset
        st.subheader(get_translation("custom_shots_visualization", lang))

        # Visualization type fixed to Shot Map (dropdown removed)
        selected_custom_viz_type = 'shot_map'

        # Create interactive visualization for all custom shots (full pitch)
        fig_custom, ax_custom = create_visualization_by_type(
            df_custom,
            selected_custom_viz_type,
            get_translation("custom_shots_visualization", lang),
            half_pitch=False,
            custom_shots=True,
            interactive=True
        )

        # Display interactive visualization
        if ax_custom is None:  # This is a Plotly figure
            st.plotly_chart(
                fig_custom,
                use_container_width=True,
                config=get_plotly_chart_config(filename_base="custom_shots", scale=2, compact=True),
            )
        else:  # This is a matplotlib figure (fallback)
            st.pyplot(fig_custom)

        # Add Half-pitch visualization for all custom shots
        st.subheader(get_translation("half_pitch_map", lang))
        fig_custom_half, ax_custom_half = create_visualization_by_type(
            df_custom,
            selected_custom_viz_type,
            get_translation('half_pitch_map', lang),
            half_pitch=True,
            custom_shots=True,
            interactive=True
        )
        if ax_custom_half is None:
            st.plotly_chart(
                fig_custom_half,
                use_container_width=True,
                config=get_plotly_chart_config(filename_base="custom_shots_half", scale=2, compact=True),
            )
        else:
            st.pyplot(fig_custom_half)

        # Download options
        col_download, col_action = st.columns(2)

        with col_download:
            if shots_count >= 3:  # Only show CSV download if 3 or more shots
                csv_data = prepare_custom_shots_for_download()
                st.download_button(
                    label=f"📊 {get_translation('download_custom_shots', lang)} (CSV)",
                    data=csv_data,
                    file_name=create_download_filename("custom_shots_collection", "csv"),
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info(f"Need at least 3 shots for CSV download (current: {shots_count})")

        with col_action:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button(f"🗑️ {get_translation('clear_all_shots', lang)}", use_container_width=True):
                clear_all_custom_shots()
                st.success("All custom shots cleared!")
                st.rerun()
