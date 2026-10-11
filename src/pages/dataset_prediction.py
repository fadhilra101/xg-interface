"""
Dataset prediction page for the xG prediction application.
"""

import streamlit as st
import pandas as pd

from ..utils.constants import REQUIRED_COLUMNS
from ..utils.data_processing import validate_columns, preprocess_shot_data
from ..utils.visualization import create_shot_map, create_half_pitch_shot_map, save_figure_to_bytes, create_download_filename, prepare_csv_download, create_visualization_by_type, get_plotly_chart_config
from ..utils.language import get_translation
from ..models.model_manager import predict_xg
from ..utils.plotly_export import fig_to_png_bytes_plotly


def render_dataset_prediction_page(model, lang="en"):
    """
    Render the dataset prediction page.
    
    Args:
        model: Trained model pipeline
        lang: Language code ('en' or 'id')
    """
    st.header(get_translation("predict_batch_header", lang))
    st.markdown(get_translation("predict_batch_desc", lang))

    # Show required columns before upload
    st.write(f"### {get_translation('required_columns', lang)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{get_translation('time_context', lang)}**")
        st.write(f"• `minute` - {get_translation('minute_desc', lang)}")
        st.write(f"• `second` - {get_translation('second_desc', lang)}")
        st.write(f"• `period` - {get_translation('period_desc', lang)}")
        st.write(f"• `play_pattern` - {get_translation('play_pattern_desc', lang)}")
        st.write(f"• `type_before` - {get_translation('type_before_desc', lang)}")
        
        st.write(f"**{get_translation('player_position', lang)}**")
        st.write(f"• `position` - {get_translation('position_desc', lang)}")
        st.write(f"• `under_pressure` - {get_translation('under_pressure_desc', lang)}")
        
        st.write(f"**{get_translation('shot_technique', lang)}**")
        st.write(f"• `shot_technique` - {get_translation('shot_technique_desc', lang)}")
        st.write(f"• `shot_first_time` - {get_translation('shot_first_time_desc', lang)}")
        st.write(f"• `shot_body_part` - {get_translation('shot_body_part_desc', lang)}")
        st.write(f"• `shot_type` - {get_translation('shot_type_desc', lang)}")
        st.write(f"• `shot_key_pass` - {get_translation('shot_key_pass_desc', lang)}")
    
    with col2:
        st.write(f"**{get_translation('shot_location', lang)}**")
        st.write(f"• `start_x` - {get_translation('start_x_desc', lang)}")
        st.write(f"• `start_y` - {get_translation('start_y_desc', lang)}")
        
        st.write(f"**{get_translation('shot_context', lang)}**")
        st.write(f"• `shot_open_goal` - {get_translation('shot_open_goal_desc', lang)}")
        st.write(f"• `shot_one_on_one` - {get_translation('shot_one_on_one_desc', lang)}")
        st.write(f"• `shot_aerial_won` - {get_translation('shot_aerial_won_desc', lang)}")
        
        st.write(f"**{get_translation('optional_analysis', lang)}**")
        st.write(f"• `shot_outcome` - {get_translation('shot_outcome_desc', lang)}")
        st.write(f"• `team_name` - {get_translation('team_name_desc', lang)}")
        st.write(f"• `player_name` - {get_translation('player_name_desc', lang)}")
    
    st.info(get_translation("tip_message", lang))

    # Visual divider
    st.markdown("---")
    
    # Download CSV template (required + optional columns)
    try:
        optional_columns = ['shot_outcome', 'team_name', 'player_name']
        template_columns = list(REQUIRED_COLUMNS) + [c for c in optional_columns if c not in REQUIRED_COLUMNS]
        template_df = pd.DataFrame(columns=template_columns)
        template_csv_bytes = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=("📥 Unduh Template CSV" if lang == 'id' else "📥 Download CSV Template"),
            data=template_csv_bytes,
            file_name=create_download_filename("xg_dataset_template", "csv"),
            mime="text/csv",
            help=(
                "Template berisi kolom wajib dan opsional (mis. shot_outcome, team_name, player_name)" 
                if lang == 'id' else 
                "Template includes required and optional columns (e.g., shot_outcome, team_name, player_name)"
            ),
            use_container_width=True
        )
    except Exception:
        pass
    
    # StatsBomb reference section
    if lang == "id":
        st.markdown("""
        ### 📚 Referensi Format Data
        
        Data yang digunakan dalam aplikasi ini mengikuti format **StatsBomb Open Data**. 
        
        🔗 **Untuk referensi lengkap tentang format ID dan struktur data:**
        - [StatsBomb Open Data GitHub Repository](https://github.com/statsbomb/open-data)
        - [Dokumentasi Spesifikasi Data](https://github.com/statsbomb/open-data/blob/master/doc/Open%20Data%20Specification%20v1.1.0.pdf)
        
        💡 **Tips:** Gunakan repositori StatsBomb untuk memahami mapping ID yang tepat untuk setiap field (play_pattern, position, shot_technique, dll.)
        """)
    else:
        st.markdown("""
        ### 📚 Data Format Reference
        
        The data used in this application follows the **StatsBomb Open Data** format.
        
        🔗 **For complete reference on ID formats and data structure:**
        - [StatsBomb Open Data GitHub Repository](https://github.com/statsbomb/open-data)
        - [Data Specification Documentation](https://github.com/statsbomb/open-data/blob/master/doc/Open%20Data%20Specification%20v1.1.0.pdf)
        
        💡 **Tips:** Use the StatsBomb repository to understand the correct ID mapping for each field (play_pattern, position, shot_technique, etc.)
        """)
    
    st.markdown("---")
    
    with st.expander(get_translation("detailed_descriptions", lang)):
        st.write(get_translation("play_pattern_ids", lang))
        st.write(get_translation("play_pattern_values", lang))
        
        st.write(get_translation("position_ids", lang))
        st.write(get_translation("position_values", lang))
        
        st.write(get_translation("shot_technique_ids", lang))
        st.write(get_translation("shot_technique_values", lang))
        
        st.write(get_translation("body_part_ids", lang))
        st.write(get_translation("body_part_values", lang))
        
        st.write(get_translation("shot_type_ids", lang))
        st.write(get_translation("shot_type_values", lang))
        
        st.write(get_translation("event_before_ids", lang))
        st.write(get_translation("event_before_values", lang))

    uploaded_file = st.file_uploader(get_translation("choose_csv", lang), type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"### {get_translation('upload_preview', lang)}")
        st.dataframe(df.head())

        # Validate columns
        missing_cols = validate_columns(df, REQUIRED_COLUMNS)

        if missing_cols:
            st.error(f"{get_translation('missing_columns', lang)} {', '.join(missing_cols)}")
        else:
            # Process the uploaded data directly without session state storage
            
            with st.spinner(get_translation('calculating_xg', lang)):
                try:
                    # Apply preprocessing
                    processed_df = preprocess_shot_data(df)
                    
                    # Predict probabilities - now with safe handling and timing
                    predictions, prediction_time = predict_xg(model, processed_df)
                    
                    if predictions is None:
                        # Model went missing during prediction
                        st.session_state.model_missing_mid_work = True
                        st.rerun()
                        return
                        
                    if predictions is not None:
                        processed_df['xG'] = predictions
                        
                        # Display model performance timing
                        col_perf1, col_perf2, col_perf3 = st.columns(3)
                        with col_perf1:
                            st.metric(f"⚡ {get_translation('model_performance', lang)}", f"{prediction_time:.4f}s")
                        with col_perf2:
                            shots_per_second = len(processed_df) / prediction_time if prediction_time > 0 else 0
                            st.metric("🏃‍♂️ Shots/Second", f"{shots_per_second:.1f}")
                        with col_perf3:
                            st.metric("📊 Total Shots", len(processed_df))

                        st.write(f"### {get_translation('results_predictions', lang)}")
                        
                        try:
                            # Reorder columns for better analyst view
                            # Key columns first: xG, shot_outcome, then important shot details
                            key_columns = []
                            
                            # Add xG first
                            key_columns.append('xG')
                            
                            # Add shot_outcome if it exists
                            if 'shot_outcome' in processed_df.columns:
                                key_columns.append('shot_outcome')
                            elif 'is_goal' in processed_df.columns:
                                key_columns.append('is_goal')
                            
                            # Add important shot context columns
                            important_cols = ['minute', 'start_x', 'start_y', 'shot_type', 'shot_body_part', 
                                            'shot_technique', 'position', 'play_pattern',
                                            'player_name', 'team_name']
                            
                            for col in important_cols:
                                if col in processed_df.columns and col not in key_columns:
                                    key_columns.append(col)
                            
                            # Add remaining columns
                            remaining_cols = [col for col in processed_df.columns if col not in key_columns]
                            
                            # Final column order
                            display_columns = key_columns + remaining_cols
                            display_df = processed_df[display_columns].copy()
                            
                            # Format xG to 3 decimal places for better readability
                            if 'xG' in display_df.columns:
                                display_df['xG'] = display_df['xG'].round(3)
                            
                        except Exception as e:
                            st.warning(f"{get_translation('column_reorder_error', lang)} {e}")
                            display_df = processed_df.copy()
                            if 'xG' in display_df.columns:
                                display_df['xG'] = display_df['xG'].round(3)
                        
                        # Display with improved styling
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=400,
                            column_config={
                                "xG": st.column_config.NumberColumn(
                                    "xG",
                                    help="Expected Goals probability",
                                    format="%.3f",
                                    width="small"
                                ),
                                "shot_outcome": st.column_config.TextColumn(
                                    "Shot Outcome",
                                    help="Actual shot result",
                                    width="medium"
                                ),
                                "player_name": st.column_config.TextColumn(
                                    get_translation("player_name", lang),
                                    help=get_translation("player_name_desc", lang),
                                    width="medium"
                                ),
                                "team_name": st.column_config.TextColumn(
                                    get_translation("team_name", lang),
                                    help=get_translation("team_name_desc", lang),
                                    width="medium"
                                ),
                                "is_goal": st.column_config.NumberColumn(
                                    "Goal",
                                    help="1 = Goal, 0 = No Goal",
                                    width="small"
                                ),
                                "minute": st.column_config.NumberColumn(
                                    get_translation("minute", lang),
                                    help=get_translation("minute_desc", lang),
                                    width="small"
                                ),
                                "start_x": st.column_config.NumberColumn(
                                    "X",
                                    help="X coordinate",
                                    width="small"
                                ),
                                "start_y": st.column_config.NumberColumn(
                                    "Y", 
                                    help="Y coordinate",
                                    width="small"
                                )
                            }
                        )
                        
                        # Add summary statistics for analysts
                        st.write(f"### {get_translation('summary_stats', lang)}")
                        
                        try:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                total_shots = len(display_df)
                                st.metric(get_translation("total_shots", lang), total_shots)
                            
                            with col2:
                                avg_xg = display_df['xG'].mean()
                                st.metric(get_translation("average_xg", lang), f"{avg_xg:.3f}")
                            
                            with col3:
                                total_xg = display_df['xG'].sum()
                                st.metric(get_translation("total_xg", lang), f"{total_xg:.2f}")
                            
                            with col4:
                                try:
                                    if 'shot_outcome' in display_df.columns:
                                        # shot_outcome contains 1 (goal) and 0 (no goal)
                                        goals_scored = int(display_df['shot_outcome'].sum())
                                    else:
                                        goals_scored = get_translation("no_shot_outcome", lang)
                                except Exception as e:
                                    goals_scored = f"{get_translation('error_prefix', lang)} {str(e)}"
                                st.metric(get_translation("goals_scored", lang), goals_scored)
                        
                        except Exception as e:
                            st.warning(f"{get_translation('summary_stats_error', lang)} {e}")
                            st.info(get_translation("predictions_accurate", lang))

                        # Visualization section (fixed to Shot Map, selector removed)
                        st.write(f"### {get_translation('shot_map', lang)}")
                        selected_viz_type = 'shot_map'
                        
                        # Create the interactive full pitch visualization
                        fig_full, ax_full = create_visualization_by_type(
                            processed_df, 
                            selected_viz_type, 
                            get_translation('shot_map', lang),
                            half_pitch=False,
                            interactive=True
                        )
                        
                        # Display interactive visualization using plotly_chart
                        if ax_full is None:  # This is a Plotly figure
                            st.plotly_chart(
                                fig_full,
                                use_container_width=True,
                                config=get_plotly_chart_config(filename_base="shot_map_full", scale=2, compact=True),
                            )
                        else:  # This is a matplotlib figure (fallback)
                            st.pyplot(fig_full)
                        
                        # Create half pitch visualization  
                        st.write(f"### {get_translation('half_pitch_map', lang)}")
                        fig_half, ax_half = create_visualization_by_type(
                            processed_df, 
                            selected_viz_type, 
                            get_translation('half_pitch_map', lang),
                            half_pitch=True,
                            interactive=True
                        )
                        
                        # Display interactive half pitch
                        if ax_half is None:  # This is a Plotly figure
                            st.plotly_chart(
                                fig_half,
                                use_container_width=True,
                                config=get_plotly_chart_config(filename_base="shot_map_half", scale=2, compact=True),
                            )
                        else:  # This is a matplotlib figure (fallback)
                            st.pyplot(fig_half)
                        
                        # Download section
                        st.write(f"### {get_translation('download_section', lang)}")
                        
                        # Only CSV download; use Plotly's built-in modebar for PNG
                        csv_data = prepare_csv_download(display_df)
                        st.download_button(
                            label=get_translation("download_csv", lang),
                            data=csv_data,
                            file_name=create_download_filename("xg_predictions", "csv"),
                            mime="text/csv",
                            help=get_translation("download_csv_desc", lang),
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"{get_translation('prediction_error', lang)} {e}")
                    st.info(get_translation('data_mismatch', lang))
