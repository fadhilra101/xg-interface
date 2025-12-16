"""
Alternative visualization using Seaborn and manual pitch drawing.
This approach gives us full control over coordinates and marker visibility.
Uses consistent theming with main visualization module.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64

# Import consistent theme from main visualization module
try:
    from .visualization import PITCH_BG_COLOR, PAPER_BG_COLOR, XG_COLORS
except ImportError:
    # Fallback definitions if import fails
    PITCH_BG_COLOR = '#0a1f14'
    PAPER_BG_COLOR = '#0d1117'
    XG_COLORS = {
        'very_high': '#ff4757',
        'high': '#ff6348',
        'medium': '#ffa502',
        'low': '#3742fa',
        'very_low': '#70a1ff',
    }


def draw_manual_pitch(ax, half_pitch=False):
    """
    Manually draw a football pitch with exact coordinate control and consistent theming.
    
    Args:
        ax: Matplotlib axis
        half_pitch: Whether to draw half pitch only
    """
    # Set consistent pitch color and style
    ax.set_facecolor(PITCH_BG_COLOR)
    
    if half_pitch:
        # Half pitch dimensions (attacking half)
        pitch_width = 80
        pitch_length = 60  # Half of 120
        x_min, x_max = 60, 120
        y_min, y_max = 0, 80
        
        # Draw pitch outline with consistent background
        pitch_rect = patches.Rectangle((60, 0), 60, 80, 
                                     linewidth=4, edgecolor='white', 
                                     facecolor=PITCH_BG_COLOR, fill=True)
        ax.add_patch(pitch_rect)
        
        # Goal area (18-yard box)
        goal_area = patches.Rectangle((102, 22), 18, 36, 
                                    linewidth=3, edgecolor='white', 
                                    facecolor='none')
        ax.add_patch(goal_area)
        
        # Six-yard box
        six_yard = patches.Rectangle((114, 30), 6, 20, 
                                   linewidth=2, edgecolor='white', 
                                   facecolor='none')
        ax.add_patch(six_yard)
        
        # Goal
        goal = patches.Rectangle((120, 36), 2, 8, 
                               linewidth=4, edgecolor='white', 
                               facecolor='white')
        ax.add_patch(goal)
        
        # Center line
        ax.plot([60, 60], [0, 80], color='white', linewidth=3)
        
        # Center circle (partial)
        circle = patches.Circle((60, 40), 9.15, linewidth=2, 
                              edgecolor='white', facecolor='none')
        ax.add_patch(circle)
        
        # Set limits
        ax.set_xlim(55, 125)
        ax.set_ylim(-5, 85)
        
    else:
        # Full pitch
        x_min, x_max = 0, 120
        y_min, y_max = 0, 80
        
        # Draw pitch outline with consistent background
        pitch_rect = patches.Rectangle((0, 0), 120, 80, 
                                     linewidth=4, edgecolor='white', 
                                     facecolor=PITCH_BG_COLOR, fill=True)
        ax.add_patch(pitch_rect)
        
        # Center line
        ax.plot([60, 60], [0, 80], color='white', linewidth=3)
        
        # Center circle
        circle = patches.Circle((60, 40), 9.15, linewidth=2, 
                              edgecolor='white', facecolor='none')
        ax.add_patch(circle)
        
        # Center spot
        ax.plot(60, 40, 'o', color='white', markersize=3)
        
        # Left goal area
        left_goal_area = patches.Rectangle((0, 22), 18, 36, 
                                         linewidth=2, edgecolor='white', 
                                         facecolor='none')
        ax.add_patch(left_goal_area)
        
        # Left six-yard box
        left_six_yard = patches.Rectangle((0, 30), 6, 20, 
                                        linewidth=2, edgecolor='white', 
                                        facecolor='none')
        ax.add_patch(left_six_yard)
        
        # Right goal area
        right_goal_area = patches.Rectangle((102, 22), 18, 36, 
                                          linewidth=2, edgecolor='white', 
                                          facecolor='none')
        ax.add_patch(right_goal_area)
        
        # Right six-yard box
        right_six_yard = patches.Rectangle((114, 30), 6, 20, 
                                         linewidth=2, edgecolor='white', 
                                         facecolor='none')
        ax.add_patch(right_six_yard)
        
        # Goals
        left_goal = patches.Rectangle((-2, 36), 2, 8, 
                                    linewidth=4, edgecolor='white', 
                                    facecolor='white')
        ax.add_patch(left_goal)
        
        right_goal = patches.Rectangle((120, 36), 2, 8, 
                                     linewidth=4, edgecolor='white', 
                                     facecolor='white')
        ax.add_patch(right_goal)
        
        # Set limits
        ax.set_xlim(-5, 125)
        ax.set_ylim(-5, 85)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def create_seaborn_shot_map(df: pd.DataFrame, title: str = "Shot Map with xG", 
                           half_pitch: bool = False) -> tuple:
    """
    Create shot map using Seaborn with manual pitch drawing and ultra-bright markers.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        half_pitch: Whether to show half pitch only
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Debug info
    print(f"DEBUG - Seaborn Shot Map:")
    print(f"- Input shots: {len(df)}")
    if len(df) > 0:
        print(f"- X range: {df['start_x'].min():.2f} to {df['start_x'].max():.2f}")
        print(f"- Y range: {df['start_y'].min():.2f} to {df['start_y'].max():.2f}")
        print(f"- xG range: {df['xG'].min():.3f} to {df['xG'].max():.3f}")
    
    # Validate coordinates for StatsBomb pitch (0-120 x 0-80)
    if half_pitch:
        valid_shots = df[
            (df['start_x'] >= 60) & (df['start_x'] <= 120) & 
            (df['start_y'] >= 0) & (df['start_y'] <= 80)
        ].copy()
    else:
        valid_shots = df[
            (df['start_x'] >= 0) & (df['start_x'] <= 120) & 
            (df['start_y'] >= 0) & (df['start_y'] <= 80)
        ].copy()
    
    print(f"- Valid shots: {len(valid_shots)}")
    
    if len(valid_shots) == 0:
        print("ERROR: No valid shots for visualization!")
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.text(0.5, 0.5, "No valid shots to display", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=20, color='white', fontweight='bold')
        ax.set_facecolor(PITCH_BG_COLOR)
        fig.patch.set_facecolor(PAPER_BG_COLOR)
        return fig, ax
    
    # Create figure
    if half_pitch:
        fig, ax = plt.subplots(figsize=(12, 16))
    else:
        fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set consistent backgrounds
    fig.patch.set_facecolor(PAPER_BG_COLOR)
    ax.set_facecolor(PITCH_BG_COLOR)
    
    # Draw the pitch manually
    draw_manual_pitch(ax, half_pitch=half_pitch)
    
    # Prepare data for visualization
    valid_shots = valid_shots.copy()
    
    # Create color mapping based on xG using consistent theme colors
    def xg_to_color(xg):
        if xg >= 0.7:
            return XG_COLORS['very_high']  # Enhanced red
        elif xg >= 0.5:
            return XG_COLORS['high']       # Coral red-orange
        elif xg >= 0.3:
            return XG_COLORS['medium']     # Bright orange
        elif xg >= 0.1:
            return XG_COLORS['low']        # Bright blue
        else:
            return XG_COLORS['very_low']   # Light blue
    
    # Create size mapping based on xG
    def xg_to_size(xg):
        return (xg * 2000) + 500  # Large sizes
    
    # Apply mappings
    valid_shots['color'] = valid_shots['xG'].apply(xg_to_color)
    valid_shots['size'] = valid_shots['xG'].apply(xg_to_size)
    
    # Create scatter plot using seaborn/matplotlib
    for idx, row in valid_shots.iterrows():
        ax.scatter(row['start_x'], row['start_y'], 
                  s=row['size'],
                  c=row['color'],
                  alpha=1.0,
                  edgecolors='white',
                  linewidths=6,
                  zorder=100)  # Very high z-order
        
        # Add xG annotation for each shot with consistent styling
        ax.annotate(f"{row['xG']:.2f}", 
                   (row['start_x'], row['start_y']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, color='white', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=PAPER_BG_COLOR, 
                           edgecolor='white', alpha=0.8))
    
    # Create legend with consistent colors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['very_high'], 
                  markeredgecolor='white', markeredgewidth=3, markersize=15, 
                  linestyle='None', label='Very High xG (≥0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['high'], 
                  markeredgecolor='white', markeredgewidth=3, markersize=12, 
                  linestyle='None', label='High xG (0.5-0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['medium'], 
                  markeredgecolor='white', markeredgewidth=3, markersize=10, 
                  linestyle='None', label='Medium xG (0.3-0.5)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['low'], 
                  markeredgecolor='white', markeredgewidth=3, markersize=8, 
                  linestyle='None', label='Low xG (0.1-0.3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['very_low'], 
                  markeredgecolor='white', markeredgewidth=3, markersize=6, 
                  linestyle='None', label='Very Low xG (<0.1)')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(1.02, 1), fontsize=11,
                      facecolor=PAPER_BG_COLOR, edgecolor='white', labelcolor='white')
    legend.get_frame().set_linewidth(2)
    
    # Title with consistent styling
    ax.set_title(title, fontsize=20, color='white', fontweight='bold', pad=20,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=PAPER_BG_COLOR, 
                        edgecolor="white", linewidth=2))
    
    # Summary info with consistent styling
    total_xg = valid_shots['xG'].sum()
    summary_text = f"{len(valid_shots)} shots | Total xG: {total_xg:.2f}"
    
    if half_pitch:
        ax.text(90, 82, summary_text, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PAPER_BG_COLOR, 
                        edgecolor="white", alpha=0.8))
    else:
        ax.text(60, 82, summary_text, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PAPER_BG_COLOR, 
                        edgecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig, ax


def create_bokeh_shot_map(df: pd.DataFrame, title: str = "Interactive Shot Map", 
                         half_pitch: bool = False):
    """
    Create interactive shot map using Bokeh as alternative to Plotly.
    
    Args:
        df: DataFrame containing shot data
        title: Title for the plot
        half_pitch: Whether to show half pitch only
        
    Returns:
        Bokeh figure object or matplotlib figure if bokeh is not installed
    """
    try:
        # Optional import using importlib to avoid static import warnings when Bokeh isn't installed
        import importlib
        bokeh_plotting = importlib.import_module('bokeh.plotting')
        bokeh_models = importlib.import_module('bokeh.models')
    except Exception:
        print("Bokeh package not installed. Please install with: pip install bokeh")
        print("Falling back to matplotlib visualization.")
        return create_seaborn_shot_map(df, title, half_pitch)
    
    try:
        # Aliases from imported modules
        figure = bokeh_plotting.figure
        HoverTool = bokeh_models.HoverTool
        
        # Validate coordinates
        if half_pitch:
            valid_shots = df[
                (df['start_x'] >= 60) & (df['start_x'] <= 120) & 
                (df['start_y'] >= 0) & (df['start_y'] <= 80)
            ].copy()
            x_range = (55, 125)
            y_range = (-5, 85)
            plot_width = 600
            plot_height = 800
        else:
            valid_shots = df[
                (df['start_x'] >= 0) & (df['start_x'] <= 120) & 
                (df['start_y'] >= 0) & (df['start_y'] <= 80)
            ].copy()
            x_range = (-5, 125)
            y_range = (-5, 85)
            plot_width = 800
            plot_height = 600
        
        # Create figure
        p = figure(width=plot_width, height=plot_height,
                  x_range=x_range, y_range=y_range,
                  title=title, background_fill_color="black")
        
        # Minimal pitch depiction could be added here if needed
        
        if len(valid_shots) > 0:
            # Color mapping
            colors = []
            for xg in valid_shots['xG']:
                if xg >= 0.7:
                    colors.append('#FF0040')
                elif xg >= 0.5:
                    colors.append('#FF8000')
                elif xg >= 0.3:
                    colors.append('#FFFF00')
                elif xg >= 0.1:
                    colors.append('#00FFFF')
                else:
                    colors.append('#4080FF')
            
            # Size mapping
            sizes = (valid_shots['xG'] * 50 + 20).tolist()
            
            # Create scatter plot
            p.circle(valid_shots['start_x'], valid_shots['start_y'],
                    size=sizes, color=colors, alpha=0.8,
                    line_color='white', line_width=2)
            
            # Add hover tool
            hover = HoverTool(tooltips=[
                ("xG", "@xG{0.000}"),
                ("Position", "(@start_x, @start_y)")
            ])
            p.add_tools(hover)
        
        return p
        
    except Exception:
        print("Bokeh not available or failed to render, falling back to matplotlib")
        return create_seaborn_shot_map(df, title, half_pitch)
