"""
Visualization utilities for the xG prediction application.
All visualizations use vertical pitch orientation via mplsoccer VerticalPitch.
"""

import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde

# --- Unified visualization theme helpers (UI/UX tidy and consistent) ---
PITCH_BG_COLOR = '#0a1f14'  # dark green tone for better contrast
PAPER_BG_COLOR = '#0d1117'  # consistent dark background
LINE_COLOR = 'white'
FONT_FAMILY = 'Inter, Segoe UI, Arial, sans-serif'

# Enhanced high-contrast palette for xG bins - optimized for dark backgrounds
XG_COLORS = {
    'very_high': '#ff4757',  # bright red - maximum contrast
    'high': '#ff6348',       # coral red-orange
    'medium': '#ffa502',     # bright orange (instead of yellow for better visibility)
    'low': '#3742fa',        # bright blue
    'very_low': '#70a1ff',   # lighter blue
}


def apply_plotly_theme(fig: go.Figure, title: str | None = None, legend: bool = True, y_range=None):
    """Apply a consistent dark pitch theme to Plotly figures.
    Keeps visuals clean, readable, and elegant across pages.
    """
    # Prepare title object pinned to top-left to avoid legend overlap
    title_text = None
    if title is not None:
        title_text = title
    elif getattr(fig.layout, 'title', None) and getattr(fig.layout.title, 'text', None):
        title_text = fig.layout.title.text

    fig.update_layout(
        title=dict(text=title_text, x=0.0, xanchor='left', y=0.98, yanchor='top', font=dict(size=18, color='white', family=FONT_FAMILY)) if title_text else None,
        plot_bgcolor=PITCH_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
        font=dict(color='white', family=FONT_FAMILY, size=13),
        showlegend=legend,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1.0,
            font=dict(color='white', size=12),
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='rgba(255,255,255,0.35)', borderwidth=1,
            itemsizing='constant',
            itemclick='toggleothers',
            itemdoubleclick='toggle',
        ),
        margin=dict(l=12, r=12, t=68, b=12),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor=PAPER_BG_COLOR,
            bordercolor='rgba(255,255,255,0.35)',
            font=dict(color='white', family=FONT_FAMILY, size=11),
            namelength=-1,
        ),
        transition=dict(duration=200, easing='cubic-in-out'),
    )
    # Axes tidy
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, range=[-5, 85])
    if y_range is not None:
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False, range=y_range, scaleanchor='x', scaleratio=1)
    else:
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False, scaleanchor='x', scaleratio=1)
    return fig


def get_plotly_chart_config(filename_base: str = 'figure', scale: int = 2, *, compact: bool = False, hide_modebar: bool = False) -> dict:
    """Return a standard Plotly config for Streamlit plotly_chart.
    - PNG export enabled via modebar (toImage)
    - Hide Plotly logo
    - Optional compact modebar (only toImage button)
    - Optional hidden modebar (no modebar at all; for small, non-exportable helper plots)
    """
    if hide_modebar:
        return {
            "displayModeBar": False,
            "displaylogo": False,
            "staticPlot": True,
            "responsive": True,
        }

    config: dict = {
        "displayModeBar": True,
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename_base,
            "scale": scale,
        },
    }

    if compact:
        # Keep only the toImage button by removing common interaction buttons
        config["modeBarButtonsToRemove"] = [
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
            "toggleSpikelines",
            "resetViews",
            "resetViewMapbox",
        ]
    return config


# Import alternative visualization approaches
try:
    from .visualization_seaborn import create_seaborn_shot_map, create_bokeh_shot_map
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def create_shot_map(df: pd.DataFrame, title: str = "Shot Map with xG") -> tuple:
    """
    Create a professional vertical shot map visualization showing shots with high contrast design.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # DEBUG: Print data info
    print(f"DEBUG - Shot Map Data Info:")
    print(f"- Number of shots: {len(df)}")
    print(f"- Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"- X range: {df['start_x'].min():.2f} to {df['start_x'].max():.2f}")
        print(f"- Y range: {df['start_y'].min():.2f} to {df['start_y'].max():.2f}")
        print(f"- xG range: {df['xG'].min():.3f} to {df['xG'].max():.3f}")
        print(f"- Sample data:")
        print(df[['start_x', 'start_y', 'xG']].head())
    
    # Validate coordinate ranges for StatsBomb pitch (0-120 x 0-80)
    valid_shots = df[
        (df['start_x'] >= 0) & (df['start_x'] <= 120) & 
        (df['start_y'] >= 0) & (df['start_y'] <= 80)
    ].copy()
    
    if len(valid_shots) < len(df):
        print(f"WARNING: Filtered out {len(df) - len(valid_shots)} shots with invalid coordinates")
    
    if len(valid_shots) == 0:
        print("ERROR: No valid shots to display!")
        # Create empty plot
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='white', linewidth=2.5)
        fig, ax = pitch.draw(figsize=(12, 18))
        fig.patch.set_facecolor(PAPER_BG_COLOR)
        ax.set_facecolor(PITCH_BG_COLOR)
        ax.text(60, 40, "No valid shots to display", ha='center', va='center', 
                fontsize=18, color='white', fontweight='semibold')
        return fig, ax
    
    # Create the vertical pitch with softer styling
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='white', linewidth=2.5)
    fig, ax = pitch.draw(figsize=(12, 18), constrained_layout=True, tight_layout=False)
    
    # Apply unified backgrounds
    fig.patch.set_facecolor(PAPER_BG_COLOR)
    ax.set_facecolor(PITCH_BG_COLOR)

    # Softer marker styling by xG bins
    def color_for_xg(xg):
        if xg >= 0.7: return XG_COLORS['very_high']
        if xg >= 0.5: return XG_COLORS['high']
        if xg >= 0.3: return XG_COLORS['medium']
        if xg >= 0.1: return XG_COLORS['low']
        return XG_COLORS['very_low']

    for _, row in valid_shots.iterrows():
        x, y, xg = row['start_x'], row['start_y'], row['xG']
        color = color_for_xg(xg)
        size = (xg * 550) + 90
        ax.scatter(
            x, y,
            s=size,
            c=color,
            alpha=0.95,
            edgecolors='white',
            linewidths=2.2,
            zorder=12,
        )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['very_high'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=14, 
                  linestyle='None', label='Very High (≥0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['high'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=12, 
                  linestyle='None', label='High (0.5–0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['medium'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=11, 
                  linestyle='None', label='Medium (0.3–0.5)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['low'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=10, 
                  linestyle='None', label='Low (0.1–0.3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['very_low'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=9, 
                  linestyle='None', label='Very Low (<0.1)')
    ]
    
    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.02, 1), 
        fontsize=11,
        facecolor=PAPER_BG_COLOR,
        edgecolor='white',
        labelcolor='white'
    )
    legend.get_frame().set_linewidth(1)

    # Title
    ax.set_title(
        title,
        fontsize=22,
        color='white',
        fontweight='semibold',
        pad=28,
    )
    
    return fig, ax


def create_half_pitch_shot_map(df: pd.DataFrame, title: str = "Half Pitch Shot Map") -> tuple:
    """
    Create a professional vertical half-pitch shot map with ultra-bright visualization.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # DEBUG: Print data info
    print(f"DEBUG - Half Pitch Data Info:")
    print(f"- Number of shots: {len(df)}")
    if len(df) > 0:
        print(f"- X range: {df['start_x'].min():.2f} to {df['start_x'].max():.2f}")
        print(f"- Y range: {df['start_y'].min():.2f} to {df['start_y'].max():.2f}")
    
    # Filter for attacking half (x >= 60) and valid coordinates
    valid_shots = df[
        (df['start_x'] >= 60) & (df['start_x'] <= 120) & 
        (df['start_y'] >= 0) & (df['start_y'] <= 80)
    ].copy()
    
    print(f"DEBUG - Valid shots in attacking half: {len(valid_shots)}")
    
    if len(valid_shots) == 0:
        print("ERROR: No valid shots in attacking half!")
        # Create empty plot
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='white', 
                              linewidth=2.5, half=True)
        fig, ax = pitch.draw(figsize=(14, 14))
        fig.patch.set_facecolor(PAPER_BG_COLOR)
        ax.set_facecolor(PITCH_BG_COLOR)
        ax.text(40, 90, "No shots in attacking half", ha='center', va='center', 
                fontsize=18, color='white', fontweight='semibold')
        return fig, ax
    
    # Create vertical half pitch with softer styling
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='white', 
                          linewidth=2.5, half=True)
    fig, ax = pitch.draw(figsize=(14, 14), constrained_layout=True, tight_layout=False)
    
    # Apply unified backgrounds
    fig.patch.set_facecolor(PAPER_BG_COLOR)
    ax.set_facecolor(PITCH_BG_COLOR)

    # Softer marker styling
    def color_for_xg(xg):
        if xg >= 0.7: return XG_COLORS['very_high']
        if xg >= 0.5: return XG_COLORS['high']
        if xg >= 0.3: return XG_COLORS['medium']
        if xg >= 0.1: return XG_COLORS['low']
        return XG_COLORS['very_low']
    
    for _, row in valid_shots.iterrows():
        x, y, xg = row['start_x'], row['start_y'], row['xG']
        color = color_for_xg(xg)
        size = (xg * 2200) + 600
        ax.scatter(
            x, y,
            s=size,
            c=color,
            alpha=0.95,
            edgecolors='white',
            linewidths=2.4,
            zorder=12,
        )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['very_high'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=14, 
                  linestyle='None', label='Very High (≥0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['high'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=12, 
                  linestyle='None', label='High (0.5–0.7)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['medium'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=11, 
                  linestyle='None', label='Medium (0.3–0.5)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['low'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=10, 
                  linestyle='None', label='Low (0.1–0.3)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=XG_COLORS['very_low'], 
                  markeredgecolor='white', markeredgewidth=2, markersize=9, 
                  linestyle='None', label='Very Low (<0.1)')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                      fontsize=11, facecolor=PAPER_BG_COLOR, edgecolor='white', labelcolor='white')
    legend.get_frame().set_linewidth(1)

    # Title and subtitle
    ax.set_title(title, fontsize=22, color='white', fontweight='semibold', pad=28)
    total_xg = valid_shots['xG'].sum()
    ax.text(90, 55, f'Final Third Focus | {len(valid_shots)} shots | Total xG: {total_xg:.2f}', 
            ha='center', va='center', fontsize=13, color='white', fontweight='semibold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PAPER_BG_COLOR, edgecolor="white", alpha=0.6))
    
    return fig, ax


def create_single_shot_visualization(x: float, y: float, xg_value: float, 
                                   title: str = "Shot Location Preview") -> tuple:
    """
    Create a revolutionary single shot visualization with dramatic glow effects.
    
    Args:
        x: X coordinate of the shot
        y: Y coordinate of the shot
        xg_value: xG value of the shot (optional, for annotation)
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Create professional vertical pitch styling with consistent dark background
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='white', 
                          linewidth=4)
    fig, ax = pitch.draw(figsize=(10, 16), constrained_layout=True, tight_layout=False)
    
    # Set consistent dark background for maximum contrast
    fig.patch.set_facecolor(PAPER_BG_COLOR)
    ax.set_facecolor(PITCH_BG_COLOR)
    
    # Determine color based on xG value using enhanced high-contrast colors
    if xg_value >= 0.7:
        base_color = '#ff4757'  # Bright Red - Very high xG
        glow_color = '#ff6b7a'  # Red glow
        ring_color = '#ff9aa7'  # Light red ring
    elif xg_value >= 0.5:
        base_color = '#ff6348'  # Coral Red-Orange - High xG
        glow_color = '#ff8066'  # Orange glow
        ring_color = '#ffa899'  # Light orange ring
    elif xg_value >= 0.3:
        base_color = '#ffa502'  # Bright Orange - Medium xG
        glow_color = '#ffb733'  # Orange glow
        ring_color = '#ffce66'  # Light orange ring
    elif xg_value >= 0.1:
        base_color = '#3742fa'  # Bright Blue - Low-Medium xG
        glow_color = '#5a67fb'  # Blue glow
        ring_color = '#8890fc'  # Light blue ring
    else:
        base_color = '#70a1ff'  # Light Blue - Very low xG
        glow_color = '#8fb3ff'  # Blue glow
        ring_color = '#b3ccff'  # Light blue ring
    
    # Calculate dramatic size for single shot
    base_size = xg_value * 5000 + 1000
    
    # MASSIVE MULTI-LAYER GLOW EFFECT for single shot
    # Outer rings for dramatic effect
    for i in range(5, 0, -1):
        alpha = 0.1 + (i * 0.05)
        size_mult = 2.0 + (i * 0.5)
        ax.scatter(x, y, s=base_size * size_mult, c=ring_color, alpha=alpha, 
                  edgecolors='none', zorder=5+i)
    
    # Main glow layers
    ax.scatter(x, y, s=base_size * 2.5, c=glow_color, alpha=0.4, 
              edgecolors='none', zorder=12)
    ax.scatter(x, y, s=base_size * 1.8, c=glow_color, alpha=0.6, 
              edgecolors='none', zorder=13)
    ax.scatter(x, y, s=base_size * 1.3, c=glow_color, alpha=0.8, 
              edgecolors='none', zorder=14)
    
    # Main marker with ultra-thick white outline
    ax.scatter(x, y, s=base_size, c=base_color, alpha=1.0,
              edgecolors='white', linewidths=10, zorder=20)
    
    # Bright white core for extra drama
    ax.scatter(x, y, s=base_size * 0.25, c='white', alpha=1.0,
              edgecolors='none', zorder=21)
    
    # Add dramatic annotation with xG value
    ax.annotate(f"xG: {xg_value:.3f}", 
                xy=(x, y), 
                xytext=(15, -30), textcoords='offset points', 
                fontsize=20, color='white', ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.8", facecolor=base_color, edgecolor="white", 
                         alpha=0.9, linewidth=4),
                zorder=25,
                arrowprops=dict(arrowstyle='->', color='white', lw=3))

    # Dramatic title styling with consistent background
    ax.set_title(title, fontsize=24, color='white', fontweight='bold', pad=40,
                bbox=dict(boxstyle="round,pad=0.8", facecolor=PAPER_BG_COLOR, edgecolor="white", linewidth=4))
    
    # Add coordinate info with consistent styling
    ax.text(x, -15, f'Position: ({x:.1f}, {y:.1f})', 
            ha='center', va='center', fontsize=16, color='white', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PAPER_BG_COLOR, edgecolor="white", alpha=0.8))
    
    return fig, ax


def create_preview_pitch(x: float, y: float) -> tuple:
    """
    Create a simple preview pitch showing shot location.
    
    Args:
        x: X coordinate of the shot
        y: Y coordinate of the shot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#f4f4f4', line_color='grey')
    fig, ax = pitch.draw(figsize=(6, 9))
    pitch.scatter(x, y, ax=ax, s=150, color='red', edgecolors='black', zorder=2)
    ax.set_title("Shot Location Preview")
    
    return fig, ax


def save_figure_to_bytes(fig, format='png', dpi=300):
    """
    Save matplotlib figure to bytes for download.
    
    Args:
        fig: Matplotlib figure object
        format: File format ('png', 'pdf', 'svg')
        dpi: Resolution for raster formats
        
    Returns:
        Bytes object of the saved figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', 
                facecolor=PAPER_BG_COLOR, edgecolor='none')
    buf.seek(0)
    return buf.getvalue()


def save_plotly_figure_to_bytes(fig, format_type: str = 'png', dpi: int = 300) -> bytes:
    """
    Save Plotly figure to bytes for download.
    
    Args:
        fig: Plotly Figure object
        format_type: File format ('png', 'jpg', 'pdf', 'svg')
        dpi: Resolution for raster formats
        
    Returns:
        Bytes data of the saved figure
    """
    import plotly.io as pio
    try:
        if format_type.lower() in ['png', 'jpg', 'jpeg']:
            # Use scale to emulate DPI; width/height chosen for vertical pitch
            img_bytes = pio.to_image(fig, format=format_type, width=1200, height=1800, scale=2)
        elif format_type.lower() == 'pdf':
            img_bytes = pio.to_image(fig, format='pdf', width=1200, height=1800)
        elif format_type.lower() == 'svg':
            img_bytes = pio.to_image(fig, format='svg', width=1200, height=1800)
        else:
            # Default to PNG
            img_bytes = pio.to_image(fig, format='png', width=1200, height=1800, scale=2)
        return img_bytes
    except Exception as e:
        # Kaleido/engine may be unavailable on some platforms (e.g., Streamlit Cloud)
        # Silence warnings; let caller choose fallback
        # print(f"WARNING: Plotly static image export failed: {e}")
        raise


def save_plotly_figure_to_html_bytes(fig) -> bytes:
    """Return an interactive HTML (standalone) representation as bytes for download.
    Useful fallback when Kaleido/static export is unavailable.
    """
    html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return html.encode('utf-8')


def create_download_filename(prefix, extension):
    """
    Create a timestamped filename for downloads.
    
    Args:
        prefix: Filename prefix
        extension: File extension (without dot)
        
    Returns:
        Formatted filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def prepare_csv_download(df: pd.DataFrame) -> str:
    """
    Prepare DataFrame for CSV download.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        CSV string
    """
    return df.to_csv(index=False)


def create_custom_shots_visualization(df: pd.DataFrame, title: str = "Custom Shots Analysis") -> tuple:
    """
    Create a professional vertical visualization for multiple custom shots.
    
    Args:
        df: DataFrame containing custom shots with 'start_x', 'start_y', and 'xg_value' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    if df.empty:
        # Return professional empty plot if no data with consistent theme
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='#30363d', 
                      linewidth=2)
        fig, ax = pitch.draw(figsize=(10, 16))
        fig.patch.set_facecolor(PAPER_BG_COLOR)
        ax.set_facecolor(PITCH_BG_COLOR)
        ax.set_title("No custom shots to display", fontsize=22, color='#8b949e', fontweight='bold')
        ax.text(60, 40, 'Create some custom shots to see the shot map visualization', 
                ha='center', va='center', fontsize=14, color='#8b949e', style='italic')
        return fig, ax
    
    # Create professional vertical pitch styling with consistent background
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='#30363d', 
                  linewidth=2)
    fig, ax = pitch.draw(figsize=(10, 16), constrained_layout=True, tight_layout=False)
    
    # Set consistent dark background for professional look
    fig.patch.set_facecolor(PAPER_BG_COLOR)
    ax.set_facecolor(PITCH_BG_COLOR)

    # Plot the shots with professional styling based on xG value - mplsoccer handles vertical orientation
    size = df['xg_value'] * 900 + 150  # Size based on xG value, larger for visibility
    
    sc = pitch.scatter(df.start_x, df.start_y,
                       s=size,
                       c=df['xg_value'],
                       cmap='hot',  # Hot colormap for maximum contrast against green
                       ax=ax,
                       edgecolors='white',  # White edges for maximum contrast
                       linewidths=2.5,
                       alpha=1.0,
                       vmin=0, vmax=1,
                       zorder=10)  # High z-order to ensure shots are on top

    # Professional colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Expected Goals (xG) Value', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    cbar.ax.yaxis.set_ticklabels([f'{x:.2f}' for x in cbar.get_ticks()], color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Add shot labels if there are few shots (professional styling) using original coordinates
    if len(df) <= 8:  # Reduced number for clarity
        for _, shot in df.iterrows():
            pitch.annotate(
                f"{shot.get('shot_name', 'Shot')}\n{shot['xg_value']:.3f}",
                xy=(shot['start_x'], shot['start_y']),
                ax=ax,
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=10,
                color='white',
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="white", alpha=0.8, linewidth=1),
                zorder=15  # Higher z-order than shot markers
            )

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    return fig, ax


def create_shot_heat_map(df: pd.DataFrame, title: str = "Shot Heat Map") -> tuple:
    """
    Create a professional smooth heat map visualization showing shot density.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Create the pitch with professional styling
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='#30363d', linewidth=2)
    fig, ax = pitch.draw(figsize=(10, 16), constrained_layout=True, tight_layout=False)
    
    # Set consistent dark background for professional look
    fig.patch.set_facecolor(PAPER_BG_COLOR)
    ax.set_facecolor(PITCH_BG_COLOR)

    # Create smooth continuous heat map using kde
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create custom red-to-white colormap (original) for better visibility on dark background
    colors = ['#000000', '#1a0000', '#330000', '#4d0000', '#660000', '#800000', 
              '#990000', '#b30000', '#cc0000', '#e60000', '#ff0000', '#ff1a1a', 
              '#ff3333', '#ff4d4d', '#ff6666', '#ff8080', '#ff9999', '#ffb3b3', '#ffcccc', '#ffffff']
    custom_cmap = LinearSegmentedColormap.from_list('smooth_red', colors, N=256)
    
    # Create higher resolution grid for smoother heat map (adjusted for vertical pitch)
    x_grid = np.linspace(0, 80, 134)  # Width of vertical pitch
    y_grid = np.linspace(0, 120, 200)  # Height of vertical pitch
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Transform coordinates for vertical pitch display
    # For VerticalPitch: horizontal x becomes vertical y, horizontal y becomes vertical x
    df_transformed = df.copy()
    df_transformed['start_x_vertical'] = df['start_y']  # horizontal y becomes vertical x
    df_transformed['start_y_vertical'] = df['start_x']  # horizontal x becomes vertical y (NO inversion)
    
    # Create positions array with transformed coordinates
    positions = np.column_stack([df_transformed.start_x_vertical.values, df_transformed.start_y_vertical.values])
    
    if len(positions) > 1:
        # Create KDE with optimal bandwidth for smooth coverage
        kde = gaussian_kde(positions.T)
        kde.set_bandwidth(bw_method=0.4)  # Larger bandwidth for smoother, more connected areas
        
        # Evaluate KDE on grid
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        # Normalize Z to a good range for visualization
        Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())
        
        # Apply power transformation for better visual contrast
        Z_enhanced = np.power(Z_normalized, 0.7)
        
        # Create smooth heat map with many levels for smoothness
        levels = np.linspace(0.05, 1.0, 50)  # Start from 0.05 to avoid showing noise
        im = ax.contourf(X, Y, Z_enhanced, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')
        
        # Add very subtle contour lines for definition
        contour_levels = np.linspace(0.2, 1.0, 8)
        ax.contour(X, Y, Z_enhanced, levels=contour_levels, colors='white', alpha=0.15, linewidths=0.5)
        
    else:
        # Fallback for single point - create a smooth gradient around it (use transformed coordinates)
        x_center, y_center = df_transformed.start_x_vertical.iloc[0], df_transformed.start_y_vertical.iloc[0]
        
        # Create circular gradient around the point
        distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
        Z_single = np.exp(-distance**2 / (2 * 8**2))  # Gaussian with sigma=8
        
        im = ax.contourf(X, Y, Z_single, levels=30, cmap=custom_cmap, alpha=0.85, extend='max')

    # Professional colorbar styling
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Shot Density', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    return fig, ax


def create_half_pitch_heat_map(df: pd.DataFrame, title: str = "Half Pitch Heat Map") -> tuple:
    """
    Create a professional smooth half-pitch heat map visualization.
    
    Args:
        df: DataFrame containing shot data with 'start_x', 'start_y', and 'xG' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    # Create half pitch with professional styling and consistent background
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='#30363d', 
                  linewidth=2, half=True)
    fig, ax = pitch.draw(figsize=(10, 8), constrained_layout=True, tight_layout=False)
    
    # Set consistent dark background for professional look
    fig.patch.set_facecolor(PAPER_BG_COLOR)
    ax.set_facecolor(PITCH_BG_COLOR)

    # Create smooth continuous heat map using kde
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create custom red-to-white colormap (original) for better visibility on dark background
    colors = ['#000000', '#1a0000', '#330000', '#4d0000', '#660000', '#800000', 
              '#990000', '#b30000', '#cc0000', '#e60000', '#ff0000', '#ff1a1a', 
              '#ff3333', '#ff4d4d', '#ff6666', '#ff8080', '#ff9999', '#ffb3b3', '#ffcccc', '#ffffff']
    custom_cmap = LinearSegmentedColormap.from_list('smooth_red', colors, N=256)
    
    # Create grid for smooth heat map (half pitch - adjusted for vertical)
    x_grid = np.linspace(0, 80, 134)  # Width of vertical pitch
    y_grid = np.linspace(60, 120, 120)  # Upper half of vertical pitch (attacking half)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Transform coordinates for vertical pitch display
    # For VerticalPitch: horizontal x becomes vertical y, horizontal y becomes vertical x
    df_transformed = df.copy()
    df_transformed['start_x_vertical'] = df['start_y']  # horizontal y becomes vertical x
    df_transformed['start_y_vertical'] = df['start_x']  # horizontal x becomes vertical y (NO inversion)
    
    # Create positions array with transformed coordinates
    positions = np.column_stack([df_transformed.start_x_vertical.values, df_transformed.start_y_vertical.values])
    
    if len(positions) > 1:
        # Create KDE with optimal bandwidth for smooth coverage
        kde = gaussian_kde(positions.T)
        kde.set_bandwidth(bw_method=0.35)  # Slightly smaller for half pitch
        
        # Evaluate KDE on grid
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        # Normalize Z to a good range for visualization
        Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())
        
        # Apply power transformation for better visual contrast
        Z_enhanced = np.power(Z_normalized, 0.7)
        
        # Create smooth heat map with many levels for smoothness
        levels = np.linspace(0.05, 1.0, 40)  # Start from 0.05 to avoid showing noise
        im = ax.contourf(X, Y, Z_enhanced, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')
        
        # Add very subtle contour lines for definition
        contour_levels = np.linspace(0.2, 1.0, 6)
        ax.contour(X, Y, Z_enhanced, levels=contour_levels, colors='white', alpha=0.15, linewidths=0.5)
        
    else:
        # Fallback for single point - create a smooth gradient around it (use transformed coordinates)
        x_center, y_center = df_transformed.start_x_vertical.iloc[0], df_transformed.start_y_vertical.iloc[0]
        
        # Create circular gradient around the point
        distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
        Z_single = np.exp(-distance**2 / (2 * 6**2))  # Gaussian with sigma=6 for half pitch
        
        im = ax.contourf(X, Y, Z_single, levels=25, cmap=custom_cmap, alpha=0.85, extend='max')

    # Professional colorbar styling
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=15, pad=0.02)
    cbar.set_label('Shot Density', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    return fig, ax


def create_custom_shots_heat_map(df: pd.DataFrame, title: str = "Custom Shots Heat Map") -> tuple:
    """
    Create a professional smooth heat map visualization for multiple custom shots.
    
    Args:
        df: DataFrame containing custom shots with 'start_x', 'start_y', and 'xg_value' columns
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axis) objects
    """
    if df.empty:
        # Return professional empty plot if no data with proper aspect ratio and consistent theme
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='#30363d', linewidth=2)
        fig, ax = pitch.draw(figsize=(10, 16))  # Proper aspect ratio for vertical pitch
        fig.patch.set_facecolor(PAPER_BG_COLOR)
        ax.set_facecolor(PITCH_BG_COLOR)
        ax.set_title("No custom shots to display", fontsize=22, color='#8b949e', fontweight='bold')
        ax.text(60, 40, 'Create some custom shots to see the heat map visualization', 
                ha='center', va='center', fontsize=14, color='#8b949e', style='italic')
        return fig, ax
    
    # Create professional pitch styling with proper aspect ratio and consistent background
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=PITCH_BG_COLOR, line_color='#30363d', linewidth=2)
    fig, ax = pitch.draw(figsize=(10, 16))  # Proper aspect ratio for vertical pitch
    
    # Set consistent dark background for professional look
    fig.patch.set_facecolor(PAPER_BG_COLOR)
    ax.set_facecolor(PITCH_BG_COLOR)

    # Create smooth continuous heat map using kde
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create custom red-to-white colormap (original) for better visibility on dark background
    colors = ['#000000', '#1a0000', '#330000', '#4d0000', '#660000', '#800000', 
              '#990000', '#b30000', '#cc0000', '#e60000', '#ff0000', '#ff1a1a', 
              '#ff3333', '#ff4d4d', '#ff6666', '#ff8080', '#ff9999', '#ffb3b3', '#ffcccc', '#ffffff']
    custom_cmap = LinearSegmentedColormap.from_list('smooth_red', colors, N=256)
    
    # Create higher resolution grid for smoother heat map (adjusted for vertical pitch)
    x_grid = np.linspace(0, 80, 160)   # Width of vertical pitch
    y_grid = np.linspace(0, 120, 240)  # Height of vertical pitch
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Transform coordinates for vertical pitch display
    # For VerticalPitch: horizontal x becomes vertical y, horizontal y becomes vertical x
    df_transformed = df.copy()
    df_transformed['start_x_vertical'] = df['start_y']  # horizontal y becomes vertical x
    df_transformed['start_y_vertical'] = df['start_x']  # horizontal x becomes vertical y (NO inversion)
    
    # Create positions array with transformed coordinates
    positions = np.column_stack([df_transformed.start_x_vertical.values, df_transformed.start_y_vertical.values])
    
    if len(positions) > 1:
        try:
            # Check if we have enough unique points for KDE
            unique_positions = np.unique(positions, axis=0)
            
            if len(unique_positions) > 1:
                # Check for sufficient variance in the data
                x_var = np.var(positions[:, 0])
                y_var = np.var(positions[:, 1])
                
                if x_var > 1e-6 and y_var > 1e-6:  # Sufficient variance threshold
                    # Create KDE with adaptive bandwidth based on data distribution
                    kde = gaussian_kde(positions.T)
                    
                    # Calculate adaptive bandwidth to prevent overly stretched heat maps
                    n_points = len(positions)
                    if n_points > 10:
                        bw_factor = 0.3  # Smaller bandwidth for many points
                    elif n_points > 5:
                        bw_factor = 0.4  # Medium bandwidth
                    else:
                        bw_factor = 0.5  # Larger bandwidth for few points
                        
                    kde.set_bandwidth(bw_method=bw_factor)
                    
                    # Evaluate KDE on grid
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                    
                    # Apply square root scaling to prevent extreme stretching
                    Z_sqrt = np.sqrt(Z)
                    Z_normalized = (Z_sqrt - Z_sqrt.min()) / (Z_sqrt.max() - Z_sqrt.min() + 1e-12)
                    
                    # Apply mild power transformation
                    Z_enhanced = np.power(Z_normalized, 0.6)
                    
                    # Create heat map with circular-friendly levels
                    levels = np.linspace(0.05, 1.0, 45)
                    im = ax.contourf(X, Y, Z_enhanced, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')
                    
                    # Add subtle contour lines
                    contour_levels = np.linspace(0.15, 0.9, 5)
                    ax.contour(X, Y, Z_enhanced, levels=contour_levels, colors='white', alpha=0.25, linewidths=0.6)
                    
                else:
                    # Fallback: Points too concentrated, use manual kernel method
                    raise np.linalg.LinAlgError("Insufficient variance for KDE")
            else:
                # Fallback: Not enough unique points, use manual kernel method
                raise np.linalg.LinAlgError("Insufficient unique points for KDE")
                
        except (np.linalg.LinAlgError, ValueError):
            # Fallback method: Create manual heat map using individual point kernels
            Z_manual = np.zeros_like(X)
            
            # Adaptive sigma based on number of points
            n_points = len(positions)
            if n_points > 5:
                sigma = 6  # Smaller kernel for many points
            else:
                sigma = 8  # Larger kernel for few points
            
            for i, (x_pos, y_pos) in enumerate(positions):
                # Create circular Gaussian kernel around each point
                distance = np.sqrt((X - x_pos)**2 + (Y - y_pos)**2)
                kernel = np.exp(-distance**2 / (2 * sigma**2))
                Z_manual += kernel
            
            # Apply square root scaling to prevent stretching
            Z_manual_sqrt = np.sqrt(Z_manual)
            if Z_manual_sqrt.max() > 0:
                Z_manual_normalized = Z_manual_sqrt / Z_manual_sqrt.max()
            else:
                Z_manual_normalized = Z_manual_sqrt
            
            # Create circular heat map with appropriate levels
            levels = np.linspace(0.05, 1.0, 35)
            im = ax.contourf(X, Y, Z_manual_normalized, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')
            
    else:
        # Single point fallback - create a circular gradient (use transformed coordinates)
        x_center, y_center = df_transformed.start_x_vertical.iloc[0], df_transformed.start_y_vertical.iloc[0]
        
        # Create circular gradient around the point
        distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
        Z_single = np.exp(-distance**2 / (2 * 8**2))  # Circular sigma=8
        
        # Create circular heat map
        levels = np.linspace(0.05, 1.0, 30)
        im = ax.contourf(X, Y, Z_single, levels=levels, cmap=custom_cmap, alpha=0.85, extend='max')

    # Professional colorbar styling
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Shot Density', color='white', fontsize=14, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)

    # Professional title styling
    ax.set_title(title, fontsize=22, color='white', fontweight='bold', pad=30)
    
    return fig, ax


def get_visualization_options(lang: str = 'en') -> dict:
    """Return mapping of localized visualization option labels to internal types.

    Pages call this to populate selectboxes. Ensure keys match translation keys
    defined in language.py (shot_map_option, heat_map_option).
    """
    from .language import get_translation  # local import to avoid circular
    # Globally disable Heat Map: return only Shot Map option
    return {
        get_translation('shot_map_option', lang): 'shot_map',
    }


def _select_shot_map_function(custom_shots: bool, half_pitch: bool):
    """Internal helper choosing appropriate shot map function.
    Preference order: Seaborn alternative (manual pitch) if available, else mplsoccer.
    """
    if custom_shots:
        # For custom shots we keep the distinct styling function already present.
        return create_custom_shots_visualization if not half_pitch else create_custom_shots_visualization
    # Dataset shots
    if SEABORN_AVAILABLE:
        return create_seaborn_shot_map if not half_pitch else create_seaborn_shot_map
    return create_shot_map if not half_pitch else create_half_pitch_shot_map


def _select_heat_map_function(custom_shots: bool, half_pitch: bool):
    if custom_shots:
        return create_custom_shots_heat_map if not half_pitch else create_custom_shots_heat_map  # half version not separate for custom; reuse full
    return create_shot_heat_map if not half_pitch else create_half_pitch_heat_map


def create_visualization_by_type(
    df: pd.DataFrame,
    viz_type: str,
    title: str,
    half_pitch: bool = False,
    interactive: bool = True,
    custom_shots: bool = False,
):
    """Factory wrapper used by pages to build the requested visualization.

    Returns (figure, axis) for matplotlib based visualizations OR (plotly_fig, None)
    for interactive Plotly versions. Currently alternative Seaborn/manual approach
    returns matplotlib figures for reliability & marker visibility.
    """
    # Defensive copy
    local_df = df.copy() if df is not None else pd.DataFrame()

    # Normalize expected column naming for custom shots (xg_value vs xG)
    if custom_shots and 'xG' not in local_df.columns and 'xg_value' in local_df.columns:
        local_df = local_df.rename(columns={'xg_value': 'xG'})

    # Prefer Plotly for interactive shot maps (hover, bright markers)
    if viz_type == 'shot_map' and interactive:
        fig, _ = create_plotly_shot_map(local_df, title=title, half_pitch=half_pitch, custom_shots=custom_shots)
        return fig, None

    if viz_type == 'shot_map':
        func = _select_shot_map_function(custom_shots, half_pitch)
        # Seaborn alternative uses same signature (df, title, half_pitch=bool)
        if func.__name__ == 'create_seaborn_shot_map':
            return func(local_df, title=title, half_pitch=half_pitch)
        # Custom shots visualization already adapts
        if func.__name__ == 'create_custom_shots_visualization':
            return func(local_df, title=title)
        # Default mplsoccer versions
        return func(local_df, title=title)
    elif viz_type == 'heat_map':
        if interactive:
            fig, _ = create_plotly_heat_map(local_df, title=title, half_pitch=half_pitch, custom_shots=custom_shots)
            return fig, None
        func = _select_heat_map_function(custom_shots, half_pitch)
        return func(local_df, title=title)
    else:
        func = _select_shot_map_function(custom_shots, half_pitch)
        if func.__name__ == 'create_seaborn_shot_map':
            return func(local_df, title=title, half_pitch=half_pitch)
        if func.__name__ == 'create_custom_shots_visualization':
            return func(local_df, title=title)
        return func(local_df, title=title)


# === NEW: Plotly interactive shot map for hover support ===

def create_plotly_shot_map(df: pd.DataFrame, title: str = "Shot Map", half_pitch: bool = False, custom_shots: bool = False):
    """Create vertical pitch shot map using Plotly with hover tooltips.
    Improvements: debug output, robust filtering, fill NaNs, guaranteed marker rendering.
    Returns (fig, None) so caller treats it as interactive figure.
    """
    # Defensive copy
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title + " (no data)")
        fig = apply_plotly_theme(fig)
        return fig, None

    # Ensure numeric columns
    work = df.copy()
    # Normalize xG column
    if 'xG' not in work.columns and 'xg_value' in work.columns:
        work = work.rename(columns={'xg_value': 'xG'})
    if 'xG' not in work.columns:
        # Create placeholder xG so we can still plot locations
        work['xG'] = 0.05

    # Fill NaNs
    work['xG'] = pd.to_numeric(work['xG'], errors='coerce').fillna(0.05)
    work['start_x'] = pd.to_numeric(work['start_x'], errors='coerce')
    work['start_y'] = pd.to_numeric(work['start_y'], errors='coerce')

    # Filter valid coordinates
    if half_pitch:
        valid = work[(work['start_x'] >= 60) & (work['start_x'] <= 120) & (work['start_y'] >= 0) & (work['start_y'] <= 80)].copy()
    else:
        valid = work[(work['start_x'].between(0, 120)) & (work['start_y'].between(0, 80))].copy()

    print(f"DEBUG Plotly Shot Map - total rows: {len(work)}, valid rows: {len(valid)}, half_pitch={half_pitch}")

    if valid.empty:
        fig = go.Figure()
        fig.update_layout(title=title + " (no valid shots)")
        fig = apply_plotly_theme(fig)
        return fig, None

    # Ensure xG column name
    if 'xG' not in valid.columns and 'xg_value' in valid.columns:
        valid = valid.rename(columns={'xg_value': 'xG'})

    # Softer color mapping
    def color_for_xg(xg):
        if xg >= 0.7: return XG_COLORS['very_high']
        if xg >= 0.5: return XG_COLORS['high']
        if xg >= 0.3: return XG_COLORS['medium']
        if xg >= 0.1: return XG_COLORS['low']
        return XG_COLORS['very_low']

    valid['color'] = valid['xG'].apply(color_for_xg)
    valid['size'] = (valid['xG'] * (34 if half_pitch else 22)) + (14 if half_pitch else 9)

    # Transform to vertical pitch
    valid['display_x'] = valid['start_y']
    valid['display_y'] = valid['start_x']

    # Choose label column for hover
    label_col = None
    for cand in ['player_name', 'shot_name', 'player', 'shooter', 'name']:
        if cand in valid.columns:
            label_col = cand
            break
    if label_col is None:
        valid[label_col := 'temp_label'] = 'Player'

    fig = go.Figure()

    # Pitch base shapes (vertical orientation)
    if half_pitch:
        fig.add_shape(type="rect", x0=0, y0=60, x1=80, y1=120, line=dict(color=LINE_COLOR, width=3), fillcolor="rgba(0,0,0,0)", layer='below')
        fig.add_shape(type="rect", x0=22, y0=102, x1=58, y1=120, line=dict(color=LINE_COLOR, width=2), fillcolor="rgba(0,0,0,0)", layer='below')
        fig.add_shape(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color=LINE_COLOR, width=2), fillcolor="rgba(0,0,0,0)", layer='below')
        fig.add_shape(type="rect", x0=36, y0=120, x1=44, y1=122, line=dict(color=LINE_COLOR, width=2), fillcolor='white', layer='below')
        y_range = [60, 122]
    else:
        fig.add_shape(type="rect", x0=0, y0=0, x1=80, y1=120, line=dict(color=LINE_COLOR, width=3), fillcolor="rgba(0,0,0,0)", layer='below')
        fig.add_shape(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color=LINE_COLOR, width=2), layer='below')
        fig.add_shape(type="circle", x0=30, y0=50, x1=50, y1=70, line=dict(color=LINE_COLOR, width=2), layer='below')
        for y0 in (0, 102):
            fig.add_shape(type="rect", x0=22, y0=y0, x1=58, y1=y0+18, line=dict(color=LINE_COLOR, width=2), fillcolor="rgba(0,0,0,0)", layer='below')
        for y0 in (0, 114):
            fig.add_shape(type="rect", x0=30, y0=y0, x1=50, y1=y0+6, line=dict(color=LINE_COLOR, width=2), fillcolor="rgba(0,0,0,0)", layer='below')
        fig.add_shape(type="rect", x0=36, y0=-2, x1=44, y1=0, line=dict(color=LINE_COLOR, width=2), fillcolor='white', layer='below')
        fig.add_shape(type="rect", x0=36, y0=120, x1=44, y1=122, line=dict(color=LINE_COLOR, width=2), fillcolor='white', layer='below')
        y_range = [-2, 122]

    customdata = np.stack([
        valid[label_col].fillna('Unknown'),
        valid['xG'],
        valid['start_x'],
        valid['start_y']
    ], axis=1)

    # Legend bins
    bins = [
        ("Very High (≥0.7)", valid['xG'] >= 0.7, XG_COLORS['very_high']),
        ("High (0.5–0.7)", (valid['xG'] >= 0.5) & (valid['xG'] < 0.7), XG_COLORS['high']),
        ("Medium (0.3–0.5)", (valid['xG'] >= 0.3) & (valid['xG'] < 0.5), XG_COLORS['medium']),
        ("Low (0.1–0.3)", (valid['xG'] >= 0.1) & (valid['xG'] < 0.3), XG_COLORS['low']),
        ("Very Low (<0.1)", valid['xG'] < 0.1, XG_COLORS['very_low']),
    ]

    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "xG: %{customdata[1]:.3f}<br>"
        "x: %{customdata[2]:.1f}, y: %{customdata[3]:.1f}<extra></extra>"
    )

    for label, mask, color in bins:
        sub = valid[mask]
        if sub.empty:
            continue
        sub_custom = np.stack([
            sub[label_col].fillna('Unknown'),
            sub['xG'],
            sub['start_x'],
            sub['start_y']
        ], axis=1)
        size_series = (sub['xG'] * (34 if half_pitch else 22)) + (14 if half_pitch else 9)
        fig.add_trace(go.Scatter(
            x=sub['display_x'],
            y=sub['display_y'],
            mode='markers',
            marker=dict(
                size=size_series.tolist(),
                color=color,
                line=dict(width=1.2, color='white'),
                opacity=0.9
            ),
            customdata=sub_custom,
            hovertemplate=hovertemplate,
            name=label
        ))

    fig = apply_plotly_theme(fig, title=title, legend=True, y_range=y_range)

    return fig, None

def create_plotly_heat_map(df: pd.DataFrame, title: str = "Shot Heat Map", half_pitch: bool = False, custom_shots: bool = False):
    """Create a smooth vertical pitch heat map using Plotly with improved coloring.
    - Uses KDE with optional xG weighting
    - Masks very low density so pitch color is preserved (no yellow wash)
    - Uses 'Inferno' colorscale for better contrast on dark background
    """
    # Defensive copy and validation
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title + " (no data)")
        fig = apply_plotly_theme(fig)
        return fig, None

    work = df.copy()
    work['start_x'] = pd.to_numeric(work['start_x'], errors='coerce')
    work['start_y'] = pd.to_numeric(work['start_y'], errors='coerce')

    # Filter valid coordinates and half pitch constraint
    if half_pitch:
        valid = work[(work['start_x'] >= 60) & (work['start_x'] <= 120) & (work['start_y'].between(0, 80))].copy()
    else:
        valid = work[(work['start_x'].between(0, 120)) & (work['start_y'].between(0, 80))].copy()

    if valid.empty:
        fig = go.Figure()
        fig.update_layout(title=title + " (no valid shots)")
        fig = apply_plotly_theme(fig)
        return fig, None

    # Transform to vertical pitch coordinates
    valid['display_x'] = valid['start_y']
    valid['display_y'] = valid['start_x']

    # Optional xG weighting (if available)
    xg_weights = None
    if 'xG' in valid.columns:
        try:
            xg_weights = pd.to_numeric(valid['xG'], errors='coerce').fillna(0.0).clip(lower=0.0).values
            if np.all(xg_weights == 0):
                xg_weights = None
        except Exception:
            xg_weights = None

    # Build pitch and y-range
    fig = go.Figure()
    if half_pitch:
        fig.add_shape(type="rect", x0=0, y0=60, x1=80, y1=120, line=dict(color=LINE_COLOR, width=3), fillcolor="rgba(0,0,0,0)", layer='above')
        fig.add_shape(type="rect", x0=22, y0=102, x1=58, y1=120, line=dict(color=LINE_COLOR, width=2), fillcolor="rgba(0,0,0,0)", layer='above')
        fig.add_shape(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color=LINE_COLOR, width=2), fillcolor="rgba(0,0,0,0)", layer='above')
        fig.add_shape(type="rect", x0=36, y0=120, x1=44, y1=122, line=dict(color=LINE_COLOR, width=2), fillcolor='white', layer='above')
        y_range = [60, 122]
        y_min, y_max = 60, 120
    else:
        fig.add_shape(type="rect", x0=0, y0=0, x1=80, y1=120, line=dict(color=LINE_COLOR, width=3), fillcolor="rgba(0,0,0,0)", layer='above')
        fig.add_shape(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color=LINE_COLOR, width=2), layer='above')
        fig.add_shape(type="circle", x0=30, y0=50, x1=50, y1=70, line=dict(color=LINE_COLOR, width=2), layer='above')
        for y0 in (0, 102):
            fig.add_shape(type="rect", x0=22, y0=y0, x1=58, y1=y0+18, line=dict(color=LINE_COLOR, width=2), fillcolor="rgba(0,0,0,0)", layer='above')
        for y0 in (0, 114):
            fig.add_shape(type="rect", x0=30, y0=y0, x1=50, y1=y0+6, line=dict(color=LINE_COLOR, width=2), fillcolor="rgba(0,0,0,0)", layer='above')
        fig.add_shape(type="rect", x0=36, y0=-2, x1=44, y1=0, line=dict(color=LINE_COLOR, width=2), fillcolor='white', layer='above')
        fig.add_shape(type="rect", x0=36, y0=120, x1=44, y1=122, line=dict(color=LINE_COLOR, width=2), fillcolor='white', layer='above')
        y_range = [-2, 122]
        y_min, y_max = 0, 120

    # Build grid
    x_grid = np.linspace(0, 80, 180)
    y_grid = np.linspace(y_min, y_max, 260)
    X, Y = np.meshgrid(x_grid, y_grid)

    positions = np.column_stack([valid['display_x'].values, valid['display_y'].values])

    # Compute density (with fallbacks) and normalize nicely
    Z = None
    try:
        uniq = np.unique(positions, axis=0)
        if len(uniq) > 1 and np.var(positions[:, 0]) > 1e-6 and np.var(positions[:, 1]) > 1e-6:
            kde = gaussian_kde(positions.T, weights=xg_weights)
            n = len(positions)
            bw = 0.32 if half_pitch else 0.36
            if n > 20:
                bw *= 0.85
            elif n < 6:
                bw *= 1.25
            kde.set_bandwidth(bw_method=bw)
            Z_raw = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            zmin_raw, zmax_raw = Z_raw.min(), Z_raw.max()
            Z_norm = (Z_raw - zmin_raw) / (zmax_raw - zmin_raw + 1e-12)
            low, high = np.quantile(Z_norm, 0.02), np.quantile(Z_norm, 0.995)
            Z_clip = np.clip((Z_norm - low) / (high - low + 1e-12), 0, 1)
            Z = np.power(Z_clip, 0.65)
        else:
            raise np.linalg.LinAlgError("insufficient variance")
    except (np.linalg.LinAlgError, ValueError):
        # Manual radial kernels (optionally weighted)
        Z_manual = np.zeros_like(X)
        sigma = 6 if half_pitch else 8
        if xg_weights is None:
            weights_iter = np.ones(len(positions))
        else:
            weights_iter = xg_weights
        for (x0, y0), w in zip(positions, weights_iter):
            D = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
            Z_manual += w * np.exp(-(D ** 2) / (2 * sigma ** 2))
        Z_manual = np.sqrt(Z_manual)
        m = Z_manual.max()
        Z = (Z_manual / m) if m > 0 else Z_manual

    # Mask very low densities to keep pitch background visible
    base_thr = 0.06 if half_pitch else 0.08
    adaptive_thr = float(np.quantile(Z[~np.isnan(Z)], 0.15)) * 0.9
    thr = min(0.12, max(base_thr, adaptive_thr))
    Z_plot = np.where(Z >= thr, Z, np.nan)

    # Use 'Inferno' colorscale for Plotly heatmap (original)
    colorscale = 'Inferno'

    # Heat layer
    heat = go.Heatmap(
        x=x_grid,
        y=y_grid,
        z=Z_plot,
        colorscale=colorscale,
        reversescale=False,
        showscale=True,
        opacity=0.85,
        zsmooth='best',
        zmin=0.0,
        zmax=1.0,
        hoverinfo='skip',
        colorbar=dict(
            title=dict(text='Shot Density', side='right', font=dict(color='white')),
            thickness=14,
            x=1.02,
            xanchor='left',
            outlinecolor='rgba(255,255,255,0.35)',
            outlinewidth=1,
            tickfont=dict(color='white')
        )
    )
    fig.add_trace(heat)

    # Subtle contour lines for extra definition (above heat layer)
    try:
        fig.add_trace(go.Contour(
            x=x_grid,
            y=y_grid,
            z=Z_plot,
            showscale=False,
            contours=dict(coloring='lines', showlines=True, start=thr, end=0.95, size=0.15),
            line=dict(color='rgba(255,255,255,0.25)', width=1)
        ))
    except Exception:
        pass

    fig = apply_plotly_theme(fig, title=title, legend=False, y_range=y_range)
    return fig, None
