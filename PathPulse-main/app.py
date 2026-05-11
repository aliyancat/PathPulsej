"""
PathPulse — Advanced Hospital Navigation & Heart Disease Prediction
====================================================================
Enhanced Streamlit app with advanced UI, customizable graph 
visualizations, color themes, and smooth animations.
"""

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from pathlib import Path
import base64
from typing import Optional, Dict, List, Tuple
import time

# Import navigation modules
from navigation.grid import (
    HOSPITAL_GRID, ROOMS, START_ROOM, START_COORDS, 
    ROOM_COLORS, get_destination_rooms, GRID_ROWS, GRID_COLS
)
from navigation import astar, greedy_bfs

# Import prediction modules - Advanced Version
from prediction.preprocess import HEALTHY_DEFAULTS, FEATURE_LABELS, ALL_FEATURE_NAMES
from prediction.predict import predict_risk

# ---------------------------------------------------------------------------
# Graph Color Themes
# ---------------------------------------------------------------------------
GRAPH_THEMES = {
    "neon": {
        "walkable": "#1a1a25", "wall": "#050508", "path": "#6366f1",
        "path_glow": "#8b5cf6", "explored": "#f59e0b", "start": "#10b981",
        "end": "#ef4444", "grid_line": "rgba(99, 102, 241, 0.15)",
    },
    "fire": {
        "walkable": "#1a0a0a", "wall": "#050303", "path": "#ef4444",
        "path_glow": "#f97316", "explored": "#fb923c", "start": "#22c55e",
        "end": "#eab308", "grid_line": "rgba(239, 68, 68, 0.15)",
    },
    "ice": {
        "walkable": "#0a1a1a", "wall": "#030505", "path": "#06b6d4",
        "path_glow": "#22d3ee", "explored": "#67e8f9", "start": "#10b981",
        "end": "#a78bfa", "grid_line": "rgba(6, 182, 212, 0.15)",
    },
    "cyberpunk": {
        "walkable": "#0f0f1a", "wall": "#050510", "path": "#ec4899",
        "path_glow": "#f472b6", "explored": "#fbbf24", "start": "#34d399",
        "end": "#818cf8", "grid_line": "rgba(236, 72, 153, 0.2)",
    },
    "matrix": {
        "walkable": "#0a0f0a", "wall": "#030503", "path": "#22c55e",
        "path_glow": "#4ade80", "explored": "#86efac", "start": "#06b6d4",
        "end": "#f97316", "grid_line": "rgba(34, 197, 94, 0.15)",
    },
}


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PathPulse — Advanced Hospital Navigation",
    page_icon="page_logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS Injection
# ---------------------------------------------------------------------------
def load_custom_css() -> None:
    """Load and inject the custom dark theme CSS."""
    css_path: Path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as css_file:
            st.markdown(
                f"<style>{css_file.read()}</style>",
                unsafe_allow_html=True,
            )

load_custom_css()

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize all session state variables."""
    destinations = get_destination_rooms()
    defaults: dict = {
        "nav_start": START_ROOM,
        "nav_destination": destinations[0] if destinations else None,
        "nav_algorithm": "A*",
        "nav_result": None,
        "pred_result": None,
        "pred_submitted": False,
        "custom_rooms": dict(ROOMS),
        "animate_next": False,
        "anim_delay": 50,
        # Graph customization options
        "show_explored": True,
        "show_path": True,
        "show_labels": True,
        "show_grid": True,
        "show_room_markers": True,
        "node_size": 18,
        "path_width": 4,
        "explored_alpha": 0.6,
        "color_theme": "neon",
        "show_legend": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ---------------------------------------------------------------------------
# Advanced Visualization Functions
# ---------------------------------------------------------------------------
def get_base64_of_bin_file(bin_file: str) -> str:
    """Convert image to base64 for inline embedding."""
    bin_path: Path = Path(__file__).parent / bin_file
    with open(bin_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def plot_hospital_grid_advanced(
    path: Optional[List[Tuple[int, int]]] = None,
    explored: Optional[List[Tuple[int, int]]] = None,
    start_room: Optional[str] = None,
    destination_room: Optional[str] = None,
    theme: str = "neon"
) -> plt.Figure:
    """Renders the hospital grid with advanced customization."""
    theme_colors = GRAPH_THEMES.get(theme, GRAPH_THEMES["neon"])
    
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(theme_colors["walkable"])
    ax.set_facecolor(theme_colors["walkable"])
    
    cmap_custom = ListedColormap([theme_colors["walkable"], theme_colors["wall"]])
    ax.imshow(HOSPITAL_GRID, cmap=cmap_custom)
    
    if st.session_state.get("show_grid", True):
        ax.set_xticks(np.arange(-.5, GRID_COLS, 1), minor=True)
        ax.set_yticks(np.arange(-.5, GRID_ROWS, 1), minor=True)
        ax.grid(which="minor", color=theme_colors["grid_line"], linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
    
    ax.set_xticks([])
    ax.set_yticks([])

    if st.session_state.get("show_room_markers", True):
        for room, (r, c) in st.session_state["custom_rooms"].items():
            color = ROOM_COLORS.get(room, '#ffffff')
            ax.plot(c, r, marker='s', color=color, 
                   markersize=st.session_state.get("node_size", 18), 
                   alpha=0.4, zorder=2)

    if st.session_state.get("show_explored", True) and explored:
        ex_r = [node[0] for node in explored]
        ex_c = [node[1] for node in explored]
        ax.scatter(ex_c, ex_r, marker='s', c=theme_colors["explored"], 
                  s=st.session_state.get("node_size", 18) * 0.5,
                  alpha=st.session_state.get("explored_alpha", 0.6),
                  zorder=3)

    if st.session_state.get("show_path", True) and path:
        path_r = [node[0] for node in path]
        path_c = [node[1] for node in path]
        ax.plot(path_c, path_r, color=theme_colors["path_glow"], 
               linewidth=st.session_state.get("path_width", 4) + 2, 
               alpha=0.3, zorder=3)
        ax.plot(path_c, path_r, color=theme_colors["path"], 
               linewidth=st.session_state.get("path_width", 4), 
               zorder=4)
        ax.scatter(path_c, path_r, marker='o', c=theme_colors["path"], 
                  s=st.session_state.get("node_size", 18) * 0.6,
                  zorder=5, edgecolors='white', linewidths=1)

    start_room = start_room or START_ROOM
    start_r, start_c = st.session_state["custom_rooms"][start_room]
    ax.scatter([start_c], [start_r], marker='s', c=theme_colors["start"],
              s=st.session_state.get("node_size", 18) * 2,
              edgecolors='white', linewidths=3, zorder=6)
    ax.annotate('START', (start_c, start_r), xytext=(start_c, start_r-1.2),
               ha='center', va='top', fontsize=10, fontweight='bold',
               color=theme_colors["start"],
               bbox=dict(boxstyle='round,pad=0.3', facecolor=theme_colors["wall"], 
                        edgecolor=theme_colors["start"], alpha=0.9))
    
    if destination_room:
        dest_r, dest_c = st.session_state["custom_rooms"][destination_room]
        ax.scatter([dest_c], [dest_r], marker='s', c=theme_colors["end"],
                  s=st.session_state.get("node_size", 18) * 2,
                  edgecolors='white', linewidths=3, zorder=6)
        ax.annotate('GOAL', (dest_c, dest_r), xytext=(dest_c, dest_r-1.2),
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   color=theme_colors["end"],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=theme_colors["wall"], 
                            edgecolor=theme_colors["end"], alpha=0.9))

    if st.session_state.get("show_labels", True):
        for room, (r, c) in st.session_state["custom_rooms"].items():
            if room not in [start_room, destination_room]:
                label_text = room.replace(" ", "\n")
                ax.text(c, r - 0.85, label_text, color='#ffffff', 
                       fontsize=8, ha='center', va='bottom', weight='bold', 
                       bbox=dict(facecolor=theme_colors["wall"], alpha=0.85, 
                                edgecolor=ROOM_COLORS.get(room, '#ffffff'), 
                                boxstyle='round,pad=0.2'))

    if st.session_state.get("show_legend", True):
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=theme_colors["start"], 
                      markersize=10, label='Start', linestyle='None'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=theme_colors["end"], 
                      markersize=10, label='Destination', linestyle='None'),
        ]
        if st.session_state.get("show_explored", True):
            legend_elements.append(
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=theme_colors["explored"], 
                          markersize=8, alpha=0.6, label='Explored', linestyle='None')
            )
        if st.session_state.get("show_path", True):
            legend_elements.append(
                plt.Line2D([0], [0], color=theme_colors["path"], linewidth=3, label='Path')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                 facecolor=theme_colors["wall"], edgecolor=theme_colors["grid_line"],
                 labelcolor='#ffffff', framealpha=0.95, fontsize=9)
    
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig



def plot_feature_importance_advanced(importances: Dict[str, float], theme: str = "neon") -> plt.Figure:
    """Plots an advanced horizontal bar chart with value labels."""
    theme_colors = GRAPH_THEMES.get(theme, GRAPH_THEMES["neon"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')


    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    labels = [FEATURE_LABELS.get(f, f) for f in features]
    
    max_score = max(scores) if scores else 1
    normalized_scores = [s / max_score for s in scores]

    colors = []
    for score in normalized_scores:
        if score > 0.8:
            colors.append(theme_colors["path"])
        elif score > 0.5:
            colors.append(theme_colors["explored"])
        else:
            colors.append('#374151')

    bars = ax.barh(labels, scores, color=colors, height=0.7, edgecolor='none')
    
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f}', va='center', ha='left', 
               color='#ffffff', fontsize=9, fontweight='bold')


    ax.set_xlabel('Importance Score', color='#a1a1aa', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', colors='#a1a1aa', labelsize=10)
    ax.tick_params(axis='y', colors='#ffffff', labelsize=10)
    ax.spines['bottom'].set_color('#1f2937')
    ax.spines['left'].set_color('#1f2937')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, color='#1f2937', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1)
    
    plt.tight_layout()
    return fig


def plot_risk_gauge(probability: float) -> plt.Figure:
    """Create a visual risk gauge with polar projection."""
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    
    theta = np.linspace(0.5 * np.pi, 2.5 * np.pi, 100)
    ax.fill_between(theta, 0.3, 0.35, color='#1f2937', alpha=0.5)
    
    prob_theta = 0.5 * np.pi + probability * np.pi
    
    if probability < 0.3:
        color = '#10b981'
    elif probability < 0.6:
        color = '#f59e0b'
    else:
        color = '#ef4444'
    
    ax.annotate('', xy=(prob_theta, 0.33), xytext=(0.5 * np.pi, 0.1),
               arrowprops=dict(arrowstyle='->', color=color, lw=3))
    
    ax.fill_between(np.linspace(0.5 * np.pi, 1.83 * np.pi, 50), 0.28, 0.32, 
                   color='#10b981', alpha=0.3)
    ax.fill_between(np.linspace(1.83 * np.pi, 2.17 * np.pi, 50), 0.28, 0.32, 
                   color='#f59e0b', alpha=0.3)
    ax.fill_between(np.linspace(2.17 * np.pi, 2.5 * np.pi, 50), 0.28, 0.32, 
                   color='#ef4444', alpha=0.3)
    
    ax.text(0.5 * np.pi, 0.15, '0%', ha='center', va='top', color='#a1a1aa', fontsize=10)
    ax.text(2.5 * np.pi, 0.15, '100%', ha='center', va='top', color='#a1a1aa', fontsize=10)
    ax.text(np.pi, 0.15, '50%', ha='center', va='top', color='#a1a1aa', fontsize=10)
    
    ax.text(np.pi, 0.5, f'{probability*100:.1f}%', ha='center', va='center', 
           color=color, fontsize=20, fontweight='bold')
    ax.text(np.pi, 0.4, 'Risk Level', ha='center', va='center', 
           color='#71717a', fontsize=10)
    
    ax.set_ylim(0, 0.5)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# App Header
# ---------------------------------------------------------------------------
logo_base64 = get_base64_of_bin_file("pathpulse_logo.png")
page_logo_base64 = get_base64_of_bin_file("page_logo.png")

st.markdown(
    f"""
    <div class="ambient-glow"></div>
    <div class="ambient-glow glow-2"></div>
    <div class="custom-header">
        <img src="data:image/png;base64,{page_logo_base64}" class="header-logo" alt="Logo" />
        <span class="header-title">Path<span style="font-weight: 700;">Pulse</span></span>
    </div>
    <div style="text-align: center; margin-bottom: 2rem; padding-top: 0.5rem; position: relative; z-index: 10;">
        <div class="liquid-glass-logo-container animate-float">
            <img src="data:image/png;base64,{logo_base64}" class="liquid-glass-logo" alt="PathPulse Logo" />
        </div>
        <h1 style="margin-top: 1.5rem; font-size: 2.2rem; font-weight: 800; letter-spacing: 1px; color: white;">
            Intelligent Hospital Navigation
        </h1>
        <p style="font-size: 1rem; color: #a1a1aa; max-width: 500px; margin: 0.5rem auto 1.5rem;">
            Advanced pathfinding algorithms & ML-powered risk assessment
        </p>
        <div style="display: flex; gap: 12px; justify-content: center;">
            <span class="badge badge-primary">A* Search</span>
            <span class="badge badge-success">Random Forest</span>
            <span class="badge badge-danger">Real-time Animation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Primary Tab Layout
# ---------------------------------------------------------------------------
tab_nav, tab_pred = st.tabs([" 🚗 Navigation", " ❤️ Risk Prediction"])

# -- Navigation Tab --------------------------------------------------------
with tab_nav:
    # Section header
    st.markdown("""
    <div class="glass-card card-highlight" style="margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="font-size: 2rem;">🏥</div>
            <div>
                <h2 style="font-size: 1.5rem; font-weight: 700; color: #ffffff; margin: 0;">
                    Hospital Path Finder
                </h2>
                <p style="font-size: 0.9rem; color: #a1a1aa; margin: 4px 0 0 0;">
                    Navigate using advanced pathfinding algorithms with customizable visualization
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_controls, col_visualization = st.columns([1, 2.2])
    
    with col_controls:
        # === Pathfinding Controls ===
        st.markdown("""
        <div class="form-section">
            <div class="form-section-title">🎯 Route Configuration</div>
        """, unsafe_allow_html=True)
        
        all_rooms = list(st.session_state["custom_rooms"].keys())
        curr_start = st.session_state.get("nav_start", START_ROOM)
        start_idx = all_rooms.index(curr_start) if curr_start in all_rooms else 0
        
        st.session_state["nav_start"] = st.selectbox(
            "📍 Start Location", options=all_rooms, index=start_idx,
            help="Select the starting room for navigation"
        )
        
        destinations = [rm for rm in all_rooms if rm != st.session_state["nav_start"]]
        curr_dest = st.session_state["nav_destination"]
        dest_idx = destinations.index(curr_dest) if curr_dest in destinations else 0
        
        st.session_state["nav_destination"] = st.selectbox(
            "🏁 Destination", options=destinations, index=dest_idx,
            help="Select the destination room"
        )
        
        algo_options = {"A* (Optimal Path)": "A*", "Greedy Best-First": "Greedy BFS"}
        current_algo = st.session_state["nav_algorithm"]
        algo_idx = 0 if current_algo == "A*" else 1
        
        selected_algo = st.selectbox(
            "⚡ Algorithm", options=list(algo_options.keys()), index=algo_idx,
            help="Choose the pathfinding algorithm"
        )
        st.session_state["nav_algorithm"] = algo_options[selected_algo]
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # === Animation Controls ===
        st.markdown("""
        <div class="form-section" style="margin-top: 16px;">
            <div class="form-section-title">🎬 Animation Settings</div>
        """, unsafe_allow_html=True)
        
        st.session_state["anim_delay"] = st.slider(
            "⏱️ Animation Speed (ms)", 
            min_value=0, max_value=300, 
            value=st.session_state.get("anim_delay", 50), 
            step=10,
            help="Delay between animation frames. Set 0 for instant display."
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # === Graph Customization ===
        with st.expander("⚙️ Customize Visualization"):
            theme_options = list(GRAPH_THEMES.keys())
            current_theme = st.session_state.get("color_theme", "neon")
            theme_idx = theme_options.index(current_theme) if current_theme in theme_options else 0
            
            st.session_state["color_theme"] = st.selectbox(
                "🎨 Color Theme", options=theme_options, index=theme_idx,
                help="Select the color scheme for the visualization"
            )
            
            st.markdown("<br/><div class='graph-options-title'>Display Options</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state["show_explored"] = st.checkbox("Show Explored", value=st.session_state.get("show_explored", True))
                st.session_state["show_labels"] = st.checkbox("Show Labels", value=st.session_state.get("show_labels", True))
            with col2:
                st.session_state["show_path"] = st.checkbox("Show Path", value=st.session_state.get("show_path", True))
                st.session_state["show_legend"] = st.checkbox("Show Legend", value=st.session_state.get("show_legend", True))
            
            st.markdown("<br/><div class='graph-options-title'>Size Adjustments</div>", unsafe_allow_html=True)
            st.session_state["node_size"] = st.slider("Node Size", 8, 30, st.session_state.get("node_size", 18), 2)
            st.session_state["path_width"] = st.slider("Path Width", 1, 10, st.session_state.get("path_width", 4), 1)
            st.session_state["explored_alpha"] = st.slider("Explored Opacity", 0.1, 1.0, st.session_state.get("explored_alpha", 0.6), 0.1)
        
        # === Room Customization ===
        with st.expander("🏗️ Customize Room Positions"):
            st.markdown("<p style='color: #a1a1aa; font-size: 0.85rem; margin-bottom: 16px;'>Adjust room coordinates on the grid (0-19)</p>", unsafe_allow_html=True)
            
            edited_rooms = {}
            for room, (orig_r, orig_c) in st.session_state["custom_rooms"].items():
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.markdown(f"<div style='font-size: 0.85rem; padding-top: 12px; color: #ffffff;'>{room}</div>", unsafe_allow_html=True)
                new_r = c2.number_input("Row", min_value=0, max_value=19, value=orig_r, key=f"r_{room}", label_visibility="collapsed")
                new_c = c3.number_input("Col", min_value=0, max_value=19, value=orig_c, key=f"c_{room}", label_visibility="collapsed")
                edited_rooms[room] = (new_r, new_c)
            
            if st.button("💾 Update Layout", use_container_width=True, type="primary"):
                valid = True
                for rm, (nr, nc) in edited_rooms.items():
                    if HOSPITAL_GRID[nr, nc] == 1:
                        st.error(f"❌ Cannot place **{rm}** at ({nr}, {nc}) - it's a wall!")
                        valid = False
                        break
                
                if valid:
                    st.session_state["custom_rooms"] = edited_rooms
                    st.session_state["nav_result"] = None
                    st.success("✅ Layout updated successfully!")
                    st.rerun()
        
        # === Find Path Button ===
        st.markdown("<br/>", unsafe_allow_html=True)
        
        if st.button("🚀 Find Optimal Path", use_container_width=True, type="primary"):
            with st.spinner("🧠 Computing optimal route..."):
                start_coords = st.session_state["custom_rooms"][st.session_state["nav_start"]]
                goal_coords = st.session_state["custom_rooms"][st.session_state["nav_destination"]]
                
                try:
                    if st.session_state["nav_algorithm"] == "A*":
                        path, exp_cnt, exp_ord = astar.search(HOSPITAL_GRID, start_coords, goal_coords)
                    else:
                        path, exp_cnt, exp_ord = greedy_bfs.search(HOSPITAL_GRID, start_coords, goal_coords)
                    
                    st.session_state["nav_result"] = {
                        "path": path,
                        "explored_count": exp_cnt,
                        "explored_order": exp_ord,
                        "start": st.session_state["nav_start"],
                        "destination": st.session_state["nav_destination"],
                        "algorithm": st.session_state["nav_algorithm"]
                    }
                    st.session_state["animate_next"] = True
                except Exception as e:
                    st.error(f"Error during pathfinding: {str(e)}")
                    
        # === Display Results ===
        if st.session_state["nav_result"]:
            res = st.session_state["nav_result"]
            st.markdown("<hr style='margin: 24px 0; border-color: rgba(255,255,255,0.1);'/>", unsafe_allow_html=True)
            
            if res["path"]:
                col1, col2, col3 = st.columns(3)
                theme_colors = GRAPH_THEMES.get(st.session_state.get("color_theme", "neon"), GRAPH_THEMES["neon"])
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value" style="background: {theme_colors['path']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {len(res['path']) - 1}
                        </div>
                        <div class="stat-label">Path Steps</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value" style="background: {theme_colors['explored']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {res['explored_count']}
                        </div>
                        <div class="stat-label">Nodes Explored</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    efficiency = (len(res['path']) - 1) / res['explored_count'] * 100 if res['explored_count'] > 0 else 0
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-value" style="background: {theme_colors['start']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {efficiency:.1f}%
                        </div>
                        <div class="stat-label">Efficiency</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"✨ Route found from **{res['start']}** to **{res['destination']}** using **{res['algorithm']}**")
            else:
                st.error("❌ No valid path exists to the selected destination.")
    
    with col_visualization:
        plot_placeholder = st.empty()
        theme = st.session_state.get("color_theme", "neon")
        
        if st.session_state["nav_result"]:
            res = st.session_state["nav_result"]
            
            if st.session_state.get("animate_next") and st.session_state.get("anim_delay", 0) > 0:
                delay_sec = st.session_state["anim_delay"] / 1000.0
                explored_so_far = []
                batch_size = max(1, int(10 * delay_sec + 1))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(0, len(res["explored_order"]), batch_size):
                    chunk = res["explored_order"][i:i+batch_size]
                    explored_so_far.extend(chunk)
                    
                    progress = min(100, int((i + len(chunk)) / len(res["explored_order"]) * 50))
                    progress_bar.progress(progress)
                    status_text.text(f"🔍 Exploring... {len(explored_so_far)} nodes")
                    
                    fig = plot_hospital_grid_advanced(
                        path=None, explored=explored_so_far,
                        start_room=res["start"], destination_room=res["destination"],
                        theme=theme
                    )
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)
                    time.sleep(delay_sec)
                
                if res["path"]:
                    path_so_far = []
                    for i in range(len(res["path"])):
                        path_so_far.append(res["path"][i])
                        progress = 50 + int((i + 1) / len(res["path"]) * 50)
                        progress_bar.progress(progress)
                        status_text.text(f"🛤️ Drawing path... {i+1}/{len(res['path'])} steps")
                        
                        fig = plot_hospital_grid_advanced(
                            path=path_so_far, explored=res["explored_order"],
                            start_room=res["start"], destination_room=res["destination"],
                            theme=theme
                        )
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)
                            time.sleep(delay_sec)
                
                progress_bar.progress(100)
                status_text.text("✅ Pathfinding complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.session_state["animate_next"] = False
            else:
                fig = plot_hospital_grid_advanced(
                    path=res["path"], explored=res["explored_order"],
                    start_room=res["start"], destination_room=res["destination"],
                    theme=theme
                )
                plot_placeholder.pyplot(fig)
                plt.close(fig)
        else:
            fig = plot_hospital_grid_advanced(
                start_room=st.session_state["nav_start"],
                destination_room=st.session_state["nav_destination"],
                theme=theme
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig)


# -- Risk Prediction Tab ---------------------------------------------------
with tab_pred:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 24px;">
            <h2 style="font-size: 1.50rem; font-weight: 600; color: #ffffff; margin-bottom: 8px;">
                Heart Disease Risk Assessment
            </h2>
            <p style="font-size: 0.95rem; color: #e5e7eb; margin: 0;">
                Enter patient clinical data below to receive an ML-backed
                risk prediction powered by a Random Forest classifier.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pred_form, col_pred_res = st.columns([1, 1.2])

    with col_pred_form:
        with st.form("prediction_form"):
            st.markdown("<h3 style='color: white; margin-bottom: 16px;'>Patient Vitals</h3>", unsafe_allow_html=True)
            
            # Using 2 columns for dense layout of 13 fields
            c1, c2 = st.columns(2)
            
            patient_data = {}
            with c1:
                patient_data["age"] = st.number_input(FEATURE_LABELS["age"], min_value=1, max_value=120, value=HEALTHY_DEFAULTS["age"])
                patient_data["sex"] = st.selectbox(FEATURE_LABELS["sex"], options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female", index=1 if HEALTHY_DEFAULTS["sex"]==1 else 0)
                patient_data["trestbps"] = st.number_input(FEATURE_LABELS["trestbps"], min_value=50, max_value=250, value=HEALTHY_DEFAULTS["trestbps"])
                patient_data["chol"] = st.number_input(FEATURE_LABELS["chol"], min_value=100, max_value=600, value=HEALTHY_DEFAULTS["chol"])
                patient_data["fbs"] = st.selectbox(FEATURE_LABELS["fbs"], options=[0, 1], format_func=lambda x: "Yes (> 120)" if x==1 else "No", index=HEALTHY_DEFAULTS["fbs"])
                patient_data["thalach"] = st.number_input(FEATURE_LABELS["thalach"], min_value=60, max_value=220, value=HEALTHY_DEFAULTS["thalach"])
                patient_data["exang"] = st.selectbox(FEATURE_LABELS["exang"], options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No", index=HEALTHY_DEFAULTS["exang"])
            with c2:
                patient_data["cp"] = st.selectbox(FEATURE_LABELS["cp"], options=[0, 1, 2, 3], index=HEALTHY_DEFAULTS["cp"], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
                patient_data["restecg"] = st.selectbox(FEATURE_LABELS["restecg"], options=[0, 1, 2], index=HEALTHY_DEFAULTS["restecg"])
                patient_data["oldpeak"] = st.number_input(FEATURE_LABELS["oldpeak"], min_value=0.0, max_value=10.0, value=float(HEALTHY_DEFAULTS["oldpeak"]), step=0.1)
                patient_data["slope"] = st.selectbox(FEATURE_LABELS["slope"], options=[0, 1, 2], index=HEALTHY_DEFAULTS["slope"])
                patient_data["ca"] = st.selectbox(FEATURE_LABELS["ca"], options=[0, 1, 2, 3], index=HEALTHY_DEFAULTS["ca"])
                patient_data["thal"] = st.selectbox(FEATURE_LABELS["thal"], options=[0, 1, 2, 3], index=HEALTHY_DEFAULTS["thal"], help="0: Unknown, 1: Normal, 2: Fixed Defect, 3: Reversable Defect") 

            st.markdown("<br/>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("Assess Risk", use_container_width=True)
            
            if submit_button:
                st.session_state["pred_submitted"] = True
                try:
                    res = predict_risk(patient_data)
                    st.session_state["pred_result"] = res
                except FileNotFoundError:
                    st.error("Model files not found! Please run `python -m prediction.train` to generate the ML models.")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

    with col_pred_res:
        if st.session_state["pred_submitted"] and st.session_state["pred_result"]:
            res = st.session_state["pred_result"]
            risk_class = res["risk_class"]
            prob = res["probability"]
            importances = res["feature_importance"]
            
            if risk_class == 1:
                st.markdown(
                    f"""
                    <div class="risk-high neon-pulse-high">
                        <div class="pulse-ring red-ring"></div>
                        <h2 style="color: #ff4d4d; margin: 0 0 10px 0; font-size: 2.2rem; text-shadow: 0 0 15px rgba(255, 77, 77, 0.6); position: relative; z-index: 2;">High Risk Detected</h2>
                        <p style="font-size: 1.15rem; color: #e5e7eb; margin: 0; position: relative; z-index: 2;">
                            The model predicts a <strong style="color: #ffffff; font-size: 1.25rem;">{prob*100:.1f}%</strong> probability of heart disease.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-low neon-pulse-low">
                        <div class="pulse-ring green-ring"></div>
                        <h2 style="color: #4dfa7f; margin: 0 0 10px 0; font-size: 2.2rem; text-shadow: 0 0 15px rgba(77, 250, 127, 0.6); position: relative; z-index: 2;">Low Risk</h2>
                        <p style="font-size: 1.15rem; color: #e5e7eb; margin: 0; position: relative; z-index: 2;">
                            The model predicts a <strong style="color: #ffffff; font-size: 1.25rem;">{prob*100:.1f}%</strong> probability of heart disease.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="glass-card" style="padding: 16px 24px;">
                    <h3 style="color: white; margin-top: 0;">Feature Importances</h3>
                    <p style="color: #e5e7eb; font-size: 0.9rem;">The factors most heavily influencing this prediction model.</p>
                </div>
                """, unsafe_allow_html=True)
            
            fig_importances = plot_feature_importance(importances)
            st.pyplot(fig_importances)
            plt.close(fig_importances)
        else:
            st.info("Submit the patient vitals form to see the prediction results and feature analysis.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <hr style="margin-top: 64px;" />
    <p style="
        text-align: center;
        font-size: 0.75rem;
        color: #d1d5db;
        font-weight: 500;
        letter-spacing: 0.25px;
        text-transform: uppercase;
    ">
        PathPulse Made by: Hadi Armughan(23k-0041) - Aliyan Munawar(23I-0641)
    </p>
    """,
    unsafe_allow_html=True,
)
