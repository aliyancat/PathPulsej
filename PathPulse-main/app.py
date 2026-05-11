"""
PathPulse — Intelligent Hospital Navigation & Heart Disease Prediction
======================================================================
Main Streamlit entry point. Configures the page layout, loads custom
styling, initializes session state, and routes users between the
Navigation and Risk Prediction tabs.
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
    ROOM_COLORS, get_destination_rooms
)
from navigation import astar, greedy_bfs

# Import prediction modules - Advanced Version
from prediction.preprocess import HEALTHY_DEFAULTS, FEATURE_LABELS
from prediction.predict import predict_risk

# ---------------------------------------------------------------------------
# Page Configuration (MUST be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PathPulse — Hospital Navigation & Risk Prediction",
    page_icon="page_logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS Injection
# ---------------------------------------------------------------------------
def load_custom_css() -> None:
    """Load and inject the custom Sentry-inspired dark theme CSS."""
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
    """Initialize all session state variables used across tabs."""
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
        "anim_delay": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------
def plot_hospital_grid(
    path: Optional[List[Tuple[int, int]]] = None,
    explored: Optional[List[Tuple[int, int]]] = None,
    start_room: Optional[str] = None,
    destination_room: Optional[str] = None
) -> plt.Figure:
    """Renders the hospital grid using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    
    # Base grid: 0=walkable, 1=wall
    cmap = ListedColormap(['#1e1e1e', '#050505']) # Walkable: elevated dark, Wall: abyss black
    ax.imshow(HOSPITAL_GRID, cmap=cmap)

    # Plot all rooms lightly
    for room, (r, c) in st.session_state["custom_rooms"].items():
        color = ROOM_COLORS.get(room, '#ffffff')
        ax.plot(c, r, marker='s', color=color, markersize=16, alpha=0.3)
        # Add text label above the cell
        label_text = room.replace(" ", "\n")  # Wrap text for long names
        ax.text(c, r - 0.75, label_text, color='#ffffff', fontsize=10, ha='center', va='bottom', weight='bold', 
                bbox=dict(facecolor='#050505', alpha=0.85, edgecolor=color, boxstyle='round,pad=0.3'))

    # Highlight Explored Nodes
    if explored:
        ex_r = [node[0] for node in explored]
        ex_c = [node[1] for node in explored]
        ax.plot(ex_c, ex_r, marker='s', color='#4a90e2', markersize=8, alpha=0.6, linestyle='None')

    # Highlight Path
    if path:
        path_r = [node[0] for node in path]
        path_c = [node[1] for node in path]
        ax.plot(path_c, path_r, color='#ff5252', linewidth=4, zorder=3)
        ax.plot(path_c, path_r, marker='o', color='#ff5252', markersize=5, linestyle='None', zorder=4)

    # Highlight Start and End tightly
    start_room = start_room or START_ROOM
    start_r, start_c = st.session_state["custom_rooms"][start_room]
    ax.plot(start_c, start_r, marker='s', color=ROOM_COLORS.get(start_room, '#ffffff'), markersize=18, markeredgecolor='white', markeredgewidth=2, zorder=5, label=start_room)
    
    if destination_room:
        dest_r, dest_c = st.session_state["custom_rooms"][destination_room]
        ax.plot(dest_c, dest_r, marker='s', color=ROOM_COLORS[destination_room], markersize=18, markeredgecolor='white', markeredgewidth=2, zorder=5, label=destination_room)

    # Grid lines and formatting
    ax.set_xticks(np.arange(-.5, 20, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 20, 1), minor=True)
    ax.grid(which="minor", color="#362d59", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Legend
    ax.legend(loc='upper right', frameon=True, facecolor='#050505', edgecolor='#262626', labelcolor='#ffffff', title_fontproperties={'weight':'bold'}, framealpha=0.9)
    
    # Ensure static layout bounds so the image size never fluctuates
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    return fig


def plot_feature_importance(importances: Dict[str, float]) -> plt.Figure:
    """Plots a horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    # Sort features
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    bars = ax.barh(labels, scores, color='#2a2a2a')
    
    # Top 3 get neon crimson accent
    for i in range(-1, -4, -1):
        if len(bars) >= abs(i):
            bars[i].set_color('#ff2a2a')

    ax.set_xlabel('Importance Score', color='#e5e7eb', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', colors='#e5e7eb')
    ax.tick_params(axis='y', colors='#e5e7eb', labelsize=10)
    ax.spines['bottom'].set_color('#262626')
    ax.spines['left'].set_color('#262626')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# App Header
# ---------------------------------------------------------------------------
def get_base64_of_bin_file(bin_file):
    bin_path = Path(__file__).parent / bin_file
    with open(bin_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

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
    <div class="hero-container" style="text-align: center; margin-bottom: 2rem; padding-top: 1rem; position: relative; z-index: 10;">
        <div class="liquid-glass-logo-container">
            <img src="data:image/png;base64,{logo_base64}" class="liquid-glass-logo" alt="PathPulse Logo" />
        </div>
        <h1 class="hero-text-glow" style="margin-top: 1.5rem; font-size: 2.5rem; font-weight: 800; letter-spacing: 2px; color: white;">
            PATHPULSE NEURAL <span style="color: #ff2a2a; text-shadow: 0 0 15px rgba(255,42,42,0.8);">ENGINE</span>
        </h1>
        <p style="font-size: 1.15rem; color: #d1d5db; max-width: 600px; margin: 0.5rem auto 2.5rem; opacity: 0.8;">
            Advanced Hospital Navigation & ML Risk Prediction
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Primary Tab Layout
# ---------------------------------------------------------------------------
tab_nav, tab_pred = st.tabs([" Navigation", " Risk Prediction"])

# -- Navigation Tab --------------------------------------------------------
with tab_nav:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 24px;">
            <h2 style="font-size: 1.50rem; font-weight: 600; color: #ffffff; margin-bottom: 8px;">
                Hospital Pathfinder
            </h2>
            <p style="font-size: 0.95rem; color: #e5e7eb; margin: 0;">
                Select a destination department and algorithm to find the optimal
                route from the Dispensary through the hospital grid.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_nav_controls, col_nav_plot = st.columns([1, 2])
    
    with col_nav_controls:
        # --- ROOM LOCATION EDITOR ---
        with st.expander("Customize Rooms"):
            st.markdown("<p style='font-size: 0.85rem; color: #d1d5db; margin-bottom: 12px;'>Modify the grid coordinates (0-19) for any room. Ensure the cell is not a wall!</p>", unsafe_allow_html=True)
            
            edited_rooms = {}
            for room, (orig_r, orig_c) in st.session_state["custom_rooms"].items():
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.markdown(f"<div style='font-size: 0.85rem; padding-top: 8px;'>{room}</div>", unsafe_allow_html=True)
                new_r = c2.number_input(f"Row", min_value=0, max_value=19, value=orig_r, key=f"r_{room}", label_visibility="collapsed")
                new_c = c3.number_input(f"Col", min_value=0, max_value=19, value=orig_c, key=f"c_{room}", label_visibility="collapsed")
                edited_rooms[room] = (new_r, new_c)
            
            if st.button("Update Layout", type="secondary", use_container_width=True):
                valid = True
                for rm, (nr, nc) in edited_rooms.items():
                    if HOSPITAL_GRID[nr, nc] == 1:
                        st.error(f"Cannot place **{rm}** at ({nr}, {nc}) because it is a wall.")
                        valid = False
                        break
                
                if valid:
                    st.session_state["custom_rooms"] = edited_rooms
                    st.session_state["nav_result"] = None  # Reset path on map change
                    st.success("Layout updated successfully!")
                    st.rerun()

        # Form-like inputs without actual st.form, as we want immediate response or simple button
        all_rooms = list(st.session_state["custom_rooms"].keys())
        curr_start = st.session_state.get("nav_start", START_ROOM)
        
        st.session_state["nav_start"] = st.selectbox(
            "Start Room",
            options=all_rooms,
            index=all_rooms.index(curr_start) if curr_start in all_rooms else 0
        )
        
        destinations = [rm for rm in all_rooms if rm != st.session_state["nav_start"]]
        curr_dest = st.session_state["nav_destination"]
        
        st.session_state["nav_destination"] = st.selectbox(
            "Destination Room",
            options=destinations,
            index=destinations.index(curr_dest) if curr_dest in destinations else 0
        )
        
        st.session_state["nav_algorithm"] = st.selectbox(
            "Search Algorithm",
            options=["A*", "Greedy BFS"],
            index=0 if st.session_state["nav_algorithm"] == "A*" else 1
        )
        
        st.session_state["anim_delay"] = st.slider(
            "Animation Delay (ms)", 
            min_value=0, 
            max_value=200, 
            value=st.session_state.get("anim_delay", 0), 
            step=10,
            help="Set > 0 to watch the algorithm explore the map in real-time."
        )
        
        if st.button("Find Path", type="primary", use_container_width=True):
            with st.spinner("Calculating optimal route..."):
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
                    
        if st.session_state["nav_result"]:
            res = st.session_state["nav_result"]
            st.markdown("<br/>", unsafe_allow_html=True)
            if res["path"]:
                st.metric(label="Path Length (Steps)", value=len(res["path"]) - 1)
                st.metric(label="Nodes Explored", value=res["explored_count"])
                st.success(f"Route found from **{res['start']}** to **{res['destination']}** using **{res['algorithm']}**.")
            else:
                st.metric(label="Nodes Explored", value=res["explored_count"])
                st.error("No valid path exists to the selected destination.")

    with col_nav_plot:
        plot_placeholder = st.empty()
        
        if st.session_state["nav_result"]:
            res = st.session_state["nav_result"]
            
            if st.session_state.get("animate_next") and st.session_state.get("anim_delay", 0) > 0:
                delay_sec = st.session_state["anim_delay"] / 1000.0
                explored_so_far = []
                
                # Batch sizes for rendering to avoid freezing on long paths
                # If delay is very small, we update fewer times to keep it fast
                batch_size = max(1, int(0.05 / (delay_sec + 0.001)))
                
                # Animate Exploration
                for i in range(0, len(res["explored_order"]), batch_size):
                    chunk = res["explored_order"][i:i+batch_size]
                    explored_so_far.extend(chunk)
                    
                    fig = plot_hospital_grid(
                        path=None,
                        explored=explored_so_far,
                        start_room=res["start"],
                        destination_room=res["destination"]
                    )
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)
                    if delay_sec > 0:
                        time.sleep(delay_sec)
                        
                # Animate Path
                if res["path"]:
                    path_so_far = []
                    for i in range(len(res["path"])):
                        path_so_far.append(res["path"][i])
                        fig = plot_hospital_grid(
                            path=path_so_far,
                            explored=res["explored_order"],
                            start_room=res["start"],
                            destination_room=res["destination"]
                        )
                        plot_placeholder.pyplot(fig)
                        plt.close(fig)
                        if delay_sec > 0:
                            time.sleep(delay_sec)
                
                st.session_state["animate_next"] = False
            else:
                fig = plot_hospital_grid(
                    path=res["path"],
                    explored=res["explored_order"],
                    start_room=res["start"],
                    destination_room=res["destination"]
                )
                plot_placeholder.pyplot(fig)
                plt.close(fig)
        else:
            fig = plot_hospital_grid(
                start_room=st.session_state["nav_start"],
                destination_room=st.session_state["nav_destination"]
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
