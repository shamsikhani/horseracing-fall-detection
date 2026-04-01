"""
Stage 5: Interactive Human Review - Streamlit Application.

Provides a web interface for reviewing model proposals, confirming annotations,
and triggering active learning retraining.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    CONFIG, OUTPUT_DIR, PROPOSALS_DIR, ANNOTATIONS_FILE,
    CLIP_META_DIR, MODEL_DIR, PREPROCESSED_DIR, DATA_ROOT,
    FEATURES_DIR,
)
from pipeline.active_learning import ActiveLearningController
from pipeline.metrics_tracker import get_tracker

logger = logging.getLogger(__name__)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Horse Racing Incident Detector",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ── Light, readable theme ─────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .stApp { background-color: #f8f9fb; }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e3e8;
    }
    section[data-testid="stSidebar"] * { color: #1e293b !important; }
    .stMarkdown, .stText, p, span, li, label, .stMetricValue, .stMetricLabel {
        color: #1e293b !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }

    /* ── Header ── */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a !important;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #dc2626;
        margin-bottom: 1rem;
    }

    /* ── Cards ── */
    .info-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 0.75rem;
    }
    .info-card h4 { margin: 0 0 0.5rem 0; color: #0f172a !important; }
    .info-card p, .info-card li { color: #334155 !important; font-size: 0.92rem; line-height: 1.6; }
    .info-card code { background: #f1f5f9; color: #dc2626; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85rem; }

    /* ── Badges ── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-positive { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
    .badge-negative { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
    .badge-coldstart { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
    .badge-model { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
    .badge-info { background: #f0f9ff; color: #0284c7; border: 1px solid #bae6fd; }

    /* ── Metric overrides ── */
    [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #64748b !important; }
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    /* ── Dataframe ── */
    .stDataFrame { border: 1px solid #e2e8f0; border-radius: 8px; }

    /* ── Pipeline stage cards ── */
    .stage-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        border-left: 4px solid #2563eb;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stage-card.done { border-left-color: #16a34a; }
    .stage-card.pending { border-left-color: #d97706; }
    .stage-card h4 { margin: 0 0 0.4rem 0; color: #0f172a !important; font-size: 1.05rem; }
    .stage-card p { color: #475569 !important; margin: 0.2rem 0; font-size: 0.9rem; line-height: 1.55; }
    .stage-card .stat { color: #0f172a !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────────────────────

def seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def mmss_to_seconds(mmss: str) -> float:
    """Convert MM:SS format to seconds."""
    try:
        parts = mmss.split(':')
        if len(parts) == 2:
            mins, secs = parts
            return int(mins) * 60 + int(secs)
        return 0.0
    except:
        return 0.0

@st.cache_data
def load_grief_data() -> dict:
    """Load grief events from Excel file and classify by actual incident types."""
    excel_path = Path(r"c:\Users\shams\Documents\Horseracing\video_selection_100.xlsx")
    if not excel_path.exists():
        return {}
    
    # Define fall-type incidents (positive cases)
    FALL_INCIDENTS = {'fell', 'brought down', 'unseated rider', 'slipped up'}
    
    try:
        df = pd.read_excel(excel_path, sheet_name='Video Selection')
        grief_map = {}
        for _, row in df.iterrows():
            video_id = row['Video_ID']
            grief_events_str = str(row['All_Grief_Events'])
            video_class = row['Class']
            num_falls = row['Num_Falls']
            
            # Parse grief events and determine if video contains fall-type incidents
            events_list = [e.strip() for e in grief_events_str.split(',')]
            has_fall_incident = any(event in FALL_INCIDENTS for event in events_list)
            
            # Override class based on actual grief events
            actual_class = 'FALL' if has_fall_incident else 'NO_FALL'
            
            grief_map[video_id] = {
                'grief_events': grief_events_str,
                'class': video_class,  # Original folder-based class
                'actual_class': actual_class,  # Reclassified based on grief events
                'num_falls': int(num_falls) if pd.notna(num_falls) else 0,
                'events_list': events_list,
            }
        return grief_map
    except Exception as e:
        st.error(f"Error loading grief data: {e}")
        return {}

def load_proposals() -> dict:
    """Load all proposals from disk."""
    combined = PROPOSALS_DIR / "all_proposals.json"
    if combined.exists():
        with open(combined) as f:
            return json.load(f)
    return {}


def load_annotations() -> list:
    """Load all annotations from JSONL file."""
    annotations = []
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        annotations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return annotations


def save_annotation(annotation: dict):
    """Append an annotation to the JSONL file."""
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, "a") as f:
        f.write(json.dumps(annotation) + "\n")
        f.flush()


COMPLETED_FILE = OUTPUT_DIR / "completed_videos.json"


def load_completed_videos() -> set:
    """Load set of video IDs that have been marked as 'Done'."""
    if not COMPLETED_FILE.exists():
        return set()
    with open(COMPLETED_FILE) as f:
        return set(json.load(f))


def save_completed_video(video_id: str):
    """Mark a video as completed (Done with this video)."""
    completed = load_completed_videos()
    completed.add(video_id)
    COMPLETED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPLETED_FILE, "w") as f:
        json.dump(sorted(completed), f, indent=2)


def get_annotated_video_ids() -> set:
    """Return set of video IDs marked as completed by the user."""
    return load_completed_videos()


def find_video_file(video_id: str) -> Path:
    """Find the actual video file for playback."""
    from pipeline.preprocessing import resolve_video_path
    return resolve_video_path(video_id)


def load_clip_metadata(video_id: str) -> list:
    """Load clip metadata for a specific video."""
    clip_file = CLIP_META_DIR / f"{video_id}_clips.json"
    if clip_file.exists():
        with open(clip_file) as f:
            return json.load(f)
    return []


def get_feature_stats() -> dict:
    """Gather statistics about extracted features."""
    stats = {"total_files": 0, "total_clips": 0, "dim": 0, "per_video": {}}
    if FEATURES_DIR.exists():
        for f in sorted(FEATURES_DIR.glob("*.npy")):
            arr = np.load(f)
            vid_id = f.stem
            stats["per_video"][vid_id] = {"clips": arr.shape[0], "dim": arr.shape[1]}
            stats["total_files"] += 1
            stats["total_clips"] += arr.shape[0]
            stats["dim"] = arr.shape[1]
    return stats


def get_training_history() -> dict:
    """Load training history if available."""
    hist_file = MODEL_DIR / "training_history.json"
    if hist_file.exists():
        with open(hist_file) as f:
            return json.load(f)
    return {}


# ── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with video navigation and system status."""
    st.sidebar.markdown("## 🏇 Incident Detector")
    st.sidebar.markdown("---")

    # Active Learning Status
    al_controller = ActiveLearningController()
    status = al_controller.get_status()

    mode_badge = "badge-coldstart" if status["mode"] == "cold_start" else "badge-model"
    mode_label = "Cold Start" if status["mode"] == "cold_start" else "Model-Based"
    st.sidebar.markdown(
        f'<span class="status-badge {mode_badge}">{mode_label}</span>',
        unsafe_allow_html=True,
    )

    st.sidebar.metric("Total Annotations", status["total_annotations"])
    st.sidebar.metric("Since Last Train", status["annotations_since_last_train"])
    st.sidebar.metric("Retraining Cycles", status["total_retrains"])

    if status["retrain_needed"]:
        st.sidebar.warning("Retraining threshold reached!")
        if st.sidebar.button("🔄 Trigger Retraining", type="primary"):
            with st.spinner("Retraining model..."):
                success = al_controller.trigger_retrain()
                if success:
                    st.sidebar.success("Retraining complete! Refresh to see new proposals.")
                    st.rerun()
                else:
                    st.sidebar.error("Retraining failed.")

    st.sidebar.markdown("---")

    # Video Navigation
    st.sidebar.markdown("### Video Navigation")
    proposals = load_proposals()

    if not proposals:
        st.sidebar.warning("No proposals found. Run the pipeline first.")
        return None, proposals

    # Load grief data for better filtering
    grief_data = load_grief_data()
    annotated_ids = get_annotated_video_ids()
    
    # Better filter options
    st.sidebar.markdown("### 🎯 Filter Videos")
    filter_option = st.sidebar.radio(
        "Show:",
        ["📝 To Annotate", "✅ Completed", "🔴 Fall Videos", "🟢 No-Fall Videos", "📋 All Videos"],
        index=0,
    )

    # Sort by bag probability (descending)
    sorted_videos = sorted(
        proposals.items(),
        key=lambda x: x[1].get("bag_prob", 0),
        reverse=True,
    )

    # Apply filters
    filtered_videos = []
    for vid_id, prop in sorted_videos:
        # Use actual_class (based on grief events) instead of folder-based class
        video_class = grief_data.get(vid_id, {}).get('actual_class', 
                      grief_data.get(vid_id, {}).get('class', 
                      'FALL' if prop.get("label_name") == "fell" else 'NO_FALL'))
        
        if filter_option == "📝 To Annotate" and vid_id in annotated_ids:
            continue
        if filter_option == "✅ Completed" and vid_id not in annotated_ids:
            continue
        if filter_option == "🔴 Fall Videos" and video_class != "FALL":
            continue
        if filter_option == "🟢 No-Fall Videos" and video_class != "NO_FALL":
            continue
        filtered_videos.append((vid_id, prop))

    # Show counts
    total = len(proposals)
    to_annotate = total - len(annotated_ids)
    st.sidebar.markdown(f'''
    <div style="background: #f1f5f9; padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;">
        <p style="margin: 0; font-size: 0.85rem; color: #64748b;">Showing: <strong style="color: #0f172a;">{len(filtered_videos)}</strong> videos</p>
        <p style="margin: 0; font-size: 0.8rem; color: #64748b;">To annotate: <strong style="color: #dc2626;">{to_annotate}</strong> / {total}</p>
    </div>
    ''', unsafe_allow_html=True)

    # Video list with better visual hierarchy
    for idx, (vid_id, prop) in enumerate(filtered_videos):
        # Use actual_class (based on grief events) for icon display
        video_class = grief_data.get(vid_id, {}).get('actual_class', 
                      grief_data.get(vid_id, {}).get('class', 'FALL'))
        is_annotated = vid_id in annotated_ids
        
        # Visual indicators
        status_icon = "✅" if is_annotated else "📝"
        class_icon = "🔴" if video_class == "FALL" else "🟢"
        
        # Compact button label
        label_text = f"{status_icon} {class_icon} {vid_id[:18]}..."
        
        if st.sidebar.button(
            label_text,
            key=f"btn_{vid_id}",
            use_container_width=True,
        ):
            st.session_state["selected_video"] = vid_id

    return st.session_state.get("selected_video"), proposals


# ── Main Content ────────────────────────────────────────────────────────────

def render_probability_timeline(proposal: dict, clips: list):
    """Render the interactive probability timeline chart."""
    all_probs = proposal.get("all_clip_probs", [])
    all_attn = proposal.get("all_attention_weights", [])
    candidates = proposal.get("candidates", [])

    if not all_probs or not clips:
        st.info("No clip probabilities available.")
        return

    # Time axis
    times = [c["start_time"] for c in clips[:len(all_probs)]]

    fig = go.Figure()

    # Create hover text with MM:SS format
    hover_times = [seconds_to_mmss(t) for t in times]
    
    # Clip probabilities line
    fig.add_trace(go.Scatter(
        x=times,
        y=all_probs[:len(times)],
        mode="lines+markers",
        name="Clip Probability",
        line=dict(color="#dc2626", width=2),
        marker=dict(size=3),
        customdata=hover_times,
        hovertemplate="Time: %{customdata} (%{x:.1f}s)<br>Prob: %{y:.3f}<extra></extra>",
    ))

    # Attention weights (secondary axis)
    if all_attn:
        fig.add_trace(go.Bar(
            x=times,
            y=all_attn[:len(times)],
            name="Attention Weight",
            marker_color="rgba(37, 99, 235, 0.25)",
            yaxis="y2",
            customdata=hover_times,
            hovertemplate="Time: %{customdata} (%{x:.1f}s)<br>Attention: %{y:.4f}<extra></extra>",
        ))

    # Highlight candidate regions
    for cand in candidates:
        fig.add_vrect(
            x0=cand["start_time"],
            x1=cand["end_time"],
            fillcolor="rgba(220, 38, 38, 0.08)",
            layer="below",
            line_width=2,
            line_color="rgba(220, 38, 38, 0.4)",
            annotation_text=f"#{cand.get('rank', '?')}",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="#dc2626",
        )

    fig.update_layout(
        title=dict(text="Clip-Level Probability Timeline", font=dict(color="#0f172a")),
        xaxis_title="Time (MM:SS)",
        yaxis_title="Clip Probability",
        yaxis2=dict(title="Attention Weight", overlaying="y", side="right"),
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )

    st.plotly_chart(fig, use_container_width=True, key="timeline_chart")


def render_candidate_cards(proposal: dict, video_id: str):
    """Render expandable candidate cards with clickable 'Use' buttons."""
    candidates = proposal.get("candidates", [])
    if not candidates:
        st.info("No candidates for this video.")
        return

    st.markdown("### Candidate Proposals")
    st.caption("Click a candidate to auto-fill the annotation form below.")

    # Show selected candidate indicator
    sel = st.session_state.get("selected_candidate")
    if sel and sel.get("video_id") == video_id:
        sel_rank = sel.get("rank", 0)
        st.success(f"Selected: Candidate #{sel_rank} ({seconds_to_mmss(sel['start_time'])} - {seconds_to_mmss(sel['end_time'])})")

    cols = st.columns(min(len(candidates), 3))
    for idx, cand in enumerate(candidates):
        col = cols[idx % len(cols)]
        with col:
            prob = cand.get("clip_prob", 0)
            rank = cand.get("rank", idx + 1)
            is_selected = (sel and sel.get("video_id") == video_id and sel.get("rank") == rank)
            border_color = "#2563eb" if is_selected else ("#dc2626" if prob > 0.5 else "#d97706" if prob > 0.3 else "#94a3b8")
            bg = "#eff6ff" if is_selected else ("#fef2f2" if prob > 0.5 else "#fffbeb" if prob > 0.3 else "#f8fafc")

            start_mmss = seconds_to_mmss(cand['start_time'])
            end_mmss = seconds_to_mmss(cand['end_time'])
            
            st.markdown(f"""
            <div style="background:{bg}; border-radius:10px; padding:1rem;
                        border-left:4px solid {border_color}; margin-bottom:0.5rem;
                        border-top:1px solid #e2e8f0; border-right:1px solid #e2e8f0;
                        border-bottom:1px solid #e2e8f0;">
                <strong style="color:{border_color}; font-size:1rem;">Rank #{rank}</strong><br>
                <span style="color:#64748b;">Time:</span>
                <span style="color:#0f172a; font-weight:600;">{start_mmss} - {end_mmss}</span><br>
                <span style="color:#64748b;">Probability:</span>
                <span style="color:#0f172a; font-weight:700;">{prob:.3f}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Use #{rank}", key=f"use_cand_{video_id}_{rank}", use_container_width=True):
                st.session_state["selected_candidate"] = {
                    "video_id": video_id,
                    "start_time": cand["start_time"],
                    "end_time": cand["end_time"],
                    "rank": rank,
                }
                st.rerun()


def render_annotation_form(video_id: str, proposal: dict):
    """Render the annotation form - allows multiple annotations per video."""
    
    # Track annotation start time
    if f"annotation_start_{video_id}" not in st.session_state:
        st.session_state[f"annotation_start_{video_id}"] = datetime.utcnow()
        get_tracker().log_annotation_start(video_id)
    
    # Show existing annotations for this video
    existing_annotations = [a for a in load_annotations() if a["video_id"] == video_id]
    
    if existing_annotations:
        st.markdown(f"#### Labels added so far: **{len(existing_annotations)}**")
        FALL_LABELS = {"fall", "brought_down", "unseated_rider", "slipped_up"}
        HARD_NEG_LABELS = {"pulled_up", "refused", "carried_out", "ran_out", "pull_up", "no_fall"}
        JUNK_LABELS = {"not_visible", "non_race_footage", "riderless_horse"}
        for i, ann in enumerate(existing_annotations, 1):
            ts = seconds_to_mmss(ann.get('timestamp', 0))
            end_ts = seconds_to_mmss(ann.get('end_timestamp', 0)) if ann.get('end_timestamp') else "-"
            ltype = ann.get('label_type', '?')
            if ltype in FALL_LABELS:
                border, bg, icon = "#dc2626", "#fef2f2", "🔴"
            elif ltype in HARD_NEG_LABELS:
                border, bg, icon = "#d97706", "#fffbeb", "🟡"
            elif ltype == "no_incident":
                border, bg, icon = "#16a34a", "#f0fdf4", "🟢"
            elif ltype in JUNK_LABELS:
                border, bg, icon = "#374151", "#f3f4f6", "⚫"
            else:
                border, bg, icon = "#94a3b8", "#f8fafc", "⚪"
            st.markdown(f'''
            <div style="background:{bg}; padding:0.4rem 0.6rem; margin-bottom:0.25rem; border-radius:4px; border-left:3px solid {border}; font-size:0.85rem;">
                <strong>#{i}</strong> {icon} {ltype} &nbsp;|&nbsp; {ts} - {end_ts}
            </div>
            ''', unsafe_allow_html=True)
    
    candidates = proposal.get("candidates", [])

    # Check if a candidate was clicked — use its time as default
    sel = st.session_state.get("selected_candidate")
    if sel and sel.get("video_id") == video_id:
        default_start = seconds_to_mmss(sel["start_time"])
        default_end = seconds_to_mmss(sel["end_time"])
        default_rank = sel.get("rank", 0)
    elif candidates:
        default_start = seconds_to_mmss(candidates[0]["start_time"])
        default_end = seconds_to_mmss(candidates[0]["end_time"])
        default_rank = candidates[0].get("rank", 1)
    else:
        default_start = "00:00"
        default_end = "00:00"
        default_rank = 0
    
    # Use a unique form key that increments so form clears after submit
    form_counter = st.session_state.get(f"form_counter_{video_id}", 0)

    with st.form(key=f"annotation_form_{video_id}_{form_counter}"):
        col1, col2 = st.columns(2)

        with col1:
            timestamp_mmss = st.text_input(
                "Incident Start (MM:SS)",
                value=default_start,
                help="Enter time in MM:SS format (e.g., 03:45)",
            )
            end_timestamp_mmss = st.text_input(
                "Incident End (MM:SS)",
                value=default_end,
                help="End time in MM:SS format",
            )

        with col2:
            LABEL_OPTIONS = [
                "fall", "brought_down", "unseated_rider", "slipped_up",
                "pulled_up", "refused", "carried_out", "ran_out",
                "no_incident",
                "not_visible", "non_race_footage", "riderless_horse",
            ]
            LABEL_DISPLAY = {
                "fall":              "🔴 Fall",
                "brought_down":      "🔴 Brought down",
                "unseated_rider":    "🔴 Unseated rider",
                "slipped_up":        "🔴 Slipped up",
                "pulled_up":         "🟡 Pulled up",
                "refused":           "🟡 Refused",
                "carried_out":       "🟡 Carried out",
                "ran_out":           "🟡 Ran out",
                "no_incident":       "🟢 No incident (clean)",
                "not_visible":       "⚫ Not visible (off-camera)",
                "non_race_footage":  "⚫ Non-race footage (junk)",
                "riderless_horse":   "⚫ Riderless horse (post-incident)",
            }
            label_type = st.selectbox(
                "Incident Type",
                LABEL_OPTIONS,
                index=0,
                format_func=lambda x: LABEL_DISPLAY.get(x, x),
            )
            rank_options = [0] + [c.get("rank", i + 1) for i, c in enumerate(candidates)]
            default_rank_idx = rank_options.index(default_rank) if default_rank in rank_options else 0
            candidate_rank = st.selectbox(
                "Which candidate matched?",
                rank_options,
                index=default_rank_idx,
                format_func=lambda x: "Manual (0)" if x == 0 else f"Candidate #{x}",
            )

        notes = st.text_area("Notes (optional)", placeholder="...", height=68)

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            add_annotation = st.form_submit_button("➕ Add Annotation", type="primary", use_container_width=True)
        with c2:
            done_with_video = st.form_submit_button("✅ Done with Video", use_container_width=True)
        with c3:
            no_incident = st.form_submit_button("❌ No Incident", use_container_width=True)

    # ── Handle form actions OUTSIDE the form context ──
    if add_annotation:
        timestamp = mmss_to_seconds(timestamp_mmss)
        end_timestamp = mmss_to_seconds(end_timestamp_mmss)
        
        annotation = {
            "video_id": video_id,
            "timestamp": timestamp,
            "end_timestamp": end_timestamp if end_timestamp > timestamp else None,
            "label_type": label_type,
            "candidate_rank": candidate_rank,
            "notes": notes,
            "annotated_at": datetime.utcnow().isoformat(),
            "confirmed": True,
            "bag_prob": proposal.get("bag_prob", 0),
            "source": proposal.get("source", "unknown"),
        }
        save_annotation(annotation)
        
        # Log metrics
        try:
            al_controller = ActiveLearningController()
            al_status = al_controller.get_status()
            get_tracker().log_annotation(
                video_id=video_id,
                label_type=label_type,
                incident_timestamp=timestamp,
                incident_end_timestamp=end_timestamp if end_timestamp > timestamp else None,
                confirmed=True,
                candidate_rank=candidate_rank,
                bag_prob=proposal.get("bag_prob", 0),
                source=proposal.get("source", "unknown"),
                iteration=al_status["total_retrains"],
                total_annotations=al_status["total_annotations"] + 1,
            )
        except Exception:
            pass
        
        # Increment form counter so a fresh form appears, keep video selected
        st.session_state[f"form_counter_{video_id}"] = form_counter + 1
        st.session_state["selected_video"] = video_id  # keep video open
        st.session_state.pop("selected_candidate", None)  # clear candidate selection
        st.success(f"Annotation added at {timestamp_mmss}! Add more or click Done.")
        st.rerun()
    
    if done_with_video:
        # Mark video as completed
        save_completed_video(video_id)
        
        if not existing_annotations:
            annotation = {
                "video_id": video_id,
                "timestamp": 0,
                "end_timestamp": None,
                "label_type": "no_fall",
                "candidate_rank": 0,
                "notes": "No incidents found",
                "annotated_at": datetime.utcnow().isoformat(),
                "confirmed": False,
                "bag_prob": proposal.get("bag_prob", 0),
                "source": proposal.get("source", "unknown"),
            }
            save_annotation(annotation)
        
        # Clear selection so next unannotated video shows
        if "selected_video" in st.session_state:
            del st.session_state["selected_video"]
        st.success(f"Video {video_id} marked as complete!")
        st.rerun()
    
    if no_incident:
        annotation = {
            "video_id": video_id,
            "timestamp": 0,
            "end_timestamp": None,
            "label_type": "no_fall",
            "candidate_rank": 0,
            "notes": "No incident found",
            "annotated_at": datetime.utcnow().isoformat(),
            "confirmed": False,
            "bag_prob": proposal.get("bag_prob", 0),
            "source": proposal.get("source", "unknown"),
        }
        save_annotation(annotation)
        save_completed_video(video_id)
        
        if "selected_video" in st.session_state:
            del st.session_state["selected_video"]
        st.success(f"Video {video_id} marked as no incident and completed!")
        st.rerun()


def render_video_detail(video_id: str, proposals: dict):
    """Render the main video detail view with compact layout."""
    proposal = proposals.get(video_id, {})
    if not proposal:
        st.warning(f"No proposals found for {video_id}")
        return

    # Load grief data from Excel
    grief_data = load_grief_data()
    video_grief = grief_data.get(video_id, {})
    
    # Header
    label = proposal.get("label_name", "unknown")
    bag_prob = proposal.get("bag_prob", 0)
    source = proposal.get("source", "unknown")
    # Use actual_class (based on grief events) instead of folder-based class
    video_class = video_grief.get('actual_class', video_grief.get('class', 'FALL' if label == 'fell' else 'NO_FALL'))

    st.markdown(f'<div class="main-header">{video_id}</div>', unsafe_allow_html=True)

    # ===== COMPACT 2-COLUMN LAYOUT =====
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # Video player
        video_file = find_video_file(video_id)
        if video_file and video_file.exists():
            st.video(str(video_file))
        else:
            st.warning(f"Video file not found")
        
        # Annotation form directly below video
        st.markdown("### 📝 Annotate Incident")
        render_annotation_form(video_id, proposal)
    
    with right_col:
        # Compact grief info card
        if video_grief:
            grief_events = video_grief.get('grief_events', 'N/A')
            num_falls = video_grief.get('num_falls', 0)
            badge_class = 'badge-positive' if video_class == 'FALL' else 'badge-negative'
            
            st.markdown(f'''
            <div class="info-card" style="margin-bottom: 1rem;">
                <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem;">📋 Video Info</h4>
                <p style="margin: 0.25rem 0;"><strong>Class:</strong> <span class="status-badge {badge_class}">{video_class}</span></p>
                <p style="margin: 0.25rem 0; font-size: 0.85rem;"><strong>Falls:</strong> {num_falls}</p>
                <p style="margin: 0.25rem 0; font-size: 0.85rem;"><strong>Events:</strong> {grief_events}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Model metrics
        st.markdown(f'''
        <div class="info-card" style="margin-bottom: 1rem;">
            <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem;">🤖 Model Output</h4>
            <p style="margin: 0.25rem 0; font-size: 0.85rem;"><strong>Bag Prob:</strong> {bag_prob:.3f}</p>
            <p style="margin: 0.25rem 0; font-size: 0.85rem;"><strong>Clips:</strong> {proposal.get("num_clips", 0)}</p>
            <p style="margin: 0.25rem 0; font-size: 0.85rem;"><strong>Source:</strong> {"Cold Start" if "cold" in source else "Model"}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Candidate proposals (compact) — clickable to auto-fill form
        candidates = proposal.get("candidates", [])
        if candidates:
            sel = st.session_state.get("selected_candidate")
            st.markdown(f"**Top {len(candidates)} Candidates:**")
            st.caption("Click to auto-fill annotation form")
            for idx, cand in enumerate(candidates):
                prob = cand.get("clip_prob", 0)
                rank = cand.get("rank", idx + 1)
                start_mmss = seconds_to_mmss(cand['start_time'])
                end_mmss = seconds_to_mmss(cand['end_time'])
                is_selected = (sel and sel.get("video_id") == video_id and sel.get("rank") == rank)
                color = "#2563eb" if is_selected else ("#dc2626" if prob > 0.5 else "#d97706" if prob > 0.3 else "#94a3b8")
                bg = "#eff6ff" if is_selected else "#f8fafc"
                st.markdown(f'''
                <div style="background: {bg}; border-left: 3px solid {color}; padding: 0.3rem 0.5rem; margin-bottom: 0.3rem; border-radius: 4px;">
                    <strong style="color: {color}; font-size: 0.85rem;">#{rank}</strong>
                    <span style="font-size: 0.8rem; color: #64748b;"> {start_mmss}-{end_mmss}</span>
                    <span style="font-size: 0.75rem; color: #0f172a;"> ({prob:.3f})</span>
                </div>
                ''', unsafe_allow_html=True)
                if st.button(f"Use #{rank}", key=f"sidebar_cand_{video_id}_{rank}", use_container_width=True):
                    st.session_state["selected_candidate"] = {
                        "video_id": video_id,
                        "start_time": cand["start_time"],
                        "end_time": cand["end_time"],
                        "rank": rank,
                    }
                    st.rerun()
        
        # Annotation count for this video
        ann_count = len([a for a in load_annotations() if a["video_id"] == video_id])
        if ann_count > 0:
            st.markdown(f'''
            <div style="background:#ecfdf5; padding:0.5rem; border-radius:6px; text-align:center;">
                <strong style="color:#059669;">{ann_count} annotation(s) added</strong>
            </div>
            ''', unsafe_allow_html=True)
    
    # Probability timeline below (collapsible)
    with st.expander("📊 View Probability Timeline", expanded=False):
        clips = load_clip_metadata(video_id)
        render_probability_timeline(proposal, clips)


# ── Dashboard Overview ──────────────────────────────────────────────────────

def render_dashboard(proposals: dict):
    """Render the overview dashboard when no video is selected."""
    st.markdown('<div class="main-header">Dashboard Overview</div>', unsafe_allow_html=True)

    if not proposals:
        st.info(
            "No proposals found. Run the pipeline first:\n\n"
            "```\npython -m pipeline.run_pipeline\n```"
        )
        return

    # Load grief data to get accurate class counts
    grief_data = load_grief_data()
    
    # Summary metrics
    total = len(proposals)
    fall_count = sum(1 for vid_id in proposals.keys() if grief_data.get(vid_id, {}).get('actual_class', grief_data.get(vid_id, {}).get('class')) == 'FALL')
    no_fall_count = total - fall_count
    annotated = len(get_annotated_video_ids())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Videos", total)
    col2.metric("Fall Videos", fall_count)
    col3.metric("No-Fall Videos", no_fall_count)
    col4.metric("Annotated", f"{annotated}/{total}")

    st.markdown("---")

    # Bag probability distribution
    st.markdown("### Bag-Level Probability Distribution")
    fall_probs = [p["bag_prob"] for vid_id, p in proposals.items() 
                  if grief_data.get(vid_id, {}).get('actual_class', grief_data.get(vid_id, {}).get('class')) == 'FALL']
    no_fall_probs = [p["bag_prob"] for vid_id, p in proposals.items() 
                     if grief_data.get(vid_id, {}).get('actual_class', grief_data.get(vid_id, {}).get('class')) == 'NO_FALL']

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=fall_probs, name="Fall", marker_color="#dc2626",
                                opacity=0.7, nbinsx=20))
    fig.add_trace(go.Histogram(x=no_fall_probs, name="No-Fall", marker_color="#16a34a",
                                opacity=0.7, nbinsx=20))
    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        xaxis_title="Bag Probability",
        yaxis_title="Count",
        height=300,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    st.plotly_chart(fig, use_container_width=True, key="dist_chart")

    # Video table
    st.markdown("### All Videos")
    table_data = []
    for vid_id, prop in sorted(proposals.items(),
                                key=lambda x: x[1].get("bag_prob", 0), reverse=True):
        annotated_ids = get_annotated_video_ids()
        table_data.append({
            "Video ID": vid_id,
            "Label": prop.get("label_name", "?"),
            "Bag Prob": f"{prop.get('bag_prob', 0):.3f}",
            "Clips": prop.get("num_clips", 0),
            "Candidates": len(prop.get("candidates", [])),
            "Source": prop.get("source", "?"),
            "Annotated": "Yes" if vid_id in annotated_ids else "No",
        })

    st.dataframe(table_data, use_container_width=True, key="video_table")


# ── ML Pipeline Summary ─────────────────────────────────────────────────────

def render_ml_summary():
    """Render a detailed summary of all ML work done so far."""
    st.markdown('<div class="main-header">ML Pipeline Summary</div>', unsafe_allow_html=True)

    # ── Overview ──
    st.markdown("""
    <div class="info-card">
        <h4>Project Overview</h4>
        <p>
            This system performs <strong>Weakly Supervised Temporal Action Localisation (W-TAL)</strong>
            for horse racing safety incidents. Given only <em>video-level</em> labels
            ("Fell" = positive, "Pulled-up" = negative), the pipeline learns to localise
            the precise temporal segments where incidents occur, without requiring
            frame-level annotations.
        </p>
        <p>
            The approach uses <strong>Multiple Instance Learning (MIL)</strong>: each video
            is treated as a "bag" of overlapping temporal clips. A positive bag (Fell) contains
            at least one clip where the incident happens; a negative bag (Pulled-up) contains
            no incident clips. The model learns to assign high attention and probability to
            the incident clips within positive bags.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stage 1: Preprocessing ──
    meta_file = OUTPUT_DIR / "video_metadata" / "all_videos.json"
    video_meta = []
    if meta_file.exists():
        with open(meta_file) as f:
            video_meta = json.load(f)

    fell_vids = [v for v in video_meta if v.get("binary_label") == 1]
    pull_vids = [v for v in video_meta if v.get("binary_label") == 0]
    total_duration = sum(v.get("duration", 0) for v in video_meta)

    st.markdown(f"""
    <div class="stage-card done">
        <h4>Stage 1 &mdash; Video Ingestion &amp; Metadata Extraction &nbsp;
            <span class="status-badge badge-negative">Complete</span></h4>
        <p>
            Scanned the <code>Fell/</code> and <code>Pulled-up/</code> directories and extracted
            metadata (duration, FPS, resolution) from each original video file using PyAV.
            <strong>No re-encoding was performed</strong> &mdash; the original high-resolution
            files are used directly for feature extraction, saving significant time and disk space.
        </p>
        <p>
            <span class="stat">{len(video_meta)}</span> videos discovered
            (<span class="stat">{len(fell_vids)}</span> Fell,
             <span class="stat">{len(pull_vids)}</span> Pulled-up) &bull;
            Total duration: <span class="stat">{total_duration/60:.1f} minutes</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stage 2: Segmentation ──
    clips_file = OUTPUT_DIR / "clip_metadata" / "all_clips.json"
    total_clips = 0
    clip_data = {}
    if clips_file.exists():
        with open(clips_file) as f:
            clip_data = json.load(f)
            total_clips = sum(len(c) for c in clip_data.values())

    st.markdown(f"""
    <div class="stage-card done">
        <h4>Stage 2 &mdash; Temporal Segmentation &nbsp;
            <span class="status-badge badge-negative">Complete</span></h4>
        <p>
            Applied a <strong>sliding window</strong> approach to divide each video into
            overlapping temporal clips. Each clip is <code>{CONFIG.segmentation.clip_duration}s</code>
            long with a stride of <code>{CONFIG.segmentation.stride}s</code>, creating
            dense temporal coverage.
        </p>
        <p>
            <span class="stat">{total_clips:,}</span> clips generated across
            <span class="stat">{len(clip_data)}</span> videos &bull;
            Average: <span class="stat">{total_clips // max(len(clip_data), 1)}</span> clips/video
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stage 3: Feature Extraction ──
    feat_stats = get_feature_stats()

    st.markdown(f"""
    <div class="stage-card done">
        <h4>Stage 3 &mdash; Spatiotemporal Feature Extraction (VideoMAE) &nbsp;
            <span class="status-badge badge-negative">Complete</span></h4>
        <p>
            Each clip was encoded through a <strong>frozen, pretrained VideoMAE-Base</strong>
            (<code>MCG-NJU/videomae-base</code>) backbone &mdash; a Vision Transformer (ViT-B/16)
            pretrained with <em>self-supervised masked autoencoding</em> on video data.
        </p>
        <p>
            <strong>How it works:</strong> For each 3-second clip, 16 frames are uniformly
            sampled and resized to 224&times;224. These are fed into the VideoMAE encoder
            (12 transformer layers, 86M parameters). The output token embeddings are
            <strong>mean-pooled</strong> to produce a single <code>{feat_stats['dim']}</code>-dimensional
            feature vector per clip.
        </p>
        <p>
            <strong>Pretrained backbone:</strong> VideoMAE was pretrained using a masked
            autoencoding objective on the Kinetics-400 dataset (240K videos, 400 action classes).
            During pretraining, 90% of video patches are randomly masked and the model learns
            to reconstruct them. This forces the encoder to learn rich spatiotemporal
            representations. The encoder weights are <em>frozen</em> (not fine-tuned) in our
            pipeline &mdash; we only use it as a fixed feature extractor.
        </p>
        <p>
            <span class="stat">{feat_stats['total_files']}</span> feature files saved &bull;
            <span class="stat">{feat_stats['total_clips']:,}</span> clip embeddings &bull;
            Dimension: <span class="stat">{feat_stats['dim']}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Per-video feature table
    if feat_stats["per_video"]:
        with st.expander("Feature extraction details per video", expanded=False):
            feat_table = []
            for vid_id, info in feat_stats["per_video"].items():
                label = "Fell" if any(v.get("video_id") == vid_id and v.get("binary_label") == 1 for v in video_meta) else "Pulled-up"
                feat_table.append({
                    "Video ID": vid_id,
                    "Label": label,
                    "Clips": info["clips"],
                    "Feature Dim": info["dim"],
                    "Feature Shape": f"({info['clips']}, {info['dim']})",
                })
            st.dataframe(feat_table, use_container_width=True, key="feat_table")

    # ── Stage 4: Clip Classifier Training ──
    history = get_training_history()
    model_file = MODEL_DIR / "attention_mil.pt"
    clip_clf_file = MODEL_DIR / "clip_classifier.pkl"
    clip_clf_metrics_file = MODEL_DIR / "clip_classifier_metrics.json"
    model_exists = clip_clf_file.exists() or model_file.exists()
    al_state_file = OUTPUT_DIR / "active_learning_state.json"
    al_state = {}
    if al_state_file.exists():
        with open(al_state_file) as f:
            al_state = json.load(f)

    # Load clip classifier metrics if available
    clf_metrics = {}
    if clip_clf_metrics_file.exists():
        with open(clip_clf_metrics_file) as f:
            clf_metrics = json.load(f)

    mode = al_state.get("mode", "cold_start")

    mode_explanation = ""
    if mode == "cold_start":
        mode_explanation = (
            "The system is in <strong>cold-start mode</strong>. Initial proposals were generated "
            "using heuristics (focusing on the 40%&ndash;80% region of each video, where incidents "
            "are statistically most likely)."
        )
    else:
        mode_explanation = (
            "The system is in <strong>model-based mode</strong>. The temporal-difference clip "
            "classifier has been trained on human-confirmed annotations and is generating proposals "
            "based on temporal anomaly features (embedding change magnitudes, context deviation)."
        )

    train_samples = clf_metrics.get("train_samples", "N/A")
    train_pos = clf_metrics.get("train_positives", "N/A")
    train_neg = clf_metrics.get("train_negatives", "N/A")
    train_acc = clf_metrics.get("train_accuracy", None)
    train_recall = clf_metrics.get("train_recall", None)

    st.markdown(f"""
    <div class="stage-card {'done' if model_exists else 'pending'}">
        <h4>Stage 4 &mdash; Temporal-Difference Clip Classifier &nbsp;
            <span class="status-badge {'badge-negative' if model_exists else 'badge-coldstart'}">{'Complete' if model_exists else 'Pending'}</span></h4>
        <p>
            The <strong>Temporal-Difference Clip Classifier</strong> uses 6 scalar features
            computed from VideoMAE embedding dynamics to detect incident clips:
        </p>
        <ul style="color:#475569; font-size:0.9rem; line-height:1.6;">
            <li><strong>Temporal change magnitudes</strong> &mdash; L2 distance to previous/next clip embeddings</li>
            <li><strong>Context deviation</strong> &mdash; how different a clip is from its local neighbourhood</li>
            <li><strong>Temporal position</strong> &mdash; normalised position in video (race-phase prior)</li>
            <li><strong>Max local change</strong> &mdash; largest visual discontinuity in the local window</li>
        </ul>
        <p>
            <strong>Model:</strong> Gradient Boosted Decision Trees (100 trees, max depth 3).
            Race-phase filter suppresses first 5% and last 15% of each video.
        </p>
        <p>{mode_explanation}</p>
        <p>
            Training clips: <span class="stat">{train_samples}</span> ({train_pos} fall, {train_neg} no-fall) &bull;
            Train accuracy: <span class="stat">{f'{train_acc:.1%}' if train_acc else 'N/A'}</span> &bull;
            Train recall: <span class="stat">{f'{train_recall:.1%}' if train_recall else 'N/A'}</span> &bull;
            Retraining cycles: <span class="stat">{al_state.get('total_retrains', 0)}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Training curves
    if history.get("train_loss") and len(history["train_loss"]) > 1:
        with st.expander("Training curves", expanded=True):
            fig = go.Figure()
            epochs = list(range(1, len(history["train_loss"]) + 1))
            fig.add_trace(go.Scatter(
                x=epochs, y=history["train_loss"],
                mode="lines+markers", name="Train Loss",
                line=dict(color="#dc2626", width=2), marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=epochs, y=history["val_loss"],
                mode="lines+markers", name="Val Loss",
                line=dict(color="#2563eb", width=2), marker=dict(size=5),
            ))
            fig.update_layout(
                title=dict(text="Training & Validation Loss", font=dict(color="#0f172a")),
                xaxis_title="Epoch", yaxis_title="Loss",
                template="plotly_white",
                height=300,
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                font=dict(color="#334155"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True, key="train_chart")

    # ── Stage 5: Active Learning ──
    st.markdown(f"""
    <div class="stage-card {'done' if mode != 'cold_start' else 'pending'}">
        <h4>Stage 5 &mdash; Active Learning &amp; Human Review &nbsp;
            <span class="status-badge badge-info">In Progress</span></h4>
        <p>
            This is the <strong>human-in-the-loop</strong> stage. The system presents candidate
            temporal proposals for each video. A human reviewer watches the video, confirms or
            rejects each proposal, and provides precise incident timestamps.
        </p>
        <p>
            After <code>{CONFIG.active_learning.retraining_threshold}</code> new annotations are
            collected, the system automatically retrains the AttentionMIL model, incorporating
            the confirmed temporal annotations as additional supervision. This iterative cycle
            progressively improves localisation accuracy.
        </p>
        <p>
            Annotations so far: <span class="stat">{al_state.get('last_training_count', 0)}</span> &bull;
            Retraining threshold: <span class="stat">{CONFIG.active_learning.retraining_threshold}</span> &bull;
            Mode: <span class="stat">{mode.replace('_', ' ').title()}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── What's next ──
    st.markdown("""
    <div class="info-card">
        <h4>What To Do Next</h4>
        <ul style="color:#475569; font-size:0.9rem; line-height:1.8;">
            <li>Go to the <strong>Review Videos</strong> tab and select a video from the sidebar</li>
            <li>Watch the video and examine the candidate proposals (highlighted time regions)</li>
            <li>Use the <strong>Annotation Form</strong> to confirm or reject each proposal with precise timestamps</li>
            <li>After 10 annotations, the system will offer to <strong>retrain</strong> the model</li>
            <li>Each retraining cycle improves the model's ability to localise incidents</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ── Main App ────────────────────────────────────────────────────────────────

def main():
    tab_review, tab_dashboard, tab_ml = st.tabs([
        "🎬 Review Videos", "📊 Dashboard", "🧠 ML Pipeline Summary"
    ])

    selected_video, proposals = render_sidebar()

    with tab_review:
        if selected_video and selected_video in proposals:
            render_video_detail(selected_video, proposals)
        else:
            st.info("Select a video from the sidebar to begin reviewing.")

    with tab_dashboard:
        render_dashboard(proposals)

    with tab_ml:
        render_ml_summary()


if __name__ == "__main__":
    main()
else:
    main()
