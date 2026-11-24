import os
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="PHANTOM X | Image Authenticity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fail-safe CSS Theme loading (Embeds full Cyber Dark styling so ImportErrors can never occur)
try:
    from frontend.styles import DARK_THEME_CSS
except Exception:
    DARK_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] { background-color: #080b11 !important; color: #f1f5f9 !important; font-family: 'Inter', sans-serif !important; }
    header[data-testid="stHeader"], .stAppHeader { background-color: #080b11 !important; border-bottom: 1px solid rgba(56, 189, 248, 0.15) !important; }
    [data-testid="stSidebar"] { background-color: #0b0f17 !important; border-right: 1px solid rgba(56, 189, 248, 0.15) !important; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .phantom-brand-container { padding: 22px 26px; background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.7) 100%); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; margin-bottom: 24px; position: relative; }
    .phantom-brand-container::before { content: ""; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, #00f2fe 0%, #38bdf8 100%); }
    .brand-title { font-size: 2.1rem; font-weight: 800; letter-spacing: 2px; background: linear-gradient(90deg, #ffffff 0%, #38bdf8 60%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
    .brand-subtitle { font-size: 0.9rem; color: #94a3b8; font-weight: 600; letter-spacing: 1px; margin-top: 5px; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

from phantom_x.inference import (
    load_forensic_model,
    run_inference,
    extract_image_metadata,
    DEFAULT_CHECKPOINT_PATH
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Analyze Image"

# Cached Model Loader (Silent backend loading without exposing internal model specifications)
@st.cache_resource(show_spinner="Initializing PHANTOM X Core...")
def get_cached_model():
    try:
        model, meta = load_forensic_model(DEFAULT_CHECKPOINT_PATH)
        return model, meta, None
    except Exception as e:
        return None, None, str(e)

# Render Sidebar (Clean, minimal, zero model disclosure)
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 20px;">
            <div style="font-size: 1.45rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace; display: flex; align-items: center; gap: 8px;">
                <span>⬡</span> PHANTOM X
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px;">
                IMAGE AUTHENTICITY DETECTOR
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation
    page_selection = st.radio(
        "Navigation",
        ["Home", "Analyze Image", "Analysis History"],
        index=0 if st.session_state.nav_page == "Home" else (1 if st.session_state.nav_page == "Analyze Image" else 2)
    )
    st.session_state.nav_page = page_selection

    st.markdown("---")

    # Clean system status
    model, meta, model_err = get_cached_model()
    if model_err:
        st.markdown(f"""
        <div style="background: rgba(127, 29, 29, 0.4); border: 1px solid #ef4444; border-radius: 8px; padding: 10px; color: #fca5a5; font-size: 0.8rem;">
            ❌ Core Service Offline
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; padding: 10px 14px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; font-size: 0.82rem; color: #34d399; font-family: 'JetBrains Mono', monospace;">
            <span style="font-size: 0.65rem;">●</span> System Ready
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# PAGE: HOME (Clean, modern overview without walls of text)
# -------------------------------------------------------------------------------------------------
if st.session_state.nav_page == "Home":
    st.markdown("""
        <div class="phantom-brand-container">
            <h1 class="brand-title">⬡ PHANTOM X</h1>
            <div class="brand-subtitle">AI-Generated vs Real Image Detector</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 24px; margin-bottom: 24px;">
        <h3 style="margin-top: 0; color: #f8fafc; font-size: 1.25rem;">System Overview</h3>
        <p style="font-size: 1.0rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 0;">
            <strong>PHANTOM X</strong> is an intelligent image classification system designed to determine whether a digital image is an 
            <strong>authentic camera capture (REAL)</strong> or <strong>synthetically generated (AI-GENERATED)</strong>. 
            The system analyzes spatial patterns, frequency distributions, and generation traces to provide a clear authenticity assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <div class="feature-title">Instant Verification</div>
            <div class="feature-desc">Upload JPG, JPEG, PNG, or WEBP images up to 20MB for fast on-device analysis.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚖️</div>
            <div class="feature-title">Confidence Scoring</div>
            <div class="feature-desc">Receive clear percentage scores, probability distributions, and boundary uncertainty alerts.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📋</div>
            <div class="feature-title">Scan History & Export</div>
            <div class="feature-desc">Keep track of all analyzed files in your active session and export logs to CSV.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

    if st.button("🚀 Launch Image Analysis", use_container_width=True):
        st.session_state.nav_page = "Analyze Image"
        st.rerun()

    st.markdown("""
        <div class="disclaimer-banner">
            ⚠️ <strong>Disclaimer:</strong> Model scores are probabilistic indicators produced by an experimental machine learning system. 
            Predictions represent algorithmic evaluations, not absolute proof of authenticity or synthetic fabrication.
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# PAGE: ANALYZE IMAGE
# -------------------------------------------------------------------------------------------------
elif st.session_state.nav_page == "Analyze Image":
    st.markdown("""
        <div class="phantom-brand-container">
            <h1 class="brand-title">🔍 Image Analysis</h1>
            <div class="brand-subtitle">Upload digital image to test whether it is Real or AI-Generated</div>
        </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_sample = st.tabs(["📁 Upload Image File", "🧪 Select Built-in Test Sample"])

    selected_image = None
    source_filename = None
    file_bytes_len = 0

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Drag and drop an image file (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Maximum supported file size: 20MB. Formats: JPEG, PNG, WEBP."
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_bytes_len = len(file_bytes)
            if file_bytes_len > 20 * 1024 * 1024:
                st.error("Uploaded file exceeds 20MB limit. Please provide a smaller image.")
            else:
                try:
                    selected_image = Image.open(uploaded_file)
                    source_filename = uploaded_file.name
                except Exception as e:
                    st.error(f"Invalid image file: {e}")

    with tab_sample:
        st.write("Or evaluate pre-loaded test samples:")
        val_samples = {
            "Real Camera Sample (data/val/REAL/080_007532_2f1bef11.jpg)": "data/val/REAL/080_007532_2f1bef11.jpg",
            "Real Camera Sample 2 (data/val/REAL/081_008293_62d881b0.jpg)": "data/val/REAL/081_008293_62d881b0.jpg",
            "AI-Generated Sample (data/val/FAKE/080_020450_9d55e017.png)": "data/val/FAKE/080_020450_9d55e017.png",
            "AI-Generated Sample 2 (data/val/FAKE/081_003606_89a6726b.jpg)": "data/val/FAKE/081_003606_89a6726b.jpg",
        }
        chosen_sample_key = st.selectbox("Choose sample image:", list(val_samples.keys()))
        if st.button("Load Selected Sample"):
            sample_path = Path(val_samples[chosen_sample_key])
            if sample_path.exists():
                selected_image = Image.open(sample_path)
                source_filename = sample_path.name
                file_bytes_len = sample_path.stat().st_size
                st.info(f"Loaded sample: {sample_path}")
            else:
                st.error(f"Sample not found on disk at {sample_path}")

    # Process and display image diagnostics
    if selected_image is not None and source_filename is not None:
        st.markdown("---")
        col_img, col_forensics = st.columns([1, 1], gap="large")

        metadata = extract_image_metadata(selected_image, source_filename, file_bytes_len)

        with col_img:
            st.markdown("### Image Preview & Details")
            
            # Constrained image preview to prevent layout distortion on tall portrait images
            st.image(selected_image, use_container_width=True, caption=f"Source: {source_filename}")

            st.markdown("#### Structural Attributes")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class="meta-box">
                    <div class="meta-label">Resolution</div>
                    <div class="meta-value">{metadata['width']} × {metadata['height']}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"""
                <div class="meta-box">
                    <div class="meta-label">Format / Mode</div>
                    <div class="meta-value">{metadata['format']} / {metadata['mode']}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col3:
                st.markdown(f"""
                <div class="meta-box">
                    <div class="meta-label">File Size</div>
                    <div class="meta-value">{metadata['file_size_kb']} KB</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 Camera & EXIF Metadata"):
                if metadata["has_exif"] and metadata["exif_data"]:
                    st.json(metadata["exif_data"])
                else:
                    st.info("No EXIF header tags detected. (Common in AI-generated imagery or web-compressed media).")

        with col_forensics:
            st.markdown("### Analysis Result")

            model, meta, model_err = get_cached_model()
            if model_err or model is None:
                st.error("Cannot perform inference: System core unavailable.")
            else:
                with st.spinner("Analyzing image patterns..."):
                    result = run_inference(model, selected_image, meta)

                pred_label = result["predicted_label"]
                score_pct = result["model_score_pct"]
                ai_score = result["scores"]["AI-GENERATED"]
                real_score = result["scores"]["REAL"]

                # UNCERTAINTY BAND:
                # If predicted confidence is between 50% and 65%, flag as inconclusive
                is_borderline = (score_pct <= 65.0)

                if is_borderline:
                    card_class = "result-card-uncertain"
                    badge_class = "badge-uncertain"
                    badge_icon = "⚠️ INCONCLUSIVE / BORDERLINE"
                    headline_text = f"Result: BORDERLINE ({pred_label} indication at {score_pct}%)"
                    status_explanation = (
                        "Ambiguous detection signal. The computed score lies near the decision threshold (50%–65%). "
                        "This commonly occurs when inspecting modern generative synthesis engines or images "
                        "with subtle generation traits and post-processing."
                    )
                elif pred_label == "REAL":
                    card_class = "result-card-real"
                    badge_class = "badge-real"
                    badge_icon = "🛡️ REAL"
                    headline_text = f"Authenticity Score: {score_pct}%"
                    status_explanation = "Visual inspection indicates photographic camera sensor characteristics."
                else:
                    card_class = "result-card-fake"
                    badge_class = "badge-fake"
                    badge_icon = "⚠️ AI-GENERATED"
                    headline_text = f"Synthetic Detection Score: {score_pct}%"
                    status_explanation = "Visual inspection indicates synthetic generative artifacts."

                # Render Result Card
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span class="{badge_class}">{badge_icon}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #cbd5e1;">
                            Latency: {result['latency_ms']} ms
                        </span>
                    </div>
                    <div style="font-size: 1.15rem; color: #f8fafc; margin-bottom: 6px; font-weight: 600;">
                        {headline_text}
                    </div>
                    <div style="font-size: 0.85rem; color: #e2e8f0; line-height: 1.5;">
                        {status_explanation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if is_borderline:
                    st.markdown("""
                    <div class="uncertain-warning-box">
                        <strong>⚠️ Notice: Boundary Uncertainty</strong><br>
                        The input image exhibits features close to the 50% boundary. 
                        This image is flagged as <em>INCONCLUSIVE</em>.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### Probability Distribution")
                col_bar1, col_bar2 = st.columns(2)
                with col_bar1:
                    st.markdown(f"<span style='color: #f87171; font-weight: 600;'>AI-Generated: {ai_score}%</span>", unsafe_allow_html=True)
                    st.progress(ai_score / 100.0)
                with col_bar2:
                    st.markdown(f"<span style='color: #34d399; font-weight: 600;'>Real Camera: {real_score}%</span>", unsafe_allow_html=True)
                    st.progress(real_score / 100.0)

                # Append to Session History
                history_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Filename": source_filename,
                    "Predicted Class": "INCONCLUSIVE" if is_borderline else pred_label,
                    "Model Score": f"{score_pct}%",
                    "AI Score": f"{ai_score}%",
                    "Real Score": f"{real_score}%",
                    "Dimensions": f"{metadata['width']}x{metadata['height']}",
                    "Format": metadata["format"],
                    "Status": "Inconclusive / Borderline" if is_borderline else "High Confidence"
                }

                if not st.session_state.history or st.session_state.history[-1]["Filename"] != source_filename:
                    st.session_state.history.append(history_entry)

                st.markdown("""
                <div class="disclaimer-banner">
                    ⚠️ <strong>Disclaimer:</strong> The prediction scores above represent algorithmic estimates from the PHANTOM X vision system. 
                    These scores are probabilistic indicators, <strong>not absolute proof of authenticity or synthetic fabrication</strong>.
                </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# PAGE: ANALYSIS HISTORY
# -------------------------------------------------------------------------------------------------
elif st.session_state.nav_page == "Analysis History":
    st.markdown("""
        <div class="phantom-brand-container">
            <h1 class="brand-title">📋 Analysis History</h1>
            <div class="brand-subtitle">Session scan telemetry and logs</div>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No scans recorded in this session. Go to 'Analyze Image' to begin inspecting files.")
    else:
        df_history = pd.DataFrame(st.session_state.history)

        total_scans = len(df_history)
        ai_scans = len(df_history[df_history["Predicted Class"] == "AI-GENERATED"])
        real_scans = len(df_history[df_history["Predicted Class"] == "REAL"])
        uncertain_scans = len(df_history[df_history["Predicted Class"] == "INCONCLUSIVE"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Scans", total_scans)
        with c2:
            st.metric("AI-Generated", ai_scans)
        with c3:
            st.metric("Real Camera", real_scans)
        with c4:
            st.metric("Inconclusive / Borderline", uncertain_scans)

        st.markdown("### Detailed Session Log")
        st.dataframe(df_history, use_container_width=True)

        col_clear, col_export = st.columns([1, 4])
        with col_clear:
            if st.button("🗑️ Clear Session History", type="secondary"):
                st.session_state.history = []
                st.rerun()
        with col_export:
            csv_data = df_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Session Log (CSV)",
                data=csv_data,
                file_name=f"phantom_x_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
