"""Clean Cyber Dark Theme CSS for PHANTOM X."""

DARK_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Dark Theme Settings */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #080b11 !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Streamlit Top Header Bar */
    header[data-testid="stHeader"], .stAppHeader {
        background-color: #080b11 !important;
        background: #080b11 !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15) !important;
    }
    
    [data-testid="stToolbar"] {
        color: #94a3b8 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }

    code, .stCode, .tech-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Branding Banner */
    .phantom-brand-container {
        padding: 22px 26px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .phantom-brand-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: linear-gradient(180deg, #00f2fe 0%, #38bdf8 100%);
    }

    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #ffffff 0%, #38bdf8 60%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 1px;
        margin-top: 5px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }

    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        background-color: #0b0f17 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
    }

    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label span {
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px dashed rgba(56, 189, 248, 0.35) !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    [data-testid="stFileUploader"] * {
        color: #cbd5e1 !important;
    }

    /* HOME OVERVIEW CARDS */
    .feature-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 20px;
        height: 100%;
        transition: all 0.2s ease-in-out;
    }
    .feature-card:hover {
        border-color: rgba(56, 189, 248, 0.45);
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.1);
    }
    .feature-icon {
        font-size: 1.6rem;
        margin-bottom: 10px;
    }
    .feature-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 6px;
    }
    .feature-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* RESULT CARDS */
    .result-card-real {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(16, 185, 129, 0.12) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 24px rgba(16, 185, 129, 0.2);
    }
    .badge-real {
        display: inline-block;
        background: rgba(16, 185, 129, 0.25);
        color: #34d399 !important;
        border: 1px solid #10b981;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 1px;
        font-family: 'JetBrains Mono', monospace;
    }

    .result-card-fake {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(239, 68, 68, 0.12) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 24px rgba(239, 68, 68, 0.2);
    }
    .badge-fake {
        display: inline-block;
        background: rgba(239, 68, 68, 0.25);
        color: #f87171 !important;
        border: 1px solid #ef4444;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 1px;
        font-family: 'JetBrains Mono', monospace;
    }

    .result-card-uncertain {
        background: linear-gradient(135deg, rgba(146, 64, 14, 0.35) 0%, rgba(245, 158, 11, 0.12) 100%);
        border: 1px solid rgba(245, 158, 11, 0.6);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 24px rgba(245, 158, 11, 0.2);
    }
    .badge-uncertain {
        display: inline-block;
        background: rgba(245, 158, 11, 0.25);
        color: #fbbf24 !important;
        border: 1px solid #f59e0b;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 1px;
        font-family: 'JetBrains Mono', monospace;
    }

    .uncertain-warning-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 12px;
        font-size: 0.85rem;
        color: #fef08a !important;
        line-height: 1.5;
    }

    .meta-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .meta-label {
        font-size: 0.72rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .meta-value {
        font-size: 0.98rem;
        color: #f1f5f9 !important;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2px;
    }

    .disclaimer-banner {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 8px;
        color: #cbd5e1 !important;
        font-size: 0.84rem;
        margin-top: 20px;
        line-height: 1.55;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        border-color: #38bdf8 !important;
        color: #f8fafc !important;
    }
</style>
"""

# Backward and cross-compatibility aliases
DARK_CYBER_THEME_CSS = DARK_THEME_CSS
LIGHT_LAB_THEME_CSS = DARK_THEME_CSS

def get_theme_css(*args, **kwargs) -> str:
    return DARK_THEME_CSS

__all__ = ["DARK_THEME_CSS", "DARK_CYBER_THEME_CSS", "LIGHT_LAB_THEME_CSS", "get_theme_css"]
