"""
Custom styles for IIITH Retail Checkout Assistant
Modern, professional UI with IIITH branding
"""

import streamlit as st

# IIITH Brand Colors
COLORS = {
    "primary": "#1a365d",        # Deep navy blue
    "primary_light": "#2c5282",  # Lighter navy
    "secondary": "#38a169",      # Green for success/actions
    "accent": "#ed8936",         # Orange accent
    "background": "#f7fafc",     # Light gray background
    "surface": "#ffffff",        # White surface
    "text_primary": "#1a202c",   # Dark text
    "text_muted": "#718096",     # Muted text
    "border": "#e2e8f0",         # Light border
    "error": "#e53e3e",          # Red for errors
    "warning": "#dd6b20",        # Warning orange
    "success": "#38a169",        # Success green
    "info": "#3182ce",           # Info blue
}


def apply_custom_styles():
    """Apply custom CSS styles to the Streamlit app"""
    st.markdown(f"""
    <style>
        /* ==================== GLOBAL STYLES ==================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        /* Remove default padding from main container */
        .main .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 1rem;
            max-width: 1200px;
        }}
        
        /* ==================== HIDE DEFAULT MULTI-PAGE NAV ==================== */
        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}
        
        /* Also hide the nav section */
        [data-testid="stSidebar"] [data-testid="stSidebarNavItems"],
        [data-testid="stSidebar"] nav {{
            display: none !important;
        }}
        
        /* ==================== SIDEBAR STYLES ==================== */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            padding-top: 0;
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 0;
        }}
        
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}
        
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {{
            color: white !important;
            font-weight: 600;
        }}
        
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.2);
            margin: 0.75rem 0;
        }}
        
        [data-testid="stSidebar"] .stButton > button {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.3);
            color: white !important;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
            width: 100%;
            text-align: left;
        }}
        
        [data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(255,255,255,0.2);
            border-color: rgba(255,255,255,0.5);
            transform: translateX(4px);
        }}
        
        /* Nav button styling */
        .nav-button {{
            background: rgba(255,255,255,0.15) !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.875rem 1rem !important;
            font-size: 0.95rem !important;
        }}
        
        .nav-button:hover {{
            background: rgba(255,255,255,0.25) !important;
        }}
        
        /* ==================== CHAT STYLES ==================== */
        [data-testid="stChatMessage"] {{
            background: {COLORS['surface']};
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid {COLORS['border']};
        }}
        
        /* Chat input styling */
        [data-testid="stChatInput"] {{
            border-radius: 12px;
            border: 2px solid {COLORS['border']};
        }}
        
        [data-testid="stChatInput"]:focus-within {{
            border-color: {COLORS['primary']};
            box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
        }}
        
        /* ==================== BUTTON STYLES ==================== */
        .stButton > button {{
            border-radius: 8px;
            font-weight: 500;
            padding: 0.5rem 1.5rem;
            transition: all 0.2s ease;
        }}
        
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            border: none;
            color: white;
        }}
        
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(26, 54, 93, 0.3);
        }}
        
        /* ==================== INPUT STYLES ==================== */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {{
            border-radius: 8px;
            border: 2px solid {COLORS['border']};
            padding: 0.75rem 1rem;
            font-size: 1rem;
        }}
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: {COLORS['primary']};
            box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
        }}
        
        /* ==================== TAB STYLES ==================== */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            background: {COLORS['background']};
            border-radius: 12px;
            padding: 4px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 500;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: {COLORS['surface']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        /* ==================== METRIC STYLES ==================== */
        [data-testid="stMetric"] {{
            background: {COLORS['surface']};
            padding: 1.25rem;
            border-radius: 12px;
            border: 1px solid {COLORS['border']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        [data-testid="stMetric"] label {{
            color: {COLORS['text_muted']} !important;
            font-size: 0.9rem !important;
        }}
        
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {COLORS['primary']} !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
        }}
        
        /* ==================== EXPANDER STYLES ==================== */
        .streamlit-expanderHeader {{
            background: {COLORS['background']};
            border-radius: 8px;
            font-weight: 500;
        }}
        
        /* ==================== FILE UPLOADER ==================== */
        [data-testid="stFileUploader"] {{
            border: 2px dashed {COLORS['border']};
            border-radius: 10px;
            padding: 0.5rem;
            background: {COLORS['background']};
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: {COLORS['primary']};
        }}
        
        /* ==================== CUSTOM HEADER - COMPACT ==================== */
        .iiith-header-compact {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 0;
            margin-bottom: 0.5rem;
        }}
        
        .iiith-logo-small {{
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_light']} 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.25rem;
            font-weight: 700;
        }}
        
        .iiith-title-compact {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {COLORS['primary']};
            margin: 0;
        }}
        
        .iiith-subtitle-compact {{
            font-size: 0.75rem;
            color: {COLORS['text_muted']};
            margin: 0;
        }}
        
        /* ==================== IMAGE PREVIEW CARD ==================== */
        .image-preview-card {{
            background: white;
            border: 2px solid {COLORS['border']};
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        
        .image-preview-card.ready {{
            border-color: {COLORS['secondary']};
            background: linear-gradient(135deg, rgba(56, 161, 105, 0.05) 0%, rgba(56, 161, 105, 0.1) 100%);
        }}
        
        /* ==================== LOADER STYLES ==================== */
        .loader-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}
        
        .loader {{
            width: 50px;
            height: 50px;
            border: 4px solid {COLORS['border']};
            border-top-color: {COLORS['primary']};
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        .loader-text {{
            margin-top: 1rem;
            color: {COLORS['text_muted']};
            font-size: 0.9rem;
        }}
        
        /* ==================== PROCESSING OVERLAY ==================== */
        .processing-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255,255,255,0.9);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_header_compact(title="Retail Assistant", subtitle=None):
    """Render a compact branded header"""
    subtitle_html = f'<p class="iiith-subtitle-compact">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="iiith-header-compact">
        <div class="iiith-logo-small">🛒</div>
        <div>
            <h1 class="iiith-title-compact">{title}</h1>
            {subtitle_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_logo():
    """Render sidebar logo at top"""
    st.markdown("""
    <div style="
        text-align: center;
        padding: 1.5rem 1rem 1rem;
        margin: -1rem -1rem 0 -1rem;
        background: rgba(0,0,0,0.1);
    ">
        <div style="font-size: 2.5rem; margin-bottom: 0.25rem;">🛒</div>
        <div style="font-size: 1rem; font-weight: 600;">Retail Assistant</div>
    </div>
    """, unsafe_allow_html=True)


def render_user_profile_bottom(email):
    """Render user profile section at bottom of sidebar"""
    initials = email[0].upper() if email else "U"
    st.markdown(f"""
    <div style="
        background: rgba(0,0,0,0.15);
        border-radius: 10px;
        padding: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-top: auto;
    ">
        <div style="
            width: 36px;
            height: 36px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.9rem;
        ">{initials}</div>
        <div style="overflow: hidden; flex: 1;">
            <div style="font-size: 0.7rem; opacity: 0.7;">Signed in as</div>
            <div style="font-size: 0.8rem; font-weight: 500; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{email}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_nav_button(icon, label, key, active=False):
    """Render a navigation button with icon"""
    bg = "rgba(255,255,255,0.25)" if active else "rgba(255,255,255,0.1)"
    st.markdown(f"""
    <style>
        div[data-testid="stButton"] button[key="{key}"] {{
            background: {bg};
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.95rem;
            padding: 0.75rem 1rem;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_stat_card(title, value, icon="📊", delta=None):
    """Render a styled stat card using Streamlit native components"""
    with st.container():
        st.metric(
            label=f"{icon} {title}",
            value=value,
            delta=f"{delta}%" if delta else None,
            delta_color="normal" if delta and delta > 0 else "inverse"
        )


def render_empty_state(message, icon="📭"):
    """Render an empty state message"""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        background: {COLORS['background']};
        border-radius: 12px;
        border: 2px dashed {COLORS['border']};
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <div style="color: {COLORS['text_muted']}; font-size: 1.1rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_image_preview_card(image_count):
    """Render image preview card with count"""
    return f"""
    <div class="image-preview-card ready">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="font-size: 1.5rem;">📷</div>
            <div style="flex: 1;">
                <div style="font-weight: 600; color: {COLORS['primary']};">{image_count} image(s) attached</div>
                <div style="font-size: 0.8rem; color: {COLORS['text_muted']};">Ready to analyze</div>
            </div>
            <div style="
                background: {COLORS['secondary']};
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 500;
            ">Ready</div>
        </div>
    </div>
    """


def render_processing_status(message="Processing..."):
    """Render processing status indicator"""
    return st.status(message, expanded=True)
