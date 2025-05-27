import streamlit as st
import base64
from translation import TRANSLATIONS
from utils.style import apply_style
from pathlib import Path

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Stale Fruit Detector", layout="wide")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background():
    bin_str = get_base64_of_bin_file(Path(__file__).parent / "static" / "fruit-pattern-bg.png")
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: 300px;
        background-repeat: repeat;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%%;
        height: 100%%;
        background: rgba(255, 255, 255, 0.9);
        z-index: -1;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Apply clean, consolidated background styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff !important;
    }
    
    /* Glass card effect */
    .glass-card {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin: 20px auto;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        max-width: 1000px;
    }

    /* Container styling */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        margin-bottom: 2rem;
        color: #1E1E1E;
        font-size: 2.5rem;
        font-weight: bold;
    }

    /* Content styling */
    .content-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #333;
    }

    /* Footer styling */
    .footer-text {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load and apply CSS
def load_css():
    css_file = Path(__file__).parent / "style.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply CSS at startup
load_css()

# Apply shared styles
apply_style()

# Set background
set_background()

# --- Language Selection ---
lang = st.sidebar.selectbox("🌐 Select Language", ["English", "Telugu", "Hindi"])
tr = TRANSLATIONS["main"][lang]

# Title
with st.container():
    st.markdown("""
        <div class="glass-card">
            <h1 class="main-title">Stale Fruit Detector</h1>
        </div>
    """, unsafe_allow_html=True)

# Create three columns for better layout
left_col, main_col, right_col = st.columns([1, 6, 1])

with main_col:
    # Welcome section
    st.markdown(f"""
        <div class="glass-card">
            <h2 class="section-title">Welcome</h2>
            <p class="content-text">{tr['desc1']}</p>
            <p class="content-text">{tr['desc2']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Features section
    st.markdown(f"""
        <div class="glass-card">
            <h2 class="section-title">Features</h2>
            <p class="content-text">{tr['desc3']}</p>
            <p class="content-text">{tr['desc4']}</p>
            <p class="content-text">{tr['desc5']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Call-to-action section
    st.markdown(f"""
        <div class="glass-card">
            <h2 class="section-title">Get Started</h2>
            <p class="cta-text">{tr['scan_line']}</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
    <div class="glass-card footer-card">
        <p class="footer-text">{tr['footer']}</p>
    </div>
""", unsafe_allow_html=True)

# Update the CSS styles
st.markdown("""
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem auto;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        max-width: 800px;
    }

    .main-title {
        color: #2C5282;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }

    .section-title {
        color: #2C5282;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-align: center;
    }

    .content-text {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #2D3748;
        margin: 1rem 0;
    }

    .cta-text {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2C5282;
        text-align: center;
        margin: 1rem 0;
    }

    .footer-card {
        background: rgba(255, 255, 255, 0.7);
        text-align: center;
    }

    .footer-text {
        color: #4A5568;
        font-size: 0.9rem;
        margin: 0;
    }

    @media (max-width: 768px) {
        .glass-card {
            margin: 1rem;
            padding: 1.5rem;
        }

        .main-title {
            font-size: 2.5rem;
        }

        .section-title {
            font-size: 1.5rem;
        }

        .content-text {
            font-size: 1rem;
        }

        .cta-text {
            font-size: 1.2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)