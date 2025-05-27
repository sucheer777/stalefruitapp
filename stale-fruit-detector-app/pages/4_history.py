import streamlit as st
import pandas as pd
from datetime import datetime
import base64
from translation import TRANSLATIONS
from db_utils import get_predictions_by_user
from utils.style import apply_style
from utils.auth_utils import check_auth, logout
import sqlite3
from PIL import Image
import os
from utils.db_utils import db
import time

# Set page config must be the first Streamlit command
st.set_page_config(page_title="History - Stale Fruit Detector", layout="wide")

# Apply shared styles first
apply_style()

# Additional history-specific styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Inter:wght@400;500&display=swap');
    
    /* History Container Styling */
    .history-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        padding: 1.75rem;
        margin: 1rem auto;
        max-width: 1000px;
    }

    /* Image Preview Styling */
    .image-hover-text {
        position: relative;
        cursor: pointer;
        color: #4A5568;
    }

    .image-preview {
        display: none;
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: white;
        padding: 5px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        margin-top: 10px;
    }

    .image-preview img {
        max-width: 300px;
        max-height: 300px;
        border-radius: 4px;
    }

    .image-hover-text:hover .image-preview {
        display: block;
    }

    /* History Item Styling */
    .history-item {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }

    .history-item:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }

    /* Table Styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    .stDataFrame table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
        width: 100% !important;
    }

    .stDataFrame th {
        background: rgba(108, 99, 255, 0.1) !important;
        color: #2C3E50 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        text-align: left !important;
    }

    .stDataFrame td {
        font-family: 'Inter', sans-serif !important;
        padding: 0.75rem 1rem !important;
        border-top: 1px solid rgba(0, 0, 0, 0.05) !important;
        position: relative !important;
    }

    /* Fresh/Stale Row Styling */
    .fresh-row {
        background-color: rgba(72, 187, 120, 0.1) !important;
    }

    .stale-row {
        background-color: rgba(245, 101, 101, 0.1) !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @media (max-width: 768px) {
        .history-container {
            margin: 1rem;
            padding: 1rem;
        }
        
        .image-preview img {
            max-width: 200px;
            max-height: 200px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Check authentication state
check_auth()

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'email' not in st.session_state:
    st.session_state.email = None

# Apply shared styles
apply_style()

# --- Language Selection ---
lang = st.sidebar.selectbox("🌐 Select Language", ["English", "Telugu", "Hindi"])
tr = TRANSLATIONS["history"][lang]

# Add logout option in sidebar if logged in
if st.session_state.logged_in:
    with st.sidebar:
        if st.button("Logout"):
            logout()

# Check if user is logged in
if not st.session_state.logged_in:
    st.error(tr["login_required"])
    if st.button(tr["go_to_login"]):
        st.switch_page("pages/2_login.py")
    st.stop()

# Main Content
st.markdown(f"""
    <div class="header-container">
        <div style="flex: 1;"></div>
        <h1 class="page-title">{tr['title']}</h1>
        <div style="flex: 1;"></div>
    </div>
""", unsafe_allow_html=True)

def format_timestamp(timestamp):
    """Format timestamp in a consistent way"""
    try:
        if isinstance(timestamp, str):
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        else:
            dt = timestamp
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except Exception as e:
        return str(timestamp)

def format_image_cell(image_path):
    if image_path and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                # Create a smaller thumbnail
                img.thumbnail((300, 300))
                # Convert image to base64
                import io
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                img_byte_arr = img_byte_arr.getvalue()
                encoded = base64.b64encode(img_byte_arr).decode()
                
                return f"""
                    <span class="image-hover-text">
                        View Image
                        <div class="image-preview">
                            <img src="data:image/{img.format.lower()};base64,{encoded}" alt="Preview">
                        </div>
                    </span>
                """
        except Exception as e:
            return "Error loading image"
    return "No image available"

def format_model_type(model_type):
    """Format model type with consistent styling"""
    return f'<span class="model-type">{model_type}</span>'

def main():
    # Apply shared styles
    apply_style()

    try:
        # Get predictions
        predictions = get_predictions_by_user(st.session_state["email"])
        
        if not predictions:
            st.info(tr["no_predictions"])
            st.stop()
        
        # Convert predictions to DataFrame
        df = pd.DataFrame(predictions, columns=[
            tr["result"],
            tr["timestamp"],
            "Model",
            "Condition",
            "Storage",
            "Image"
        ])
        
        # Format timestamp
        df[tr["timestamp"]] = df[tr["timestamp"]].apply(format_timestamp)
        
        # Format model type
        df["Model"] = df["Model"].apply(format_model_type)
        
        # Format image column
        df["Image"] = df["Image"].apply(format_image_cell)
        
        # Replace NaN values
        df = df.fillna("Not Available")
        
        # Create download version (clean version without HTML tags)
        df_download = df.copy()
        df_download["Model"] = df_download["Model"].apply(lambda x: x.replace('<span class="model-type">', '').replace('</span>', ''))
        df_download["Image"] = df_download["Image"].apply(lambda x: "Image Available" if "View Image" in x else "No Image")
        
        # Add download button
        with st.container():
            st.download_button(
                label="📥 Download History",
                data=df_download.to_csv(index=False).encode('utf-8'),
                file_name=f"fruit_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        
        # Add some space after the download button
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Convert DataFrame to HTML and apply row colors
        html = df.to_html(escape=False, index=False)
        
        # Apply row colors based on result
        for i, row in df.iterrows():
            result = row[tr["result"]]
            if "FRESH" in result.upper():
                html = html.replace(f'<tr><td>{result}', f'<tr class="fresh-row"><td>{result}')
            elif "STALE" in result.upper():
                html = html.replace(f'<tr><td>{result}', f'<tr class="stale-row"><td>{result}')
        
        st.write(html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading prediction history: {str(e)}")
        st.error("Please try refreshing the page. If the problem persists, contact support.")

if __name__ == "__main__":
    main() 