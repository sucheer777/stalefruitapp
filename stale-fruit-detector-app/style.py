# import streamlit as st
# import base64
# from pathlib import Path

# def get_base64_encoded_image(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded = base64.b64encode(image_file.read()).decode()
#     return f"data:image/png;base64,{encoded}"

# def load_css():
#     return '''
#         <style>
#             .stApp {
#     try:
#         image_path = Path(__file__).parent / "static" / "fruit-pattern-bg.png"
#         background_image = get_base64_encoded_image(str(image_path))
#     except:
#         # Fallback pattern if image loading fails
#         background_image = "none"
    
#     return f"""
#         <style>
#             /* Global Styles */
#             .stApp {{
#                 background-image: url('{background_image}');
#                 background-repeat: repeat;
#                 background-size: 300px;
#                 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#                 color: #1a1a1a;
#             }}

#             /* Add a light overlay to improve readability */
#             .stApp::before {{
#                 content: '';
#                 position: fixed;
#                 top: 0;
#                 left: 0;
#                 width: 100%;
#                 height: 100%;
#                 background: rgba(255, 255, 255, 0.9);
#                 z-index: -1;
#             }}

#             /* Modern Card Styles */
#             .glass-card {{
#                 background: rgba(255, 255, 255, 0.1);
#                 backdrop-filter: blur(12px);
#                 -webkit-backdrop-filter: blur(12px);
#                 border-radius: 20px;
#                 border: 1px solid rgba(255, 255, 255, 0.2);
#                 padding: 2rem;
#                 margin: 1.5rem 0;
#                 box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
#                 color: #1a1a1a;
#                 transition: all 0.3s ease;
#             }}

#             .glass-card:hover {{
#                 transform: translateY(-5px);
#                 box-shadow: 0 12px 48px 0 rgba(31, 38, 135, 0.5);
#                 border: 1px solid rgba(255, 255, 255, 0.3);
#             }}

#             /* Header Styles */
#             .page-header {{
#                 background: rgba(255, 255, 255, 0.1);
#                 padding: 2rem;
#                 border-radius: 20px;
#                 margin-bottom: 2rem;
#                 text-align: center;
#                 border: 1px solid rgba(255, 255, 255, 0.1);
#             }}

#             .page-header h1 {{
#                 color: #1a1a1a;
#                 font-size: 2.5rem;
#                 margin: 0;
#                 font-weight: 600;
#                 text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
#             }}

#             .page-header p {{
#                 color: #333333;
#                 margin-top: 1rem;
#                 font-size: 1.1rem;
#             }}

#             /* Form Styles */
#             .form-container {{
#                 background: rgba(255, 255, 255, 0.1);
#                 padding: 2rem;
#                 border-radius: 20px;
#                 margin: 1rem auto;
#                 max-width: 300px;
#                 aspect-ratio: 1;
#                 display: flex;
#                 flex-direction: column;
#                 justify-content: center;
#                 align-items: center;
#                 backdrop-filter: blur(12px);
#                 -webkit-backdrop-filter: blur(12px);
#                 border: 1px solid rgba(255, 255, 255, 0.2);
#                 box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
#             }}

#             .inner-glass-container {{
#                 background: rgba(255, 255, 255, 0.15);
#                 padding: 1.5rem;
#                 border-radius: 15px;
#                 width: 90%;
#                 backdrop-filter: blur(8px);
#                 -webkit-backdrop-filter: blur(8px);
#                 border: 1px solid rgba(255, 255, 255, 0.3);
#                 box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.25);
#             }}

#             .form-field {{
#                 margin: 0.8rem 0;
#                 max-width: 100%;
#             }}

#             .form-field label {{
#                 color: #1a1a1a;
#                 font-size: 1rem;
#                 margin-bottom: 0.5rem;
#                 display: block;
#                 text-align: center;
#             }}

#             .form-field input {{
#                 width: 100%;
#                 max-width: 200px;
#                 padding: 0.75rem;
#                 border-radius: 10px;
#                 border: 1px solid rgba(0, 0, 0, 0.2);
#                 background: rgba(255, 255, 255, 0.8);
#                 color: #1a1a1a;
#                 font-size: 1rem;
#                 display: block;
#                 margin: 0 auto;
#             }}

#             /* Button container for form */
#             .form-button-container {{
#                 margin-top: 1rem;
#                 text-align: center;
#                 width: 100%;
#             }}

#             /* Adjust button width in forms */
#             .form-container .custom-button {{
#                 width: 80%;
#                 max-width: 180px;
#             }}

#             /* Button Styles */
#             .custom-button {{
#                 background: linear-gradient(135deg, #6366f1, #4f46e5);
#                 color: #ffffff;
#                 padding: 0.75rem 1.5rem;
#                 border-radius: 10px;
#                 border: none;
#                 font-weight: 500;
#                 cursor: pointer;
#                 transition: all 0.3s ease;
#                 text-align: center;
#                 display: inline-block;
#                 margin: 0.5rem 0;
#                 text-decoration: none;
#             }}

#             .custom-button:hover {{
#                 transform: translateY(-2px);
#                 box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
#                 background: linear-gradient(135deg, #4f46e5, #4338ca);
#             }}

#             /* Alert and Message Styles */
#             .success-message {{
#                 background: rgba(34, 197, 94, 0.2);
#                 border: 1px solid rgba(34, 197, 94, 0.3);
#                 color: #166534;
#                 padding: 1rem;
#                 border-radius: 10px;
#                 margin: 1rem 0;
#             }}

#             .error-message {{
#                 background: rgba(239, 68, 68, 0.2);
#                 border: 1px solid rgba(239, 68, 68, 0.3);
#                 color: #991b1b;
#                 padding: 1rem;
#                 border-radius: 10px;
#                 margin: 1rem 0;
#             }}

#             /* Navigation Styles */
#             .nav-link {{
#                 color: #1a1a1a;
#                 text-decoration: none;
#                 padding: 0.5rem 1rem;
#                 border-radius: 8px;
#                 transition: all 0.3s ease;
#             }}

#             .nav-link:hover {{
#                 background: rgba(255, 255, 255, 0.2);
#             }}

#             /* Card Grid */
#             .card-grid {{
#                 display: grid;
#                 grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
#                 gap: 1.5rem;
#                 margin: 1.5rem 0;
#             }}

#             /* Responsive Design */
#             @media (max-width: 768px) {{
#                 .page-header h1 {{
#                     font-size: 2rem;
#                 }}

#                 .glass-card {{
#                     padding: 1.5rem;
#                 }}

#                 .card-grid {{
#                     grid-template-columns: 1fr;
#                 }}
#             }}

#             /* Animation Classes */
#             .fade-in {{
#                 animation: fadeIn 0.5s ease-in;
#             }}

#             .slide-up {{
#                 animation: slideUp 0.5s ease-out;
#             }}

#             @keyframes fadeIn {{
#                 from {{ opacity: 0; }}
#                 to {{ opacity: 1; }}
#             }}

#             @keyframes slideUp {{
#                 from {{
#                     opacity: 0;
#                     transform: translateY(20px);
#                 }}
#                 to {{
#                     opacity: 1;
#                     transform: translateY(0);
#                 }}
#             }}

#             /* Custom Scrollbar */
#             ::-webkit-scrollbar {{
#                 width: 8px;
#             }}

#             ::-webkit-scrollbar-track {{
#                 background: rgba(255, 255, 255, 0.1);
#             }}

#             ::-webkit-scrollbar-thumb {{
#                 background: rgba(0, 0, 0, 0.3);
#                 border-radius: 4px;
#             }}

#             ::-webkit-scrollbar-thumb:hover {{
#                 background: rgba(0, 0, 0, 0.4);
#             }}

#             /* Loading Spinner */
#             .loading-spinner {{
#                 border: 4px solid rgba(0, 0, 0, 0.1);
#                 border-left: 4px solid #1a1a1a;
#                 border-radius: 50%;
#                 width: 40px;
#                 height: 40px;
#                 animation: spin 1s linear infinite;
#             }}

#             @keyframes spin {{
#                 0% {{ transform: rotate(0deg); }}
#                 100% {{ transform: rotate(360deg); }}
#             }}
#         </style>
#     """

# def apply_style():
#     st.markdown(load_css(), unsafe_allow_html=True) 


import streamlit as st
import base64
from pathlib import Path

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

def load_css():
    try:
        image_path = Path("static/fruit-pattern-bg.png").resolve()
        background_image = get_base64_encoded_image(str(image_path))
    except:
        background_image = "none"

    return f"""
        <style>
            /* Global App Styling */
            .stApp {{
                background-image: url('{background_image}');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #1a1a1a;
            }}

            .stApp::before {{
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(255, 255, 255, 0.85);
                z-index: -1;
            }}

            /* Card */
            .glass-card {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(12px);
                border-radius: 20px;
                padding: 2rem;
                margin: 1rem 0;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                transition: 0.3s ease;
            }}

            .glass-card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 12px 48px rgba(31, 38, 135, 0.4);
            }}

            /* Button */
            .custom-button {{
                background: linear-gradient(135deg, #6366f1, #4f46e5);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 10px;
                border: none;
                cursor: pointer;
                font-weight: 500;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                transition: 0.3s ease;
            }}

            .custom-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
                background: linear-gradient(135deg, #4f46e5, #4338ca);
            }}

            /* Header */
            .page-header {{
                text-align: center;
                padding: 2rem;
                margin-bottom: 2rem;
            }}

            .page-header h1 {{
                font-size: 2.5rem;
                font-weight: 700;
                color: #1a1a1a;
            }}

            .page-header p {{
                font-size: 1.1rem;
                color: #333;
            }}
        </style>
    """

def apply_style():
    st.markdown(load_css(), unsafe_allow_html=True)
