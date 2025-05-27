import streamlit as st
from utils.db_utils import db
from utils.auth_utils import check_auth, create_login_token, logout
from translation import TRANSLATIONS
from utils.style import apply_style
import time

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Login - Stale Fruit Detector", layout="wide")

# Apply shared styles first
apply_style()

# Additional login-specific styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Inter:wght@400;500&display=swap');
    
    /* Modern Container Styling */
    .login-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        padding: 1.75rem;
        margin: 1rem auto;
        max-width: 380px;
    }

    /* Heading Styles */
    .login-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .login-header h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #333;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }

    .login-header p {
        font-family: 'Inter', sans-serif;
        color: #666;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Form Field Styling */
    .stTextInput > label {
        font-family: 'Inter', sans-serif !important;
        color: #2C3E50 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.25rem !important;
        display: block !important;
    }

    /* Input Styling */
    .stTextInput > div > div > input {
        font-family: 'Inter', sans-serif !important;
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 10px !important;
        padding: 0.8rem 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        color: #2C3E50 !important;
        font-weight: 500 !important;
        caret-color: #2C3E50 !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #F34949 !important;
        box-shadow: 0 0 15px rgba(243, 73, 73, 0.15) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #95A5A6 !important;
        opacity: 0.8 !important;
    }

    /* Hide Streamlit's default form messages */
    .stMarkdown div[data-testid="stMarkdownContainer"] > p,
    div[data-baseweb="base-input"] + div small {
        display: none !important;
    }

    /* Button Styling */
    .stButton > button {
        font-family: 'Poppins', sans-serif !important;
        background: #F34949 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        transform: scale(1) !important;
        width: 100% !important;
        margin-bottom: 0.5rem !important;
    }

    .stButton > button:hover {
        background: #E43D3D !important;
        transform: scale(1.02) !important;
        box-shadow: 0 5px 15px rgba(243, 73, 73, 0.2) !important;
    }

    /* Alternative Action Text */
    .alt-action {
        text-align: center;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
        color: #666;
        font-size: 0.9rem;
    }

    /* Sign Up Button */
    .signup-button > button {
        background: #6C63FF !important;
    }

    .signup-button > button:hover {
        background: #5B52E0 !important;
        box-shadow: 0 5px 15px rgba(108, 99, 255, 0.2) !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Form Container */
    div[data-testid="stForm"] {
        background: transparent !important;
        box-shadow: none !important;
    }

    @media (max-width: 768px) {
        .login-container {
            margin: 1rem;
            padding: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Check authentication state
check_auth()

# --- Language Selection ---
lang = st.sidebar.selectbox("🌐 Select Language", ["English", "Telugu", "Hindi"])
tr = TRANSLATIONS["login"][lang]

# Add logout option in sidebar if logged in
if st.session_state.logged_in:
    with st.sidebar:
        if st.button("Logout"):
            logout()

# Show current login status and redirect if already logged in
if st.session_state.logged_in:
    st.info(f"Currently logged in as: {st.session_state.email}")
    st.switch_page("pages/app.py")

# Center container
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Create the glass-style container
    st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <h1>Welcome Back</h1>
                <p>Continue detecting fruit freshness with us</p>
            </div>
    """, unsafe_allow_html=True)
    
    # Login Form
    with st.form("login_form", clear_on_submit=True):
        email = st.text_input(
            "Email:",
            placeholder="Enter your email",
            key="email_input"
        )
        password = st.text_input(
            "Password:",
            type="password",
            placeholder="Enter your password",
            key="password_input"
        )
        
        submit = st.form_submit_button("Login", use_container_width=True)
        
        st.markdown(
            '''
            <div class="alt-action">
                Don't have an account?
            </div>
            ''',
            unsafe_allow_html=True
        )
        
        signup = st.form_submit_button("Sign Up", use_container_width=True)

        if submit:
            if not email or not password:
                st.markdown(
                    """
                    <div class="error-box">
                        <ul>
                            <li>Please fill in all fields</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                try:
                    if db.verify_user(email, password):
                        st.markdown(
                            f"""
                            <div class="success-message">
                                Welcome back, {email}! Redirecting to app...
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.session_state.logged_in = True
                        st.session_state.email = email
                        time.sleep(1)  # Brief pause to show success message
                        st.switch_page("pages/app.py")  # Redirect to app
                    else:
                        st.markdown(
                            """
                            <div class="error-box">
                                <ul>
                                    <li>Incorrect email or password</li>
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.markdown(
                        """
                        <div class="error-box">
                            <ul>
                                <li>Login failed. Please try again.</li>
                                <li>If the problem persists, please contact support.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        if signup:
            st.switch_page("pages/3_signup.py")
    
    # Close the glass-style container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    pass




