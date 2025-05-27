import streamlit as st
import hmac
import hashlib
import time
import json
from typing import Optional

# Secret key for token signing (in a real app, this should be in a secure config)
SECRET_KEY = "your-secret-key-12345"

def init_auth_state():
    """Initialize authentication state in Streamlit session"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'email' not in st.session_state:
        st.session_state.email = None
    if 'auth_token' not in st.session_state:
        st.session_state.auth_token = None

def create_login_token(email: str) -> str:
    """Create a secure token for persistent login"""
    timestamp = str(int(time.time()))
    message = f"{email}:{timestamp}"
    signature = hmac.new(
        SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{message}:{signature}"

def verify_login_token(token: str) -> Optional[str]:
    """Verify the login token and return the email if valid"""
    try:
        if not token:
            return None
        
        email, timestamp, signature = token.split(':')
        message = f"{email}:{timestamp}"
        expected_signature = hmac.new(
            SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if hmac.compare_digest(signature, expected_signature):
            # Check if token is not too old (e.g., 30 days)
            if int(time.time()) - int(timestamp) < 30 * 24 * 60 * 60:
                return email
        return None
    except Exception:
        return None

def get_auth_js():
    """Return JavaScript code for authentication"""
    return """
    <script>
    function getLoginToken() {
        return localStorage.getItem('login_token');
    }

    function setLoginToken(token) {
        localStorage.setItem('login_token', token);
    }

    function clearLoginToken() {
        localStorage.removeItem('login_token');
    }

    // Check for stored token and add to URL if found
    window.addEventListener('load', function() {
        const token = getLoginToken();
        if (token && !window.location.search.includes('token=')) {
            const separator = window.location.search ? '&' : '?';
            window.location.search += separator + 'token=' + encodeURIComponent(token);
        }
    });
    </script>
    """

def check_auth():
    """Check authentication state and handle token verification"""
    init_auth_state()
    
    # If already logged in, no need to check token
    if st.session_state.logged_in:
        return
    
    # Check for stored token in session state
    if st.session_state.auth_token:
        email = verify_login_token(st.session_state.auth_token)
        if email:
            st.session_state.logged_in = True
            st.session_state.email = email
            return

def login_user(email: str, remember_me: bool = False):
    """Handle user login"""
    st.session_state.logged_in = True
    st.session_state.email = email
    
    if remember_me:
        token = create_login_token(email)
        st.session_state.auth_token = token

def logout():
    """Handle user logout"""
    st.session_state.logged_in = False
    st.session_state.email = None
    st.session_state.auth_token = None
    st.rerun() 