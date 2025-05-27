import streamlit as st

def load_css():
    return """
        <style>
            /* Global Styles */
            .stApp {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f0f8ff !important;
            }

            /* Main Container */
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

            /* Modern Card Styles */
            .glass-card {
                background: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                border: 1px solid rgba(224, 224, 224, 0.5);
                padding: 1.5rem;
                margin: 1rem auto;
                max-width: 1000px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                color: #333;
                backdrop-filter: blur(10px);
            }

            .glass-card:hover {
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            /* Header Styles */
            .page-header {
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                text-align: center;
            }

            .page-header h1 {
                color: #333;
                font-size: 2rem;
                margin: 0;
                font-weight: 600;
            }

            .page-header p {
                color: #666;
                margin-top: 0.5rem;
                font-size: 1.1rem;
            }

            /* Form Styles */
            .form-container {
                background: rgba(255, 255, 255, 0.9);
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem auto;
                max-width: 600px;
                border: 1px solid #e0e0e0;
                backdrop-filter: blur(10px);
            }

            .form-field {
                margin: 1rem 0;
            }

            .form-field label {
                color: #333;
                font-size: 1rem;
                margin-bottom: 0.5rem;
                display: block;
            }

            .form-field input {
                width: 100%;
                padding: 0.75rem;
                border-radius: 5px;
                border: 1px solid #e0e0e0;
                background: white;
                color: #333;
                font-size: 1rem;
            }

            /* Button Styles */
            .custom-button {
                background: #1a73e8;
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 5px;
                border: none;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.3s ease;
                text-align: center;
                display: inline-block;
                margin: 0.5rem 0;
                text-decoration: none;
            }

            .custom-button:hover {
                background: #1557b0;
            }

            /* Alert and Message Styles */
            .success-message {
                background: rgba(230, 244, 234, 0.9);
                border: 1px solid #34a853;
                color: #34a853;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
                backdrop-filter: blur(10px);
            }

            .error-message {
                background: rgba(252, 232, 230, 0.9);
                border: 1px solid #ea4335;
                color: #ea4335;
                padding: 1rem;
                border-radius: 5px;
                margin: 1rem 0;
                backdrop-filter: blur(10px);
            }

            /* Navigation Styles */
            .nav-link {
                color: #1a73e8;
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                transition: background 0.3s ease;
            }

            .nav-link:hover {
                background: rgba(241, 243, 244, 0.8);
            }

            /* Card Grid */
            .card-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin: 1.5rem 0;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .page-header h1 {
                    font-size: 1.75rem;
                }

                .glass-card {
                    padding: 1rem;
                }

                .card-grid {
                    grid-template-columns: 1fr;
                }

                .main-container {
                    padding: 1rem;
                }
            }

            /* Animation Classes */
            .fade-in {
                animation: fadeIn 0.3s ease-in;
            }

            .slide-up {
                animation: slideUp 0.3s ease-out;
            }

            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* Custom Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }

            ::-webkit-scrollbar-track {
                background: #f1f1f1;
            }

            ::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;
            }
        </style>
    """

def apply_style():
    st.markdown(load_css(), unsafe_allow_html=True) 