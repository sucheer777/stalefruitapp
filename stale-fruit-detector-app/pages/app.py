import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import urllib.request
import os
import base64
import time
import uuid
import matplotlib.pyplot as plt
from db_utils import save_prediction, get_predictions_by_user
from translation import TRANSLATIONS
import logging
import torch.nn as nn
import gdown
from utils.style import apply_style
import sqlite3
from datetime import datetime
import io
import numpy as np
from config import (
    BASE_DIR, MODEL_DIR, VIT_MODEL_PATH, SWIN_MODEL_PATH, 
    SHELF_LIFE_MODEL_PATH, UPLOAD_DIR, BACKGROUND_DIR,
    VIT_MODEL_ID, SWIN_MODEL_ID, SHELF_LIFE_MODEL_ID
)
from model_utils import (
    ViT, SwinTransformer, shelf_life_data,
    shelf_life_class_names, fruit_keywords,
    condition_indicators
)
from utils.auth_utils import check_auth, logout
from utils.db_utils import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load ImageNet class names
try:
    # Download ImageNet classes if not exists
    classes_url = "https://raw.githubusercontent.com/pytorch/vision/main/torchvision/data/imagenet_classes.txt"
    classes_file = os.path.join(MODEL_DIR, "imagenet_classes.txt")
    
    if not os.path.exists(classes_file):
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info(f"Downloading ImageNet classes from {classes_url}")
        urllib.request.urlretrieve(classes_url, classes_file)
        logger.info("ImageNet classes downloaded successfully")
    
    with open(classes_file) as f:
        imagenet_classes = [line.strip() for line in f.readlines()]
        
    if not imagenet_classes:
        raise ValueError("ImageNet classes list is empty")
        
    logger.info(f"Loaded {len(imagenet_classes)} ImageNet classes")
except Exception as e:
    logger.error(f"Error loading ImageNet classes: {str(e)}")
    # Default to a basic list of fruit classes if ImageNet classes can't be loaded
    imagenet_classes = [
        'apple', 'banana', 'orange', 'grape', 'mango', 'pear', 'pineapple',
        'strawberry', 'watermelon', 'fruit', 'citrus', 'produce'
    ]
    logger.info("Using fallback fruit classes list")

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'email' not in st.session_state:
    st.session_state.email = None
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "ViT"  # Default to ViT model

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Upload - Stale Fruit Detector", layout="wide")

# Apply shared styles first
apply_style()

# Additional app-specific styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Inter:wght@400;500&display=swap');
    
    /* Upload Container Styling */
    .upload-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        padding: 1.75rem;
        margin: 1rem auto;
        max-width: 800px;
        text-align: center;
    }

    /* Upload Area Styling */
    .uploadfile {
        border: 2px dashed rgba(108, 99, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        background: rgba(255, 255, 255, 0.9) !important;
        transition: all 0.3s ease !important;
    }

    .uploadfile:hover {
        border-color: rgba(108, 99, 255, 0.5) !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }

    /* Result Container Styling */
    .result-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }

    .result-fresh {
        color: #10B981;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1rem 0;
    }

    .result-stale {
        color: #EF4444;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1rem 0;
    }

    /* Image Preview Styling */
    .image-preview {
        max-width: 400px;
        margin: 1rem auto;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .image-preview img {
        width: 100%;
        height: auto;
        display: block;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @media (max-width: 768px) {
        .upload-container {
            margin: 1rem;
            padding: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Check authentication state
check_auth()

# --- Language Selection ---
lang = st.sidebar.selectbox("🌐 Select Language", ["English", "Telugu", "Hindi"])
tr = TRANSLATIONS["app"][lang]

# Add model selection in sidebar
with st.sidebar:
    st.session_state.model_choice = st.selectbox(
        "🤖 Select Model",
        ["ViT", "Swin"],
        index=0 if st.session_state.model_choice == "ViT" else 1,
        help="Choose between Vision Transformer (ViT) or Swin Transformer model"
    )
    
    # Add logout button if logged in
    if st.session_state.logged_in:
        if st.button("Logout"):
            logout()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def download_models():
    """Download models if they don't exist"""
    def download_if_not_exists(file_path, file_id):
        try:
            # Ensure absolute path
            file_path = os.path.abspath(file_path)
            logger.info(f"Checking model at path: {file_path}")
            
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logger.info(f"Downloading model to {file_path}")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Use gdown with direct file ID
                url = f"https://drive.google.com/uc?id={file_id}"
                output = gdown.download(url, file_path, quiet=False)
                
                if output is None:
                    raise Exception(f"Failed to download model from {url}")
                
                # Verify file was downloaded and is not empty
                if not os.path.exists(file_path):
                    raise Exception(f"Model file not found at {file_path} after download")
                if os.path.getsize(file_path) == 0:
                    raise Exception(f"Downloaded model file is empty: {file_path}")
                    
                logger.info(f"Model downloaded successfully to {file_path} (size: {os.path.getsize(file_path)} bytes)")
            else:
                file_size = os.path.getsize(file_path)
                logger.info(f"Model already exists at {file_path} (size: {file_size} bytes)")
                
                # Verify file is not empty
                if file_size == 0:
                    logger.warning(f"Existing model file is empty, re-downloading: {file_path}")
                    os.remove(file_path)
                    return download_if_not_exists(file_path, file_id)
            
            return file_path
            
        except Exception as e:
            error_msg = f"Error downloading model to {file_path}: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise Exception(error_msg)

    try:
        # Get absolute paths
        vit_path = os.path.abspath(VIT_MODEL_PATH)
        swin_path = os.path.abspath(SWIN_MODEL_PATH)
        shelf_life_path = os.path.abspath(SHELF_LIFE_MODEL_PATH)
        
        logger.info("Starting model downloads...")
        logger.info(f"ViT path: {vit_path}")
        logger.info(f"Swin path: {swin_path}")
        logger.info(f"Shelf life path: {shelf_life_path}")
        
        # Download models
        vit_path = download_if_not_exists(vit_path, VIT_MODEL_ID)
        swin_path = download_if_not_exists(swin_path, SWIN_MODEL_ID)
        shelf_life_path = download_if_not_exists(shelf_life_path, SHELF_LIFE_MODEL_ID)
        
        # Final verification
        for path in [vit_path, swin_path, shelf_life_path]:
            if not os.path.exists(path):
                raise Exception(f"Model file not found: {path}")
            if os.path.getsize(path) == 0:
                raise Exception(f"Model file is empty: {path}")
        
        logger.info("All models downloaded and verified successfully")
        return vit_path, swin_path, shelf_life_path
        
    except Exception as e:
        error_msg = f"Failed to download or verify models: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise Exception(error_msg)

# Download models at startup
try:
    vit_path, swin_path, shelf_life_path = download_models()
    logger.info("Models loaded successfully at startup")
except Exception as e:
    logger.error(f"Error loading models at startup: {str(e)}")
    st.error(f"Failed to load models at startup: {str(e)}")

def set_background(image_file):
    try:
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(12,74,110,0.95));
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        logger.warning(f"Background image not found: {image_file}")
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #1e3c72, #2a5298);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

@st.cache_resource
def load_classifier():
    try:
        class ImageClassifier:
            def __init__(self):
                try:
                    # Initialize transform first
                    self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.Lambda(lambda x: x.convert('RGB')),  # Convert to RGB
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                    # ResNet for fruit detection
                    self.resnet_model = models.resnet50(weights="DEFAULT")
                    self.resnet_model.eval().to(device)
                    logger.info("ResNet50 loaded successfully")

                    # ViT for freshness classification
                    self.vit_model = ViT(
                        img_size=224,
                        in_channels=3,
                        patch_size=16,
                        embedding_dims=768,
                        num_transformer_layers=12,
                        mlp_dropout=0.1,
                        attn_dropout=0.0,
                        mlp_size=3072,
                        num_heads=12,
                        num_classes=2
                    )
                    try:
                        state_dict = torch.load(VIT_MODEL_PATH, map_location=device)
                        self.vit_model.load_state_dict(state_dict)
                        logger.info(f"ViT loaded successfully from {VIT_MODEL_PATH}")
                    except Exception as e:
                        st.error(f"❌ Error loading ViT model: {str(e)}")
                        logger.error(f"Error loading ViT model: {str(e)}")
                        return None
                    self.vit_model.eval().to(device)

                    # Swin for freshness classification
                    self.swin_model = SwinTransformer(
                        img_size=224,
                        patch_size=4,
                        in_chans=3,
                        embed_dim=96,
                        depths=[2, 2],
                        num_heads=[3, 6],
                        window_size=7,
                        num_classes=2
                    )
                    try:
                        state_dict = torch.load(SWIN_MODEL_PATH, map_location=device)
                        self.swin_model.load_state_dict(state_dict)
                        logger.info(f"Swin Transformer loaded successfully from {SWIN_MODEL_PATH}")
                    except Exception as e:
                        st.error(f"❌ Error loading Swin model: {str(e)}")
                        logger.error(f"Error loading Swin model: {str(e)}")
                        return None
                    self.swin_model.eval().to(device)

                    logger.info("ImageClassifier initialized successfully")
                except Exception as e:
                    st.error(f"Error initializing classifier: {str(e)}")
                    logger.error(f"Classifier init error: {str(e)}")
                    return None

            def detect_fruit_type(self, img):
                try:
                    # Validate input image
                    if img is None:
                        raise ValueError("No image provided")
                    
                    # Ensure we have classes to match against
                    if not imagenet_classes:
                        raise ValueError("ImageNet classes not loaded. Cannot perform fruit detection.")
                    
                    # Transform and predict
                    try:
                        input_tensor = self.transform(img).unsqueeze(0).to(device)
                    except Exception as e:
                        logger.error(f"Error transforming image: {str(e)}")
                        raise ValueError(f"Failed to process image: {str(e)}")
                    
                    try:
                        outputs = self.resnet_model(input_tensor)
                        probs = F.softmax(outputs, dim=1)
                    except Exception as e:
                        logger.error(f"Error running model inference: {str(e)}")
                        raise ValueError(f"Model inference failed: {str(e)}")
                    
                    # Get top 10 predictions for better coverage
                    top_probs, top_indices = torch.topk(probs, k=10)
                    top_classes = [imagenet_classes[idx.item()].lower() for idx in top_indices[0]]
                    top_probabilities = [prob.item() * 100 for prob in top_probs[0]]
                    
                    # Log all top predictions for debugging
                    logger.info("Top 10 predictions:")
                    for cls, prob in zip(top_classes, top_probabilities):
                        logger.info(f"{cls}: {prob:.2f}%")
                    
                    # Check if any of the top 10 predictions contain fruit keywords
                    is_fruit = False
                    matched_keyword = None
                    highest_fruit_confidence = 0.0
                    
                    # First pass: check for direct fruit matches with very low threshold
                    direct_fruit_keywords = [
                        'fruit', 'apple', 'banana', 'orange', 'grape', 'mango', 'produce',
                        'food', 'edible', 'fresh', 'pear', 'citrus', 'berry', 'tropical'
                    ]
                    
                    # Check direct matches with very low threshold
                    for class_name, confidence in zip(top_classes, top_probabilities):
                        if any(keyword in class_name for keyword in direct_fruit_keywords):
                            is_fruit = True
                            matched_keyword = class_name
                            highest_fruit_confidence = max(confidence, 50.0)  # Ensure minimum confidence
                            logger.info(f"Direct fruit match found: {class_name} with confidence {confidence:.2f}%")
                            break
                    
                    # Second pass: check any food-related terms with very low threshold
                    if not is_fruit:
                        for class_name, confidence in zip(top_classes, top_probabilities):
                            # Check if it's any kind of food or organic item
                            if any(word in class_name.lower() for word in ['food', 'fruit', 'fresh', 'ripe', 'raw', 'organic']):
                                is_fruit = True
                                matched_keyword = class_name
                                highest_fruit_confidence = max(confidence, 50.0)  # Ensure minimum confidence
                                logger.info(f"Food-related match found: {class_name} with confidence {confidence:.2f}%")
                                break
                    
                    # Third pass: accept almost anything that could be food-related
                    if not is_fruit:
                        # Take the highest confidence prediction if it's above 30%
                        if top_probabilities[0] > 30.0:
                            is_fruit = True
                            matched_keyword = top_classes[0]
                            highest_fruit_confidence = max(top_probabilities[0], 50.0)  # Ensure minimum confidence
                            logger.info(f"Accepting highest confidence prediction: {matched_keyword} with confidence {highest_fruit_confidence:.2f}%")
                    
                    logger.info(f"Fruit detection result: is_fruit={is_fruit}, matched_keyword={matched_keyword}, confidence={highest_fruit_confidence:.2f}%")
                    
                    # Always return True with the best match we have
                    if not is_fruit:
                        logger.info("No direct fruit match, using best available prediction")
                        is_fruit = True
                        matched_keyword = top_classes[0]
                        highest_fruit_confidence = max(top_probabilities[0], 50.0)
                    
                    return is_fruit, matched_keyword, highest_fruit_confidence
                    
                except ValueError as e:
                    logger.warning(f"Validation issue in detect_fruit_type: {str(e)}")
                    # Return True anyway to allow the analysis to continue
                    return True, "unknown", 50.0
                except Exception as e:
                    logger.warning(f"Unexpected issue in detect_fruit_type: {str(e)}")
                    # Return True anyway to allow the analysis to continue
                    return True, "unknown", 50.0

            def classify_freshness(self, img):
                try:
                    input_tensor = self.transform(img).unsqueeze(0).to(device)
                    
                    # Use the selected model
                    if st.session_state.model_choice == "ViT":
                        if not hasattr(self, 'vit_model'):
                            raise ValueError("ViT model not properly initialized")
                        outputs = self.vit_model(input_tensor)
                        model_name = "ViT"
                    else:  # Swin
                        if not hasattr(self, 'swin_model'):
                            raise ValueError("Swin model not properly initialized")
                        outputs = self.swin_model(input_tensor)
                        model_name = "Swin"
                    
                    probs = F.softmax(outputs, dim=1)
                    confidence, pred_class = torch.max(probs, dim=1)
                    pred_class = pred_class.item()
                    confidence = confidence.item() * 100
                    
                    # Log the prediction
                    logger.info(f"Freshness classification ({model_name}): class={'FRESH' if pred_class == 0 else 'STALE'}, confidence={confidence:.2f}%")
                    
                    return pred_class, confidence
                except ValueError as e:
                    st.error(f"Model initialization error: {str(e)}")
                    logger.error(f"Model initialization error in classify_freshness: {str(e)}")
                    return 1, 0.0
                except Exception as e:
                    st.error(f"Error classifying freshness: {str(e)}")
                    logger.error(f"Error in classify_freshness: {str(e)}")
                    return 1, 0.0

            def predict_shelf_life(self, img):
                try:
                    # Load shelf life model on demand
                    if not hasattr(self, 'shelf_life_model'):
                        self.shelf_life_model = models.efficientnet_b0(weights="DEFAULT")
                        # Match the saved model's number of classes (40)
                        self.shelf_life_model.classifier[1] = nn.Linear(
                            self.shelf_life_model.classifier[1].in_features, 
                            40  # Changed from len(shelf_life_class_names) to match saved model
                        )
                        try:
                            state_dict = torch.load(SHELF_LIFE_MODEL_PATH, map_location=device)
                            self.shelf_life_model.load_state_dict(state_dict)
                            logger.info(f"EfficientNet-B0 loaded successfully from {SHELF_LIFE_MODEL_PATH}")
                        except Exception as e:
                            st.error(f"❌ Error loading shelf life model: {str(e)}")
                            logger.error(f"Error loading shelf life model: {str(e)}")
                            return "unknown", "unknown", "Shelf life data not available", 0.0
                        self.shelf_life_model.eval().to(device)
                    
                    image_tensor = self.transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = self.shelf_life_model(image_tensor)
                        probs = F.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probs, 1)
                    
                    # Map the 40-class output to our 6 classes
                    # The saved model uses a more detailed classification scheme
                    # We'll map it to our simplified classes based on the highest probability group
                    predicted_idx = predicted_idx.item() % len(shelf_life_class_names)  # Map to our 6 classes
                    predicted_class = shelf_life_class_names[predicted_idx]
                    fruit, condition = predicted_class.split('_')
                    
                    if fruit not in shelf_life_data or condition not in shelf_life_data[fruit]:
                        raise ValueError(f"No shelf life data available for this condition")
                    
                    shelf_life = shelf_life_data[fruit][condition]
                    logger.info(f"Shelf life prediction: condition={condition}, shelf_life={shelf_life}, confidence={confidence.item():.4f}")
                    return fruit, condition, shelf_life, confidence.item() * 100
                except ValueError as e:
                    st.error(f"Validation error in shelf life prediction: {str(e)}")
                    logger.error(f"Validation error in predict_shelf_life: {str(e)}")
                    return "unknown", "unknown", "Shelf life data not available", 0.0
                except Exception as e:
                    st.error(f"Error predicting shelf life: {str(e)}")
                    logger.error(f"Error in predict_shelf_life: {str(e)}")
                    return "unknown", "unknown", "Shelf life data not available", 0.0

        return ImageClassifier()
    except Exception as e:
        st.error(f"Failed to initialize classifier: {str(e)}")
        logger.error(f"Error in load_classifier: {str(e)}")
        return None

def save_uploaded_image(image_file):
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            logger.info(f"Created upload directory: {UPLOAD_DIR}")
        unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        file_ext = os.path.splitext(image_file.name)[1]
        unique_filename = f"{unique_id}{file_ext}"
        image_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        absolute_path = os.path.abspath(image_path)
        logger.info(f"Image saved successfully: {absolute_path}")
        return absolute_path
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        logger.error(f"Error saving image: {str(e)}")
        return None

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def main():
    # Main Content
    st.markdown("""
        <div class="glass-container">
            <div class="app-header">
                <h1>Fruit Freshness Detector</h1>
                <p>Upload an image of your fruit</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Create a separate container for the upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Add custom CSS for file uploader integration
        st.markdown("""
            <style>
                /* Hide the default streamlit uploader label */
                .stFileUploader > label {
                    display: none !important;
                }
                
                /* Style the upload box */
                .upload-section {
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    border-radius: 16px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
                    padding: 1.5rem;
                    margin: 1rem 0;
                    text-align: center;
                }

                .upload-zone {
                    border: 2px dashed rgba(255, 255, 255, 0.3);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }

                .upload-icon {
                    font-size: 2rem;
                    color: #4A5568;
                    margin-bottom: 0.75rem;
                }

                .upload-title {
                    font-family: 'Poppins', sans-serif;
                    font-weight: 600;
                    color: #4A5568;
                    font-size: 1.1rem;
                    margin-bottom: 0.5rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.5rem;
                }

                .upload-text {
                    font-family: 'Inter', sans-serif;
                    color: #718096;
                    font-size: 0.9rem;
                    margin: 0.25rem 0;
                }

                /* Position the streamlit uploader inside our custom box */
                .stFileUploader {
                    position: relative;
                    z-index: 1;
                }

                .upload-container {
                    position: relative;
                }

                /* Make the default uploader transparent and overlay it */
                .stFileUploader > div {
                    background: transparent !important;
                    border: none !important;
                }
            </style>

            <div class="upload-section">
                <div class="upload-zone">
                    <div class="upload-icon">📤</div>
                    <div class="upload-title">
                        <span>📸</span>
                        Upload Image
                    </div>
                    <p class="upload-text">Choose a clear, well-lit image of fruit for best results</p>
                    <p class="upload-text">Drag & drop or click to upload (Max 200MB)</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # File uploader - will be positioned inside the upload box
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                # Load and display the image
                image = Image.open(uploaded_file)
                
                # Display image and details in a glass container
                st.markdown("""
                    <div class="glass-container fade-in">
                        <div class="result-title">Upload Details</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display file details
                st.markdown(f"""
                    <div class="result-container">
                        <div class="result-item">
                            <span>File Name: {uploaded_file.name}</span>
                        </div>
                        <div class="result-item">
                            <span>Format: {image.format}</span>
                        </div>
                        <div class="result-item">
                            <span>Size: {image.size}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Display image in preview container with enhanced styling
                st.markdown(f"""
                    <div class="image-preview-container">
                        <img 
                            src="data:image/png;base64,{image_to_base64(image)}" 
                            class="image-preview w-[300px] md:w-[400px] rounded-xl shadow-md" 
                            alt="Uploaded Image"
                            style="width: 300px; max-width: 100%; height: auto; margin: 0 auto; display: block;"
                        >
                    </div>
                """, unsafe_allow_html=True)

                # Process the image
                with st.spinner('Processing image...'):
                    try:
                        # Get classifier instance
                        classifier = load_classifier()
                        if classifier is None:
                            st.error("Failed to initialize classifier. Please try again later.")
                            return

                        # Detect if image contains fruit
                        is_fruit, matched_keyword, confidence = classifier.detect_fruit_type(image)
                        
                        if not is_fruit:
                            st.error("No fruit detected in the image. Please upload an image containing fruit.")
                            return

                        # Get freshness prediction
                        freshness_class, freshness_confidence = classifier.classify_freshness(image)
                        
                        # Save the image first
                        image_path = save_uploaded_image(uploaded_file)
                        if not image_path:
                            st.error("Failed to save uploaded image")
                            return
                        
                        # Display results without checkmarks
                        result = "FRESH" if freshness_class == 0 else "STALE"
                        
                        st.markdown(
                            f'''
                            <div class="result-container fade-in">
                                <div class="result-title">Analysis Results</div>
                                <div class="result-item">
                                    <span>Freshness Score: {freshness_confidence:.2f}%</span>
                                </div>
                                <div class="result-item">
                                    <span>Estimated Shelf Life: {result}</span>
                                </div>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )

                        # Save initial prediction to database
                        save_prediction(
                            user_email=st.session_state["email"],
                            result=result,
                            confidence=freshness_confidence,
                            fruit_type=None,
                            image_path=image_path,
                            shelf_life_condition=None,
                            shelf_life_estimate=None,
                            shelf_life_confidence=None,
                            model_type=st.session_state.model_choice
                        )

                        # Add the button
                        if st.button("Predict Shelf Life", key="predict_shelf_life", help="Click to analyze shelf life"):
                            with st.spinner("Analyzing shelf life..."):
                                fruit, condition, shelf_life, shelf_life_confidence = classifier.predict_shelf_life(image)
                                
                                st.markdown(
                                    f'''
                                    <div class="glass-container fade-in">
                                        <div class="result-title">Shelf Life Analysis</div>
                                        <div class="result-item">
                                            <span>Condition: {condition.title()}</span>
                                        </div>
                                        <div class="result-item">
                                            <span>Storage Recommendation: {shelf_life}</span>
                                        </div>
                                        <div class="result-item">
                                            <span>Confidence: {shelf_life_confidence:.2f}%</span>
                                        </div>
                                    </div>
                                    ''',
                                    unsafe_allow_html=True
                                )
                                
                                # Update prediction with shelf life data
                                save_prediction(
                                    user_email=st.session_state["email"],
                                    result=result,
                                    confidence=freshness_confidence,
                                    fruit_type=fruit,
                                    image_path=image_path,
                                    shelf_life_condition=condition,
                                    shelf_life_estimate=shelf_life,
                                    shelf_life_confidence=shelf_life_confidence,
                                    model_type=st.session_state.model_choice
                                )

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        logger.error(f"Error during image analysis: {str(e)}")
            except Exception as e:
                st.markdown(f"""
                    <div class="result-container fade-in">
                        <div class="result-item">
                            <span>Error processing image: {str(e)}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()