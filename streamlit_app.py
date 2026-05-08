import os
import sys
import cv2
import av
import time
import torch
import streamlit as st
import numpy as np
from PIL import Image
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ASL Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS INJECTION
# ==========================================
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 95%;
    }
    
    h1 {
        font-weight: 300;
        letter-spacing: -1px;
        margin-bottom: 0rem;
    }
    
    .terminal-box {
        background-color: #0E1117;
        color: #00FF41;
        padding: 20px;
        border-radius: 8px;
        font-size: 28px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        border: 1px solid #2D3139;
        min-height: 90px;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# PROJECT ROOT & PATH SETUP
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.models.cnn_model import ASLCNN

# ==========================================
# DEVICE & CONFIG
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(PROJECT_ROOT, "pipeline", "saved_models", "robust_model.pth")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "asl_alphabet_train", "asl_alphabet_train")

if os.path.exists(TRAIN_DIR):
    CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
else:
    CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                   'del', 'nothing', 'space']

# ==========================================
# MODEL & TRANSFORM LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_resources():
    model = ASLCNN(num_classes=29)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return model, preprocess

cnn_model, transform = load_resources()

# ==========================================
# VIDEO PROCESSOR ENGINE (HUD OVERLAY)
# ==========================================
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.previous_prediction = ""
        self.stable_count = 0
        self.last_added_time = time.time()
        self.local_sentence = "" 
        self.current_pred = "Waiting"
        self.current_conf = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        x1, y1, x2, y2 = 300, 150, 600, 450
        
        roi = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 10
        )
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        final_roi = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(final_roi)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = cnn_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        self.current_pred = CLASS_NAMES[pred.item()]
        self.current_conf = conf.item()

        curr_time = time.time()
        if self.current_conf > 0.92:
            if self.current_pred == self.previous_prediction:
                self.stable_count += 1
            else:
                self.stable_count = 0
                self.previous_prediction = self.current_pred

            if self.stable_count >= 12 and (curr_time - self.last_added_time > 1.2):
                if self.current_pred == "space": 
                    self.local_sentence += " "
                elif self.current_pred == "del": 
                    self.local_sentence = self.local_sentence[:-1]
                elif self.current_pred != "nothing": 
                    self.local_sentence += self.current_pred
                
                self.stable_count = 0
                self.last_added_time = curr_time

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        bar_x, bar_y = x1, y1 - 25
        bar_w, bar_h = x2 - x1, 15 
        fill_w = int(bar_w * self.current_conf)
        
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
        
        hud_text = f"{self.current_pred.upper()} [{self.current_conf * 100:.1f}%]"
        cv2.putText(img, hud_text, (x1, y1 - 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# STREAMLIT INTERFACE
# ==========================================
st.title("Neural ASL Engine")
st.markdown("---")

col_video, col_space, col_ui = st.columns([12, 1, 10])

with col_video:
    ctx = webrtc_streamer(
        key="asl-translator",
        video_processor_factory=ASLProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_ui:
    st.markdown("### Output Stream")
    
    # 1. Initialize empty slots for dynamic UI injection
    terminal_placeholder = st.empty()
    st.write("")
    status_placeholder = st.empty()
    st.write("")
    
    # 2. Controls remain fixed
    if st.button("Flush Buffer", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.local_sentence = ""
            ctx.video_processor.stable_count = 0
            ctx.video_processor.last_added_time = time.time()

with st.sidebar:
    st.markdown("### Engine Parameters")
    st.markdown("""
    - **Lock Delay:** 1.2s
    - **Min Confidence:** 92%
    - **Input Res:** 64x64
    """)

# ==========================================
# REAL-TIME UI SYNC LOOP
# ==========================================
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            # Safely grab the current string from the background thread
            sentence = ctx.video_processor.local_sentence if ctx.video_processor.local_sentence else "..."
            
            # Inject directly into the placeholders
            terminal_placeholder.markdown(f'<div class="terminal-box">{sentence}</div>', unsafe_allow_html=True)
            status_placeholder.metric("Buffer Status", "Active" if len(ctx.video_processor.local_sentence) > 0 else "Empty")
            
        time.sleep(0.1) # Prevents the while loop from maxing out your CPU