"""
Boggle Bot — Deep Learning CNN Edition
──────────────────────────────────────────────────────────────────────────
Pipeline:
  1. Screenshot via WebDriverAgent
  2. Crop 16 tiles -> Apply exact dataset "Shave" -> Batch to Tensor
  3. CNN Inference (predicts all 16 letters instantly)
  4. DFS solver over the curated dictionary
  5. Swipe the best word's tile path on the board
  6. Repeat on new board

Requirements:
  pip install pillow numpy requests opencv-python tensorflow
"""

import argparse
import requests
import time
import base64
import os
import threading
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Suppress TensorFlow terminal spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
WDA_URL   = "http://localhost:8100"
DICT_PATH = "Dictionary-curated.txt"
MODEL_PATH = "perfect_ocr_model.h5"

# ── Timing ──
BOARD_WAIT_OCR = 0.55   
HOLD_MS        = 50     
TILE_PAUSE_MS  = 15     
LIFT_DELAY_MS  = 120    
IDLE_TIMEOUT   = 4.5    

# ── Word filtering ──
MIN_WORD_LEN = 5   
MAX_WORD_LEN = 7

# ── Crop & Shave (CNN Params) ──
COORD_SCALE  = 3.0    
TILE_CROP_PX = 100    
SHAVE_PIXELS = 65     
CNN_TARGET_SIZE = (64, 64)
CNN_CONFIDENCE_THRESHOLD = 0.60 

# Keras image_dataset_from_directory sorts classes alphanumerically
CLASS_NAMES = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'qu', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# ── 4×4 board tile coordinates (WDA logical px) ──
TILE_COORDS = {
     0: ( 88, 393),  1: (174, 398),  2: (260, 396),  3: (349, 393),
     4: ( 82, 483),  5: (175, 482),  6: (258, 477),  7: (361, 486),
     8: ( 85, 574),  9: (173, 572), 10: (266, 572), 11: (357, 564),
    12: ( 85, 662), 13: (177, 660), 14: (261, 656), 15: (362, 664),
}

# ─────────────────────────────────────────
#  BOGGLE SCORING
# ─────────────────────────────────────────
def boggle_score(word: str) -> int:
    n = len(word)
    if n <= 4: return 1
    if n == 5: return 2
    if n == 6: return 3
    if n == 7: return 5
    return 11

LETTER_VALUE = {
    'a':1,'e':1,'i':1,'o':1,'s':1,'t':1,'r':1,'n':1,
    'l':2,'u':2,'d':2,
    'g':3,'b':3,'c':3,'m':3,'p':3,'f':3,'h':3,'w':3,'y':3,
    'v':4,'k':4,
    'j':5,'x':5,
    'q':6,'z':6,
}

def tile_score(word: str) -> int:
    return boggle_score(word) + sum(LETTER_VALUE.get(c, 1) for c in word.lower())

# ─────────────────────────────────────────
#  CNN VISION ENGINE
# ─────────────────────────────────────────
print("  [Init] Booting TensorFlow Vision Engine...")
try:
    ocr_model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"  [Error] Failed to load {MODEL_PATH}. Did you run train_model.py?")
    exit(1)

def _crop_tile(img_grey: np.ndarray, idx: int) -> np.ndarray:
    cx = int(TILE_COORDS[idx][0] * COORD_SCALE)
    cy = int(TILE_COORDS[idx][1] * COORD_SCALE)
    h, w = img_grey.shape[:2]
    x1 = max(0, cx - TILE_CROP_PX)
    y1 = max(0, cy - TILE_CROP_PX)
    x2 = min(w, cx + TILE_CROP_PX)
    y2 = min(h, cy + TILE_CROP_PX)
    return img_grey[y1:y2, x1:x2]

def _canonical_preprocess(crop: np.ndarray) -> np.ndarray:
    """Restored from your original script to match dataset generation!"""
    # 1. Scale 200x200 up to 600x600
    large = cv2.resize(crop, (TILE_CROP_PX * 6, TILE_CROP_PX * 6), interpolation=cv2.INTER_CUBIC)
    
    # 2. Threshold to pure "crazy clear" Black & White
    _, bw = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) < 127:
        bw = cv2.bitwise_not(bw)
        
    # 3. Pad to 640x640
    return cv2.copyMakeBorder(bw, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

def _preprocess_for_cnn(crops: list) -> np.ndarray:
    """Processes live game tiles identically to generate_dataset.py"""
    batch = []
    for crop in crops:
        # First, turn the raw crop into the 640x640 B&W canvas your model trained on
        canonical = _canonical_preprocess(crop)
        
        # NOW apply the exact 65px shave to the 640x640 image
        h, w = canonical.shape
        shaved = canonical[SHAVE_PIXELS : h - 10, SHAVE_PIXELS : w - SHAVE_PIXELS]
        
        # Finally, squish to 64x64 for the CNN brain
        resized = cv2.resize(shaved, CNN_TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        batch.append(np.expand_dims(resized, axis=-1))
        
    return np.array(batch, dtype=np.float32)
# Add this at the top with your configurations
DEBUG_MODE = False

def ocr_board(img: Image.Image) -> list:
    """Reads all 16 tiles instantly using a single CNN batch prediction."""
    img_grey = np.array(img.convert("L"))
    
    # Extract 16 square crops
    raw_crops = [_crop_tile(img_grey, i) for i in range(16)]
    
    # Preprocess all 16 at once
    batch_tensor = _preprocess_for_cnn(raw_crops)
    
    # --- DEBUG VISION DUMP ---
    if DEBUG_MODE:
        os.makedirs("debug_vision", exist_ok=True)
        print("\n  [Debug] Dumping CNN inputs to ./debug_vision/")
        for i, tensor in enumerate(batch_tensor):
            # Tensor is shape (64, 64, 1), we need to squeeze it to (64, 64) to save as PNG
            debug_img = tensor.astype(np.uint8).squeeze()
            cv2.imwrite(f"debug_vision/tile_{i:02d}.png", debug_img)
    
    # Inference! Pass the batch through the model
    predictions = ocr_model.predict(batch_tensor, verbose=0)
    
    letters = []
    for i, pred in enumerate(predictions):
        class_idx = np.argmax(pred)
        confidence = pred[class_idx]
        
        # --- DEBUG LOGGING ---
        if DEBUG_MODE:
            print(f"  [Debug] Tile {i:02d} Top 3:")
            top_3_idx = np.argsort(pred)[-3:][::-1]
            for idx in top_3_idx:
                print(f"          {CLASS_NAMES[idx].upper():>2} : {pred[idx]*100:>5.1f}%")
        
        if confidence >= CNN_CONFIDENCE_THRESHOLD:
            letters.append(CLASS_NAMES[class_idx])
        else:
            letters.append("?")

    # Format printout
    rows = [" ".join(f"{letters[r*4+c].upper():>2}" for c in range(4)) for r in range(4)]
    print(f"\n  OCR[CNN]:   {rows[0]}")
    for row in rows[1:]:
        print(f"              {row}")
        
    return letters

# --- UPDATE YOUR MAIN BLOCK ---
def run():
    # ... (Keep your existing dictionary loading and while loop logic exactly the same) ...
    pass # (Don't copy this pass, keep your run loop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boggle Bot — CNN Vision Edition")
    parser.add_argument("--debug", action="store_true", help="Dump CNN inputs and confidence scores.")
    args = parser.parse_args()
    
    if args.debug:
        DEBUG_MODE = True
        
    run()