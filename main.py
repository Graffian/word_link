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

def _preprocess_for_cnn(crops: list) -> np.ndarray:
    """Takes 16 raw crops, shaves them matching the dataset, and batches them."""
    batch = []
    for crop in crops:
        h, w = crop.shape
        
        # Exact shave logic from generate_dataset.py
        # Numpy slicing is [Y, X] -> [top:bottom, left:right]
        shaved = crop[SHAVE_PIXELS : h - 10, SHAVE_PIXELS : w - SHAVE_PIXELS]
        
        # Resize to 64x64 for the CNN
        resized = cv2.resize(shaved, CNN_TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Add the channel dimension so shape is (64, 64, 1)
        batch.append(np.expand_dims(resized, axis=-1))
        
    return np.array(batch, dtype=np.float32)

def ocr_board(img: Image.Image) -> list:
    """Reads all 16 tiles instantly using a single CNN batch prediction."""
    img_grey = np.array(img.convert("L"))
    
    # Extract 16 square crops
    raw_crops = [_crop_tile(img_grey, i) for i in range(16)]
    
    # Preprocess all 16 at once
    batch_tensor = _preprocess_for_cnn(raw_crops)
    
    # Inference! Pass the batch through the model
    predictions = ocr_model.predict(batch_tensor, verbose=0)
    
    letters = []
    for pred in predictions:
        class_idx = np.argmax(pred)
        confidence = pred[class_idx]
        
        if confidence >= CNN_CONFIDENCE_THRESHOLD:
            letters.append(CLASS_NAMES[class_idx])
        else:
            letters.append("?")

    # Format printout
    rows = [" ".join(f"{letters[r*4+c].upper():>2}" for c in range(4)) for r in range(4)]
    print(f"  OCR[CNN]:   {rows[0]}")
    for row in rows[1:]:
        print(f"              {row}")
        
    return letters

# ─────────────────────────────────────────
#  DICTIONARY
# ─────────────────────────────────────────
def load_dictionary(path: str = DICT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dictionary not found: '{path}'")

    with open(path, encoding="utf-8", errors="ignore") as fh:
        raw = {line.strip().lower() for line in fh if line.strip()}

    words    = {w for w in raw if len(w) >= MIN_WORD_LEN}
    prefixes: set[str] = set()
    for w in words:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])
    return words, prefixes

# ─────────────────────────────────────────
#  SOLVER & UTILS
# ─────────────────────────────────────────
def _build_neighbours():
    nb = {}
    for idx in range(16):
        r, c = divmod(idx, 4)
        nb[idx] = [
            (r + dr) * 4 + (c + dc)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= r + dr < 4 and 0 <= c + dc < 4
        ]
    return nb

NEIGHBOURS = _build_neighbours()

def solve_board(letters, words, prefixes):
    found = {}

    def dfs(idx, word, path, visited):
        if word not in prefixes:
            return
        if len(word) >= MIN_WORD_LEN and word in words:
            if word not in found:
                found[word] = list(path)
        for nb in NEIGHBOURS[idx]:
            if nb not in visited:
                ch = letters[nb]
                if ch and ch != "?":
                    visited.add(nb); path.append(nb)
                    dfs(nb, word + ch, path, visited)
                    path.pop(); visited.remove(nb)

    for i, ch in enumerate(letters):
        if ch and ch != "?":
            dfs(i, ch, [i], {i})

    return dict(sorted(found.items(), key=lambda x: (boggle_score(x[0]), len(x[0])), reverse=True))

_http       = requests.Session()
_http.headers.update({"Content-Type": "application/json"})
_session_id = None

def _create_session() -> str:
    r = _http.post(f"{WDA_URL}/session", json={"capabilities": {"alwaysMatch": {}}}, timeout=30)
    r.raise_for_status()
    sid = r.json().get("sessionId") or r.json().get("value", {}).get("sessionId")
    return sid

def get_session() -> str:
    global _session_id
    if _session_id: return _session_id
    try:
        r   = _http.get(f"{WDA_URL}/status", timeout=8)
        sid = r.json().get("sessionId") or r.json().get("value", {}).get("sessionId")
        if sid:
            _session_id = sid
            return _session_id
    except Exception: pass
    _session_id = _create_session()
    return _session_id

def take_screenshot(retries: int = 3) -> Image.Image:
    global _session_id
    for attempt in range(retries):
        try:
            r = _http.get(f"{WDA_URL}/session/{_session_id}/screenshot", timeout=10)
            if r.status_code == 404:
                _session_id = None
                get_session()
                continue
            r.raise_for_status()
            raw = r.json().get("value", "")
            if isinstance(raw, dict): raw = raw.get("value", "")
            return Image.open(BytesIO(base64.b64decode(raw)))
        except Exception:
            if attempt < retries - 1: time.sleep(0.2)
    raise RuntimeError("Screenshot failed.")

def swipe_path(indices):
    sid    = get_session()
    sx, sy = TILE_COORDS[indices[0]]
    acts   = [
        {"type": "pointerMove", "duration": 0, "x": int(sx), "y": int(sy)},
        {"type": "pointerDown"},
        {"type": "pause",       "duration": HOLD_MS},
    ]
    for idx in indices[1:]:
        tx, ty = TILE_COORDS[idx]
        acts += [{"type": "pointerMove", "duration": TILE_PAUSE_MS, "x": int(tx), "y": int(ty)}]
    acts.append({"type": "pause", "duration": LIFT_DELAY_MS})
    acts.append({"type": "pointerUp"})

    payload = {"actions": [{"type": "pointer", "id": "finger1", "parameters": {"pointerType": "touch"}, "actions": acts}]}
    try:
        r = _http.post(f"{WDA_URL}/session/{sid}/actions", json=payload, timeout=5)
        return r.status_code == 200
    except Exception: return False

def _tier(w: str) -> int:
    n = len(w)
    if n == 7: return 3
    if n == 6: return 2
    if n == 5: return 1
    return 0

def run():
    words, prefixes = load_dictionary(DICT_PATH)

    print("\n" + "=" * 54)
    print("   Boggle Bot  ⚡  CNN Vision Edition")
    print("=" * 54)
    get_session()

    played:            set[str] = set()
    last_letters:      list     = []
    results:           dict     = {}
    last_swipe_time             = time.time()
    in_game                     = False
    board_will_change: bool     = False

    _next_screenshot: list = [None]
    _prefetch_thread: list = [None]

    def _prefetch_fn(wait: float):
        time.sleep(wait)
        try: _next_screenshot[0] = take_screenshot()
        except Exception: _next_screenshot[0] = None

    def _start_prefetch(wait: float):
        t = threading.Thread(target=_prefetch_fn, args=(wait,), daemon=True)
        t.start()
        _prefetch_thread[0] = t

    def _get_img() -> Image.Image:
        if _prefetch_thread[0] is not None:
            _prefetch_thread[0].join(timeout=2.0)
            _prefetch_thread[0] = None
        if _next_screenshot[0] is not None:
            img = _next_screenshot[0]
            _next_screenshot[0] = None
            return img
        return take_screenshot()

    while True:
        try:
            img     = _get_img()
            letters = ocr_board(img)

            if letters.count("?") >= 12:
                if in_game:
                    print("\n  ── Board unreadable — round over... ──")
                    in_game = False
                    played.clear(); last_letters = []; results = {}
                time.sleep(2.0)
                continue

            if letters.count("?") > 0:
                unsettled_start = getattr(run, '_unsettled_since', None)
                now = time.time()
                if unsettled_start is None:
                    run._unsettled_since = now
                    time.sleep(0.25)
                    continue
                elif now - unsettled_start < 3.0:
                    time.sleep(0.25)
                    continue
                else:
                    run._unsettled_since = None
                    if in_game:
                        in_game = False
                        played.clear(); last_letters = []; results = {}
                    time.sleep(2.0)
                    continue
            run._unsettled_since = None

            if in_game and (time.time() - last_swipe_time) > IDLE_TIMEOUT:
                in_game = False
                played.clear(); last_letters = []; results = {}
                time.sleep(3.0)
                continue

            board_changed = (letters != last_letters) or board_will_change
            board_will_change = False

            if board_changed:
                played.clear()
                last_letters = letters[:]
                t0      = time.perf_counter()
                results = solve_board(letters, words, prefixes)
                elapsed = time.perf_counter() - t0
                top     = list(results.items())[:8]
                top_str = "  ".join(f"{w.upper()}({tile_score(w)}pts)" for w, _ in top)
                print(f"  {len(results)} words solved in ({elapsed:.3f}s) | Top: {top_str}")
                in_game = True

            unplayed = [(w, p) for w, p in results.items() if w not in played and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN]

            if not unplayed:
                print("  No playable words remaining — waiting for board update...")
                in_game = False
                time.sleep(2.0)
                continue

            unplayed.sort(key=lambda x: (_tier(x[0]), len(x[0])), reverse=True)
            word, path = unplayed[0]
            played.add(word)
            score  = tile_score(word)
            print(f"  ▶  {word.upper():<12} +{score}pt  path={path}", end="  ")
            ok = swipe_path(path)
            print("✓" if ok else "✗")

            if ok:
                last_swipe_time = time.time()
                board_will_change = True
                _start_prefetch(BOARD_WAIT_OCR)

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boggle Bot — CNN Vision Edition")
    args = parser.parse_args()
    run()