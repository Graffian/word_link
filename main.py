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

DEBUG_MODE = False

# ── Word filtering ──
MIN_WORD_LEN = 5   
MAX_WORD_LEN = 16  # no cap — board can have 8+ letter words

# ── Crop & Shave (CNN Params) ──
COORD_SCALE  = 3.0    
TILE_CROP_PX = 100    
SHAVE_PIXELS = 65     
CNN_TARGET_SIZE = (64, 64)
CNN_CONFIDENCE_THRESHOLD = 0.60 

# Keras image_dataset_from_directory sorts classes alphanumerically
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'QU', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
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
    print("Model classes (indices):", CLASS_NAMES)
    print("--- MODEL MAPPING CONFIRMATION ---")
# If you used image_dataset_from_directory, the labels are derived from folder names
# We just need to ensure the order is correct. 
# Run this once and check your terminal:

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
            predicted = CLASS_NAMES[class_idx]
            # Q alone doesn't exist in Boggle — always treat it as QU
            if predicted == 'Q':
                predicted = 'QU'
            letters.append(predicted)
        else:
            letters.append("?")

    # Format printout
    rows = [" ".join(f"{letters[r*4+c].upper():>2}" for c in range(4)) for r in range(4)]
    print(f"\n  OCR[CNN]:   {rows[0]}")
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
        raw = {line.strip().upper() for line in fh if line.strip()}

    words: set[str] = {w for w in raw if len(w) >= MIN_WORD_LEN}
    prefixes: set[str] = set()
    for w in words:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])
    return words, prefixes

def _build_prefix_set(words: set) -> set:
    prefixes: set[str] = set()
    for w in words:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])
    return prefixes

def load_fallback_dictionary(path: str = "words.txt"):
    """
    Loads the ENABLE word list and builds three tiers by frequency:
      tier_strict  — common words only (game will almost always accept)
      tier_loose   — broader set (rare but valid words)
      tier_full    — everything in ENABLE >= MIN_WORD_LEN (nuclear option)
    Returns (tier_strict_words, tier_strict_prefixes,
             tier_loose_words,  tier_loose_prefixes,
             tier_full_words,   tier_full_prefixes)
    """
    import subprocess, sys
    ENABLE_URL = "https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt"

    try:
        from wordfreq import word_frequency
    except ImportError:
        print("  [Fallback Dict] Installing wordfreq...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wordfreq", "-q"])
        from wordfreq import word_frequency

    if not os.path.exists(path):
        print("  [Fallback Dict] Downloading ENABLE word list...")
        r = subprocess.run(["curl", "-s", "-o", path, ENABLE_URL], capture_output=True)
        if r.returncode != 0:
            print(f"  [Fallback Dict] Download failed — no fallback available.")
            empty = set()
            return empty, empty, empty, empty, empty, empty
        print(f"  [Fallback Dict] Downloaded → {path}")

    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = {w.strip().upper() for w in f if MIN_WORD_LEN <= len(w.strip()) <= 16}

    tier_strict: set[str] = set()
    tier_loose:  set[str] = set()

    for w in raw:
        freq = word_frequency(w.lower(), "en")
        n    = len(w)
        # Strict: words the game will almost certainly accept
        if n <= 6:
            if freq >= 5e-6: tier_strict.add(w)
            if freq >= 1e-6: tier_loose.add(w)
        else:
            if freq >= 5e-7: tier_strict.add(w)
            if freq >= 1e-7: tier_loose.add(w)

    print(f"  [Fallback Dict] strict={len(tier_strict):,}  loose={len(tier_loose):,}  full={len(raw):,}")
    return (
        tier_strict, _build_prefix_set(tier_strict),
        tier_loose,  _build_prefix_set(tier_loose),
        raw,         _build_prefix_set(raw),
    )

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
    """Score by length — longer is always better."""
    return len(w)

def run():
    words, prefixes = load_dictionary(DICT_PATH)
    (fb_strict, fb_strict_pre,
     fb_loose,  fb_loose_pre,
     fb_full,   fb_full_pre)  = load_fallback_dictionary()

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

                # ── Cascading dictionary: curated → strict → loose → full ──
                results = solve_board(letters, words, prefixes)
                fallback_used = None

                if not results and fb_strict:
                    print("  [Dict] Curated empty — trying fallback strict...")
                    results = solve_board(letters, fb_strict, fb_strict_pre)
                    fallback_used = "strict"

                if not results and fb_loose:
                    print("  [Dict] Strict empty — trying fallback loose...")
                    results = solve_board(letters, fb_loose, fb_loose_pre)
                    fallback_used = "loose"

                if not results and fb_full:
                    print("  [Dict] Loose empty — trying full ENABLE list...")
                    results = solve_board(letters, fb_full, fb_full_pre)
                    fallback_used = "full"

                elapsed = time.perf_counter() - t0
                top     = list(results.items())[:8]
                top_str = "  ".join(f"{w.upper()}({tile_score(w)}pts)" for w, _ in top)
                src     = f" [{fallback_used}]" if fallback_used else ""
                print(f"  {len(results)} words solved in ({elapsed:.3f}s){src} | Top: {top_str}")
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
    parser.add_argument("--debug", action="store_true", help="Dump CNN inputs and confidence scores.")
    parser.add_argument("--test", metavar="IMAGE", help="Run OCR on a local screenshot instead of connecting to device.")
    args = parser.parse_args()

    if args.debug:
        DEBUG_MODE = True

    if args.test:
        print(f"\n  [Test Mode] Loading screenshot: {args.test}")
        if not os.path.exists(args.test):
            print(f"  [ERROR] File not found: {args.test}")
            exit(1)
        img = Image.open(args.test)
        print(f"  [Test Mode] Image size: {img.size}")
        letters = ocr_board(img)
        print("\n  [Test Mode] Raw letters list:")
        print(f"  {letters}")
    else:
        run()