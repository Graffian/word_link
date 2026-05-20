"""
Boggle Bot — Dynamic Contour & Tesseract Edition
──────────────────────────────────────────────────────────────────────────
Pipeline:
  1. Screenshot via WebDriverAgent
  2. OpenCV Contour Detection dynamically finds the 16 tiles (ignores UI shifts)
  3. Crop each dynamic board tile → Tesseract OCR (single-char mode) → letter
  4. DFS solver over the curated dictionary
  5. Swipe the best word's tile path on the board using dynamic WDA coordinates
  6. Repeat on new board

Requirements:
  pip install pillow numpy requests opencv-python pytesseract
  brew install tesseract   (or apt install tesseract-ocr)
"""

import argparse
import requests
import time
import base64
import os
import threading
import numpy as np
import cv2
import pytesseract
from PIL import Image
from io import BytesIO

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
WDA_URL   = "http://localhost:8100"
DICT_PATH = "Dictionary-curated.txt"

# ── Timing ──
BOARD_WAIT_OCR = 0.55   
HOLD_MS        = 50     
TILE_PAUSE_MS  = 15     
LIFT_DELAY_MS  = 120    
IDLE_TIMEOUT   = 4.5    

# ── Word filtering ──
MIN_WORD_LEN = 5   
MAX_WORD_LEN = 7

# ── Crop & Scale ──
COORD_SCALE  = 3.0    # physical px = WDA logical px × scale (3.0 for Retina)

# ── Tesseract ──
TEMPLATES_DIR = "templates"
_TESS_CONFIG  = "--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ── Dynamic Coordinates (Populated at Runtime) ──
# Maps tile index (0-15) to its (x, y) logical WDA swipe coordinate
DYNAMIC_TILE_COORDS = {}

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
#  DYNAMIC CONTOUR DETECTION
# ─────────────────────────────────────────
def find_dynamic_grid(img_grey: np.ndarray):
    """
    Uses OpenCV to find 16 square tiles on the screen.
    Returns a list of 16 bounding boxes (x, y, w, h) sorted in a 4x4 grid.
    """
    # 1. Edge detection to find the hard borders of the tiles
    edges = cv2.Canny(img_grey, 50, 150)
    
    # Dilate slightly to connect broken edge lines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 2. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = float(w) / h
        
        # 3. Filter for squares (Adjust area boundaries based on screen size if needed)
        # 10000 to 120000 pixels roughly covers a 100x100 to 340x340 physical tile
        if 10000 < area < 120000 and 0.85 <= aspect_ratio <= 1.15:
            squares.append((x, y, w, h))
            
    # 4. Remove overlapping bounding boxes (keep the best ones)
    squares.sort(key=lambda s: s[2]*s[3], reverse=True)
    filtered_squares = []
    
    for s1 in squares:
        overlap = False
        cx1, cy1 = s1[0] + s1[2]//2, s1[1] + s1[3]//2
        for s2 in filtered_squares:
            cx2, cy2 = s2[0] + s2[2]//2, s2[1] + s2[3]//2
            dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if dist < 50: # If centers are within 50px, it's the same tile
                overlap = True
                break
        if not overlap:
            filtered_squares.append(s1)
        if len(filtered_squares) == 16:
            break
            
    if len(filtered_squares) != 16:
        return None # Grid not found on this frame
        
    # 5. Sort into 4x4 reading order
    # Sort all by Y (top to bottom)
    filtered_squares.sort(key=lambda s: s[1])
    
    grid_boxes = []
    for i in range(4):
        # Take chunks of 4 (a row) and sort them by X (left to right)
        row = filtered_squares[i*4 : (i+1)*4]
        row.sort(key=lambda s: s[0])
        grid_boxes.extend(row)
        
    return grid_boxes

# ─────────────────────────────────────────
#  TESSERACT OCR ENGINE
# ─────────────────────────────────────────
def _preprocess_for_tess(crop: np.ndarray) -> np.ndarray:
    _, bw_full = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_full, connectivity=8)
    clean = np.zeros_like(bw_full)
    tile_area = crop.shape[0] * crop.shape[1]
    
    # Filter out small noise (like point dots)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] > tile_area * 0.01: 
            clean[labels == lbl] = 255
            
    crop = cv2.bitwise_not(clean)
    large = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) < 127:                                       
        bw = cv2.bitwise_not(bw)
    return cv2.copyMakeBorder(bw, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

def _read_tile(img_grey: np.ndarray, box: tuple, idx: int) -> str:
    x, y, w, h = box
    
    # Shrink the box by 10% to completely eliminate the outer tile border from the crop
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    crop = img_grey[y + margin_y : y + h - margin_y, x + margin_x : x + w - margin_x]
    
    if crop.size == 0:
        return "?"

    ready = _preprocess_for_tess(crop)
    pil   = Image.fromarray(ready)
    raw   = pytesseract.image_to_string(pil, config=_TESS_CONFIG).strip().upper()

    letter = next((c for c in raw if c.isalpha()), None)
    if letter:
        return letter.lower()

    return "?"

def ocr_board(img: Image.Image) -> list:
    global DYNAMIC_TILE_COORDS
    img_grey = np.array(img.convert("L"))
    
    # Dynamically find the grid on this specific frame
    grid_boxes = find_dynamic_grid(img_grey)
    
    if not grid_boxes:
        print("  [Vision] Could not lock onto the 4x4 grid. UI might be transitioning.")
        return ["?"] * 16
        
    letters = []
    for i, box in enumerate(grid_boxes):
        x, y, w, h = box
        
        # Calculate WDA logical coordinate for swiping (center of the box / scale)
        cx_logical = (x + w // 2) / COORD_SCALE
        cy_logical = (y + h // 2) / COORD_SCALE
        DYNAMIC_TILE_COORDS[i] = (cx_logical, cy_logical)
        
        letters.append(_read_tile(img_grey, box, i))

    rows = [" ".join(f"{letters[r*4+c].upper():>2}" for c in range(4)) for r in range(4)]
    print(f"  OCR[Tess]:  {rows[0]}")
    for row in rows[1:]:
        print(f"              {row}")
    return letters

# ─────────────────────────────────────────
#  DICTIONARY & SOLVER
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

# ─────────────────────────────────────────
#  WDA DRIVER LOGIC
# ─────────────────────────────────────────
_http       = requests.Session()
_http.headers.update({"Content-Type": "application/json"})
_session_id = None

def _create_session() -> str:
    r = _http.post(f"{WDA_URL}/session", json={"capabilities": {"alwaysMatch": {}}}, timeout=30)
    r.raise_for_status()
    return r.json().get("sessionId") or r.json().get("value", {}).get("sessionId")

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
    sid = get_session()
    
    # Ensure coordinates exist from the contour detector
    if not DYNAMIC_TILE_COORDS:
        return False
        
    sx, sy = DYNAMIC_TILE_COORDS[indices[0]]
    acts   = [
        {"type": "pointerMove", "duration": 0, "x": int(sx), "y": int(sy)},
        {"type": "pointerDown"},
        {"type": "pause",       "duration": HOLD_MS},
    ]
    for idx in indices[1:]:
        tx, ty = DYNAMIC_TILE_COORDS[idx]
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
    if n >= 7: return 3
    if n == 6: return 2
    if n == 5: return 1
    return 0

# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────
def run():
    words, prefixes = load_dictionary(DICT_PATH)

    print("\n" + "=" * 60)
    print("   Boggle Bot  ⚡  Dynamic Contour + Tesseract Edition")
    print("=" * 60)
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
    parser = argparse.ArgumentParser(description="Boggle Bot — Dynamic Contour Edition")
    args = parser.parse_args()
    run()