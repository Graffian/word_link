"""
Boggle Bot — Shift-Tolerant Template Matching Edition
──────────────────────────────────────────────────────────────────────────
Pipeline:
  1. Screenshot via WebDriverAgent
  2. Crop each board tile → Shift-tolerant Template Matching → letter (A-Z or QU)
  3. DFS solver over the curated dictionary
  4. Swipe the best word's tile path on the board
  5. Repeat on new board

Requirements:
  pip install pillow numpy requests opencv-python

No Tesseract binary required!
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

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
WDA_URL   = "http://localhost:8100"
DICT_PATH = "Dictionary-curated.txt"

# ── Timing ──
BOARD_WAIT_OCR = 0.55   # wait after swipe before next screenshot (tile animation)
HOLD_MS        = 50     # initial press on first tile
TILE_PAUSE_MS  = 15     # slide duration between tiles (>0 = interpolated)
LIFT_DELAY_MS  = 120    # hold on last tile before lifting

IDLE_TIMEOUT   = 4.5    # seconds with no successful swipe → assume round over

# ── Word filtering ──
MIN_WORD_LEN = 5   # only play 5-7 letter words (best score/risk ratio)
MAX_WORD_LEN = 7

# ── Crop ──
COORD_SCALE  = 3.0    # physical px = WDA logical px × scale (3.0 for Retina)
TILE_CROP_PX = 100    # half-side of square crop around each tile centre

# ── Template matching ──
TEMPLATES_DIR         = "templates"
_TMPL_MATCH_THRESHOLD = 0.85  # Perfect fits yield > 0.95. Anything below 0.85 is unrecognized.

# ── 4×4 board tile coordinates (WDA logical px) ──
TILE_COORDS = {
     0: ( 84, 401),  1: (170, 406),  2: (256, 404),  3: (345, 401),
     4: ( 78, 491),  5: (171, 490),  6: (254, 485),  7: (357, 494),
     8: ( 81, 582),  9: (169, 580), 10: (262, 580), 11: (353, 572),
    12: ( 81, 670), 13: (173, 668), 14: (257, 664), 15: (358, 672),
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
#  SHIFTS-TOLERANT MATCHING ENGINE
# ─────────────────────────────────────────
SIDE = TILE_CROP_PX * 2
_auto_templates: dict = {}


def _canonical_preprocess(crop: np.ndarray) -> np.ndarray:
    """Creates a standardized binary canvas layout matching reference files."""
    large    = cv2.resize(crop, (SIDE * 3, SIDE * 3), interpolation=cv2.INTER_CUBIC)
    _, bw    = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) < 127:
        bw = cv2.bitwise_not(bw)
    return cv2.copyMakeBorder(bw, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def load_templates(folder: str = TEMPLATES_DIR) -> None:
    """Loads your manually named alphabet files from disk into memory."""
    global _auto_templates
    os.makedirs(folder, exist_ok=True)
    loaded = {}
    for fname in os.listdir(folder):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in (".png", ".jpg", ".jpeg") or name.startswith("tile_"):
            continue
        label = name.upper()
        path  = os.path.join(folder, fname)
        img   = Image.open(path).convert("L").resize((SIDE * 3 + 40, SIDE * 3 + 40), Image.LANCZOS)
        loaded[label.lower()] = np.array(img, dtype=np.float32)
    _auto_templates = loaded
    if loaded:
        print(f"  [OCR] Loaded {len(loaded)} curated templates: {sorted(k.upper() for k in loaded)}")
    else:
        print(f"  [WARNING] No letter templates found! Please rename tile_xx.png files to letters (e.g., A.png)")


def _crop_tile(img_grey: np.ndarray, idx: int) -> np.ndarray:
    cx = int(TILE_COORDS[idx][0] * COORD_SCALE)
    cy = int(TILE_COORDS[idx][1] * COORD_SCALE)
    h, w = img_grey.shape[:2]
    x1 = max(0, cx - TILE_CROP_PX)
    y1 = max(0, cy - TILE_CROP_PX)
    x2 = min(w, cx + TILE_CROP_PX)
    y2 = min(h, cy + TILE_CROP_PX)
    return img_grey[y1:y2, x1:x2]


def _template_match(canonical: np.ndarray) -> tuple:
    """Slides the master templates across a padded tile area to handle shifts."""
    if not _auto_templates:
        return "", -1.0
    
    # Pad search area by 12px to let templates safely slide if coordinates bounce
    search_area = cv2.copyMakeBorder(canonical, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    search_area = search_area.astype(np.float32)
    
    best_letter, best_score = "", -1.0
    for letter, tmpl in _auto_templates.items():
        if tmpl.shape != canonical.shape:
            continue
            
        res = cv2.matchTemplate(search_area, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_score:
            best_score, best_letter = max_val, letter
            
    return best_letter, best_score


def _read_tile(img_grey: np.ndarray, idx: int) -> str:
    crop = _crop_tile(img_grey, idx)
    if crop.size == 0:
        return "?"

    canonical = _canonical_preprocess(crop)
    letter, score = _template_match(canonical)
    
    if score >= _TMPL_MATCH_THRESHOLD:
        return letter

    print(f"  [OCR] Unknown character layout at tile {idx:02d} (Best guess: '{letter.upper()}' @ {score:.2f})")
    return "?"


def ocr_board(img: Image.Image) -> list:
    """Reads all 16 tiles instantly over memory."""
    img_grey = np.array(img.convert("L"))
    letters = [_read_tile(img_grey, i) for i in range(16)]

    rows = [" ".join(f"{letters[r*4+c].upper():>2}" for c in range(4)) for r in range(4)]
    print(f"  OCR[Match]: {rows[0]}")
    for row in rows[1:]:
        print(f"              {row}")
    return letters


# ─────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────
def calibrate():
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    print("  [calibrate] Connecting to WDA…")
    get_session()
    print("  [calibrate] Taking screenshot…")
    img      = take_screenshot()
    img_grey = np.array(img.convert("L"))
    
    for i in range(16):
        crop = _crop_tile(img_grey, i)
        if crop.size == 0:
            continue
        canonical = _canonical_preprocess(crop)
        out_path  = os.path.join(TEMPLATES_DIR, f"tile_{i:02d}.png")
        cv2.imwrite(out_path, canonical)
        
    print(f"\n  Saved 16 processed crops to ./{TEMPLATES_DIR}/")
    print("  Please map these files to true letters (e.g., rename tile_00.png -> A.png)")


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
    load_templates()
    words, prefixes = load_dictionary(DICT_PATH)

    print("\n" + "=" * 54)
    print("   Boggle Bot  ⚡  Template Matching Edition")
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
    parser = argparse.ArgumentParser(description="Boggle Bot — Template Matching Edition")
    parser.add_argument("--calibrate", action="store_true", help="Extract templates.")
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
    else:
        run()