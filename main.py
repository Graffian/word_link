import requests
import time
import base64
import re
import threading
import os




import numpy as np
from PIL import Image
from io import BytesIO

import tensorflow as tf


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
WDA_URL        = "http://localhost:8100"
DICT_PATH      = "Dictionary-curated.txt"
MODEL_PATH     = "perfect_ocr_model.h5"

# ── Timing ──
BOARD_WAIT_OCR  = 0.50
BOARD_WAIT_FAST = 0.11
HOLD_MS         = 50   # slightly longer initial press so game registers touch
TILE_PAUSE_MS   = 15   # ms to slide between each tile
LIFT_DELAY_MS   = 120  # ms to hold on the LAST tile before lifting

IDLE_TIMEOUT     = 4.5
DIRTY_TILE_LIMIT = 10

# ── Word filtering ──
MIN_WORD_LEN = 3   # game requires 3+

TILE_COORDS = {
     0: (  84,  383),  1: ( 170,  388),  2: ( 256,  386),  3: ( 345,  383),
     4: (  78,  473),  5: ( 171,  472),  6: ( 254,  467),  7: ( 357,  476),
     8: (  81,  564),  9: ( 169,  562), 10: ( 262,  562), 11: ( 353,  554),
    12: (  81,  652), 13: ( 173,  650), 14: ( 257,  646), 15: ( 358,  654),
}

COORD_SCALE  = 3.0
TILE_CROP_PX = 320    # Radius of square crop around centre to get full 640x640 base tile

# ── OCR Debug ──
# Set to True once to dump every preprocessed tile to debug_tiles/ so you
# can visually verify what the model is actually seeing.
DEBUG_OCR         = False
DEBUG_OCR_DIR     = "debug_tiles"
# Predictions below this confidence are returned as "?" (likely off-board)
CONFIDENCE_THRESH = 0.50

# Exact alphabetical order matching tf.keras.preprocessing.image_dataset_from_directory
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'QU', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# ── Load Neural Network ──
print("🧠 Loading perfect_ocr_model.h5...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✓ CNN Brain initialized successfully.")


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
    """Total letter value + length bonus — approximates actual game score."""
    letter_pts = sum(LETTER_VALUE.get(c, 1) for c in word.lower())
    return boggle_score(word) + letter_pts


# ─────────────────────────────────────────
#  SESSION MANAGEMENT
# ─────────────────────────────────────────
_http       = requests.Session()
_http.headers.update({"Content-Type": "application/json"})
_session_id = None


def _create_session() -> str:
    r = _http.post(f"{WDA_URL}/session",
                   json={"capabilities": {"alwaysMatch": {}}}, timeout=30)
    r.raise_for_status()
    data = r.json()
    sid  = data.get("sessionId") or data.get("value", {}).get("sessionId")
    if not sid:
        raise RuntimeError(f"No sessionId in WDA response: {data}")
    print(f"  [WDA] New session: {sid}")
    return sid


def get_session() -> str:
    global _session_id
    if _session_id:
        return _session_id
    try:
        r   = _http.get(f"{WDA_URL}/status", timeout=8)
        sid = r.json().get("sessionId") or r.json().get("value", {}).get("sessionId")
        if sid:
            print(f"  [WDA] Reusing session: {sid}")
            _session_id = sid
            return _session_id
    except Exception:
        pass
    _session_id = _create_session()
    return _session_id


# ─────────────────────────────────────────
#  SCREENSHOT
# ─────────────────────────────────────────
def take_screenshot(retries: int = 3) -> Image.Image:
    global _session_id
    for attempt in range(retries):
        try:
            r = _http.get(f"{WDA_URL}/session/{_session_id}/screenshot", timeout=10)
            if r.status_code == 404:
                print("  [screenshot] Session gone — resetting")
                _session_id = None
                get_session()
                continue
            r.raise_for_status()
            raw = r.json().get("value", "")
            if isinstance(raw, dict):
                raw = raw.get("value", "")
            return Image.open(BytesIO(base64.b64decode(raw)))
        except Exception as e:
            print(f"  [screenshot] attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(0.2)
    raise RuntimeError("Screenshot failed after all retries.")


# ─────────────────────────────────────────
#  HIGH-SPEED CNN INFERENCE OCR
# ─────────────────────────────────────────

def _validate_coord_scale(w_img: int, h_img: int) -> None:
    """
    Warn once if COORD_SCALE looks wrong for this screenshot.
    WDA can return screenshots at 1x, 2x, or 3x depending on device + config.
    A 3x iPhone 14 Pro is 1290×2796; 2x would be 828×1792; 1x ~390×844.
    The rightmost tile centre (tile 3) in logical coords is x=345.  At the
    current COORD_SCALE that maps to x={int(345*COORD_SCALE)}.
    If that pixel is outside the screenshot width, COORD_SCALE is wrong.
    """
    max_logical_x = max(x for x, _ in TILE_COORDS.values())
    max_logical_y = max(y for _, y in TILE_COORDS.values())
    if int(max_logical_x * COORD_SCALE) >= w_img or int(max_logical_y * COORD_SCALE) >= h_img:
        # Try to suggest the correct scale
        for scale in (1.0, 2.0, 3.0):
            if int(max_logical_x * scale) < w_img and int(max_logical_y * scale) < h_img:
                print(f"  ⚠️  [OCR] COORD_SCALE={COORD_SCALE} maps tiles outside the "
                      f"{w_img}×{h_img} screenshot! "
                      f"Try setting COORD_SCALE = {scale}")
                break
        else:
            print(f"  ⚠️  [OCR] COORD_SCALE={COORD_SCALE} looks wrong for {w_img}×{h_img} "
                  f"screenshot. Check TILE_COORDS match the current device.")

_scale_checked = False  # only validate once per run


def _crop_tile(img: Image.Image, tile_idx: int) -> Image.Image:
    """
    Crop and preprocess one tile to exactly match the training data:
    tight crop -> proportional shave -> threshold to pure B&W -> 64x64.
    """
    cx = int(TILE_COORDS[tile_idx][0] * COORD_SCALE)
    cy = int(TILE_COORDS[tile_idx][1] * COORD_SCALE)
    w_img, h_img = img.size

    # 1. Base tile size calculation (43 logical radius isolates exactly one tile safely)
    logical_radius = 43
    phys_radius = int(logical_radius * COORD_SCALE)
    side = phys_radius * 2

    # 2. Extract strictly the single tile, padded with white if it hits an edge
    full = Image.new("RGB", (side, side), (255, 255, 255))
    src_x1 = max(0, cx - phys_radius)
    src_y1 = max(0, cy - phys_radius)
    src_x2 = min(w_img, cx + phys_radius)
    src_y2 = min(h_img, cy + phys_radius)
    
    paste_x = src_x1 - (cx - phys_radius)
    paste_y = src_y1 - (cy - phys_radius)
    full.paste(img.crop((src_x1, src_y1, src_x2, src_y2)), (paste_x, paste_y))

    # 3. Proportional shave to match the 640x640 squish ratio from training
    w, h = full.size
    shave_l = int(w * (65 / 640))
    shave_t = int(h * (65 / 640))
    shave_r = int(w * (65 / 640))
    shave_b = int(h * (10 / 640))

    full = full.crop((shave_l, shave_t, w - shave_r, h - shave_b))

    # 4. Resize and convert to grayscale first
    full = full.resize((64, 64)).convert("L")

    # 5. B&W THRESHOLDING (The "Just an E and a dot" magic)
    # Any pixel lighter than 150 turns pure white (255), darker turns pure black (0)
    # Adjust 150 up or down slightly if the dots are getting wiped out or gray is surviving
    full = full.point(lambda p: 255 if p > 150 else 0)

    return full


def ocr_board(img: Image.Image) -> list[str]:
    global _scale_checked

    w_img, h_img = img.size

    # One-time sanity check that COORD_SCALE is plausible for this device
    if not _scale_checked:
        _scale_checked = True
        print(f"  [OCR] Screenshot: {w_img}×{h_img} | "
              f"tile_0 pixel centre: "
              f"({int(TILE_COORDS[0][0]*COORD_SCALE)}, "
              f"{int(TILE_COORDS[0][1]*COORD_SCALE)}) | "
              f"COORD_SCALE={COORD_SCALE}")
        _validate_coord_scale(w_img, h_img)

    # Build a batch of all 16 preprocessed tiles in one pass
    batch    = []   # will become shape (16, 64, 64, 1)
    crops    = []   # keep PIL images for optional debug save
    errored  = {}   # tile_idx -> True for any crop that failed

    for i in range(16):
        try:
            tile_pil = _crop_tile(img, i)
            arr = np.array(tile_pil, dtype=np.float32)   # (64, 64), values 0-255
            batch.append(arr)
            crops.append(tile_pil)
        except Exception as e:
            print(f"  [OCR] Tile {i} crop error: {e}")
            batch.append(np.zeros((64, 64), dtype=np.float32))
            crops.append(None)
            errored[i] = True

    # Single model.predict() call for all 16 tiles at once
    batch_arr  = np.stack(batch, axis=0)           # (16, 64, 64)
    batch_arr  = np.expand_dims(batch_arr, axis=-1) # (16, 64, 64, 1)
    all_preds  = model.predict(batch_arr, verbose=0) # (16, num_classes)

    letters = []
    for i in range(16):
        if i in errored:
            letters.append("?")
            continue

        confidence = float(np.max(all_preds[i]))
        class_idx  = int(np.argmax(all_preds[i]))
        letter     = CLASS_NAMES[class_idx].lower()

        # Low confidence → treat as unreadable rather than a wrong guess
        if confidence < CONFIDENCE_THRESH:
            letter = "?"

        letters.append(letter)

        # ── Debug: save the preprocessed 64×64 image the model received ──
        if DEBUG_OCR and crops[i] is not None:
            os.makedirs(DEBUG_OCR_DIR, exist_ok=True)
            fname = f"{DEBUG_OCR_DIR}/tile_{i:02d}_{letter.upper()}_{confidence:.2f}.png"
            # Save upscaled so it's easy to inspect
            crops[i].resize((256, 256), Image.NEAREST).save(fname)

    s = "".join(l.upper() for l in letters)
    print(f"  OCR[CNN]: {s[:4]} {s[4:8]} {s[8:12]} {s[12:]}")
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

    return found  # caller sorts by tile_score; no need to pre-sort here


# ─────────────────────────────────────────
#  SWIPE
# ─────────────────────────────────────────
def swipe_path(indices: list[int]) -> bool:
    sid    = get_session()
    sx, sy = TILE_COORDS[indices[0]]
    acts   = [
        {"type": "pointerMove", "duration": 0, "x": int(sx), "y": int(sy)},
        {"type": "pointerDown"},
        {"type": "pause",       "duration": HOLD_MS},
    ]
    for idx in indices[1:]:
        tx, ty = TILE_COORDS[idx]
        acts.append({"type": "pointerMove", "duration": TILE_PAUSE_MS,
                     "x": int(tx), "y": int(ty)})
    acts.append({"type": "pause",    "duration": LIFT_DELAY_MS})
    acts.append({"type": "pointerUp"})

    payload = {"actions": [{"type": "pointer", "id": "finger1",
                             "parameters": {"pointerType": "touch"}, "actions": acts}]}
    try:
        r = _http.post(f"{WDA_URL}/session/{sid}/actions", json=payload, timeout=20)
        return r.status_code == 200
    except Exception as e:
        print(f"    [swipe] exception: {e}")
        return False


# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────
def run():
    words, prefixes = load_dictionary(DICT_PATH)

    print("\n" + "=" * 54)
    print("   Boggle Bot  ⚡  CNN OCR Autonomous Edition")
    print("=" * 54)
    print("  ✓  Using 100% accurate perfect_ocr_model.h5 local brain.")
    print("  ✓  No external APIs. Blazing-fast inference active.")
    print("Ctrl+C to stop.\n")

    try:
        get_session()
    except Exception as e:
        print(f"  [WDA] Could not connect: {e}")
        print("  Make sure WebDriverAgent is running on port 8100.\n")

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
        try:
            _next_screenshot[0] = take_screenshot()
        except Exception as e:
            print(f"  [prefetch] {e}")
            _next_screenshot[0] = None

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
            # ── OCR ──────────────────────────────────────────────────────────
            img     = _get_img()
            letters = ocr_board(img)

            # Board not visible (between rounds / loading screen)
            if letters.count("?") >= 12:
                if in_game:
                    print("\n  ── Board unreadable — round over, waiting... ──")
                    in_game = False
                    played.clear(); last_letters = []; results = {}
                time.sleep(2.0)
                continue

            # Idle safety
            if in_game and (time.time() - last_swipe_time) > IDLE_TIMEOUT:
                print("\n  ── Idle timeout — round over, waiting... ──")
                in_game = False
                played.clear(); last_letters = []; results = {}
                time.sleep(3.0)
                continue

            # Re-solve on genuine board change OR when a swipe just completed.
            # IMPORTANT: only clear `played` on a real letter change — not after
            # every swipe — otherwise the bot would endlessly replay the same word.
            force_resolv      = board_will_change
            board_changed     = (letters != last_letters)
            board_will_change = False

            if board_changed or force_resolv:
                n_diff = sum(1 for a, b in zip(letters, last_letters) if a != b) if last_letters else 16
                if last_letters and board_changed:
                    print(f"  [board] {n_diff}/16 tiles refreshed — new board detected")
                elif force_resolv and not board_changed:
                    print(f"  [board] re-solving after swipe (same letters, {len(played)} words already played)")

                if board_changed:
                    # Genuine new board → reset everything
                    played.clear()
                    last_letters = letters[:]

                t0      = time.perf_counter()
                results = solve_board(letters, words, prefixes)
                elapsed = time.perf_counter() - t0
                top     = list(results.items())[:8]
                top_str = "  ".join(f"{w.upper()}({tile_score(w)}pts)" for w, _ in top)
                pts     = sum(boggle_score(w) for w in results)
                print(f"  {len(results)} words  {pts} pts  ({elapsed:.3f}s)")
                print(f"  Top: {top_str}")
                in_game = True

            # ── Pick best unplayed word ──────────────────────────────────────
            # Priority tiers: 8+ letter → 7-letter → 6-letter → 5-letter
            def _tier(w: str) -> int:
                n = len(w)
                if n >= 8: return 4   # Boggle max score (11 pts) — never skip these
                if n == 7: return 3
                if n == 6: return 2
                if n == 5: return 1
                return 0

            candidates = [
                (w, p) for w, p in results.items()
                if w not in played and len(w) >= 5   # include all 5+ letter words
            ]

            remaining = sorted(
                candidates,
                key=lambda x: (_tier(x[0]), tile_score(x[0])),
                reverse=True,
            )

            if not remaining:
                print("  No unplayed 5+ letter words on this board — waiting for next...")
                in_game = False
                time.sleep(2.0)
                continue

            word, path = remaining[0]
            played.add(word)
            score = tile_score(word)
            print(f"  ▶  {word.upper():<12} +{score}pt  path={path}", end="  ")
            ok = swipe_path(path)
            print("✓" if ok else "✗")

            if ok:
                last_swipe_time   = time.time()
                board_will_change = True
                _start_prefetch(BOARD_WAIT_OCR)

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    run()