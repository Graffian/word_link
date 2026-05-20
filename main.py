"""
Boggle Bot — PaddleOCR tile OCR + curated dictionary + swipe input
───────────────────────────────────────────────────────────────────
Pipeline:
  1. Screenshot via WebDriverAgent
  2. Crop each board tile → PaddleOCR → letter (A-Z or QU)
  3. DFS solver over the curated dictionary
  4. Swipe the best word's tile path on the board
  5. Repeat on new board

Requirements:
  pip install paddleocr pillow numpy requests
"""

import requests
import time
import base64
import os
import threading
import numpy as np
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
TILE_PAUSE_MS  = 15     # slide duration between tiles (>0 = interpolated, not teleport)
LIFT_DELAY_MS  = 120    # hold on last tile before lifting

IDLE_TIMEOUT   = 4.5    # seconds with no successful swipe → assume round over

# ── Word filtering ──
MIN_WORD_LEN = 5   # only play 5-7 letter words (best score/risk ratio)
MAX_WORD_LEN = 7

# ── Crop ──
COORD_SCALE = 3.0     # physical px = WDA logical px × scale (3.0 for Retina)
TILE_CROP_PX = 60     # half-side of square crop around each tile centre

# ── 4×4 board tile coordinates (WDA logical px) ──
TILE_COORDS = {
     0: ( 84, 383),  1: (170, 388),  2: (256, 386),  3: (345, 383),
     4: ( 78, 473),  5: (171, 472),  6: (254, 467),  7: (357, 476),
     8: ( 81, 564),  9: (169, 562), 10: (262, 562), 11: (353, 554),
    12: ( 81, 652), 13: (173, 650), 14: (257, 646), 15: (358, 654),
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
    """Length bonus + letter values — approximates actual game score."""
    return boggle_score(word) + sum(LETTER_VALUE.get(c, 1) for c in word.lower())


# ─────────────────────────────────────────
#  PADDLEOCR TILE CLASSIFIER
# ─────────────────────────────────────────
_ocr = None


def load_paddle_ocr() -> None:
    global _ocr
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        import subprocess, sys
        print("  Installing paddleocr…")
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "paddleocr", "-q"])
        from paddleocr import PaddleOCR
    # v3.x API: use_textline_orientation replaces use_angle_cls.
    # Tiles are always upright so all orientation/unwarping models are off.
    _ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print("  [OCR] PaddleOCR ready.")


def _crop_tile(img_grey: np.ndarray, idx: int) -> np.ndarray:
    cx = int(TILE_COORDS[idx][0] * COORD_SCALE)
    cy = int(TILE_COORDS[idx][1] * COORD_SCALE)
    h, w = img_grey.shape[:2]
    x1, y1 = max(0, cx - TILE_CROP_PX), max(0, cy - TILE_CROP_PX)
    x2, y2 = min(w, cx + TILE_CROP_PX), min(h, cy + TILE_CROP_PX)
    return img_grey[y1:y2, x1:x2]


def _run_ocr(rgb: np.ndarray) -> list[str]:
    """Call PaddleOCR predict and return rec_texts, or [] on failure."""
    try:
        result = _ocr.predict(rgb)
    except Exception:
        return []
    if not result:
        return []
    texts = []
    for res in result:
        texts.extend(res["rec_texts"])
    return texts


def _ocr_tile(img_grey: np.ndarray, idx: int) -> str:
    """Run PaddleOCR on one tile crop. Returns lowercase letter/qu or '?'.
    Retries with a 2× upscaled crop when the first pass returns nothing —
    thin letters like I are often missed at native resolution.
    """
    if _ocr is None:
        raise RuntimeError("Call load_paddle_ocr() first.")
    crop = _crop_tile(img_grey, idx)
    if crop.size == 0:
        return "?"

    pil_crop = Image.fromarray(crop)
    rgb = np.array(pil_crop.convert("RGB"))
    texts = _run_ocr(rgb)

    # Retry at 2× scale — catches thin letters (I, L, 1) the detector misses
    if not texts:
        w, h = pil_crop.size
        rgb2x = np.array(pil_crop.resize((w * 2, h * 2), Image.LANCZOS).convert("RGB"))
        texts = _run_ocr(rgb2x)
        if texts:
            print(f"  [OCR] tile {idx}: detected on 2× retry")

    if not texts:
        return "?"
    text = "".join(c for c in texts[0].strip().upper() if c.isalpha())
    if not text:
        return "?"
    # QU is a single Boggle tile - treat Q alone as QU too
    if text in ("QU", "Q"):
        return "qu"
    return text[0].lower()  # single letter; discard OCR noise beyond first char


def ocr_board(img: Image.Image) -> list[str]:
    """
    OCR all 16 tiles with PaddleOCR.
    Returns list of lowercase letter strings ('a'..'z', 'qu') or '?'.
    """
    img_grey = np.array(img.convert("L"))
    letters  = [_ocr_tile(img_grey, i) for i in range(16)]

    # Pretty-print 4×4 grid
    rows = [" ".join(f"{letters[r*4+c].upper():>2}" for c in range(4)) for r in range(4)]
    print(f"  OCR[Paddle]: {rows[0]}")
    for row in rows[1:]:
        print(f"               {row}")

    return letters



# ─────────────────────────────────────────
#  DICTIONARY
# ─────────────────────────────────────────
def load_dictionary(path: str = DICT_PATH):
    """
    Load Dictionary-curated.txt.
    Returns (words, prefixes) — lowercase sets, no frequency filtering needed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dictionary not found: '{path}'")

    with open(path, encoding="utf-8", errors="ignore") as fh:
        raw = {line.strip().lower() for line in fh if line.strip()}

    words    = {w for w in raw if len(w) >= MIN_WORD_LEN}
    prefixes: set[str] = set()
    for w in words:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])

    by_len = {n: sum(1 for w in words if len(w) == n) for n in range(3, 16)}
    top    = "  ".join(f"{n}L:{c:,}" for n, c in sorted(by_len.items()) if c > 0)
    print(f"  [dict] {len(words):,} words  |  {top}")
    return words, prefixes


# ─────────────────────────────────────────
#  SOLVER
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
    """
    DFS Boggle solver with QU-tile support.
    A 'qu' tile contributes the two-char string 'qu' to the word (one tile used).
    Returns {word: [tile_indices]} sorted best → worst.
    """
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

    return dict(sorted(found.items(),
                        key=lambda x: (boggle_score(x[0]), len(x[0])),
                        reverse=True))


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
#  SWIPE
# ─────────────────────────────────────────
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
        # duration > 0 makes WDA interpolate the move over that many ms so the
        # game detects every tile the finger passes through.  duration=0 is an
        # instant teleport — the game only catches the first and last positions.
        acts += [{"type": "pointerMove", "duration": TILE_PAUSE_MS, "x": int(tx), "y": int(ty)}]
    # Hold on the last tile long enough for the game to register it before lifting
    acts.append({"type": "pause", "duration": LIFT_DELAY_MS})
    acts.append({"type": "pointerUp"})

    payload = {"actions": [{"type": "pointer", "id": "finger1",
                             "parameters": {"pointerType": "touch"}, "actions": acts}]}
    try:
        r = _http.post(f"{WDA_URL}/session/{sid}/actions", json=payload, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"    [swipe] exception: {e}")
        return False


# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────
def _tier(w: str) -> int:
    """Word length priority: 7 > 6 > 5. Anything else = 0 (never played)."""
    n = len(w)
    if n == 7: return 3
    if n == 6: return 2
    if n == 5: return 1
    return 0


def run():
    load_paddle_ocr()
    words, prefixes = load_dictionary(DICT_PATH)

    print("\n" + "=" * 54)
    print("   Boggle Bot  ⚡  PaddleOCR Edition")
    print("=" * 54)
    print(f"  Dict  : {DICT_PATH}  ({len(words):,} words)")
    print("Ctrl+C to stop.\n")

    try:
        get_session()
    except Exception as e:
        print(f"  [WDA] Could not connect: {e}")
        print("  Make sure WebDriverAgent is running on port 8100.")
        return   # no point continuing without WDA

    played:            set[str] = set()
    last_letters:      list     = []
    results:           dict     = {}
    last_swipe_time             = time.time()
    in_game                     = False
    board_will_change: bool     = False  # set True after every successful swipe

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
            # ── Always OCR first ──────────────────────────────────────────────
            # This game refills swiped tiles with new letters after each correct
            # word, so the board genuinely changes every turn. We must re-read
            # and re-solve after every swipe — there is no safe "fast path" of
            # replaying cached paths on stale tile data.
            img     = _get_img()
            letters = ocr_board(img)

            # Board not visible yet (between rounds, loading screen, etc.)
            if letters.count("?") >= 12:
                if in_game:
                    print("\n  ── Board unreadable — round over, waiting... ──")
                    in_game = False
                    played.clear(); last_letters = []; results = {}
                time.sleep(2.0)
                continue

            # Partial read — tiles are still mid-fall. Poll until the board
            # is fully settled. Tiles vanish → fall → new ones drop in from top,
            # which can take 0.6–1.2s total. We check every 0.25s, give up at 3s.
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
                    # Board has been unsettled for 3s+ — round transition or error
                    run._unsettled_since = None
                    if in_game:
                        print("\n  ── Board unsettled 3s — round over, waiting... ──")
                        in_game = False
                        played.clear(); last_letters = []; results = {}
                    time.sleep(2.0)
                    continue
            run._unsettled_since = None  # board fully settled, reset tracker

            # Idle safety: if we haven't swiped for a while the round ended
            if in_game and (time.time() - last_swipe_time) > IDLE_TIMEOUT:
                print("\n  ── Idle timeout — round over, waiting... ──")
                in_game = False
                played.clear(); last_letters = []; results = {}
                time.sleep(3.0)
                continue

            # Re-solve when the board changes — or when we know it must have
            # changed after a swipe but OCR returned the same letters (noise).
            board_changed = (letters != last_letters) or board_will_change
            board_will_change = False  # consume the flag

            if board_changed:
                n_diff = sum(1 for a, b in zip(letters, last_letters) if a != b) if last_letters else 16
                if last_letters:
                    src_tag = "refreshed" if n_diff > 0 else "OCR same but swipe ran — forcing re-solve"
                    print(f"  [board] {n_diff}/16 tiles {src_tag}")
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

    
            # Build sorted candidate list once per board; pop from front each swipe
            unplayed = [
                (w, p) for w, p in results.items()
                if w not in played and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN
            ]

            if not unplayed:
                print("  No 5-7 letter words on this board — waiting for next...")
                in_game = False
                time.sleep(2.0)
                continue

            # Sort once — best word is always first
            unplayed.sort(key=lambda x: (_tier(x[0]), len(x[0])), reverse=True)
            word, path = unplayed[0]
            played.add(word)
            score  = tile_score(word)
            est_ms = HOLD_MS + (len(path) - 1) * TILE_PAUSE_MS + LIFT_DELAY_MS
            print(f"  ▶  {word.upper():<12} +{score}pt  path={path}  (~{est_ms}ms)", end="  ")
            ok = swipe_path(path)
            print("✓" if ok else "✗")

            if ok:
                last_swipe_time = time.time()
                board_will_change = True  # next OCR must re-solve even if letters look same
                # Wait for tiles to animate back in before next screenshot
                _start_prefetch(BOARD_WAIT_OCR)

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    run()