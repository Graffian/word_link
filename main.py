import requests
import time
import base64
import re
import threading
import os
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
WDA_URL        = "http://localhost:8100"
TEMPLATES_DIR  = "tile_templates"   # saved letter crops live here
DICT_PATH      = "Dictionary-curated.txt"

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
TILE_CROP_PX = 55    # half-side of square crop around each tile centre (device px)

_OCR_RE = re.compile(r"(?:Tile\s*)?(\d+)\s*[:\-]\s*([A-Za-z])")


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
#  ML OCR — OpenCV Template Matching
# ─────────────────────────────────────────
_claude         = anthropic.Anthropic()
_templates: dict[str, np.ndarray] = {}
MATCH_THRESHOLD = 0.85


def _ensure_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        import subprocess, sys
        print("  Installing opencv-python-headless...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "opencv-python-headless", "-q"])
        import cv2
        return cv2


def _crop_tile(img_np: np.ndarray, tile_idx: int) -> np.ndarray:
    cx = int(TILE_COORDS[tile_idx][0] * COORD_SCALE)
    cy = int(TILE_COORDS[tile_idx][1] * COORD_SCALE)
    h, w = img_np.shape[:2]
    x1, y1 = max(0, cx - TILE_CROP_PX), max(0, cy - TILE_CROP_PX)
    x2, y2 = min(w, cx + TILE_CROP_PX), min(h, cy + TILE_CROP_PX)
    return img_np[y1:y2, x1:x2]


def load_templates():
    cv2 = _ensure_cv2()
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    _templates.clear()
    for fname in os.listdir(TEMPLATES_DIR):
        if fname.endswith(".png") and len(fname) == 5:
            letter = fname[0].upper()
            img    = cv2.imread(os.path.join(TEMPLATES_DIR, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _templates[letter] = img
    if _templates:
        print(f"  [templates] Loaded {len(_templates)} letters: {''.join(sorted(_templates))}")


def _save_template(letter: str, crop: np.ndarray):
    cv2  = _ensure_cv2()
    path = os.path.join(TEMPLATES_DIR, f"{letter.upper()}.png")
    if not os.path.exists(path):
        cv2.imwrite(path, crop)
        _templates[letter.upper()] = crop
        print(f"  [templates] Saved '{letter.upper()}' ({len(_templates)}/26)")


def _match_letter(crop: np.ndarray) -> tuple[str, float]:
    cv2    = _ensure_cv2()
    best_l, best_s = "?", 0.0
    for letter, tmpl in _templates.items():
        if tmpl.shape != crop.shape:
            tmpl = cv2.resize(tmpl, (crop.shape[1], crop.shape[0]))
        score = float(cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED).max())
        if score > best_s:
            best_s, best_l = score, letter
    return best_l, best_s


def _ocr_cv(img_np: np.ndarray) -> list[str]:
    letters = []
    for i in range(16):
        crop = _crop_tile(img_np, i)
        if crop.size == 0:
            letters.append("?")
        else:
            l, s = _match_letter(crop)
            letters.append(l if s >= MATCH_THRESHOLD else "?")
    return letters


def _ocr_claude(img: Image.Image, img_np: np.ndarray,
                tile_indices = None) -> list[str]:
    indices = tile_indices if tile_indices is not None else list(range(16))
    buf     = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64     = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    coord_hint = "\n".join(
        f"Tile {i} (row {i//4} col {i%4}): pixel "
        f"({int(TILE_COORDS[i][0]*COORD_SCALE)}, {int(TILE_COORDS[i][1]*COORD_SCALE)})"
        for i in indices
    )
    prompt = (
        "This is a screenshot of a 4x4 Boggle word game board.\n"
        "Read the single uppercase letter on each tile listed below.\n"
        "Distinguish carefully: I (straight) vs L (foot at bottom), "
        "O vs Q, U vs V, M vs N.\n\n"
        "Tile positions:\n" + coord_hint + "\n\n"
        f"Reply ONLY with {len(indices)} lines:\n"
        "Tile N: X\nNo extra text."
    )

    result = {i: "?" for i in indices}
    try:
        message = _claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        for line in message.content[0].text.strip().splitlines():
            m = _OCR_RE.search(line.strip())
            if m:
                idx, letter = int(m.group(1)), m.group(2).upper()
                if idx in result:
                    result[idx] = letter
                    crop = _crop_tile(img_np, idx)
                    if crop.size > 0 and letter not in _templates:
                        _save_template(letter, crop)
    except Exception as e:
        print(f"  [OCR-Claude] failed: {e}")

    return [result.get(i, "?") for i in range(16)]


def ocr_board(img: Image.Image) -> list[str]:
    cv2    = _ensure_cv2()
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    if len(_templates) < 26:
        print(f"  [OCR] Claude mode ({len(_templates)}/26 templates seen)")
        letters = _ocr_claude(img, img_np)
    else:
        letters   = _ocr_cv(img_np)
        uncertain = [i for i, l in enumerate(letters) if l == "?"]
        if uncertain:
            print(f"  [OCR] CV uncertain on {len(uncertain)} tiles → Claude")
            partial = _ocr_claude(img, img_np, tile_indices=uncertain)
            for i in uncertain:
                letters[i] = partial[i]

    if letters.count("?") == 0:
        s = "".join(l.upper() for l in letters)
        src = "CV" if len(_templates) >= 20 else "Claude"
        print(f"  OCR[{src}]: {s[:4]} {s[4:8]} {s[8:12]} {s[12:]}")

    return [l.lower() for l in letters]


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
    load_templates()
    words, prefixes = load_dictionary(DICT_PATH)

    print("\n" + "=" * 54)
    print("   Boggle Bot  ⚡  Curated-Dictionary Edition")
    print("=" * 54)
    if len(_templates) < 26:
        print(f"  ℹ  Cold start — Claude OCRs first few boards to build templates.")
        print(f"     ({len(_templates)}/26 letters in ./{TEMPLATES_DIR}/)")
    else:
        print(f"  ✓  Template library ready ({len(_templates)} letters) — full CV speed.")
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

            # Tiles still mid-fall — poll until settled
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
                        print("\n  ── Board unsettled 3s — round over, waiting... ──")
                        in_game = False
                        played.clear(); last_letters = []; results = {}
                    time.sleep(2.0)
                    continue
            run._unsettled_since = None

            # Idle safety
            if in_game and (time.time() - last_swipe_time) > IDLE_TIMEOUT:
                print("\n  ── Idle timeout — round over, waiting... ──")
                in_game = False
                played.clear(); last_letters = []; results = {}
                time.sleep(3.0)
                continue

            # Re-solve whenever the board changes (or a swipe just ran)
            board_changed    = (letters != last_letters) or board_will_change
            board_will_change = False

            if board_changed:
                n_diff = sum(1 for a, b in zip(letters, last_letters) if a != b) if last_letters else 16
                if last_letters:
                    tag = "refreshed" if n_diff > 0 else "OCR same but swipe ran — forcing re-solve"
                    print(f"  [board] {n_diff}/16 tiles {tag}")
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
            # Priority tiers: 7-letter → 6-letter → 5-letter
            def _tier(w: str) -> int:
                n = len(w)
                if n == 7: return 3
                if n == 6: return 2
                if n == 5: return 1
                return 0

            candidates = [
                (w, p) for w, p in results.items()
                if w not in played and 5 <= len(w) <= 7
            ]

            # Sort by tier then score (no frequency — dictionary is already curated)
            remaining = sorted(
                candidates,
                key=lambda x: (_tier(x[0]), tile_score(x[0])),
                reverse=True,
            )

            if not remaining:
                print("  No unplayed 5-7 letter words on this board — waiting for next...")
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