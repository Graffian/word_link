"""
auto_calibrate.py — Run this when the game board is open on your screen.
It automatically calculates the exact center coordinates for all 16 tiles
and prints out the ready-to-use TILE_COORDS dictionary.
"""

import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO

WDA_URL      = "http://localhost:8100"
COORD_SCALE  = 3.0  # Keep this matched to your device scale factor

# ── 1. Capture the current game screen ───────────────────────────────────────
print("Connecting to device and taking screenshot...")
http = requests.Session()
http.headers.update({"Content-Type": "application/json"})

sid = None
try:
    status = http.get(f"{WDA_URL}/status", timeout=8).json()
    sid = status.get("sessionId") or status.get("value", {}).get("sessionId")
except Exception:
    pass

if not sid:
    r = http.post(f"{WDA_URL}/session", json={"capabilities": {"alwaysMatch": {}}}, timeout=30)
    sid = r.json().get("sessionId") or r.json().get("value", {}).get("sessionId")

r = http.get(f"{WDA_URL}/session/{sid}/screenshot", timeout=15)
raw = r.json().get("value", "")
if isinstance(raw, dict):
    raw = raw.get("value", "")

screenshot = Image.open(BytesIO(base64.b64decode(raw)))
w_img, h_img = screenshot.size
print(f"Captured screen size: {w_img}×{h_img}")

# ── 2. Computer Vision Grid Profiling ────────────────────────────────────────
# Convert to grayscale array
arr = np.array(screenshot.convert("L"))

# Isolate the middle 60% of the screen vertically to avoid top/bottom game menus
ymin_search = int(h_img * 0.25)
ymax_search = int(h_img * 0.85)

# High-pass threshold to isolate the bright white face of the tiles
binary = arr > 220

# Generate 1D projection profiles of the pixels
col_profile = binary[ymin_search:ymax_search, :].any(axis=0)
row_profile = binary[ymin_search:ymax_search, :].any(axis=1)

def find_centers(profile, expected_count=4):
    """Detects continuous bands of white pixels and finds their midpoints."""
    intervals = []
    in_block = False
    start = 0
    
    for idx, val in enumerate(profile):
        if val and not in_block:
            start = idx
            in_block = True
        elif not val and in_block:
            intervals.append((start, idx - 1))
            in_block = False
    if in_block:
        intervals.append((start, len(profile) - 1))
        
    # If standard segmentation matches the 4 rows/cols, use it
    if len(intervals) == expected_count:
        return [int((s + e) / 2) for s, e in intervals]
    else:
        # Symmetrical fallback if background elements break the crisp segmentation
        if not intervals:
            return None
        imin, imax = intervals[0][0], intervals[-1][1]
        step = (imax - imin) / expected_count
        return [int(imin + (i + 0.5) * step) for i in range(expected_count)]

x_pixels = find_centers(col_profile, 4)
y_pixels = find_centers(row_profile, 4)

# ── 3. Output the Calibration ───────────────────────────────────────────────
if x_pixels and y_pixels:
    # Offset y positions back to absolute image coordinates
    y_pixels = [y + ymin_search for y in y_pixels]
    
    print("\n" + "═"*60)
    print(" SUCCESS! Copy and paste this dictionary into main.py and debug_ocr.py:")
    print("═"*60 + "\n")
    print("TILE_COORDS = {")
    
    idx = 0
    for r in range(4):
        line = "   "
        for c in range(4):
            # Map the raw physical pixel centers back to logical layout units
            lx = int(round(x_pixels[c] / COORD_SCALE))
            ly = int(round(y_pixels[r] / COORD_SCALE))
            line += f" {idx:>2}: ({lx:>4}, {ly:>4}),"
            idx += 1
        print(line)
        
    print("}")
    print("\n" + "═"*60)
else:
    print("\n❌ Error: Could not cleanly isolate the 4x4 tile grid.")
    print("Make sure a dynamic match board is fully visible and unobstructed on your screen.")