"""
debug_ocr.py — run this once to diagnose OCR issues.

Usage:
    python debug_ocr.py

It takes ONE screenshot via WDA, runs every tile through the same
preprocessing pipeline as main.py, saves 16 images to debug_tiles/,
and prints a confidence table so you can see exactly what the model sees.
"""

import os
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import tensorflow as tf

# ── Configuration ────────────────────────────────────────────────────────────
WDA_URL      = "http://localhost:8100"
MODEL_PATH   = "perfect_ocr_model.h5"
COORD_SCALE  = 3.0

# 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 👇 
# PASTE YOUR CALIBRATED DICTIONARY HERE
TILE_COORDS = {
     0: (  86,  391),  1: ( 174,  391),  2: ( 263,  391),  3: ( 352,  391),
     4: (  86,  480),  5: ( 174,  480),  6: ( 263,  480),  7: ( 352,  480),
     8: (  86,  569),  9: ( 174,  569), 10: ( 263,  569), 11: ( 352,  569),
    12: (  86,  658), 13: ( 174,  658), 14: ( 263,  658), 15: ( 352,  658),
}
# 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆 👆

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'QU', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = "debug_tiles"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Screenshot ─────────────────────────────────────────────────────────────
print("Taking screenshot via WDA...")
http = requests.Session()
http.headers.update({"Content-Type": "application/json"})

sid = None
try:
    status = http.get(f"{WDA_URL}/status", timeout=8).json()
    sid = status.get("sessionId") or status.get("value", {}).get("sessionId")
except Exception:
    pass

if not sid:
    r = http.post(f"{WDA_URL}/session",
                  json={"capabilities": {"alwaysMatch": {}}}, timeout=30)
    r.raise_for_status()
    data = r.json()
    sid  = data.get("sessionId") or data.get("value", {}).get("sessionId")

print(f"  Session: {sid}")
r   = http.get(f"{WDA_URL}/session/{sid}/screenshot", timeout=15)
raw = r.json().get("value", "")
if isinstance(raw, dict):
    raw = raw.get("value", "")
screenshot = Image.open(BytesIO(base64.b64decode(raw)))
w_img, h_img = screenshot.size
print(f"  Screenshot size: {w_img}×{h_img}")
screenshot.save(f"{OUT_DIR}/00_full_screenshot.png")
print(f"  Saved: {OUT_DIR}/00_full_screenshot.png")

# ── 2. Draw tile crop boxes on the screenshot ─────────────────────────────────
annotated = screenshot.copy().convert("RGB")
draw = ImageDraw.Draw(annotated)

# 38 logical pixels radius tightly bounds the letter/dots and avoids the tile edge
logical_radius = 38
phys_radius = int(logical_radius * COORD_SCALE)

for idx, (lx, ly) in TILE_COORDS.items():
    cx = int(lx * COORD_SCALE)
    cy = int(ly * COORD_SCALE)
    
    draw.rectangle(
        [cx - phys_radius, cy - phys_radius,
         cx + phys_radius, cy + phys_radius],
        outline=(255, 0, 0), width=4
    )
    r = 12
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 0), width=4)
    draw.text((cx - 10, cy - phys_radius + 5), str(idx), fill=(255, 255, 0))

scale_factor = min(1.0, 1200 / max(w_img, h_img))
preview_w = int(w_img * scale_factor)
preview_h = int(h_img * scale_factor)
annotated_small = annotated.resize((preview_w, preview_h), Image.LANCZOS)
annotated_small.save(f"{OUT_DIR}/01_tile_positions.png")
print(f"  Saved: {OUT_DIR}/01_tile_positions.png")

# ── 3. Load model ─────────────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("  Model loaded.")

# ── 4. Preprocess and predict ─────────────────────────────────────────────────
print("\nProcessing tiles...\n")
print(f"  {'Tile':>4}  {'Pred':>4}  {'Conf':>6}  {'2nd':>4}  {'2ndConf':>7}  {'Status'}")
print("  " + "─" * 54)

batch = []
tiles_pil = []

for i in range(16):
    lx, ly = TILE_COORDS[i]
    cx = int(lx * COORD_SCALE)
    cy = int(ly * COORD_SCALE)

    box = (
        max(0, cx - phys_radius),
        max(0, cy - phys_radius),
        min(w_img, cx + phys_radius),
        min(h_img, cy + phys_radius)
    )
    
    full = screenshot.crop(box)
    full = full.resize((64, 64)).convert("L")

    # The high threshold to preserve anti-aliased loops (like in P, B, R)
    full = full.point(lambda p: 255 if p > 220 else 0)

    tiles_pil.append(full)
    batch.append(np.array(full, dtype=np.float32))

batch_arr = np.stack(batch, axis=0)[:, :, :, np.newaxis]
preds     = model.predict(batch_arr, verbose=0)

for i in range(16):
    p         = preds[i]
    top2_idx  = np.argsort(p)[::-1][:2]
    pred      = CLASS_NAMES[top2_idx[0]]
    conf      = p[top2_idx[0]]
    second    = CLASS_NAMES[top2_idx[1]]
    conf2     = p[top2_idx[1]]
    status    = "✓ confident" if conf >= 0.80 else ("⚠ low" if conf >= 0.50 else "✗ very low")

    print(f"  {i:>4}  {pred:>4}  {conf:>5.1%}  {second:>4}  {conf2:>6.1%}  {status}")

    fname = f"{OUT_DIR}/tile_{i:02d}_{pred}_{conf:.2f}.png"
    tiles_pil[i].resize((256, 256), Image.NEAREST).save(fname)

print(f"\nAll 16 tiles saved to '{OUT_DIR}/'")