"""
debug_ocr.py — run this once to diagnose OCR issues.

Usage:
    python debug_ocr.py

It takes ONE screenshot via WDA, runs every tile through the same
preprocessing pipeline as main.py, saves 16 images to debug_tiles/,
and prints a confidence table so you can see exactly what the model sees.

You do NOT need to edit main.py or set any flags.
"""

import os
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import tensorflow as tf

# ── Copy these values from your main.py ──────────────────────────────────────
WDA_URL      = "http://localhost:8100"
MODEL_PATH   = "perfect_ocr_model.h5"
COORD_SCALE  = 3.0

TILE_COORDS = {
     0: (  84,  383),  1: ( 170,  388),  2: ( 256,  386),  3: ( 345,  383),
     4: (  78,  473),  5: ( 171,  472),  6: ( 254,  467),  7: ( 357,  476),
     8: (  81,  564),  9: ( 169,  562), 10: ( 262,  562), 11: ( 353,  554),
    12: (  81,  652), 13: ( 173,  650), 14: ( 257,  646), 15: ( 358,  654),
}

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

# Try to find or create a session
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

# ── 2. Draw tile crop boxes on the screenshot so you can see where we're looking ──
annotated = screenshot.copy().convert("RGB")
draw = ImageDraw.Draw(annotated)

# 43 logical pixels radius isolates exactly one tile safely
logical_radius = 43
phys_radius = int(logical_radius * COORD_SCALE)

for idx, (lx, ly) in TILE_COORDS.items():
    cx = int(lx * COORD_SCALE)
    cy = int(ly * COORD_SCALE)
    
    # Draw the tight crop boundary
    draw.rectangle(
        [cx - phys_radius, cy - phys_radius,
         cx + phys_radius, cy + phys_radius],
        outline=(255, 0, 0), width=4
    )
    # Draw the tile centre crosshair
    r = 12
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 255, 0), width=4)
    draw.text((cx - 10, cy - phys_radius + 5), str(idx), fill=(255, 255, 0))

# Scale down for easy viewing
scale_factor = min(1.0, 1200 / max(w_img, h_img))
preview_w = int(w_img * scale_factor)
preview_h = int(h_img * scale_factor)
annotated_small = annotated.resize((preview_w, preview_h), Image.LANCZOS)
annotated_small.save(f"{OUT_DIR}/01_tile_positions.png")
print(f"  Saved: {OUT_DIR}/01_tile_positions.png")
print(f"  ↳  Open this to verify the red boxes are centred on each tile.")

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

    # 1. Base tile size calculation
    side = phys_radius * 2

    # 2. Extract strictly the single tile, padded with white if it hits an edge
    full = Image.new("RGB", (side, side), (255, 255, 255))
    src_x1 = max(0, cx - phys_radius)
    src_y1 = max(0, cy - phys_radius)
    src_x2 = min(w_img, cx + phys_radius)
    src_y2 = min(h_img, cy + phys_radius)
    
    paste_x = src_x1 - (cx - phys_radius)
    paste_y = src_y1 - (cy - phys_radius)
    full.paste(screenshot.crop((src_x1, src_y1, src_x2, src_y2)), (paste_x, paste_y))

    # 3. Apply PROPORTIONAL shave to match the 640x640 training data asset squish
    w, h = full.size
    shave_l = int(w * (65 / 640))
    shave_t = int(h * (65 / 640))
    shave_r = int(w * (65 / 640))
    shave_b = int(h * (10 / 640))

    inner = full.crop((shave_l, shave_t, w - shave_r, h - shave_b))

    # 4. Same resize + grayscale mapping to 64x64
    full = inner.resize((64, 64)).convert("L")

    tiles_pil.append(full)
    batch.append(np.array(full, dtype=np.float32))

batch_arr = np.stack(batch, axis=0)[:, :, :, np.newaxis]  # (16, 64, 64, 1)
preds     = model.predict(batch_arr, verbose=0)            # (16, num_classes)

for i in range(16):
    p         = preds[i]
    top2_idx  = np.argsort(p)[::-1][:2]
    pred      = CLASS_NAMES[top2_idx[0]]
    conf      = p[top2_idx[0]]
    second    = CLASS_NAMES[top2_idx[1]]
    conf2     = p[top2_idx[1]]
    status    = "✓ confident" if conf >= 0.80 else ("⚠ low" if conf >= 0.50 else "✗ very low")

    print(f"  {i:>4}  {pred:>4}  {conf:>5.1%}  {second:>4}  {conf2:>6.1%}  {status}")

    # Save the 64×64 grayscale tile scaled up 4× for easy inspection
    fname = f"{OUT_DIR}/tile_{i:02d}_{pred}_{conf:.2f}.png"
    tiles_pil[i].resize((256, 256), Image.NEAREST).save(fname)

print(f"\nAll 16 tiles saved to '{OUT_DIR}/'")