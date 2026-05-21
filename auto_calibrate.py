"""
click_calibrate.py — A simple tool to calculate your perfect grid coordinates.
"""

import tkinter as tk
from PIL import Image, ImageTk
import os

# ── Configuration ──────────────────────────────────────────
IMG_PATH = "debug_tiles/00_full_screenshot.png"
COORD_SCALE = 3.0

if not os.path.exists(IMG_PATH):
    print(f"❌ Error: Could not find {IMG_PATH}")
    print("Run debug_ocr.py first so it saves a screenshot!")
    exit()

clicks = []

def on_click(event):
    clicks.append((event.x, event.y))
    if len(clicks) == 1:
        print(f"✓ Top-Left clicked. Now click the BOTTOM-RIGHT tile center.")
    elif len(clicks) == 2:
        root.destroy()

# ── GUI Setup ──────────────────────────────────────────────
root = tk.Tk()
root.title("Click TOP-LEFT tile center, then BOTTOM-RIGHT tile center")

# Load and scale image so it fits on laptop screens
img = Image.open(IMG_PATH)
w, h = img.size
scale = 800 / h if h > 800 else 1.0

if scale < 1.0:
    img_resized = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
else:
    img_resized = img
    scale = 1.0

tk_img = ImageTk.PhotoImage(img_resized)
canvas = tk.Canvas(root, width=tk_img.width(), height=tk_img.height(), cursor="crosshair")
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
canvas.bind("<Button-1>", on_click)

print("Opening window... (check behind your other windows if you don't see it)")
print("1️⃣  Click the exact dead-center of the TOP-LEFT tile.")
print("2️⃣  Click the exact dead-center of the BOTTOM-RIGHT tile.")
root.mainloop()

# ── Math & Output ──────────────────────────────────────────
if len(clicks) == 2:
    # Convert scaled clicks back to absolute physical pixels
    tl_x = clicks[0][0] / scale
    tl_y = clicks[0][1] / scale
    br_x = clicks[1][0] / scale
    br_y = clicks[1][1] / scale
    
    # Calculate spacing (3 gaps between 4 tiles)
    step_x = (br_x - tl_x) / 3
    step_y = (br_y - tl_y) / 3

    print("\n" + "═"*60)
    print(" SUCCESS! Copy and paste this dictionary into main.py and debug_ocr.py:")
    print("═"*60 + "\n")
    print("TILE_COORDS = {")
    
    idx = 0
    for r in range(4):
        line = "   "
        for c in range(4):
            phys_x = tl_x + (c * step_x)
            phys_y = tl_y + (r * step_y)
            
            # Convert physical pixels back to logical iOS points
            lx = int(round(phys_x / COORD_SCALE))
            ly = int(round(phys_y / COORD_SCALE))
            line += f" {idx:>2}: ({lx:>4}, {ly:>4}),"
            idx += 1
        print(line)
    print("}")
    print("\n" + "═"*60)
else:
    print("Calibration cancelled or closed early.")