"""
click_calibrate.py — A simple tool to calculate your perfect grid coordinates using matplotlib.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── Configuration ──────────────────────────────────────────
IMG_PATH = "debug_tiles/00_full_screenshot.png"
COORD_SCALE = 3.0

if not os.path.exists(IMG_PATH):
    print(f"❌ Error: Could not find {IMG_PATH}")
    print("Run debug_ocr.py first so it saves a screenshot!")
    exit()

print("Opening window... (Check behind your terminal if it doesn't pop to the front)")
print("1️⃣  Click the exact dead-center of the TOP-LEFT tile.")
print("2️⃣  Click the exact dead-center of the BOTTOM-RIGHT tile.")

# Load the image
img = mpimg.imread(IMG_PATH)

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 10))
ax.imshow(img)
ax.set_title("1. Click TOP-LEFT tile center\n2. Click BOTTOM-RIGHT tile center\n(Window will close automatically)", fontsize=10)
ax.axis("off") # Hide the axis numbers
plt.tight_layout()

# ginput(2) waits for exactly 2 mouse clicks, timeout=0 means it waits forever
clicks = plt.ginput(2, timeout=0)
plt.close()

# ── Math & Output ──────────────────────────────────────────
if len(clicks) == 2:
    tl_x, tl_y = clicks[0]
    br_x, br_y = clicks[1]
    
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
    print("❌ Calibration cancelled. You didn't click twice.")