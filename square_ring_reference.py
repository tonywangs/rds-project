#!/usr/bin/env python3
# Minimal “square in a square” (square ring) reference image
# Writes: square_ring_ref.svg (always)
#         square_ring_ref.png (if Pillow is available)
#
# Usage: python3 square_ring_reference.py

import math

# --- Simple knobs (match your RDS proportions if you want) ---
W, H = 900, 600          # canvas size in pixels
OUTER_FRAC = 0.60        # outer square side = this * min(W, H)
INNER_FRAC = 0.38        # inner (hole) side = this * min(W, H)

# --- Geometry (centered squares) ---
cx, cy = W // 2, H // 2
side_outer = int(min(W, H) * OUTER_FRAC)
side_inner = int(min(W, H) * INNER_FRAC)

# outer square corners
x0 = cx - side_outer // 2
y0 = cy - side_outer // 2
x1 = cx + side_outer // 2
y1 = cy + side_outer // 2

# inner square corners (hole)
xi0 = cx - side_inner // 2
yi0 = cy - side_inner // 2
xi1 = cx + side_inner // 2
yi1 = cy + side_inner // 2

# --- SVG (no dependencies) ---
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
  <rect width="100%" height="100%" fill="white"/>
  <rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="black"/>
  <rect x="{xi0}" y="{yi0}" width="{xi1 - xi0}" height="{yi1 - yi0}" fill="white"/>
</svg>
"""
with open("square_ring_ref.svg", "w", encoding="utf-8") as f:
    f.write(svg)

# --- Optional PNG (requires Pillow) ---
try:
    from PIL import Image, ImageDraw
    img = Image.new("L", (W, H), 255)  # white background
    draw = ImageDraw.Draw(img)
    # PIL rectangles are inclusive on the end pixel; subtract 1 to keep exact sizes.
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=0)       # outer black
    draw.rectangle([xi0, yi0, xi1 - 1, yi1 - 1], fill=255) # inner white hole
    img.save("square_ring_ref.png")
except ImportError:
    # Pillow not installed; SVG is still produced.
    pass
