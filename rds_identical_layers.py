#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Dot Stereogram (RDS) — identical layers except color, red-only edit
===========================================================================

This script implements the slide-deck recipe:
- Start from one random-dot field.
- Make **two identical layers** of dots (same positions), but **different colors**:
  cyan (right eye) and red (left eye).
- **Leave cyan untouched**; **modify red only** inside a shape by cutting that region,
  shifting it **horizontally** (disparity), then:
    1) fill the vacated source band with fresh random red dots,
    2) clear the destination band in red before pasting the shifted patch back.
This preserves dot density and avoids monocular “giveaway” cues.

Outputs (PNG):
  - *_left_RED.png        : red dots on white (LEFT eye)
  - *_right_CYAN.png      : cyan dots on white (RIGHT eye)
  - *_MERGED_onWhite.png  : both layers drawn on white; optional small global merge
                             offset shows red+cyan dots everywhere (like in class)
                             while the **local** extra shift inside the shape makes depth.

Viewing: red lens LEFT eye, cyan lens RIGHT eye.

Dependencies: numpy, pillow
Install: pip install numpy pillow
"""

import argparse
import math
import os
from typing import Tuple

import numpy as np
from PIL import Image


# -----------------------------
# Shape (square ring)
# -----------------------------
def make_square_ring_mask(H: int, W: int, outer_frac: float, inner_frac: float) -> np.ndarray:
    """
    Centered square ring (True inside the ring, False elsewhere).

    outer_frac, inner_frac are fractions of min(H, W).
    Using a *ring* keeps us within the “not a single simple square/rectangle” rule.
    """
    cy, cx = H // 2, W // 2
    S_outer = int(min(H, W) * outer_frac)
    S_inner = int(min(H, W) * inner_frac)

    Y, X = np.ogrid[:H, :W]
    half_o = S_outer // 2
    outer = (np.abs(Y - cy) <= half_o) & (np.abs(X - cx) <= half_o)

    if S_inner <= 0:
        inner = np.zeros((H, W), dtype=bool)
    else:
        half_i = S_inner // 2
        inner = (np.abs(Y - cy) < half_i) & (np.abs(X - cx) < half_i)

    return outer & (~inner)


# -----------------------------
# Cell-grid helpers
# -----------------------------
def mask_to_cells(mask_full: np.ndarray, cell: int) -> np.ndarray:
    """
    Reduce a full-res boolean mask to cell resolution by OR within each cell block.
    A cell is 'inside' if any pixel of that cell is inside.
    """
    H, W = mask_full.shape
    h, w = math.ceil(H / cell), math.ceil(W / cell)
    padH, padW = h * cell - H, w * cell - W
    if padH or padW:
        mask_full = np.pad(mask_full, ((0, padH), (0, padW)), mode="constant", constant_values=False)
    return mask_full.reshape(h, cell, w, cell).max(axis=(1, 3))


def shift_bool_h(arr: np.ndarray, dx_cells: int) -> np.ndarray:
    """
    Shift a boolean array horizontally by integer *cells* (no wrap).
    +dx moves content to the RIGHT.
    """
    out = np.zeros_like(arr, dtype=bool)
    if dx_cells > 0:
        out[:, dx_cells:] = arr[:, :-dx_cells]
    elif dx_cells < 0:
        s = -dx_cells
        out[:, :-s] = arr[:, s:]
    else:
        out = arr.copy()
    return out


# -----------------------------
# Rendering (colored dots on white)
# -----------------------------
def paint_blocks_on_white(small_bool: np.ndarray, H: int, W: int, cell: int,
                          color_rgb: Tuple[int, int, int], x_offset_cells: int = 0) -> np.ndarray:
    """
    Draw filled square 'dots' for True cells, on a WHITE background.
    Optional x_offset_cells lets us shift a whole layer for the *merged* image only.
    """
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    h, w = small_bool.shape
    ox = x_offset_cells * cell
    for i in range(h):
        y0, y1 = i * cell, min((i + 1) * cell, H)
        cols = np.where(small_bool[i])[0]
        for j in cols:
            x0, x1 = j * cell + ox, j * cell + ox + cell
            if x1 <= 0 or x0 >= W:
                continue  # fully off-canvas
            x0c, x1c = max(0, x0), min(W, x1)
            img[y0:y1, x0c:x1c, :] = color_rgb
    return img


def merge_on_white(left_bool: np.ndarray, right_bool: np.ndarray,
                   H: int, W: int, cell: int, merge_offset_cells: int) -> np.ndarray:
    """
    Draw left (red) and right (cyan) on the same white page.
    We paint RED with no offset, then CYAN with a small global horizontal offset
    so both colors are visible across the page (as shown in class).
    """
    merged = paint_blocks_on_white(left_bool, H, W, cell, (255, 0, 0), x_offset_cells=0)
    cyan = paint_blocks_on_white(right_bool, H, W, cell, (0, 255, 255),
                                 x_offset_cells=merge_offset_cells)
    # Cyan on top (simple overwrite is fine for display)
    mask = (cyan != 255).any(axis=2)
    merged[mask] = cyan[mask]
    return merged


# -----------------------------
# Core stereo construction (identical layers; red-only edit)
# -----------------------------
def build_layers_identical_then_edit_red(H: int, W: int, cell: int, density: float,
                                         disparity_px: int, pop: str,
                                         outer_frac: float, inner_frac: float,
                                         rng: np.random.Generator):
    """
    1) Make one Bernoulli(p) field B of dot cells.
    2) CYAN := B (untouched).
       RED  := B (identical to cyan).
    3) Inside the shape mask, apply the red-only cut–shift–fill–clear–paste:
       - Cut patch = RED & M
       - Destination mask = shift(M, dx)
       - Area1 = M & ~dest   (vacated band)  -> fill with fresh red dots
       - Area3 = dest & ~M   (incoming band) -> clear before paste
       - RED[M] = False
       - RED[Area3] = False
       - RED[Area1] = Bernoulli(p)
       - RED[dest]  = shift(patch, dx) restricted to dest
    """
    # Cell grid
    h, w = math.ceil(H / cell), math.ceil(W / cell)

    # Single random field of occupied cells (both eyes share this initially)
    B = rng.random((h, w)) < density

    # Two identical layers
    cyan = B.copy()
    red = B.copy()

    # Shape mask (cells)
    M = mask_to_cells(make_square_ring_mask(H, W, outer_frac, inner_frac), cell)

    # Disparity in *cells* (horizontal only)
    if disparity_px % cell != 0:
        raise ValueError("disparity_px must be a multiple of dot-size (cell) for clean alignment.")
    dx = (disparity_px // cell)
    if pop.upper() == "IN":
        dx = -dx

    # Cut/shift/fill/clear/paste on RED only
    patch = red & M                       # cut
    dest = shift_bool_h(M, dx)            # destination mask
    area1 = M & (~dest)                   # vacated band to fill
    area3 = dest & (~M)                   # incoming band to clear

    # 1) remove original region from red
    red[M] = False
    # 2) clear area3 before paste
    red[area3] = False
    # 3) fill area1 with fresh random red dots
    fills = rng.random(area1.shape) < density
    red[area1] = fills[area1]
    # 4) paste shifted patch
    shifted_patch = shift_bool_h(patch, dx)
    red[dest] = shifted_patch[dest]

    return red, cyan


# -----------------------------
# CLI and main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="RDS with identical cyan/red layers (red-only horizontal edit inside a shape).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--out", type=str, default="rds_outputs_identical", help="Output folder")
    p.add_argument("--width", type=int, default=900, help="Canvas width (pixels)")
    p.add_argument("--height", type=int, default=600, help="Canvas height (pixels)")
    p.add_argument("--dot-size", type=int, default=3, help="Square dot size (pixels per cell)")
    p.add_argument("--density", type=float, default=0.32, help="Dot probability per cell (0..1)")
    p.add_argument("--disparity", type=int, default=12, help="Local horizontal shift INSIDE the shape (pixels)")
    p.add_argument("--pop", type=str, choices=["OUT", "IN"], default="OUT", help="Depth direction (OUT=pop out)")
    p.add_argument("--outer-frac", type=float, default=0.60, help="Square ring outer size / min(H,W)")
    p.add_argument("--inner-frac", type=float, default=0.38, help="Square ring inner size / min(H,W)")
    p.add_argument("--merge-offset", type=int, default=None,
                   help="Small GLOBAL horizontal offset (pixels) for the *merged display only*. "
                        "If None, uses 1 cell; set 0 for no offset.")
    p.add_argument("--seed", type=int, default=30, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    H, W, cell = args.height, args.width, args.dot_size
    if args.merge_offset is None:
        merge_offset = cell  # 1 cell by default (helps show both colors across the page)
    else:
        merge_offset = args.merge_offset

    rng = np.random.default_rng(args.seed)
    red_cells, cyan_cells = build_layers_identical_then_edit_red(
        H=H, W=W, cell=cell, density=args.density,
        disparity_px=args.disparity, pop=args.pop,
        outer_frac=args.outer_frac, inner_frac=args.inner_frac, rng=rng
    )

    # Render per-eye images (no global offset in the per-eye files)
    left_red_img  = paint_blocks_on_white(red_cells,  H, W, cell, (255, 0, 0), x_offset_cells=0)
    right_cyn_img = paint_blocks_on_white(cyan_cells, H, W, cell, (0, 255, 255), x_offset_cells=0)

    # Merged display (white) with small global cyan offset so dots appear as red+cyan speckles
    merged = merge_on_white(red_cells, cyan_cells, H, W, cell, merge_offset_cells=int(round(merge_offset / cell)))

    # Save
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f"squareRing_identical_d{args.disparity}_{args.pop}")

    Image.fromarray(left_red_img).save(base + "_left_RED.png")
    Image.fromarray(right_cyn_img).save(base + "_right_CYAN.png")
    Image.fromarray(merged).save(base + "_MERGED_onWhite.png")

    print("Saved:")
    print("  ", os.path.basename(base + "_left_RED.png"))
    print("  ", os.path.basename(base + "_right_CYAN.png"))
    print("  ", os.path.basename(base + "_MERGED_onWhite.png"))
    print("Folder:", os.path.abspath(args.out))
    print("Viewing: red lens LEFT, cyan lens RIGHT.")


if __name__ == "__main__":
    main()
