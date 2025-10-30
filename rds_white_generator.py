#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Dot Stereogram (RDS) generator — colored dots on white
=============================================================

This script creates *anaglyph-style* random dot stereograms for red/cyan glasses,
matching the classroom look you described: **red (left) and cyan/blue (right) dots
splattered on a white page**. No black pixels are used.

Outputs (PNG):
  - Left-eye image:    red dots on white
  - Right-eye image:   cyan dots on white  (and an optional blue variant)
  - Merged image:      union overlay on white (red + cyan/blue appear scattered)
  - (Optional) multiple disparities and a pop-IN version

By default the 3D shape is a **square ring** (a "square donut") so it is *not*
"one simple square/rectangle" per the assignment constraint, while remaining very simple.

Viewing
-------
Use red/cyan anaglyph glasses:
  - Red lens over LEFT eye
  - Cyan lens over RIGHT eye

If the ring appears "inside" instead of popping out, generate a **pop-IN** image
or flip the sign of the disparity.

Dependencies
------------
  - Python 3.9+
  - numpy
  - pillow (PIL)

Install:
  pip install numpy pillow

Quick start
-----------
  python rds_white_generator.py --out rds_outputs \
      --width 900 --height 600 --dot-size 3 --density 0.32 \
      --disparities 6 12 18 --pop OUT --seed 42

This will create, for each disparity, three files like:
  squareRing_d12_OUT_left_RED.png
  squareRing_d12_OUT_right_CYAN.png
  squareRing_d12_OUT_MERGED_redCyan.png

Notes on correctness & rubric alignment
---------------------------------------
- **Random dots**: Dots are sampled from a Bernoulli(p) field on a coarse grid.
- **Horizontal shifts only**: Inside the shape *only*, the right-eye dots are a
  horizontal shift of the left-eye dots by `disparity_px` (in units of pixels).
- **Depth from disparity**: Outside the shape, the two eye patterns are independent
  (default) so the shape is effectively invisible monocularly.
- **Duplicate layer option**: If your grader wants "the second layer is a duplicate
  of the first", set `--duplicate-background` which makes the outside-of-shape
  patterns identical. The merged page then shows overlapping colors there; the
  shape can look slightly easier to spot without glasses, so the default keeps them independent.

Implementation outline
----------------------
1. Define a shape mask (square ring) at full resolution.
2. Downsample the mask to a grid of "dot cells" (each cell becomes one colored square).
3. Sample a Bernoulli(p) map B indicating which cells should contain *a* dot (either eye).
4. Partition B randomly into disjoint left/right maps (L, R) outside the shape (or duplicate if requested).
5. Inside the shape, enforce stereo correspondence by taking the left-eye dots and **shifting them horizontally**
   to the right-eye (or the opposite direction for pop-IN). Keep only pairs that remain inside the shape.
6. Lightly fill any leftover inside-shape cells so that the *union* density remains ~uniform (reduces monocular cues).
7. Render left and right as colored square dots on a **white** canvas; overlay them to produce the merged image.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image


# -----------------------------
# Shape generation
# -----------------------------

def make_square_ring_mask(H: int, W: int, outer_frac: float, inner_frac: float) -> np.ndarray:
    """
    Return a boolean mask for a centered square ring.

    Parameters
    ----------
    H, W : int
        Full-resolution image height and width in pixels.
    outer_frac : float
        Outer square side = outer_frac * min(H, W).
    inner_frac : float
        Inner square side = inner_frac * min(H, W). If 0, this becomes a filled square
        (not recommended per assignment).

    Returns
    -------
    mask : (H, W) bool
        True for pixels inside the ring (outer minus inner), False elsewhere.
    """
    cy, cx = H // 2, W // 2
    S_outer = int(min(H, W) * outer_frac)
    S_inner = int(min(H, W) * inner_frac)

    # Build outer/inner as axis-aligned "diamonds" (squares)
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
# Grid / cell utilities
# -----------------------------

def mask_to_cells(mask_full: np.ndarray, block: int) -> np.ndarray:
    """
    Downsample a full-res boolean mask to "dot cell" resolution by OR-reducing within each block.

    Each dot is rendered as a block of size `block` x `block` pixels. We reduce the full-resolution mask
    so that a cell is considered "inside-shape" if *any* pixel of that cell is inside.

    Parameters
    ----------
    mask_full : (H, W) bool
    block : int

    Returns
    -------
    (h_cells, w_cells) bool
    """
    H, W = mask_full.shape
    h_cells, w_cells = math.ceil(H / block), math.ceil(W / block)
    # Pad to exact multiples for easy reshape
    padH = h_cells * block - H
    padW = w_cells * block - W
    if padH or padW:
        mask_full = np.pad(mask_full, ((0, padH), (0, padW)), mode="constant", constant_values=False)
    # OR within each block
    mask_cells = mask_full.reshape(h_cells, block, w_cells, block).max(axis=(1, 3))
    return mask_cells


def shift_bool_h(arr: np.ndarray, dx_cells: int) -> np.ndarray:
    """
    Shift a boolean 2D array horizontally by an integer number of *cells* (no wrap).

    Positive dx shifts content to the RIGHT (i.e., columns increase).
    Cells that would flow out are discarded; new cells are False.

    Returns a new array.
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
# Rendering helpers
# -----------------------------

def paint_blocks_rgb(small_bool: np.ndarray, H: int, W: int, block: int, color_rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Render colored square "dots" on a WHITE background.

    Parameters
    ----------
    small_bool : (h_cells, w_cells) bool
        True indicates "paint a dot in this cell".
    H, W : int
        Full-resolution size in pixels.
    block : int
        Size of each dot cell (square), in pixels.
    color_rgb : tuple
        RGB color of the dot, e.g., (255, 0, 0) for red.

    Returns
    -------
    img : (H, W, 3) uint8
        White background with colored squares where small_bool is True.
    """
    img = np.full((H, W, 3), 255, dtype=np.uint8)  # white background
    h_cells, w_cells = small_bool.shape

    # Paint each occupied cell as a filled block
    for i in range(h_cells):
        y0, y1 = i * block, min((i + 1) * block, H)
        row = small_bool[i]
        cols = np.where(row)[0]
        for j in cols:
            x0, x1 = j * block, min((j + 1) * block, W)
            img[y0:y1, x0:x1, :] = color_rgb
    return img


def overlay_union(H: int, W: int, block: int,
                  left_bool: np.ndarray, right_bool: np.ndarray,
                  left_color=(255, 0, 0), right_color=(0, 255, 255)) -> np.ndarray:
    """
    Create the merged image: **red & cyan dots on white**, by drawing the union of left and right.

    Overlaps are possible but rare when the outside-of-shape assignment is disjoint.
    We simply paint left first, then right.
    """
    img = paint_blocks_rgb(left_bool, H, W, block, left_color)
    # Draw right on top
    h_cells, w_cells = right_bool.shape
    for i in range(h_cells):
        y0, y1 = i * block, min((i + 1) * block, H)
        row = right_bool[i]
        cols = np.where(row)[0]
        for j in cols:
            x0, x1 = j * block, min((j + 1) * block, W)
            img[y0:y1, x0:x1, :] = right_color
    return img


# -----------------------------
# Core stereo construction
# -----------------------------

def build_stereo_maps(H: int, W: int, dot_size: int, density: float,
                      disparity_px: int, pop: str, outer_frac: float, inner_frac: float,
                      duplicate_background: bool, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct boolean dot maps for left and right eyes (cell resolution).

    The algorithm keeps the merged page looking like "random red & cyan dots on white":
    - We first sample which cells carry *a* dot (B), then randomly assign each occupied cell
      to either the left or right eye (disjoint) outside the shape.
    - Inside the shape, we *pair* dots: right-eye is a pure horizontal shift of left-eye.

    Parameters
    ----------
    H, W : int
        Full-resolution image size in pixels.
    dot_size : int
        Cell size in pixels.
    density : float
        Bernoulli probability for a cell to carry a dot (either eye).
    disparity_px : int
        Horizontal shift inside the shape, in *pixels*. Must be a multiple of `dot_size`.
    pop : {'OUT', 'IN'}
        Direction of depth (OUT = pop-out; IN = pop-in).
    outer_frac, inner_frac : float
        Square ring parameters (fractions of min(H, W)).
    duplicate_background : bool
        If True, make the outside-of-shape patterns identical across eyes. Default False
        (recommended) keeps them independent to hide the shape monocularly.
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    L, R : (h_cells, w_cells) bool
        Left-eye and right-eye dot occupancy maps at cell resolution.
    """
    # Cell grid size
    h_cells, w_cells = math.ceil(H / dot_size), math.ceil(W / dot_size)

    # Convert disparity from pixels to cells
    if disparity_px % dot_size != 0:
        raise ValueError("disparity_px must be a multiple of dot_size for clean cell alignment.")
    dx_cells = disparity_px // dot_size
    dx_cells = int(dx_cells)

    # Shape mask at cell resolution
    mask_full = make_square_ring_mask(H, W, outer_frac, inner_frac)
    mask_cells = mask_to_cells(mask_full, dot_size)

    # Where should there be dots at all? (Either eye.)
    B = rng.random((h_cells, w_cells)) < density

    # Initial disjoint assignment outside shape (looks best in the merged image)
    # If duplicate_background, copy left to right outside the mask.
    S = rng.random((h_cells, w_cells)) < 0.5
    L = B & S
    R = B & (~S)

    if duplicate_background:
        R[~mask_cells] = L[~mask_cells]

    # --- Inside the shape: force stereo correspondence by shifting left to right ---
    # Choose shift direction based on desired depth percept.
    # "Pop-OUT": Right-eye image corresponds to content shifted LEFT relative to left-eye,
    # so the binocular fusion yields a crossed disparity (appears nearer).
    dx = dx_cells if pop.upper() == "OUT" else -dx_cells

    # Take only left dots that remain inside shape after the shift.
    L_in = L & mask_cells
    R_from_L = shift_bool_h(L_in, -dx)  # shift left-eye dots to make the right-eye
    R_from_L &= mask_cells              # keep only targets that are still inside

    # Compute the exact set of left dots that have a valid partner
    L_paired = shift_bool_h(R_from_L, +dx) & mask_cells

    # Overwrite inside-shape occupancy with the paired correspondence
    L[mask_cells] = False
    R[mask_cells] = False
    L[L_paired] = True
    R[R_from_L] = True

    # Optional: keep union density inside shape close to B by filling leftover cells
    target_inside = B & mask_cells
    union_inside = (L | R) & mask_cells
    leftovers = target_inside & (~union_inside)
    # Split leftovers randomly between eyes (small effect; reduces monocular edges)
    left_fill = leftovers & (rng.random(leftovers.shape) < 0.5)
    right_fill = leftovers & (~left_fill)
    L[left_fill] = True
    R[right_fill] = True

    return L, R


def generate_one(out_dir: str, tag: str, H: int, W: int, dot_size: int, density: float,
                 disparity_px: int, pop: str, outer_frac: float, inner_frac: float,
                 duplicate_background: bool, right_blue_variant: bool,
                 rng: np.random.Generator) -> Dict[str, str]:
    """
    Generate one trio (left, right, merged) of images for the given disparity and depth direction.

    Returns a dict of file paths.
    """
    # Build occupancy maps at cell resolution
    L, R = build_stereo_maps(H, W, dot_size, density, disparity_px, pop,
                             outer_frac, inner_frac, duplicate_background, rng)

    # Render each eye on white
    left_rgb = paint_blocks_rgb(L, H, W, dot_size, (255, 0, 0))          # RED
    right_cyan = paint_blocks_rgb(R, H, W, dot_size, (0, 255, 255))      # CYAN
    merged_cyan = overlay_union(H, W, dot_size, L, R,
                                left_color=(255, 0, 0), right_color=(0, 255, 255))

    # Save (red/cyan)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{tag}_d{disparity_px}_{pop.upper()}")
    paths = {}
    paths["left_RED"] = f"{base}_left_RED.png"
    paths["right_CYAN"] = f"{base}_right_CYAN.png"
    paths["merged_redCyan"] = f"{base}_MERGED_redCyan.png"
    Image.fromarray(left_rgb).save(paths["left_RED"])
    Image.fromarray(right_cyan).save(paths["right_CYAN"])
    Image.fromarray(merged_cyan).save(paths["merged_redCyan"])

    # Optional BLUE variant for the right eye
    if right_blue_variant:
        right_blue = paint_blocks_rgb(R, H, W, dot_size, (0, 0, 255))  # BLUE
        merged_blue = overlay_union(H, W, dot_size, L, R,
                                    left_color=(255, 0, 0), right_color=(0, 0, 255))
        paths["right_BLUE"] = f"{base}_right_BLUE.png"
        paths["merged_redBlue"] = f"{base}_MERGED_redBlue.png"
        Image.fromarray(right_blue).save(paths["right_BLUE"])
        Image.fromarray(merged_blue).save(paths["merged_redBlue"])

    return paths


# -----------------------------
# Main / CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Random Dot Stereogram generator (colored dots on white). "
                    "Use red lens LEFT, cyan lens RIGHT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--out", type=str, default="rds_outputs", help="Output folder")
    p.add_argument("--width", type=int, default=900, help="Canvas width (pixels)")
    p.add_argument("--height", type=int, default=600, help="Canvas height (pixels)")
    p.add_argument("--dot-size", type=int, default=3, help="Square dot size (pixels per cell)")
    p.add_argument("--density", type=float, default=0.32, help="Bernoulli probability for a cell to contain a dot")
    p.add_argument("--disparities", type=int, nargs="+", default=[12], help="List of disparities in pixels (multiples of dot-size)")
    p.add_argument("--pop", type=str, choices=["OUT", "IN"], default="OUT", help="Depth direction: OUT = pop-out, IN = pop-in")
    p.add_argument("--outer-frac", type=float, default=0.60, help="Outer square side / min(H, W)")
    p.add_argument("--inner-frac", type=float, default=0.38, help="Inner square side / min(H, W) (0 = filled square)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--duplicate-background", action="store_true",
                   help="Outside the shape, make the two layers identical (duplicate). Default is independent backgrounds.")
    p.add_argument("--right-blue-variant", action="store_true",
                   help="Also output right-eye BLUE + merged red/blue images (in addition to cyan).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    all_paths = []

    for d in args.disparities:
        # Validate disparity alignment
        if d % args.dot_size != 0:
            raise SystemExit(f"--disparities must be multiples of --dot-size (= {args.dot_size}). Got {d}.")

        pack = generate_one(
            out_dir=args.out,
            tag="squareRing",
            H=args.height,
            W=args.width,
            dot_size=args.dot_size,
            density=args.density,
            disparity_px=d,
            pop=args.pop,
            outer_frac=args.outer_frac,
            inner_frac=args.inner_frac,
            duplicate_background=args.duplicate_background,
            right_blue_variant=args.right_blue_variant,
            rng=rng
        )
        all_paths.append(pack)

    # Print a compact summary for the console
    print("Saved files:")
    for pack in all_paths:
        for k, v in pack.items():
            print(f"  {os.path.basename(v)}")
    print(f"\nOutput folder: {os.path.abspath(args.out)}")
    print("Viewing: red lens LEFT, cyan lens RIGHT.")


if __name__ == "__main__":
    main()
