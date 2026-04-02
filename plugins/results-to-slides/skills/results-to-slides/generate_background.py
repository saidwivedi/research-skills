#!/usr/bin/env python3
"""Generate slide background images matching CSS theme gradients using Pillow+numpy.
No Chrome/browser dependency needed."""

import sys
import os
import numpy as np
from PIL import Image


def hex_to_rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]


def generate_warm(path, W=1280, H=720):
    """Render warm paper theme background.

    CSS equivalent:
      background:
        radial-gradient(circle at 18% 12%, rgba(12,124,153,0.10), transparent 36%),
        radial-gradient(circle at 85% 10%, rgba(228,87,46,0.10), transparent 32%),
        linear-gradient(180deg, #fbf7ee 0%, #f5ecd9 58%, #f1e5cf 100%);
      ::before  top 10px gradient bar
      ::after   amber circle accent top-right
    """
    yy = np.arange(H, dtype=np.float64)[:, None]  # (H, 1)
    xx = np.arange(W, dtype=np.float64)[None, :]   # (1, W)

    # 1) Linear gradient: #fbf7ee(0%) -> #f5ecd9(58%) -> #f1e5cf(100%)
    c1 = np.array([251, 247, 238], dtype=np.float64).reshape(1, 1, 3)
    c2 = np.array([245, 236, 217], dtype=np.float64).reshape(1, 1, 3)
    c3 = np.array([241, 229, 207], dtype=np.float64).reshape(1, 1, 3)

    t = (yy / (H - 1))[:, :, None]  # (H, 1, 1)
    t1 = np.clip(t / 0.58, 0, 1)
    t2 = np.clip((t - 0.58) / 0.42, 0, 1)
    mask = (t <= 0.58)
    gradient = mask * (c1 + (c2 - c1) * t1) + (~mask) * (c2 + (c3 - c2) * t2)
    img = np.broadcast_to(gradient, (H, W, 3)).copy()

    # 2) Radial accent blobs
    def add_radial(cx_frac, cy_frac, r_frac, color, opacity):
        cx, cy = cx_frac * W, cy_frac * H
        r = r_frac * max(W, H)
        d = np.sqrt((xx - cx)**2 + (yy - cy)**2) / r
        a = np.clip(opacity * (1 - d), 0, opacity)[:, :, None]
        img[:] = img * (1 - a) + np.array(color, dtype=np.float64) * a

    # Teal blob at 18%, 12%
    add_radial(0.18, 0.12, 0.36, [12, 124, 153], 0.10)
    # Coral blob at 85%, 10%
    add_radial(0.85, 0.10, 0.32, [228, 87, 46], 0.10)

    # Amber circle: CSS width=230px height=230px top=-110px right=-85px
    # Center in px: (W - (-85) - 115, -110 + 115) = (W+85-115, 5)
    cx_px, cy_px = (W + 85 - 115) / W, 5.0 / H
    add_radial(cx_px, cy_px, 115.0 / max(W, H), [244, 162, 89], 0.25)

    # 3) Top gradient bar (10px)
    bar_colors = np.array([
        [228, 87, 46],    # #e4572e
        [12, 124, 153],   # #0c7c99
        [88, 129, 87],    # #588157
        [244, 162, 89],   # #f4a259
    ], dtype=np.float64)
    n_stops = len(bar_colors) - 1
    for x in range(W):
        t_bar = x / max(W - 1, 1)
        seg = t_bar * n_stops
        i = min(int(seg), n_stops - 1)
        f = seg - i
        color = bar_colors[i] + (bar_colors[i + 1] - bar_colors[i]) * f
        img[:10, x] = color

    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    print(f"Generated: {path} ({W}x{H})")


def generate_light(path, W=1280, H=720):
    """Render clean near-white light theme background.

    Neutral, close to white with very subtle teal/coral accents.
    Same accent bar structure for brand consistency.
    """
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]

    # Linear gradient: near-white (#fcfcfb -> #f9f9f8 -> #f6f6f4)
    c1 = np.array([252, 252, 251], dtype=np.float64).reshape(1, 1, 3)
    c2 = np.array([249, 249, 248], dtype=np.float64).reshape(1, 1, 3)
    c3 = np.array([246, 246, 244], dtype=np.float64).reshape(1, 1, 3)

    t = (yy / (H - 1))[:, :, None]
    t1 = np.clip(t / 0.55, 0, 1)
    t2 = np.clip((t - 0.55) / 0.45, 0, 1)
    mask = (t <= 0.55)
    gradient = mask * (c1 + (c2 - c1) * t1) + (~mask) * (c2 + (c3 - c2) * t2)
    img = np.broadcast_to(gradient, (H, W, 3)).copy()

    # Radial accent blobs (muted)
    def add_radial(cx_frac, cy_frac, r_frac, color, opacity):
        cx, cy = cx_frac * W, cy_frac * H
        r = r_frac * max(W, H)
        d = np.sqrt((xx - cx)**2 + (yy - cy)**2) / r
        a = np.clip(opacity * (1 - d), 0, opacity)[:, :, None]
        img[:] = img * (1 - a) + np.array(color, dtype=np.float64) * a

    # Teal blob at 18%, 12% (very subtle)
    add_radial(0.18, 0.12, 0.36, [12, 124, 153], 0.05)
    # Coral blob at 85%, 10% (very subtle)
    add_radial(0.85, 0.10, 0.32, [228, 87, 46], 0.04)

    # Amber circle accent top-right (toned down)
    cx_px, cy_px = (W + 85 - 115) / W, 5.0 / H
    add_radial(cx_px, cy_px, 115.0 / max(W, H), [244, 162, 89], 0.10)

    # Top gradient bar (10px) — same accent colors
    bar_colors = np.array([
        [228, 87, 46],    # #e4572e
        [12, 124, 153],   # #0c7c99
        [88, 129, 87],    # #588157
        [244, 162, 89],   # #f4a259
    ], dtype=np.float64)
    n_stops = len(bar_colors) - 1
    for x in range(W):
        t_bar = x / max(W - 1, 1)
        seg = t_bar * n_stops
        i = min(int(seg), n_stops - 1)
        f = seg - i
        color = bar_colors[i] + (bar_colors[i + 1] - bar_colors[i]) * f
        img[:10, x] = color

    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    print(f"Generated: {path} ({W}x{H})")


def generate_dark(path, W=1280, H=720):
    """Render dark theme background.

    Deep charcoal-navy base with subtle luminous accent blobs.
    Same top gradient bar for brand consistency.
    Font colors should be light (white/light grey) when using this background.
    """
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]

    # Linear gradient: deep charcoal (#1a1d28 -> #1e2234 -> #20253a)
    c1 = np.array([26, 29, 40], dtype=np.float64).reshape(1, 1, 3)
    c2 = np.array([30, 34, 52], dtype=np.float64).reshape(1, 1, 3)
    c3 = np.array([32, 37, 58], dtype=np.float64).reshape(1, 1, 3)

    t = (yy / (H - 1))[:, :, None]
    t1 = np.clip(t / 0.50, 0, 1)
    t2 = np.clip((t - 0.50) / 0.50, 0, 1)
    mask = (t <= 0.50)
    gradient = mask * (c1 + (c2 - c1) * t1) + (~mask) * (c2 + (c3 - c2) * t2)
    img = np.broadcast_to(gradient, (H, W, 3)).copy()

    # Radial accent blobs — luminous against dark
    def add_radial(cx_frac, cy_frac, r_frac, color, opacity):
        cx, cy = cx_frac * W, cy_frac * H
        r = r_frac * max(W, H)
        d = np.sqrt((xx - cx)**2 + (yy - cy)**2) / r
        a = np.clip(opacity * (1 - d), 0, opacity)[:, :, None]
        img[:] = img * (1 - a) + np.array(color, dtype=np.float64) * a

    # Teal glow at 18%, 12%
    add_radial(0.18, 0.12, 0.40, [12, 124, 153], 0.12)
    # Coral/orange glow at 85%, 10%
    add_radial(0.85, 0.10, 0.34, [228, 87, 46], 0.08)
    # Subtle purple glow bottom-right
    add_radial(0.80, 0.85, 0.30, [100, 60, 160], 0.06)
    # Subtle teal glow bottom-left
    add_radial(0.12, 0.80, 0.25, [12, 100, 130], 0.05)

    # Amber circle accent top-right (subtle glow on dark)
    cx_px, cy_px = (W + 85 - 115) / W, 5.0 / H
    add_radial(cx_px, cy_px, 115.0 / max(W, H), [244, 162, 89], 0.15)

    # Top gradient bar (10px) — same accent colors, slightly brighter on dark
    bar_colors = np.array([
        [235, 95, 55],    # brighter coral
        [20, 140, 170],   # brighter teal
        [100, 145, 100],  # brighter sage
        [250, 175, 100],  # brighter amber
    ], dtype=np.float64)
    n_stops = len(bar_colors) - 1
    for x in range(W):
        t_bar = x / max(W - 1, 1)
        seg = t_bar * n_stops
        i = min(int(seg), n_stops - 1)
        f = seg - i
        color = bar_colors[i] + (bar_colors[i + 1] - bar_colors[i]) * f
        img[:10, x] = color

    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    print(f"Generated: {path} ({W}x{H})")


if __name__ == '__main__':
    bg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backgrounds')
    os.makedirs(bg_dir, exist_ok=True)

    if len(sys.argv) > 1:
        variant = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else None
        if variant == 'light':
            generate_light(out or os.path.join(bg_dir, 'slide_bg_light.png'))
        elif variant == 'warm':
            generate_warm(out or os.path.join(bg_dir, 'slide_bg_warm.png'))
        elif variant == 'dark':
            generate_dark(out or os.path.join(bg_dir, 'slide_bg_dark.png'))
        elif variant == 'all':
            generate_light(os.path.join(bg_dir, 'slide_bg_light.png'))
            generate_warm(os.path.join(bg_dir, 'slide_bg_warm.png'))
            generate_dark(os.path.join(bg_dir, 'slide_bg_dark.png'))
        else:
            generate_light(variant)  # treat as output path
    else:
        generate_light(os.path.join(bg_dir, 'slide_bg_light.png'))
