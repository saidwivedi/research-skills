# results-to-slides

A Claude Code skill that handles the grunt work of making research presentations. Discovers experiments from git history and output folders, collects images and metrics, and generates editable PowerPoint presentations (.pptx).

## What This Does

Every week you have the same chore: dig through output folders, copy the right images, put numbers in tables, make it all look decent. This skill does that for you.

The skill will:
1. Ask about slide count (5/10/20) and background theme (light/warm/dark)
2. Discover experiments from git commits and output folders in your date range
3. Collect images, metrics, and logs
4. Build a slide-by-slide script for your approval
5. Generate Marp markdown + editable PowerPoint presentation

## Usage

```
/results-to-slides 0301 0308
```

Output: `presentation/YYYY_MM_DD/slides.md` + `slides_MMDD_HHMM_theme.pptx`

## Themes

| Theme | Description |
|-------|-------------|
| Light | Clean white background with subtle accent bar |
| Warm  | Off-white, slightly warm tone |
| Dark  | Near-black dark grey with muted accent bar |

## PowerPoint Converter

```bash
python md_to_pptx.py --input slides.md --output slides.pptx \
  --bg-image backgrounds/slide_bg_light.png --theme light --base-dir /path/to/project
```

Supports themed backgrounds, rounded cards with shadows, responsive image grids, native tables, and video embedding with poster frames.

## Files

| File | Purpose |
|------|---------|
| `SKILL.md` | Skill definition (full workflow) |
| `md_to_pptx.py` | Markdown → PowerPoint converter |
| `generate_background.py` | Background image generator |
| `slide_reference.md` | Supported slide elements |
| `theme.css` | CSS theme (Marp preview) |
| `backgrounds/` | Pre-rendered background PNGs |

## Dependencies

```bash
pip install python-pptx Pillow lxml
# Optional (video poster frames): opencv-python or ffmpeg
```

## Credits

Inspired by [frontend-slides](https://github.com/zarazhangrui/frontend-slides) by [@zarazhangrui](https://github.com/zarazhangrui).
