# Slide Element Reference for notebook-status Theme

This is the complete element vocabulary supported by the theme CSS and the `md_to_pptx.py` converter. When generating slide markdown, use ONLY these elements to ensure correct PPTX conversion.

## Markdown Header (Required)

Every presentation MUST start with this YAML frontmatter:

```markdown
---
theme: notebook-status
paginate: true
size: 16:9
html: true
---
```

## Slide Separator

Slides are separated by `---` on its own line:

```markdown
---
```

## Title Slide (Lead)

```markdown
<!-- _class: lead -->

# Main Title

Subtitle text goes here. Keep to 1-2 sentences.

<div class="chips">
  <span class="chip">tag 1</span>
  <span class="chip">tag 2</span>
</div>
```

Note: Chips are ONLY allowed on the title (lead) slide.

## Content Slide (Standard)

```markdown
## Slide Heading

- Bullet point one.
- Bullet point two with **bold** and `code`.
- Keep to max 4 bullets per slide.
```

## Image Grids

### 2-column grid

```html
<div class="cols-2">
  <figure class="figure-card">
    <img src="path/to/image1.png">
    <figcaption>Caption for image 1</figcaption>
  </figure>
  <figure class="figure-card">
    <img src="path/to/image2.png">
    <figcaption>Caption for image 2</figcaption>
  </figure>
</div>
```

### 3-column grid

```html
<div class="cols-3">
  <figure class="figure-card"><img src="img1.png"><figcaption>Cap 1</figcaption></figure>
  <figure class="figure-card"><img src="img2.png"><figcaption>Cap 2</figcaption></figure>
  <figure class="figure-card"><img src="img3.png"><figcaption>Cap 3</figcaption></figure>
</div>
```

### 4-image grid (renders as 2x2)

```html
<div class="cols-4">
  <figure class="figure-card"><img src="img1.png"><figcaption>Cap 1</figcaption></figure>
  <figure class="figure-card"><img src="img2.png"><figcaption>Cap 2</figcaption></figure>
  <figure class="figure-card"><img src="img3.png"><figcaption>Cap 3</figcaption></figure>
  <figure class="figure-card"><img src="img4.png"><figcaption>Cap 4</figcaption></figure>
</div>
```

### Single image (centered, full width)

```html
<div class="cols-2">
  <figure class="figure-card"><img src="image.png"><figcaption>Caption</figcaption></figure>
</div>
```

## Tables

```html
<div class="table-box">
<table>
<tr><th>Column 1</th><th>Column 2</th><th>Column 3</th></tr>
<tr><td>Data A</td><td>Data B</td><td>Data C</td></tr>
<tr><td>Data D</td><td>Data E</td><td><strong>Data F</strong></td></tr>
</table>
</div>
```

**Table slides**: NO bullets. Use only a **single context line** (plain text, not a bullet) between the heading and the table.

Note: The converter supports 2-column and 3-column tables.

## Inline Formatting

Within any text element:
- `**bold text**` → bold with darker color
- `` `code text` `` → monospace with teal background
- Standard markdown bullets with `- `

## Content Density Limits

To keep slides readable and ensure proper PPTX conversion:

| Slide Type | Maximum Content |
|------------|-----------------|
| Title (lead) | 1 heading + 1 subtitle + chips |
| Content | 1 heading + 4 bullets + optional images |
| Comparison | 1 heading + 2-4 images with captions |
| Table | 1 heading + 1 context line + table (max 5 rows) |

## Video Support

Videos use the same `<img>` syntax as images — the converter detects video files by extension and embeds them as playable media in the PPTX.

Supported formats: `.mp4`, `.avi`, `.mov`, `.webm`, `.wmv`, `.m4v`, `.mkv`, `.flv`

```html
<div class="cols-2">
  <figure class="figure-card"><img src="outputs/experiment/result.mp4"><figcaption>Training progression</figcaption></figure>
  <figure class="figure-card"><img src="outputs/experiment/baseline.png"><figcaption>Baseline</figcaption></figure>
</div>
```

- Videos are embedded with a poster frame (auto-extracted from middle of video)
- Click to play in PowerPoint presentation mode
- Videos and images can be mixed in the same grid
- Poster frame extraction requires cv2 or ffmpeg (falls back to grey placeholder)

## Media Path Rules

- Use paths relative to the project root directory
- The converter resolves paths via `--base-dir` argument
- Missing images/videos cause the converter to exit with an error
- Images get rounded corners automatically (10px radius)
- Images use `object-fit: contain` — no cropping
- Videos are aspect-ratio preserved and centered in their cell
