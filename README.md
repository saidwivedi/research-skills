# research-skills

Claude Code skills for AI/ML researchers.

## Demos

<table>
  <tr>
    <td align="center"><strong>Research Collaborator</strong><br><br>
      <video src="https://github.com/user-attachments/assets/f09a0afd-b149-4233-8f65-4d4f193517a1" width="480" controls></video>
    </td>
    <td align="center"><strong>Results to Slides</strong><br><br>
      <video src="https://github.com/user-attachments/assets/5d583ca8-03e7-4740-ad7e-8cfef469e4fc" width="480" controls></video>
    </td>
  </tr>
</table>

## Skills

<table>
  <tr>
    <td width="280"><a href="plugins/research-collaborator/"><b>research-collaborator</b></a></td>
    <td>Guardrails your research workflow. Encodes principles from experienced researchers and applies them before you spend the GPU hours. Checks your hypothesis, catches known bugs and flags sloppy methodology.</td>
  </tr>
  <tr>
    <td width="280"><a href="plugins/results-to-slides/"><b>results-to-slides</b></a></td>
    <td>Discovers experiments from git history and output folders, collects images and metrics, generates editable PowerPoint presentations (.pptx).</td>
  </tr>
</table>

Also includes a utility skill:

<table>
  <tr>
    <td width="280"><a href="plugins/token-usage/"><b>token-usage</b></a></td>
    <td>Per-session token usage breakdown. <code>/token-usage</code> or <code>/token-usage 30</code> for last N days.</td>
  </tr>
</table>

## Installation

Add the marketplace in Claude Code:

```bash
/plugin marketplace add saidwivedi/research-skills
```

Install all plugins:

```bash
/plugin install research-collaborator@saidwivedi-research
/plugin install results-to-slides@saidwivedi-research
/plugin install token-usage@saidwivedi-research
```

Or install just the ones you need.

Skills are invoked via slash commands in Claude Code:
- `/research-collaborator` - start a research collaboration session
- `/results-to-slides 0301 0308` - generate a presentation for experiments from March 1-8
- `/token-usage 30` - show token usage over the last 30 days

## Requirements

**results-to-slides:**
- Python with `python-pptx`, `Pillow`, `lxml`
- Optional: `cv2` or `ffmpeg` for video poster frame extraction
- A git repository with experiment outputs

**research-collaborator:**
- Web search access (for literature search)
- A codebase with experiments to investigate

## License

MIT
