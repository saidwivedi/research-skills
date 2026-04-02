# research-collaborator

A Claude Code skill that guardrails your research workflow. It encodes the principles that experienced researchers follow and applies them before you spend the GPU hours.

## Why

Good research advice exists: Karpathy wrote down how to train neural networks. Lipton & Steinhardt catalogued how ML papers fool themselves. Kapoor & Narayanan mapped out every way data leaks. But it's hard to remember all that at 2AM when you are convinced your idea just needs one more run. This skill puts that knowledge in the loop: checks your hypothesis before you commit, catches known bugs before you blame the idea, and calls out sloppy methodology before a reviewer does.

## What It Does

The skill doesn't force a workflow — it figures out what you need from context. Whether you bring an idea, results, or a bug, it applies 10 behavioral overrides that change how Claude works:

- Kill criteria on every hypothesis before any experiment
- Searches 3+ query variations (including negative results) instead of relying on training knowledge
- Cheapest killing test first — never full-scale when a toy version can falsify
- Equal scrutiny for positive and negative results
- HP attribution test before trusting any improvement
- Per-class/per-component metric breakdowns to catch hidden failures
- Silent bug audit against 191 architecture-specific bugs
- Uses parallel agents to maximize throughput

## Usage

```
/research-collaborator
```

Or just talk:

```
"I have an idea for using score-guided diffusion to steer motion generation"
"My transformer's attention maps look uniform after layer 4"
"Training loss plateaus at 0.3 and won't go lower"
"Are these results solid or am I missing something?"
```

## Files

| File | Purpose |
|------|---------|
| `SKILL.md` | Behavioral overrides and methodology |
| `silent-bugs.md` | 191 silent bugs, 16 tiers, 13+ architectures |
| `sources.md` | Bibliography |
