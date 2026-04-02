#!/usr/bin/env python3
"""Show per-session token usage. Scans transcripts directly, caches in sessions.jsonl."""
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

CACHE_PATH = os.path.expanduser("~/.claude/token-tracking/sessions.jsonl")
PROJECTS_DIR = os.path.expanduser("~/.claude/projects")


def fmt_tok(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


def parse_ts(ts):
    if isinstance(ts, (int, float)):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000
        except ValueError:
            return None
    return None


def extract_session_name(transcript_path):
    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "user":
                continue
            msg = entry.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(content, list):
                text = " ".join(
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            else:
                text = str(content)
            clean = re.sub(r"<[^>]+>", "", text).strip()
            clean = re.sub(r"\s+", " ", clean)
            if not clean or len(clean) < 5:
                continue
            if any(clean.startswith(s) for s in ("Caveat:", "Set model", "/")):
                continue
            return clean[:100]
    return ""


def process_transcript(fpath):
    session_id = os.path.basename(fpath).replace(".jsonl", "")
    totals = {"input_tokens": 0, "output_tokens": 0,
              "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    api_calls = 0
    model = "unknown"
    first_ts = last_ts = None
    project = ""

    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_ts(entry.get("timestamp"))
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
            if not project and entry.get("cwd"):
                project = entry["cwd"]
            if entry.get("type") == "assistant":
                msg = entry.get("message", {})
                usage = msg.get("usage", {})
                if usage:
                    api_calls += 1
                    for k in totals:
                        totals[k] += usage.get(k, 0)
                if msg.get("model"):
                    model = msg["model"]

    if api_calls == 0:
        return None

    duration_min = 0
    if first_ts and last_ts:
        duration_min = round((last_ts - first_ts) / 60000, 1)

    session_time = datetime.now().isoformat()
    if first_ts:
        session_time = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc).isoformat()

    return {
        "session_id": session_id,
        "timestamp": session_time,
        "project": project,
        "model": model,
        "session_name": extract_session_name(fpath),
        "api_calls": api_calls,
        "duration_min": duration_min,
        **totals,
        "total_tokens": sum(totals.values()),
    }


def refresh_cache():
    """Scan for new transcripts not yet in cache."""
    cached = set()
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cached.add(json.loads(line)["session_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    new = 0
    for fpath in glob.glob(os.path.join(PROJECTS_DIR, "*", "*.jsonl")):
        if "/subagents/" in fpath:
            continue
        sid = os.path.basename(fpath).replace(".jsonl", "")
        if sid in cached:
            continue
        record = process_transcript(fpath)
        if record:
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
            new += 1
    return new


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    cutoff = datetime.now().astimezone() - timedelta(days=days)

    refresh_cache()

    if not os.path.exists(CACHE_PATH):
        print("No sessions logged yet.")
        return

    sessions = []
    with open(CACHE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ts = datetime.fromisoformat(r["timestamp"])
            if ts.tzinfo is None:
                ts = ts.astimezone()
            if ts >= cutoff:
                sessions.append(r)

    if not sessions:
        print(f"No sessions in the last {days} days.")
        return

    # "Billable" = input + cache_creation + output (cache reads are free)
    for s in sessions:
        s["billable"] = (s["input_tokens"] + s["cache_creation_input_tokens"]
                         + s["output_tokens"])

    sessions.sort(key=lambda x: x["billable"], reverse=True)

    total_billable = sum(s["billable"] for s in sessions)
    total_out = sum(s["output_tokens"] for s in sessions)
    total_cache_read = sum(s["cache_read_input_tokens"] for s in sessions)

    print(f"{len(sessions)} sessions, {days}d")
    print(f"Billable: {fmt_tok(total_billable)} (input+cache_create+output)")
    print(f"Cache reads (free): {fmt_tok(total_cache_read)}")
    print(f"Output: {fmt_tok(total_out)}")
    print()
    print("| # | Billable | Out | When | Project | Session |")
    print("|--:|--------:|----:|------|---------|---------|")

    for i, s in enumerate(sessions, 1):
        proj = s["project"].rstrip("/").split("/")[-1]
        when = datetime.fromisoformat(s["timestamp"]).strftime("%m/%d %H:%M")
        name = s.get("session_name", "") or "-"
        print(f"| {i} | {fmt_tok(s['billable'])} | {fmt_tok(s['output_tokens'])} | {when} | {proj} | {name} |")


if __name__ == "__main__":
    main()
