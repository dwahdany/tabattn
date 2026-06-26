"""Autonomous autoresearch loop for speeding up TabPFN-3 predict (runs on the pod).

Each iteration:
  1. proposer  (claude-opus-4-8 + read-only code tools) -> ONE output-preserving edit
  2. critic    (claude-opus-4-8) -> adversarial review of the diff (anti-cheat)
  3. referee   (referee.py) -> A/B speedup + calibrated correctness gate -> verdict
  4. notes/frontier/ledger updated; accepted candidates promoted to trunk + recalibrate

The referee is the trusted, mechanical judge; the proposer never runs it and can
only edit model code (architectures/*.py). Needs ANTHROPIC_API_KEY in env (or
/workspace/.anthropic_key).

    python autoloop.py --iters 6 --points typical,sample
"""
import argparse
import json
import os
import re
import subprocess
import sys

import anthropic

MODEL = "claude-opus-4-8"
OUT = "/workspace/out"
LEDGER = f"{OUT}/ledger.jsonl"
NOTES = f"{OUT}/NOTES.md"
FRONTIER = f"{OUT}/frontier.json"
HOTSPOTS = f"{OUT}/hotspots.json"
TRUNK = "master"

TPDIR = subprocess.check_output(
    [sys.executable, "-c", "import tabpfn,os;print(os.path.dirname(tabpfn.__file__))"]
).decode().strip()

if not os.environ.get("ANTHROPIC_API_KEY") and os.path.exists("/workspace/.anthropic_key"):
    os.environ["ANTHROPIC_API_KEY"] = open("/workspace/.anthropic_key").read().strip()
client = anthropic.Anthropic()


def git(*a, check=True):
    return subprocess.run(["git", *a], cwd=TPDIR, check=check,
                          capture_output=True, text=True)


def read(path):
    return open(os.path.join(TPDIR, path)).read()


def safe_path(p):
    full = os.path.realpath(os.path.join(TPDIR, p))
    return full.startswith(os.path.realpath(TPDIR)) and full.endswith(".py")


# --------------------------------------------------------------------------- #
# proposer
# --------------------------------------------------------------------------- #
TOOLS = [
    {"name": "read_file", "description": "Read lines from a source file (relative to the tabpfn package root).",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
         "required": ["path"]}},
    {"name": "grep", "description": "Regex search across the tabpfn package; returns file:line matches.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
    {"name": "submit_proposal",
     "description": "Submit ONE edit. old_str must appear VERBATIM exactly once in the file.",
     "input_schema": {"type": "object", "properties": {
         "file": {"type": "string", "description": "path under architectures/, e.g. architectures/tabpfn_v3.py"},
         "old_str": {"type": "string"}, "new_str": {"type": "string"},
         "hypothesis": {"type": "string", "description": "what & why, incl. expected speed mechanism"},
         "frontier_id": {"type": "string"}},
         "required": ["file", "old_str", "new_str", "hypothesis"]}},
]

PROPOSER_SYS = """You are a performance engineer speeding up TabPFN-3's `predict` on an H100.

GOAL: make the model forward faster while PRESERVING OUTPUTS. A trusted referee will
A/B-time your change vs trunk and check output-equivalence (argmax-agreement + Jensen-
Shannon divergence, each within the model's own run-to-run noise). It ACCEPTS only if
>=1 operating point speeds up >3% (bootstrap-CI) AND no point is confidently slower AND
outputs stay within noise.

EDIT SURFACE: only architectures/*.py (mainly architectures/tabpfn_v3.py). The model
lives there (class TabPFNV3). Do NOT touch anything else.

RULES:
- Exactly ONE focused edit per turn, via submit_proposal. old_str must match VERBATIM, once.
- Output-preserving systems optimizations ONLY: remove redundant copies/casts/.contiguous(),
  fuse elementwise, reduce kernel launches, improve overlap, reuse buffers, torch.compile
  scoped to where it helps. fp-level reordering is fine (the gate tolerates it).
- FORBIDDEN (the critic will reject): changing the math/predictions, changing output shapes,
  caching or special-casing the benchmark inputs, skipping samples/layers, lowering precision
  in a way that changes results, touching the harness/gate/data.
- Read the diagnostics, notes (what works / dead ends), and frontier first. Don't repeat dead ends.
- Explore with read_file/grep, then submit_proposal once. Be surgical and concrete."""

CRITIC_SYS = """You are an adversarial reviewer guarding an autoresearch loop that speeds up
TabPFN-3. Given a unified diff + hypothesis, decide if the change is a legitimate, output-
PRESERVING systems optimization. REJECT if it: changes the model's math so predictions could
differ beyond fp-rounding noise; changes output shapes; caches/special-cases benchmark inputs;
skips work that affects results (dropping samples/layers/heads); lowers precision in a way that
changes predictions; or edits anything outside the model code. fp-level reordering (e.g.
torch.compile, fusion) is ALLOWED. Be strict about cheating, lenient about benign numeric noise."""


def handle_tool(name, inp, captured):
    if name == "read_file":
        p = inp["path"]
        if not safe_path(p):
            return "error: path not allowed"
        try:
            lines = read(p).splitlines()
        except Exception as e:
            return f"error: {e}"
        s = max(0, inp.get("start", 1) - 1)
        e = min(len(lines), inp.get("end", s + 120))
        return "\n".join(f"{i+1}\t{lines[i]}" for i in range(s, e))[:12000]
    if name == "grep":
        r = subprocess.run(["grep", "-rnE", inp["pattern"], "architectures/"],
                           cwd=TPDIR, capture_output=True, text=True)
        return (r.stdout or "(no matches)")[:6000]
    if name == "submit_proposal":
        f = inp["file"]
        if not (f.startswith("architectures/") and safe_path(f)):
            return "error: file must be under architectures/ and .py"
        try:
            content = read(f)
        except Exception as e:
            return f"error: cannot read {f}: {e}"
        n = content.count(inp["old_str"])
        if n != 1:
            return f"error: old_str appears {n} times (must be exactly 1). Re-read and make it unique."
        captured["proposal"] = inp
        return "accepted — evaluating now."
    return "error: unknown tool"


def propose(context):
    captured = {}
    messages = [{"role": "user", "content": context}]
    MAXT = 26
    for t in range(MAXT):
        resp = client.messages.create(
            model=MODEL, max_tokens=12000, system=PROPOSER_SYS,
            thinking={"type": "adaptive"}, output_config={"effort": "high"},
            tools=TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content":
                             "You must call submit_proposal with your single best "
                             "output-preserving edit now."})
            continue
        calls = [b for b in resp.content if b.type == "tool_use"]
        print(f"  turn {t+1}: " + ",".join(c.name for c in calls), flush=True)
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in calls:
            out = handle_tool(b.name, b.input, captured)
            if b.name == "submit_proposal" and "proposal" not in captured:
                print(f"    submit rejected: {out[:90]}", flush=True)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})
        if "proposal" in captured:
            return captured["proposal"]
        if t >= MAXT - 5:  # forcing function: stop exploring, commit to an edit
            messages.append({"role": "user", "content":
                             "Exploration budget nearly exhausted. Do NOT read more. "
                             "Call submit_proposal NOW with your single best edit."})
    return None


def critic(diff, hypothesis):
    resp = client.messages.create(
        model=MODEL, max_tokens=2000, system=CRITIC_SYS,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {
                "approve": {"type": "boolean"}, "reason": {"type": "string"}},
            "required": ["approve", "reason"], "additionalProperties": False}}},
        messages=[{"role": "user", "content":
                   f"Hypothesis: {hypothesis}\n\nDiff:\n{diff[:8000]}"}])
    txt = next(b.text for b in resp.content if b.type == "text")
    return json.loads(txt)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def context_block(points):
    notes = open(NOTES).read()[-4000:] if os.path.exists(NOTES) else ""
    frontier = open(FRONTIER).read() if os.path.exists(FRONTIER) else "{}"
    hot = ""
    if os.path.exists(HOTSPOTS):
        h = json.load(open(HOTSPOTS))
        hot = "\n".join(f"  [{o['bucket']}] {o['ms']}ms x{o['count']} {o['op']}"
                        for o in h.get("by_aten_op", [])[:12])
    led = ""
    if os.path.exists(LEDGER):
        rows = [json.loads(l) for l in open(LEDGER)][-8:]
        led = "\n".join(f"  {r['verdict']:18s} {r['geomean_speedup']} {r['hypothesis'][:60]}"
                        for r in rows)
    return (f"# Diagnostics (top GPU ops, small model)\n{hot}\n\n"
            f"# NOTES (lab notebook)\n{notes}\n\n# frontier.json\n{frontier}\n\n"
            f"# recent ledger\n{led}\n\n"
            f"Propose ONE output-preserving optimization. Operating points: {points}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--points", default="typical,sample")
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()

    for i in range(a.iters):
        print(f"\n{'='*70}\n=== iteration {i+1}/{a.iters} ===\n{'='*70}", flush=True)
        git("checkout", "-q", "-f", TRUNK)
        prop = propose(context_block(a.points))
        if not prop:
            print("no proposal; skipping", flush=True)
            continue
        print(f"PROPOSAL [{prop.get('frontier_id','?')}]: {prop['hypothesis'][:120]}", flush=True)

        # apply on a branch
        branch = f"opt/auto-{i+1}"
        git("branch", "-D", branch, check=False)
        git("checkout", "-q", "-f", "-b", branch)
        f = prop["file"]
        content = read(f)
        open(os.path.join(TPDIR, f), "w").write(
            content.replace(prop["old_str"], prop["new_str"], 1))
        git("add", "-A")
        git("commit", "-q", "-m", f"auto {i+1}: {prop['hypothesis'][:60]}")
        diff = git("show", "--stat", "HEAD").stdout + "\n" + git("show", "HEAD").stdout

        # critic
        verdict_c = critic(diff, prop["hypothesis"])
        print(f"CRITIC: approve={verdict_c['approve']} — {verdict_c['reason'][:140]}", flush=True)
        if not verdict_c["approve"]:
            git("checkout", "-q", "-f", TRUNK)
            log_note(i + 1, prop, "critic-reject", verdict_c["reason"], None)
            continue

        # referee
        git("checkout", "-q", "-f", TRUNK)
        r = subprocess.run([sys.executable, "referee.py", "evaluate", "--ref", branch,
                            "--hypothesis", prop["hypothesis"], "--points", a.points,
                            "--repeats", str(a.repeats)],
                           cwd="/workspace", capture_output=True, text=True)
        rec = json.loads([l for l in open(LEDGER)][-1])
        print(f"REFEREE: {rec['verdict']}  geomean={rec['geomean_speedup']}", flush=True)

        if rec["verdict"] == "accept":
            subprocess.run([sys.executable, "referee.py", "promote", "--ref", branch],
                           cwd="/workspace", check=True)
            subprocess.run([sys.executable, "referee.py", "calibrate", "--points",
                            a.points, "--repeats", "15"], cwd="/workspace",
                           capture_output=True, text=True)
            print(f"  ACCEPTED -> promoted to trunk", flush=True)
        log_note(i + 1, prop, rec["verdict"], verdict_c["reason"], rec)
    print("\nloop complete.", flush=True)


def log_note(i, prop, verdict, critic_reason, rec):
    line = f"\n## auto iteration {i}: {verdict}\n- hypothesis: {prop['hypothesis']}\n"
    if rec:
        line += f"- geomean {rec['geomean_speedup']}; per-point: " + ", ".join(
            f"{p}={v['speedup']}(ci{v['ci']})" for p, v in rec["points"].items()) + "\n"
    if verdict.startswith("critic"):
        line += f"- critic: {critic_reason}\n"
    with open(NOTES, "a") as fh:
        fh.write(line)


if __name__ == "__main__":
    main()
