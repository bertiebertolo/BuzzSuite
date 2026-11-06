"""Rebuild D:\\BuzzSuite\\yolo_export on disk by replaying, in time order, the Write/Edit/MultiEdit calls and the
shell commands that edited it (paths remapped to OUT). Usage: python replay2.py OUT"""
import json
import re
import subprocess
import sys
from pathlib import Path

PROJ = Path(r"C:\Users\alber\.claude\projects")
OUT = Path(sys.argv[1]).resolve()
WANT = re.compile(r"(?i)^d:[\\/]+buzzsuite[\\/]+(yolo_export[\\/].*)$")
SHELL_IDS = {"toolu_01XzBg3JCJQgTrJLbJMKJRAh", "toolu_01QCQHQmB9yWELSKqGbokgr2", "toolu_01VYGDKz3wUSHoQZ7YnQH6Ra",
             "toolu_01Rq1x6RKCLtft8npoLpDPjw", "toolu_016zaHqMr1doh2RTbtmuZgML",
             "toolu_01KegqUkHz6p5srdyBeLzF7d"}
BASH = r"C:\Program Files\Git\bin\bash.exe"

events, errored = {}, set()
for f in PROJ.glob("**/*.jsonl"):
    for line in f.open(encoding="utf-8", errors="replace"):
        if '"tool_result"' in line and '"is_error":true' in line.replace(" ", ""):
            for c in (json.loads(line).get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("is_error"):
                    errored.add(c.get("tool_use_id"))
        if '"tool_use"' not in line or "yolo_export" not in line:
            continue
        d = json.loads(line)
        ts = d.get("timestamp")
        for c in (d.get("message") or {}).get("content") or []:
            if not isinstance(c, dict) or c.get("type") != "tool_use":
                continue
            inp = c.get("input") or {}
            ok = (c["name"] in ("Write", "Edit", "MultiEdit") and WANT.match(str(inp.get("file_path", "")))) \
                or c["id"] in SHELL_IDS
            if ok and (c["id"] not in events or not events[c["id"]][0]):
                events[c["id"]] = (ts, c["name"], inp)

shell_err = {i for i in SHELL_IDS if i in errored}
fails = []
posix = "/" + OUT.drive[0].lower() + OUT.as_posix()[2:]
for tid, (ts, name, inp) in sorted(events.items(), key=lambda t: t[1][0] or ""):
    if tid in errored and tid not in SHELL_IDS:
        continue
    if name in ("Bash", "PowerShell"):
        cmd = inp["command"]
        win = str(OUT)
        cmd = cmd.replace("D:\\\\BuzzSuite\\\\yolo_export", win.replace("\\", "\\\\") + "\\\\yolo_export")
        cmd = cmd.replace("D:\\BuzzSuite\\yolo_export", win + "\\yolo_export")
        cmd = cmd.replace("cd /d/BuzzSuite", "cd " + posix)
        r = subprocess.run([BASH, "-c", cmd], cwd=str(OUT), capture_output=True, text=True)
        print("SHELL", ts, tid, "(errored originally)" if tid in shell_err else "", "rc", r.returncode,
              (r.stdout + r.stderr).strip()[:400])
        continue
    rel = WANT.match(inp["file_path"]).group(1).replace("\\", "/")
    dst = OUT / rel
    if name == "Write":
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(inp.get("content", "").encode("utf-8"))
        continue
    if not dst.exists():
        fails.append((ts, name, rel, "no base"))
        continue
    text = dst.read_bytes().decode("utf-8").replace("\r\n", "\n")
    for e in (inp.get("edits") if name == "MultiEdit" else [inp]):
        old, new = e.get("old_string", ""), e.get("new_string", "")
        if old not in text:
            fails.append((ts, name, rel, old[:100]))
            continue
        text = text.replace(old, new) if e.get("replace_all") else text.replace(old, new, 1)
    dst.write_bytes(text.encode("utf-8"))
print("%d events, %d failed edits" % (len(events), len(fails)))
for x in fails:
    print("FAIL", x)
