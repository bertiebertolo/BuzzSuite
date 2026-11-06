# BuzzSuite — Status (2026-11-06, fresh start)

`D:\BuzzSuite` is gone for good. From 2026-11-06 the main BuzzSuite is **`G:\BuzzSuite-github`**, the git clone of
GitHub `main` (fff0dab, 2026-11-05). The local-only files that lived on D: (config.json, run scripts, logs, the
full DEVLOG/STATUS) were not recovered. Older copies: `G:\BuzzSuite` (code and DEVLOG up to 2026-09-25,
CLAUDE.md from 2026-07-08, copied here as the project rules).

## What is here

- Everything on GitHub `main`, including BuzzPhono (`src/buzzphono`) and BuzzSwarm.
- `yolo_export/` (generator 1.2.0), rebuilt from the Claude Code transcripts and verified against
  `G:\BuzzWatch_LiveDetector\meta\build_report.json`; see `yolo_export/REBUILT.txt`. **Not committed yet.**
- `buzzsuite_cli.py` here is GitHub's, which has no `yolo` command. Call the exporter directly:
  `python -c "from yolo_export import runner; runner.run('build', out_dir=r'G:\BuzzWatch_LiveDetector')"`.

## Live detector (Pi 5 + Hailo-10H)

- Deployed: yolov8s stack v12b (`G:\BuzzWatch_LiveDetector\hailo\stack_v12b`), 175–196 tiles/s on the chip.
- 2026-11-06: gold set for resting boxes (`G:\BuzzWatch_LiveDetector\training\gold_set\label.html`) being labelled;
  six Penzance 720p experiments being exported as aux training data (`training\aux_export.py`) for v13.

## Open

- Commit `yolo_export/` to GitHub (ask first).
- config.json for the app on this machine (start from `config.example.json`).
