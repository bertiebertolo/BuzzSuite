# ffmpeg binaries

This folder is intentionally empty in the repository — ffmpeg/ffprobe are ~210 MB each,
well past GitHub's per-file limit, so they aren't committed.

`h264_converter/convert_gui.py`'s `find_tool()` looks for the binaries in this order:

1. `ffmpeg/bin/win/ffmpeg.exe` + `ffprobe.exe` (Windows) or `ffmpeg/bin/mac/ffmpeg` + `ffprobe`
   (macOS), relative to the app — drop platform binaries here if you want a self-contained copy.
2. Falls back to `ffmpeg`/`ffprobe` on your system `PATH`.

**Easiest setup:** install ffmpeg system-wide and make sure it's on `PATH`.

- **Windows:** [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/) or `winget install ffmpeg`.
- **macOS:** `brew install ffmpeg`.

If you'd rather bundle a portable copy instead, download a static build for your platform and
place `ffmpeg`/`ffprobe` (or `ffmpeg.exe`/`ffprobe.exe` on Windows) in `bin/win/` or `bin/mac/`
here.
