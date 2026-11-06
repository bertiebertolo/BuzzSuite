"""Level-2 simulator: a whole synthetic experiment, in BuzzWatch's own file formats, with known truth.

Thirty simulated mosquitoes rest and fly in a cage filmed by a simulated Raspberry Pi; a tone plays
at minutes 40-50 of every hour. Planted effects (each can be switched off):

* **recruitment**: during the tone (after a short latency, decaying after it), landings are
  ``recruitment`` times more likely to be in the speaker zone;
* **arousal**: take-off rate × ``arousal`` during the tone;
* **directed flight**: flights during the tone end near the speaker more often (``directed``).

Tracker and recording artefacts (each can be switched off): missed detections, static specks
counted as resters, flight tracks broken into fragments, reduced detection in the first minutes
of each chunk (warm-up), identity jumps; a camera whose real frame rate is off by
``drift_s_per_day``, a first frame that comes ``first_frame_delay_s`` after the session stamp,
chunks of ``segment_s × fps + overrun`` frames with a short last one, and tone windows that failed
(recorded as failures in the playback log).

:func:`simulate_experiment` writes an experiment folder BuzzPhono can analyse like a real one, and
returns the ground truth: the true clock, and per hour-cycle the speaker-zone resting fraction
computed from the true positions (no artefacts) with the published definition.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buzzphono.io.naming import ChunkName
from buzzphono.synthetic.writers import write_tracking_file, write_zones
from buzzphono.timebase import NS_PER_HOUR, NS_PER_MIN, NS_PER_S, to_ns


@dataclass(frozen=True)
class L2Design:
    # recording
    width: int = 640
    height: int = 360
    cage: tuple[tuple[float, float], ...] = ((60, 40), (580, 40), (600, 330), (40, 330))
    fps: float = 2.0
    segment_s: int = 1200
    overrun_frames: int = 5
    hours: float = 6.0
    start: datetime = datetime(2026, 1, 10, 12, 0, 9)
    label: str = "SynthCage"
    drift_s_per_day: float = 50.0
    first_frame_delay_s: float = 12.0
    lights_on: str = "05:00"
    lights_off: str = "17:00"
    # behaviour
    colony: int = 30
    rest_mean_s: float = 600.0
    flight_mean_s: float = 12.0
    flight_speed_px_s: float = 300.0
    # stimulus and planted effects
    on_window: tuple[int, int] = (40, 50)
    frequency_hz: float = 475.0
    recruitment: float = 3.0
    arousal: float = 2.0
    directed: float = 2.0
    startle: float = 0.8  # chance a resting mosquito takes off within startle_window_s of the onset
    startle_window_s: float = 5.0
    startle_flight_s: tuple[float, float] = (30.0, 60.0)  # startled flights last this long (uniform)
    effect_latency_s: float = 20.0
    effect_decay_s: float = 300.0
    failed_hours: tuple[int, ...] = ()  # hour indices (0 = first) whose tone failed
    # tracker artefacts
    miss_prob: float = 0.0
    specks: int = 0
    fragment_mean_s: float = 0.0  # 0 = flight tracks never broken
    warmup_s: float = 0.0
    warmup_detect: float = 0.5
    jump_prob: float = 0.0
    camera_shift_px: tuple[int, int] = (0, 0)  # shift of the reference frames in the second half
    zone_size: tuple[int, int] = (36, 70)

    def zones(self) -> dict[str, np.ndarray]:
        """Speaker on the left wall, equal-size controls on the other three walls."""
        w, h = self.zone_size
        cx, cy = 320.0, 185.0
        return {
            "speaker": _rect(62, cy - h / 2, w, h),
            "opposite": _rect(578 - w, cy - h / 2, w, h),
            "top": _rect(cx - h / 2, 42, h, w),
            "bottom": _rect(cx - h / 2, 328 - w, h, w),
        }

    @property
    def nf_full(self) -> int:
        return round(self.segment_s * self.fps) + self.overrun_frames


def _rect(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


@dataclass
class SyntheticTruth:
    design: L2Design
    seed: int
    experiment: Path
    clock_offset_s: float  # model offset a (negative: the first frame comes after the stamp)
    drift_s_per_day: float
    cycles: pd.DataFrame  # per hour-cycle true ON/OFF capped speaker resting fractions
    n_frames: int
    files: list[Path] = field(default_factory=list)


def _hhmm(text: str) -> float:
    h, m = text.split(":")
    return int(h) + int(m) / 60.0


def _in_cage(rng: np.random.Generator, cage: np.ndarray, n: int) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    path = MplPath(cage)
    lo, hi = cage.min(axis=0) + 4, cage.max(axis=0) - 4
    out = np.empty((0, 2))
    while len(out) < n:
        cand = rng.uniform(lo, hi, size=(2 * n, 2))
        out = np.vstack([out, cand[path.contains_points(cand)]])
    return out[:n]


def _flight_path(
    rng: np.random.Generator,
    start: np.ndarray,
    target: np.ndarray,
    n: int,
    design: L2Design,
    cage: np.ndarray,
) -> np.ndarray:
    """A flight of ``n`` frames from ``start`` to ``target`` at roughly the design flight speed.

    The heading wanders (correlated random walk) and bounces off the cage's bounding box; over the
    last third of the flight the mosquito homes in on its landing site, where the last frame is.
    """
    lo, hi = cage.min(axis=0) + 2, cage.max(axis=0) - 2
    speed = design.flight_speed_px_s * rng.uniform(0.7, 1.3) / design.fps
    heading = rng.uniform(0, 2 * np.pi)
    p = start.astype(float).copy()
    out = np.empty((n, 2))
    for i in range(n):
        frac = (i + 1) / n
        to_target = target - p
        dist = float(np.hypot(*to_target))
        heading += rng.normal(0, 0.5)
        v = speed * np.array([np.cos(heading), np.sin(heading)])
        w = max(0.0, (frac - 2 / 3) * 3) ** 2
        if dist > 0:
            v = (1 - w) * v + w * to_target / dist * min(speed, dist)
        p = p + v
        for d in (0, 1):  # bounce off the walls
            if p[d] < lo[d] or p[d] > hi[d]:
                p[d] = np.clip(p[d], lo[d], hi[d])
                heading = np.pi - heading if d == 0 else -heading
        out[i] = p
    out[-1] = target
    return out


def simulate_experiment(
    design: L2Design,
    seed: int,
    out_dir: Path,
    *,
    write_files: bool = True,
    tiers: tuple[str, ...] = ("resting", "moving", "tracks"),
) -> SyntheticTruth:
    """Simulate one experiment into ``out_dir`` (created) and return the truth.

    ``tiers`` limits which tracking files are written (for fast tests that need only some).
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    cage = np.asarray(design.cage, dtype=float)
    zones = design.zones()
    from matplotlib.path import Path as MplPath

    speaker = MplPath(zones["speaker"])
    fps = design.fps
    b = design.drift_s_per_day
    a = -design.first_frame_delay_s
    n_frames = round(design.hours * 3600 * fps)
    t_rec = np.arange(n_frames) / fps
    wall_s = t_rec - a - b * t_rec / 86400.0  # seconds after the session stamp, Pi clock
    start = design.start
    # Same integer-nanosecond arithmetic as buzzphono.timebase.CorrectedClock, so a frame falls in
    # exactly the same minute here and in the analysis.
    start_ns = to_ns(start)
    wall_ns = start_ns + np.round(wall_s * NS_PER_S).astype(np.int64)
    minute_of_day = (wall_ns // NS_PER_MIN) % 1440
    hour_index = ((wall_ns - wall_ns % NS_PER_HOUR) - (start_ns - start_ns % NS_PER_HOUR)) // NS_PER_HOUR
    moh = (wall_ns // NS_PER_MIN) % 60
    failed = set(design.failed_hours)
    tone = (moh >= design.on_window[0]) & (moh < design.on_window[1]) & ~np.isin(hour_index, list(failed))
    # effect envelope: on after a latency, decays exponentially after the tone ends
    lat = round(design.effect_latency_s * fps)
    env = np.zeros(n_frames)
    on_idx = np.flatnonzero(tone)
    env[np.clip(on_idx + lat, 0, n_frames - 1)] = 1.0
    decay = np.exp(-1.0 / max(design.effect_decay_s * fps, 1e-9))
    for i in range(1, n_frames):
        if env[i] < env[i - 1] * decay:
            env[i] = env[i - 1] * decay
    day = (minute_of_day >= round(_hhmm(design.lights_on) * 60)) & (
        minute_of_day < round(_hhmm(design.lights_off) * 60)
    )
    onsets = np.flatnonzero(tone[1:] & ~tone[:-1]) + 1 + lat  # first frame of each tone, plus the latency

    # --- behaviour: per mosquito, alternate resting and flight bouts on the frame grid
    rest_sites = _in_cage(rng, cage, 4000)
    in_spk = speaker.contains_points(rest_sites)
    spk_sites, other_sites = rest_sites[in_spk], rest_sites[~in_spk]
    if len(spk_sites) < 5:
        spk_sites = _in_cage(rng, zones["speaker"], 200)
    state = np.zeros((design.colony, n_frames), dtype=np.int8)
    pos = np.zeros((design.colony, n_frames, 2))
    base_p_spk = 0.04
    for m in range(design.colony):
        f = 0
        here = other_sites[rng.integers(len(other_sites))]
        while f < n_frames:
            rate = (1.0 / design.rest_mean_s) * (1.5 if not day[f] else 1.0)
            rate *= 1 + (design.arousal - 1) * env[f]
            dur = max(1, int(rng.exponential(1.0 / rate) * fps))
            end = min(n_frames, f + dur)
            later = onsets[(onsets > f) & (onsets < end)]
            startled = bool(later.size) and rng.random() < design.startle
            if startled:
                end = min(end, int(later[0] + rng.uniform(0, design.startle_window_s) * fps) + 1)
            state[m, f:end] = 0
            pos[m, f:end] = here
            f = end
            if f >= n_frames:
                break
            if startled:
                fdur = int(rng.uniform(*design.startle_flight_s) * fps)
            else:
                # aroused mosquitoes also fly for longer, so the tone raises flight for minutes
                fmean = design.flight_mean_s * (1 + (design.arousal - 1) * env[f])
                fdur = max(int(2 * fps), int(rng.exponential(fmean) * fps))
            fend = min(n_frames, f + fdur)
            p_spk = (
                base_p_spk * (1 + (design.recruitment - 1) * env[f]) * (1 + (design.directed - 1) * env[f])
            )
            p_spk = min(p_spk, 0.9)
            land = (
                spk_sites[rng.integers(len(spk_sites))]
                if rng.random() < p_spk
                else other_sites[rng.integers(len(other_sites))]
            )
            pos[m, f:fend] = _flight_path(rng, here, land, fend - f, design, cage)
            state[m, f:fend] = 1
            here = land
            f = fend

    # --- truth per hour-cycle (published definition, true positions, no artefacts)
    resting = state == 0
    in_zone = np.zeros_like(resting)
    for m in range(design.colony):
        in_zone[m] = resting[m] & speaker.contains_points(pos[m])
    tot = np.minimum(resting.sum(axis=0), design.colony).astype(float)
    inz = np.minimum(in_zone.sum(axis=0), tot)
    rows = []
    for h in np.unique(hour_index):
        sel = hour_index == h
        on = sel & (moh >= design.on_window[0]) & (moh < design.on_window[1])
        off = sel & (moh >= 28) & (moh < 38)
        if on.sum() < 0.9 * 600 * fps or off.sum() < 0.9 * 600 * fps:
            continue
        hour_start = start.replace(minute=0, second=0) + timedelta(hours=int(h))
        rows.append(
            {
                "cycle_start_ns": int(start_ns - start_ns % NS_PER_HOUR + int(h) * NS_PER_HOUR),
                "cycle_start": hour_start,
                "hour": hour_start.hour,
                "tone_played": int(h) not in failed,
                "off_frac": float(inz[off].sum() / tot[off].sum()) if tot[off].sum() else np.nan,
                "on_frac": float(inz[on].sum() / tot[on].sum()) if tot[on].sum() else np.nan,
                "flyers_off": float(state[:, off].sum(axis=0).mean()),
                "flyers_on": float(state[:, on].sum(axis=0).mean()),
            }
        )
    truth = SyntheticTruth(design, seed, out_dir, a, b, pd.DataFrame(rows), n_frames)
    if write_files:
        truth.files = _write_experiment(
            design, rng, out_dir, state, pos, day, tone, hour_index, wall_s, zones, tiers
        )
    return truth


def _write_experiment(
    design: L2Design,
    rng: np.random.Generator,
    out_dir: Path,
    state: np.ndarray,
    pos: np.ndarray,
    day: np.ndarray,
    tone: np.ndarray,
    hour_index: np.ndarray,
    wall_s: np.ndarray,
    zones: dict[str, np.ndarray],
    tiers: tuple[str, ...],
) -> list[Path]:
    n_frames = state.shape[1]
    nf = design.nf_full
    stamp = design.start
    files: list[Path] = []
    specks = (
        _in_cage(rng, np.asarray(design.cage, float), design.specks) if design.specks else np.empty((0, 2))
    )
    for k in range(int(np.ceil(n_frames / nf))):
        g0, g1 = k * nf, min(n_frames, (k + 1) * nf)
        chunk = ChunkName(design.label, stamp.strftime("%Y%m%d"), stamp.strftime("%H%M%S"), None, k)
        frames = np.arange(g1 - g0)
        detect = np.ones(g1 - g0)
        if design.warmup_s > 0:
            detect[frames < design.warmup_s * design.fps] = design.warmup_detect
        resting_objs: dict[int, dict[str, Any]] = {}
        moving_objs: dict[int, dict[str, Any]] = {}
        final_objs: dict[int, dict[str, object]] = {}
        oid = 0
        for m in range(state.shape[0]):
            st = state[m, g0:g1]
            xy = pos[m, g0:g1]
            seen = (rng.random(g1 - g0) >= design.miss_prob) & (rng.random(g1 - g0) < detect)
            # final (assembled) track: the whole chunk, state per frame, NaN where missed
            coords = [
                (float(x), float(y)) if s else (np.nan, np.nan) for (x, y), s in zip(xy, seen, strict=True)
            ]
            final_objs[m] = {
                "start": 0,
                "end": g1 - g0 - 1,
                "state": [float(v) for v in st],
                "coordinates": coords,
                "start_boot": 0,
                "end_boot": 0,
            }
            # resting objects: one per resting bout; moving objects: flight bouts, maybe fragmented
            change = np.flatnonzero(np.diff(st)) + 1
            for a_i, b_i in zip(np.r_[0, change], np.r_[change, st.size], strict=True):
                keep = np.flatnonzero(seen[a_i:b_i]) + a_i
                if keep.size == 0:
                    continue
                if st[a_i] == 0:
                    resting_objs[oid] = {
                        "time_stamp": keep.tolist(),
                        "coordinates": [tuple(map(float, xy[i])) for i in keep],
                    }
                    oid += 1
                else:
                    pieces = [keep]
                    if design.fragment_mean_s > 0 and keep.size > 4:
                        cuts = (
                            np.flatnonzero(
                                rng.random(keep.size - 1) < 1 / (design.fragment_mean_s * design.fps)
                            )
                            + 1
                        )
                        pieces = np.split(keep, cuts)
                    for piece in pieces:
                        pts = [tuple(map(float, xy[i])) for i in piece]
                        if design.jump_prob > 0:
                            for j in range(len(pts)):
                                if rng.random() < design.jump_prob:
                                    pts[j] = (pts[j][0] + 250.0, pts[j][1])
                        moving_objs[oid] = {
                            "time_stamp": piece.tolist(),
                            "coordinates": pts,
                            "start_type": np.nan,
                            "end_type": np.nan,
                        }
                        oid += 1
        for sxy in specks:
            resting_objs[oid] = {
                "time_stamp": list(range(g1 - g0)),
                "coordinates": [tuple(map(float, sxy))] * (g1 - g0),
            }
            oid += 1
        if "resting" in tiers:
            files.append(write_tracking_file(out_dir, "resting", chunk, resting_objs))
        if "moving" in tiers:
            files.append(write_tracking_file(out_dir, "moving", chunk, moving_objs))
        if "tracks" in tiers:
            times = [stamp + timedelta(seconds=k * 1200 + i / design.fps) for i in range(g1 - g0)]
            files.append(
                write_tracking_file(
                    out_dir,
                    "tracks",
                    chunk,
                    final_objs,
                    time_stamp=times,
                    settings={"fps": design.fps},
                    video_name=chunk.stem,
                )
            )
        files.append(_reference_image(design, out_dir, chunk, bool(day[g0]), k * nf >= n_frames / 2))
    files.append(write_zones(out_dir / "custom_zones.json", zones))
    files.append(_settings(design, out_dir))
    files.append(_pi_config(design, out_dir))
    files.append(_sound_log(design, out_dir, tone, hour_index, wall_s))
    meta = {
        "design": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in asdict(design).items()},
        "note": "synthetic experiment written by buzzphono.synthetic.l2",
    }
    (out_dir / "synthetic_truth.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return files


def _reference_image(design: L2Design, out_dir: Path, chunk: ChunkName, lit: bool, shifted: bool) -> Path:
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(7)  # the same texture in every frame (it is the same cage)
    base = (rng.random((design.height, design.width)) * 30 + (150 if lit else 60)).astype(np.uint8)
    im = Image.fromarray(base)
    draw = ImageDraw.Draw(im)
    cage = [tuple(map(float, p)) for p in design.cage]
    draw.polygon(cage, outline=20 if lit else 5)
    if shifted and design.camera_shift_px != (0, 0):
        dx, dy = design.camera_shift_px
        im = im.transform(im.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy))
    folder = out_dir / "individual_images"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{chunk.stem}.png"
    im.save(path)
    return path


def _settings(design: L2Design, out_dir: Path) -> Path:
    import yaml

    data = {
        "fps": float(design.fps),
        "cage_border_points": [[float(x), float(y)] for x, y in design.cage],
        "track_moving": {"maxdisttracking": 200},
        "tracking": {
            "assignment": "hungarian",
            "motion_model": "constant_velocity",
            "split_blobs": True,
            "use_backward_pass": True,
        },
    }
    path = out_dir / "buzzwatch_track_settings.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _pi_config(design: L2Design, out_dir: Path) -> Path:
    on_h, on_m = design.lights_on.split(":")
    text = f"""[recording]
output_dir = /home/buzzpi/pirecorder/recordings
label = {design.label}
segment_duration = {design.segment_s}
width = {design.width}
height = {design.height}
fps = {design.fps:g}

[sound]
enabled = true
mode = tone
frequency_hz = {design.frequency_hz:g}
sound_file = /home/buzzpi/pirecorder/sounds/stimulus.wav
windows = {design.on_window[0]}-{design.on_window[1]}
active_from = 00:00
active_until = 24:00

[light]
ramp_up_start = {design.lights_on}
ramp_up_end = {int(on_h):02d}:{int(on_m) + 30:02d}
ramp_down_start = 16:30
ramp_down_end = {design.lights_off}

[paths]
logs_dir = /home/buzzpi/pirecorder/logs
"""
    path = out_dir / "pi_config.ini"
    path.write_text(text, encoding="utf-8")
    return path


def _sound_log(
    design: L2Design, out_dir: Path, tone: np.ndarray, hour_index: np.ndarray, wall_s: np.ndarray
) -> Path:
    lines = [
        f"{design.start:%Y-%m-%d %H:%M:%S} [INFO] Sound schedule running: {design.frequency_hz:g} Hz tone"
    ]
    first_hour = design.start.replace(minute=0, second=0)
    for h in range(int(hour_index.max()) + 1):
        t_on = first_hour + timedelta(hours=h, minutes=design.on_window[0])
        t_off = first_hour + timedelta(hours=h, minutes=design.on_window[1])
        if t_on < design.start or (t_off - design.start).total_seconds() > wall_s[-1]:
            continue
        if h in design.failed_hours:
            lines.append(
                f"{t_on:%Y-%m-%d %H:%M:%S} [ERROR] Could not start playback: device busy (retry in 10s)"
            )
            continue
        lines.append(f"{t_on:%Y-%m-%d %H:%M:%S} [INFO] Window open - playing {design.frequency_hz:g} Hz tone")
        lines.append(f"{t_off:%Y-%m-%d %H:%M:%S} [INFO] Window closed - playback stopped")
    path = out_dir / "sound.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
