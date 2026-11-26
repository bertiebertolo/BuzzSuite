# Feeder-zone tracking checks

Standalone diagnostics for one question: **why do the BuzzSwarm flight heatmaps look empty around
the feeders (or any zone in `custom_zones.json`)?** Is it the tracker, the BuzzSwarm filters, or
the mosquitoes?

All four tools are read-only with respect to the experiment: they read `final_tracking_data/`,
`custom_zones.json`, `plots/buzzswarm/geometry.json`, `plots/buzzswarm/day_cache/` and the raw
videos, and write only the output files you name. Run them from the repo root in `buzzwatch_env`
(Python 3.8). The BuzzSwarm clock analysis must have been run on the experiment first (for
`geometry.json` and the day caches).

| Tool | What it makes |
|---|---|
| `heatmap_before_after.py` | 4-panel figure: cage + zones, every raw moving-tracker detection (before), the real-flight points the BuzzSwarm heatmap plots (after), and the share of detections the filters keep per cell. Prints per-zone density before/after. |
| `resting_vs_flying.py` | Resting (state 0) vs flying heatmaps, plus resting density per unit area in each zone vs the other walls at the same distance from the centre, by hour. Shows feeder visits that a flight-only heatmap cannot. |
| `tracking_clip.py` | A labelled video clip: each detection coloured by its fate in the analysis (kept real flight / removed too slow / removed short-jittery piece / resting), with zones outlined. |
| `untracked_spots.py` | An independent check on raw video: dark spots vs the tracker's detections per area, the background contrast there, a classification of the zone spots the moving tracker did not record (stationary/moving, resting tracker or neither), and zoomed frame strips to look at by eye. |

```bash
E=/path/to/Analysis/<Assay>/<Location>/<Experiment>
V=/path/to/the/experiment/videos        # folder with <segment>.mp4
SEG=CageName_YYYYMMDD_HHMMSS_00051       # segment basename, no extension

python tools/feeder_check/heatmap_before_after.py "$E" -o before_after.png
python tools/feeder_check/resting_vs_flying.py   "$E" -o resting_vs_flying.png
python tools/feeder_check/tracking_clip.py       "$E" "$V" $SEG --start 300 --dur 60 -o clip.mp4
python tools/feeder_check/untracked_spots.py     "$E" "$V" $SEG --start 300 --dur 60 --outdir out/ --tag dawn
```

Days default to the experiment's saved BuzzSwarm day selection (`ld_dd_schedule.json`), else every
full day; pass `--days 2026-07-05 2026-07-06` to choose. `--ring LO HI` sets the comparison band
(px from the cage centre; default 230-354, the radius span of the Penzance feeder polygons).

## Reading the results

- **Colour scale.** BuzzSwarm's own heatmap is linear and capped at the busiest (central) cell, so
  anything under ~10% of that looks black. These figures use density relative to the cage average
  on a log scale.
- **Filters.** Only points moving at >= the experiment's real-flight threshold count as real flight
  (3 px/frame = 75 px/s at 25 fps on the old 720p camera; it scales with the camera's frame width,
  as in BuzzSwarm).
  On the walls most moving-tracker detections are mosquitoes walking or landing, so
  the filters remove most wall detections everywhere, not just at the zones. The density-artifact
  filter removes well under 1%.
- **Contrast.** The tracker sees a mosquito as a dark spot on a lighter background. Over a dark
  panel a flyer darkens the image by ~10-30 grey levels instead of ~70 over the bright centre;
  `untracked_spots.py` prints these numbers per area. Very faint flyers can be missed by both the
  tracker and the independent check.
- **Frame alignment.** A track's coordinate *k* belongs to video frame `start + k - 2`
  (`FRAME_OFFSET` in `_common.py`; same as `VideoManager.overlay_tracking`). Confirmed on video:
  detections land on the darkest pixels at -2 and not at -1 or -3.
