"""Key-results table, auto-generated README and flyer-threshold sensitivity summary (from fp_readme / fp_sens)."""
import datetime
import os

import numpy as np

from . import config as fc

# (section, label, table, metric, zone, kind) -- kind 'fold' shows OFF -> ON (xfold); 'diff' shows the
# ON-OFF change of the speaker relative to a control (difference in differences).
KEY_ROWS = [
    ("Arousal", "Flyers airborne per frame", "main", "flyers_per_frame", "all", "fold"),
    ("Arousal", "Take-offs (flight-track starts) per min", "main", "takeoffs_per_min", "all", "fold"),
    ("Near the speaker", "Share of flyers within R of the speaker, vs mean of other walls", "main", "share_near_r1", "speaker - mean(controls)", "diff"),
    ("Near the speaker", "Share of flyers within R of the speaker, vs opposite wall", "main", "share_near_r1", "speaker - right", "diff"),
    ("Near the speaker", "Share of flyers within R of the speaker, vs top wall", "main", "share_near_r1", "speaker - top", "diff"),
    ("Near the speaker", "Share of flyers within R of the speaker, vs bottom wall", "main", "share_near_r1", "speaker - bottom", "diff"),
    ("Wall sector", "Share of flyers in the speaker wall sector, vs mean of other walls", "main", "share_sector", "speaker - mean(controls)", "diff"),
    ("Wall sector", "Share of flyers in the speaker wall sector, vs top wall", "main", "share_sector", "speaker - top", "diff"),
    ("Wall sector", "Share of flyers in the speaker wall sector, vs bottom wall", "main", "share_sector", "speaker - bottom", "diff"),
    ("Entries", "Entries into the speaker neighbourhood per min, vs mean of other walls", "main", "entries_per_min", "speaker - mean(controls)", "diff"),
    ("Landings", "Flight tracks ending in the speaker zone per min, vs mean of other walls", "main", "landings_per_min", "speaker - mean(controls)", "diff"),
    ("Resting (old metric)", "Resting share in the speaker zone", "main", "rest_prop", "speaker", "fold"),
    ("Links", "Arrival probability at the speaker (bouts from elsewhere)", "links", "arrival_prob", "speaker", "fold"),
    ("Links", "Arrival probability at the speaker, vs mean of other walls", "links", "arrival_prob", "speaker - mean(controls)", "diff"),
    ("Links", "Arrival probability at the speaker, vs top wall", "links", "arrival_prob", "speaker - top", "diff"),
    ("Distance", "Average distance of flyers from the speaker (cage widths)", "dist", "mean_dist", "speaker", "fold"),
    ("Distance", "Distance change at the speaker minus mean of other walls (negative = closer)", "dist", "mean_dist", "speaker - mean(controls)", "diff"),
    ("Distance", "Distance change at the speaker minus top wall", "dist", "mean_dist", "speaker - top", "diff"),
    ("Distance", "Distance change at the speaker minus bottom wall", "dist", "mean_dist", "speaker - bottom", "diff"),
]


def pfmt(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<1e-4" if p < 1e-4 else ("%.3f" % p if p >= 0.001 else "%.1e" % p)


def _row(st, metric, zone, group, pair="ON vs OFF"):
    if st is None or not len(st):
        return None
    r = st[(st.pair == pair) & (st.metric == metric) & (st.zone == zone) & (st.group == group)
           & (st.grouping.isin(["light", "all"]))]
    return None if r.empty else r.iloc[0]


def key_results(stats, links_stats, dist_stats, traj_tab, params):
    """one row per (key metric, light phase) for the GUI table and key_results.csv."""
    import pandas as pd
    tables = {"main": stats, "links": links_stats, "dist": dist_stats}
    rows = []
    R = "%g cage widths" % params.radius
    for section, label, tab, metric, zone, kind in KEY_ROWS:
        for ph in (params.light_label, params.dark_label):
            r = _row(tables[tab], metric, zone, ph)
            row = dict(section=section, metric=label.replace("within R", "within " + R), phase=ph.split()[0],
                       comparison=fc.role_label(zone), kind=kind)
            if r is None:
                row.update(off=np.nan, on=np.nan, change=np.nan, fold=np.nan, dz=np.nan, n_on_gt_off="",
                           p_on_gt_off=np.nan, p_two_sided=np.nan)
            else:
                # contrast rows are stored as (speaker change - control change) vs 0: no OFF / ON level
                lvl = kind != "diff"
                row.update(off=r.mean_off if lvl else np.nan, on=r.mean_on if lvl else np.nan, change=r.delta,
                           fold=r.fold, dz=r.dz, n_on_gt_off="%d/%d" % (r.n_on_gt_off, r.n_cycles),
                           p_on_gt_off=r.p_on_gt_off, p_two_sided=r.p_two_sided)
            rows.append(row)
    if traj_tab is not None and len(traj_tab):
        for ph in (params.light_label, params.dark_label):
            t = traj_tab[traj_tab.phase == ph].set_index("tone")
            if "ON" not in t.index or "OFF" not in t.index:
                continue
            for r, lab in (("speaker", "the speaker"), ("right", "the opposite wall"), ("top", "the top wall"),
                           ("bottom", "the bottom wall")):
                off, on = t.loc["OFF", "pct_pass_%s" % r], t.loc["ON", "pct_pass_%s" % r]
                rows.append(dict(section="Trajectories", phase=ph.split()[0], kind="pct",
                                 metric="%% of flight tracks passing within %s of %s" % (R, lab),
                                 comparison=fc.ROLE_LABELS[r], off=off, on=on, change=on - off,
                                 fold=(on / off) if off else np.nan, dz=np.nan,
                                 n_on_gt_off="%d / %d tracks" % (t.loc["OFF", "n_flights"], t.loc["ON", "n_flights"]),
                                 p_on_gt_off=np.nan, p_two_sided=np.nan))
    return pd.DataFrame(rows)


def _md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _num(v, fmt="%.3g"):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt % v


def write_readme(path, result):
    """short README with the method, the clock fit and its caveats, the key results and the file list."""
    p = result["params_obj"]
    ck = result["clock"]
    kr = result["key_results"]
    lines = []
    lines.append("# Flight phonotaxis: %s\n" % result["label"])
    lines.append("Auto-generated by BuzzSuite (BuzzPhono tab, 'Flight phonotaxis') on %s. Re-running overwrites this file.\n"
                 % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("- Experiment: `%s`" % result["experiment_dir"])
    lines.append("- Zones: `%s` -> speaker = `%s`, opposite wall = `%s`, top wall = `%s`, bottom wall = `%s` "
                 "(roles assigned by geometry; check `geometry_check.png`)." % ((result["zones_file"],) + tuple(result["role_zones"])))
    lines.append("- Window analysed: %s to %s; tone ON minutes %d-%d, OFF baseline %d-%d of each hour; light phase %02d-%02d h."
                 % ((result["window"][0], result["window"][1]) + p.on + p.off + p.light_hours))
    lines.append("- Hour-cycles with >= 90%% frame coverage: %d (ON vs OFF pairs: %d light, %d dark)."
                 % (result["n_cycles"], result["n_cycles_light"], result["n_cycles_dark"]))
    lines.append("- Frames per chunk (read from the data): %d. Chunks: %d listed, %d used after the clock model."
                 % (ck["nf"], result["n_chunks"], result["chunks_used"]))
    lines.append("- Flyers = moving tracks with >= %d points and mean step >= %g px/frame x s x %g (s = cage width / 780 px "
                 "= %.2f for this camera). Neighbourhood radius %g cage widths (also %g and %g)."
                 % (fc.MIN_FLYER_PTS, fc.FLYER_STEP, p.flyer_mult, result["s"], p.radius, p.r_fracs()[0], p.r_fracs()[2]))
    if p.min_median > 0:
        lines.append("- **Sustained-movement filter ON:** flyers (and the flight-px activity used for the clock fit, and the "
                     "trajectory / distance metrics) must also have a median step >= %g px/frame x s. This removes tracks "
                     "that are mostly stationary with a few large jumps from linking to a neighbouring mosquito (dense cages). "
                     "The take-off / landing links come from final_tracking_data bouts and are NOT filtered by this rule; "
                     "results are not comparable with runs made with the filter off." % p.min_median)
    else:
        lines.append("- Sustained-movement filter: off (the published Penzance / Ae. aegypti setting).")
    lines.append("")
    lines.append("## Recording clock\n")
    if ck["fitted"]:
        lines.append("Calibrated on the flight burst at tone onset (minute %d): %d bursts found, %d kept by the robust fit; "
                     "drift %.1f s/day -> effective %.4f fps (chunk = %.2f s real); residual SD %.1f s. Session offsets (s): %s."
                     % (p.on[0], ck["n_bursts"], ck["n_kept"], ck["b"], ck["effective_fps"],
                        ck["nf"] / ck["effective_fps"], ck["resid_sd"],
                        ", ".join("%s: %+.1f" % (s, a) for s, a in sorted(ck["a"].items()))))
        if ck.get("dropped_sessions"):
            lines.append("\n**Sessions without a clock fit (fewer than 3 bursts) are left out:** %s."
                         % ", ".join(ck["dropped_sessions"]))
    else:
        lines.append("**Not calibrated:** %s Minute-of-hour windows may be shifted by the camera's drift (tens of seconds "
                     "per day on some cameras; none was seen in Bangkok). Treat onset-window (%d-%d) results with care."
                     % (ck["message"], p.on[0], p.on[0] + p.onset_minutes))
    lines.append("")
    lines.append("## Key results (ON vs OFF, paired per hour-cycle)\n")
    rows = []
    for r in kr.itertuples():
        if r.kind == "pct":
            val = "%.1f%% -> %.1f%% (%s)" % (r.off, r.on, r.n_on_gt_off)
        elif r.kind == "fold":
            val = "%s -> %s (x%s), %s, p=%s" % (_num(r.off), _num(r.on), _num(r.fold, "%.2f"), r.n_on_gt_off or "-",
                                                 pfmt(r.p_on_gt_off))
        else:
            val = "dz %s, %s, p=%s" % (_num(r.dz, "%+.2f"), r.n_on_gt_off or "-", pfmt(r.p_on_gt_off))
        rows.append([r.metric, r.phase, val])
    lines.append(_md_table(rows, ["metric", "phase", "OFF -> ON (cycles ON > OFF, one-sided Wilcoxon p)"]))
    lines.append("\ndz = mean(ON-OFF)/SD(ON-OFF) over cycles. For 'vs' rows the speaker's ON-OFF change minus the "
                 "control's change is tested against 0. Full tables: `stats_*.csv`.\n")
    lines.append("## Caveats\n")
    lines.append("- Moving toward the speaker is always moving away from the opposite wall, so a speaker gain is "
                 "mirrored there. Judge the speaker against the top and bottom walls, not only the mean of the controls.")
    lines.append("- Shares sum to 1 across the four walls: a wall that empties raises the others mechanically.")
    lines.append("- The camera looks along the cage axis: 'near the speaker' means near the speaker wall at any height; "
                 "the wall-sector share is the perspective-robust version.")
    lines.append("- Hour-cycles are treated as independent replicates (one cage; no random effects).")
    lines.append("- Not reported, because they failed the Ae. aegypti positive control: the approach index (mean cos of "
                 "the step direction toward a zone) and per-track toward/away (start vs end distance).")
    lines.append("- The sham pair (OFF %d-%d vs %d-%d, no sound in either) should be null; differences of similar size "
                 "elsewhere are not evidence of a response." % (p.off + p.sham))
    lines.append("- Track-level metrics use the raw tracking_moving pieces; the links metrics use the assembled "
                 "final_tracking_data tracks (state 1 = flying, 0 = resting).")
    if result.get("problems"):
        lines.append("- %d chunk(s) could not be read and are left out: %s"
                     % (len(result["problems"]), "; ".join("%s (%s)" % x for x in result["problems"][:5])))
    lines.append("")
    lines.append("## Files\n")
    lines.append("- `key_results.csv`: the table above. `stats_flight_on_vs_off.csv`, `stats_links_on_vs_off.csv`, "
                 "`stats_distance_on_vs_off.csv`: every pair x metric x zone x phase (the zone key `right` is the "
                 "opposite wall, see `zone_label`).")
    lines.append("- `csv/`: per-minute series, per-cycle metrics, clock bursts, trajectory pass-by counts.")
    lines.append("- `arousal_timecourse.png`, `flight_share_timecourse.png`, `distance_to_walls_timecourse.png`, "
                 "`zone_comparison_flight.png`, `geometry_check.png`.")
    lines.append("- `cache/`: per-chunk extraction caches (safe to delete; rebuilt on the next run).")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


SENS_METRICS = [("flyers_per_frame", "all"), ("share_near_r1", "speaker - mean(controls)"),
                ("share_sector", "speaker - mean(controls)"), ("entries_per_min", "speaker - mean(controls)"),
                ("landings_per_min", "speaker - mean(controls)"), ("share_sector", "speaker - top"),
                ("entries_per_min", "speaker - top")]


def write_sensitivity(base_out_dir, params):
    """dz of the headline metrics at every flyer threshold run so far (x1 in the base folder, others in
    flyer_x*/ subfolders) -> sensitivity_flyer_threshold.csv in the base folder. Returns the path or None."""
    import pandas as pd
    found = []
    p1 = os.path.join(base_out_dir, "stats_flight_on_vs_off.csv")
    if os.path.isfile(p1):
        found.append((1.0, p1))
    if os.path.isdir(base_out_dir):
        for d in sorted(os.listdir(base_out_dir)):
            if d.startswith("flyer_x"):
                p = os.path.join(base_out_dir, d, "stats_flight_on_vs_off.csv")
                try:
                    mult = float(d[len("flyer_x"):])
                except ValueError:
                    continue
                if os.path.isfile(p):
                    found.append((mult, p))
    if len(found) < 2:
        return None
    rows = []
    for mult, p in found:
        st = pd.read_csv(p)
        st = st[(st.pair == "ON vs OFF") & (st.grouping == "light")]
        for m, z in SENS_METRICS:
            for ln in (params.light_label, params.dark_label):
                r = st[(st.metric == m) & (st.zone == z) & (st.group == ln)]
                if len(r):
                    r = r.iloc[0]
                    rows.append(dict(flyer_threshold_mult=mult, metric=m, zone=z, group=ln, n_cycles=r.n_cycles,
                                     delta=r.delta, dz=r.dz, n_on_gt_off=r.n_on_gt_off, p_on_gt_off=r.p_on_gt_off))
    out = os.path.join(base_out_dir, "sensitivity_flyer_threshold.csv")
    pd.DataFrame(rows).sort_values(["metric", "zone", "group", "flyer_threshold_mult"]).to_csv(out, index=False)
    return out
