"""Single-experiment flight-phonotaxis pipeline: extract -> clock -> per-minute -> cycles -> stats -> outputs.

``run_flight_phonotaxis(experiment_dir, ...)`` is the one entry point used by the BuzzPhono tab and by
``buzzsuite_cli.py phono-flight``. It only reads the experiment's tracking pickles; everything it writes goes
under ``<exp>/plots/flight_phonotaxis/`` (or ``out_dir``), with the per-chunk caches in ``cache/``. Caches are
incremental: a chunk is only re-read when its cache is missing or older than its pickles, and the cache
folders are keyed by the parameters that change their content (zones, cage, flyer multiplier, radius).
Results for a flyer multiplier other than 1, or with the sustained-movement filter on, go to a ``flyer_x<mult>/`` / ``flyer_med<m>/`` subfolder so the default results stay.
"""
import hashlib
import json
import os
import time

import numpy as np

from . import build as fb
from . import clock as fk
from . import config as fc
from . import extract as fx
from . import figures as ff
from . import report as fr

CHUNK_MARGIN_H = 2.0   # chunks within this many hours of the window are read (clock drift is ~1 min/day)


def default_workers():
    n = os.cpu_count() or 2
    return max(1, min(8, n - 2))


def output_dirs(experiment_dir, flyer_mult=1.0, out_dir=None, min_median=0.0):
    """(results folder, base folder). Base holds cache/ and the x1 results; other multipliers get a subfolder."""
    base = out_dir or os.path.join(experiment_dir, fc.OUT_SUBDIR)
    parts = (["x%g" % float(flyer_mult)] if float(flyer_mult) != 1.0 else []) + (["med%g" % float(min_median)] if float(min_median) > 0 else [])
    res = os.path.join(base, "flyer_" + "_".join(parts)) if parts else base
    return res, base


def _hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def _parse_time(v, what):
    import pandas as pd
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        return pd.Timestamp(v)
    except Exception:  # noqa: BLE001
        raise fc.FlightPhonotaxisError("Could not read the %s time '%s' (use e.g. 2026-10-11 or 2026-10-11 05:00)."
                                       % (what, v))


def _chunk_spans(chunks, nf):
    import pandas as pd
    out = []
    for stem, sess, k in chunks:
        s0 = fc.session_start(sess)
        out.append((s0 + pd.Timedelta(seconds=k * nf / fc.FPS), s0 + pd.Timedelta(seconds=(k + 1) * nf / fc.FPS)))
    return out


def prepare(experiment_dir, params, zones_file=None, out_dir=None, frames_per_chunk=None, log=None):
    """Validate the inputs and work out geometry, frames per chunk, window, chunks and cache folders."""
    import pandas as pd
    exp = os.path.abspath(experiment_dir)
    if not os.path.isdir(exp):
        raise fc.FlightPhonotaxisError("Experiment folder not found: %s" % exp)
    chunks = fc.list_chunks(exp)
    if not chunks:
        raise fc.FlightPhonotaxisError(
            "No tracking data in %s: tracking_moving/ has no moving_*.pkl files. Track the experiment first." % exp)
    zones_file = fc.resolve_zones_file(exp, zones_file)
    geom = fc.load_geometry(exp, zones_file)
    G = fc.setup(geom, params.r_fracs())
    order, role_zones = fc.role_order(geom)
    nf = int(frames_per_chunk) if frames_per_chunk else fc.detect_frames_per_chunk(exp, chunks, log=log)
    nmax = fc.nmax_for(nf)
    spans = _chunk_spans(chunks, nf)
    data_start = min(a for a, _ in spans).floor("h")
    data_end = max(b for _, b in spans).ceil("h")
    w0 = _parse_time(params.start, "start") or data_start
    w1 = _parse_time(params.end, "end") or data_end
    if w1 <= w0:
        raise fc.FlightPhonotaxisError("The analysis end (%s) must be after its start (%s)." % (w1, w0))
    margin = pd.Timedelta(hours=CHUNK_MARGIN_H)
    sel = [c for c, (a, b) in zip(chunks, spans) if b >= w0 - margin and a <= w1 + margin]
    if not sel:
        raise fc.FlightPhonotaxisError("No tracked chunk falls inside %s - %s (recorded data: %s - %s)."
                                       % (w0, w1, data_start, data_end))
    res_dir, base_dir = output_dirs(exp, params.flyer_mult, out_dir, params.min_median)
    med = "" if params.min_median <= 0 else "_med%g" % params.min_median
    gh = _hash(dict(cage=np.round(geom["cage"], 3).tolist(),
                    zones=[[n, np.round(geom["zones"][n], 3).tolist()] for n in geom["names"]], nmax=nmax))
    cache = os.path.join(base_dir, "cache")
    dirs = {"main": os.path.join(cache, "main_x%g%s_r%g_%s" % (params.flyer_mult, med, params.radius, gh)),
            "links": os.path.join(cache, "links_%s" % gh),
            "traj": os.path.join(cache, "traj_x%g%s_%s" % (params.flyer_mult, med, gh))}
    ctx = dict(exp=exp, geom=geom, r_fracs=params.r_fracs(), nmax=nmax, mult=params.flyer_mult,
               min_median=params.min_median, dirs=dirs)
    return dict(exp=exp, label=os.path.basename(os.path.normpath(exp)), zones_file=zones_file, geom=geom, G=G,
                order=order, role_zones=role_zones, nf=nf, nmax=nmax, chunks=chunks, chunks_sel=sel,
                window=(w0, w1), data_span=(data_start, data_end), res_dir=res_dir, base_dir=base_dir, ctx=ctx)


def _write_cache_params(plan, params):
    for kind, d in plan["ctx"]["dirs"].items():
        os.makedirs(d, exist_ok=True)
        info = dict(kind=kind, zones_file=plan["zones_file"], zones=plan["geom"]["names"], nf=plan["nf"],
                    nmax=plan["nmax"], flyer_mult=params.flyer_mult if kind != "links" else None,
                    min_median=params.min_median if kind != "links" else None,
                    r_fracs=list(params.r_fracs()) if kind == "main" else None)
        with open(os.path.join(d, "params.json"), "w") as f:
            json.dump(info, f, indent=1)


def run_flight_phonotaxis(experiment_dir, zones_file=None, out_dir=None, start=None, end=None, on=(40, 50),
                          off=(28, 38), light_hours=(5, 17), flyer_mult=1.0, min_median=0.0, radius=0.1, workers=None,
                          frames_per_chunk=None, make_figures=True, progress=None, cancel=None, log=None):
    """Run the whole single-experiment analysis. progress(fraction, message); cancel = threading.Event;
    log(message). Returns a result dict (tables, clock fit, output paths). Raises FlightPhonotaxisError for
    problems the user can fix, Cancelled when cancel is set."""
    import pandas as pd
    t_start = time.time()
    say = log or (lambda m: None)

    def report(frac, msg):
        if progress:
            progress(frac, msg)

    def check():
        if cancel is not None and cancel.is_set():
            raise fc.Cancelled()

    params = fc.FlightParams(on=on, off=off, light_hours=light_hours, flyer_mult=flyer_mult, min_median=min_median, radius=radius,
                             start=start, end=end)
    workers = int(workers) if workers else default_workers()
    report(0.0, "Checking zones and tracking data")
    plan = prepare(experiment_dir, params, zones_file=zones_file, out_dir=out_dir,
                   frames_per_chunk=frames_per_chunk, log=say)
    check()
    w0, w1 = plan["window"]
    say("flight phonotaxis: %s, %d of %d chunks in %s - %s, %d frames/chunk, zones %s"
        % (plan["label"], len(plan["chunks_sel"]), len(plan["chunks"]), w0, w1, plan["nf"],
           os.path.basename(plan["zones_file"])))
    _write_cache_params(plan, params)

    # 1. extraction (the slow part)
    def xprog(done, total):
        report(0.03 + 0.82 * (done / float(max(total, 1))), "Reading tracking data: chunk %d of %d" % (done, total))
    statuses, n_cached = fx.run_extraction(plan["ctx"], plan["chunks_sel"], workers=workers, progress=xprog,
                                           cancel=cancel, log=say)
    problems = []
    for stem, st in sorted(statuses.items()):
        for kind, s in st.items():
            if s != "ok":
                problems.append((stem, "%s: %s" % (kind, s)))
    if problems:
        say("flight phonotaxis: %d chunk cache(s) could not be built, e.g. %s" % (len(problems), problems[0]))
    check()

    # 2. clock
    report(0.85, "Calibrating the recording clock on the tone-onset flight burst")
    dirs = plan["ctx"]["dirs"]
    clock, bursts = fk.calibrate(dirs["main"], plan["chunks_sel"], plan["nf"], w0, w1, params.on[0])
    say("clock: " + clock["message"])
    check()

    # 3. per-minute accumulation and cycle metrics
    report(0.88, "Per-minute flight metrics")
    G, order, nf, nmax = plan["G"], plan["order"], plan["nf"], plan["nmax"]
    acc = fb.accumulate(dirs["main"], plan["chunks_sel"], clock, order, params, plan["window"], nf, nmax,
                        G["s"], cancel=cancel)
    if acc["chunks_used"] == 0:
        raise fc.FlightPhonotaxisError(
            "No chunk could be placed in the analysis window on the calibrated clock (%d chunk cache problem(s)). "
            "Check the date range and the log." % len(problems))
    CM = fb.cycle_metrics(acc, params)
    ST = fb.stat_table(CM, plan["label"], params)
    report(0.92, "Take-off / landing links")
    lacc = fb.accumulate_links(dirs["links"], plan["chunks_sel"], clock, order, G, params, plan["window"], nf,
                               acc["minutes"].time, cancel=cancel)
    LCM = fb.links_cycle_metrics(lacc, params)
    LST = fb.stat_table(LCM, plan["label"], params)
    report(0.94, "Distance from each wall")
    dacc = fb.accumulate_dist(dirs["traj"], plan["chunks_sel"], clock, order, params, plan["window"], nf,
                              cancel=cancel)
    DCM = fb.dist_cycle_metrics(dacc, params)
    DST = fb.stat_table(DCM, plan["label"], params)
    report(0.95, "Trajectories passing each wall")
    traj_tab = fb.trajectory_pass_summary(dirs["traj"], plan["chunks_sel"], clock, order, G, params,
                                          plan["window"], cancel=cancel)
    check()

    # 4. tables and README
    report(0.96, "Writing tables")
    res_dir = plan["res_dir"]
    csv_dir = os.path.join(res_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    on_c = CM[CM.window == "ON"].hour
    off_c = CM[CM.window == "OFF"].hour
    paired = pd.to_datetime(pd.Index(on_c).intersection(pd.Index(off_c)))
    n_light = int(sum(params.light_of(h) == params.light_label for h in paired.hour))
    kr = fr.key_results(ST, LST, DST, traj_tab, params)
    clock_out = dict(clock)
    with open(os.path.join(res_dir, "clock_fit.json"), "w") as f:
        json.dump(clock_out, f, indent=1)
    if len(bursts):
        bursts.to_csv(os.path.join(csv_dir, "clock_bursts.csv"), index=False)
    acc["minutes"].drop(columns=["hour", "moh"]).to_csv(os.path.join(csv_dir, "flight_per_minute.csv"), index=False)
    CM.to_csv(os.path.join(csv_dir, "flight_cycle_metrics.csv"), index=False)
    LCM.to_csv(os.path.join(csv_dir, "links_cycle_metrics.csv"), index=False)
    DCM.to_csv(os.path.join(csv_dir, "distance_cycle_metrics.csv"), index=False)
    traj_tab.to_csv(os.path.join(csv_dir, "trajectory_pass_summary.csv"), index=False)
    ST.to_csv(os.path.join(res_dir, "stats_flight_on_vs_off.csv"), index=False)
    LST.to_csv(os.path.join(res_dir, "stats_links_on_vs_off.csv"), index=False)
    DST.to_csv(os.path.join(res_dir, "stats_distance_on_vs_off.csv"), index=False)
    kr.to_csv(os.path.join(res_dir, "key_results.csv"), index=False)
    with open(os.path.join(res_dir, "run_parameters.json"), "w") as f:
        json.dump(dict(params.to_dict(), experiment_dir=plan["exp"], zones_file=plan["zones_file"],
                       window=[str(w0), str(w1)], frames_per_chunk=nf, workers=workers,
                       role_zones=plan["role_zones"]), f, indent=1)
    result = dict(experiment_dir=plan["exp"], label=plan["label"], out_dir=res_dir, base_dir=plan["base_dir"],
                  params=params.to_dict(), params_obj=params, clock=clock, bursts=bursts,
                  window=(str(w0), str(w1)), data_span=tuple(str(x) for x in plan["data_span"]),
                  zones_file=plan["zones_file"], role_zones=plan["role_zones"], s=G["s"],
                  minutes=acc["minutes"], heat=acc["heat"], nblk=acc["nblk"], cycle_metrics=CM, stats=ST,
                  links_minutes=lacc["minutes"], links_cycle_metrics=LCM, links_stats=LST,
                  dist_minutes=dacc["minutes"], dist_cycle_metrics=DCM, dist_stats=DST, traj_tab=traj_tab,
                  key_results=kr, n_cycles=len(paired), n_cycles_light=n_light, n_cycles_dark=len(paired) - n_light,
                  n_chunks=len(plan["chunks"]), n_chunks_selected=len(plan["chunks_sel"]),
                  chunks_used=acc["chunks_used"], n_cached=n_cached, problems=problems, figures=[])
    fr.write_readme(os.path.join(res_dir, "README_flight_phonotaxis.md"), result)
    sens = fr.write_sensitivity(plan["base_dir"], params) if params.min_median <= 0 else None
    if sens:
        result["sensitivity_csv"] = sens
    check()

    # 5. figures
    if make_figures:
        report(0.98, "Drawing figures")
        figs = []
        try:
            figs.append(ff.write_geometry_png(G, order, plan["exp"], os.path.join(res_dir, "geometry_check.png")))
        except Exception as exc:  # noqa: BLE001
            say("flight phonotaxis: geometry figure failed: %s" % exc)
        for kind, _title, fname in ff.FIGURES:
            check()
            try:
                fig = ff.build_figure(kind, result)
                figs.append(ff.save(fig, os.path.join(res_dir, fname)))
            except Exception as exc:  # noqa: BLE001
                say("flight phonotaxis: figure %s failed: %s" % (fname, exc))
        result["figures"] = figs
    report(1.0, "Done")
    result["elapsed_s"] = time.time() - t_start
    say("flight phonotaxis: done in %.0f s -> %s" % (result["elapsed_s"], res_dir))
    return result
