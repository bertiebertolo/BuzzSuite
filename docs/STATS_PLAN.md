# BuzzPhono statistical analysis plan

Frozen on 2026-11-04, before any result from the corrected clock was looked at on real data. A
change after that date must be listed under "Changes" at the end, with the reason. BuzzPhono's
reports state which version of this plan they followed.

## 1. Question and unit of analysis

- **Question:** does the sound change where mosquitoes rest and fly?
- **Paired unit (as published):** the **hour-cycle**. Each clock hour gives one OFF window
  (minutes 28–38) and one ON window (minutes 40–50, the tone). Both windows sit inside the same
  hour, so slow drifts (light phase, time since lights-off, cage condition) cancel within the pair.
- **Hierarchy:** cycles are nested in **cage-days**, which are nested in **cages**.
  - Neighbouring hours are correlated.
  - Days of the same cage share the same mosquitoes.
  - Cycles are therefore *not* independent replicates. The plan reports the classic per-cycle test
    for continuity, and bases its conclusions on methods that respect the hierarchy.

## 2. Primary endpoint

**Δ = ON − OFF of the speaker-zone resting fraction per hour-cycle.** The fraction is the
published formula: the ratio of capped sums,
`colony_frac_in_zone = Σ min(n_in, total) / Σ total` with `total = min(n_rest, C)`, where C is the
colony size (30 unless the experiment says otherwise).

Secondary versions of the same endpoint are always reported next to it:
- the uncapped resting fraction `Σ n_in / Σ n_rest`;
- the mean of per-frame capped fractions;
- the in-zone count (mosquitoes per frame).

**Default hypothesis:** two-sided, H0: mean Δ = 0. Thesis mode keeps the published one-sided test,
H1: ON > OFF.

## 3. Methods for the primary endpoint (all reported, none chosen after seeing the data)

| Method | What it assumes | Role |
|---|---|---|
| Wilcoxon signed-rank on cycle Δ (exact up to 50 pairs, ties handled exactly; normal approximation with tie correction above) | cycles independent | continuity with the thesis |
| Block sign-flip permutation test of mean Δ (blocks = 12 consecutive clock hours within a cage-day; exact if at most 2^16 sign patterns, otherwise 100,000 random patterns) | ON/OFF labels exchangeable under H0; blocks independent | robust to correlation between neighbouring hours |
| **Wild cluster bootstrap-t** for mean Δ, cage-days as clusters, Webb six-point weights (all patterns when at most 50,000, otherwise 9,999 random); test with the null imposed, interval by percentile-t | cage-days independent; any dependence within a cage-day | **primary cluster-robust inference** |
| Linear mixed model `Δ ~ 1 + phase` with a random intercept per cage-day; cage as a fixed effect (sum-to-zero coding) when there are fewer than 5 cages, otherwise as a variance component | normal effects and residuals | sensitivity; convergence and singular fits are always printed |

**Block size.** 12 hours. The report prints the lag 1–6 hour autocorrelation of Δ. If the lag-6
autocorrelation is above 0.3, the report says the block test may be optimistic.

**Clusters of the wrong size.**
- With fewer than 6 cage-days, the wild cluster bootstrap is labelled "unreliable".
- The bootstrap is also run with whole **cages** as clusters, and reported only as a sensitivity
  check when there are fewer than 6 cages.

## 4. Effect sizes (always with their method's interval)

- Mean Δ: wild cluster bootstrap-t interval.
- Median Δ: cage-day cluster bootstrap percentile interval.
- Hodges–Lehmann estimate, matched-pairs rank-biserial correlation, Cohen's dz.
- Fold change mean(ON) / mean(OFF).
- All with the number of cycles, zero differences, cage-days and cages.

## 5. Phases and multiplicity

- **Phases** come from the experiment's light schedule: dawn = the hour of the lights-on ramp,
  dusk = the hour before lights-off, day between them, night the rest. The thesis cages use dawn
  05–06, day 06–16, dusk 16–17, night 17–05.
- A phase is tested only when it has at least 4 cycles (as published), and its row says how many
  cage-days those cycles come from.

**Families and adjustments:**

| Family | Adjustment |
|---|---|
| F1: the primary endpoint in each phase (up to 4 tests) | Holm |
| F2: the primary contrast of each secondary analysis that was run: attraction index (Δ of the index), PSTH (onset response 0–5 min after onset vs the 5 min before), approach (radial velocity toward the speaker, speaker minus controls), spatial (fraction within the speaker's radial band) | Holm |
| F3: exploratory (PSTH time bins, sensitivity grid, tuning-curve contrasts, per-cage tests) | Benjamini–Hochberg FDR |

The overall primary endpoint is a single test and is not adjusted.

## 6. Data rules (fixed before analysis)

- **Window coverage** (default mode). A window counts only if:
  - at least 80% of its expected frames are present;
  - no gap is longer than 2 minutes;
  - at least 8 of its 10 minutes contain resting detections.

  A cycle needs both windows. Excluded cycles are listed with the reason. Thesis mode uses the
  published rule instead: drop a cycle only if a window has no resting detections at all.
- **Stimulus truth.** A cycle whose tone is known not to have played (from the Pi's playback log)
  is excluded and listed. So is a cycle whose session has no consistent tone-evoked flight burst
  while the protocol says the tone was on: it is flagged "stimulus not confirmed", not dropped
  silently.
- **Clock.** The default is the corrected clock: frame counts plus the fitted session rate and
  offset. The published (thesis) clock is reported for comparison whenever a thesis experiment is
  analysed.

## 7. Controls reported with every primary result

1. **Sham comparison at the same chunk position:** ON′ = minutes 20–30 against OFF′ = minutes
   08–18, with the same tests.
   - Recordings are cut into 20-minute chunks at :00/:20/:40, so ON′ starts on a fresh chunk like
     the real ON window does. That catches tracker warm-up artefacts.
   - A sham result as large as the real one means the real one cannot be trusted.
2. **Speaker against control zones:**
   - Δ for each control zone, and Δ_speaker − mean(Δ_controls) on area-normalised fractions
     (fraction per 10,000 px²).
   - A speaker effect that the control zones show as strongly is not attraction to the speaker.
3. **Zone specification curve.** Results for the speaker zone grown and shrunk by −10…+20 px,
   shifted −40…+60 px, and the earlier (pre-adjustment) zone where one exists.

## 8. How results are worded

- The report's one-line conclusion uses the cluster-robust results (wild cluster bootstrap
  interval and p-value).
  - **"Evidence of an effect"** needs the interval to exclude 0, the block test to agree
    (p < 0.05) and the sham comparison to show no comparable effect.
  - **"Suggestive"** means only the per-cycle Wilcoxon test is below 0.05.
  - Otherwise **"no clear effect"**.
- Estimates and intervals are always shown, never only stars.

## 9. Validation of these methods

Checked by the test suite on simulated data with known truth (`buzzphono.synthetic.l1`):
- false-positive rates at α = 0.05 with serial correlation ρ ∈ {0, 0.3, 0.6};
- interval coverage;
- power, and the minimum detectable effect for a given design.

The suite also records that the per-cycle Wilcoxon test is anti-conservative when neighbouring
hours are correlated.

## Changes

- **2026-11-04: block length 4 h → 12 h.** This was a method-validation result on simulated data
  (`buzzphono.synthetic.l1`, 300 seeds per setting), made before any real-data result under this
  plan was seen.
  - With AR(1) correlation between neighbouring hours of ρ = 0.6, the false-positive rates at
    α = 0.05 were:

    | Block length | Rate |
    |---|---|
    | 4 h | 0.107 |
    | 6 h | 0.087 |
    | 8 h | 0.073 |
    | **12 h** | **0.053** |
    | whole cage-days | 0.027 (too coarse) |
  - At ρ = 0.3: 4 h 0.057, 12 h 0.043.
  - For comparison at ρ = 0.6: the wild cluster bootstrap gave 0.057 and the per-cycle Wilcoxon
    test 0.29.
- **2026-11-04 (later): fewer than 6 cage-days → t-test on cage-day means.**
  - **How it surfaced:** the first real-data runs had 2–3 cage-days per cage. Their wild cluster
    intervals looked implausibly narrow, which prompted a coverage check on simulated data. The
    change is a property of the method, not of any result: the simulator was the only evidence
    used.
  - **Coverage of the 95% interval** (`synthetic.l1`, one cage, ρ = 0.3, 300 seeds):

    | Cage-days | Wild cluster bootstrap-t | t-test on cage-day means (G − 1 df) |
    |---|---|---|
    | 2 | degenerate (unbounded width) | 0.95 |
    | 3 | 0.91 | 0.94 |
    | 4 | 0.92 | 0.95 |
    | 6 | 0.95 | 0.95 |
    | 10 | 0.96 | 0.97 |
  - **The change:** below 6 cage-days, `wild_cluster_bootstrap` returns the t-test on cage-day
    means instead, with the method named in the output. The primary cluster-robust result
    therefore stays honest for short single-cage experiments. At 6 or more cage-days nothing
    changes.
  - **Regression test:** `tests/buzzphono/stats/test_stats_methods.py::test_few_cage_days_use_the_cluster_means_t_interval`.
