# BuzzPhono

Phonotaxis analysis of BuzzWatch mosquito tracking data: does a sound draw mosquitoes to the
speaker? The headline measure is the fraction of the colony resting in the speaker zone while
the tone plays (ON) compared with just before (OFF), every hour, with cluster-robust statistics.
It also reports time courses, time of day, spatial maps, directed flight, flight phonotaxis,
reliability, sensitivity and comparisons between experiments.

```bash
conda env create -f environment-buzzphono.yml   # once
conda activate buzzphono_env
buzzphono demo --open                            # a simulated experiment with a known effect
buzzphono app                                    # the app: load, calibrate, validate, analyse, explore
buzzphono analyse <experiment folder> -a all     # the same from the command line
```

The full guide is in [`docs/BUZZPHONO_GUIDE.md`](../../docs/BUZZPHONO_GUIDE.md), and the statistical
plan in [`docs/STATS_PLAN.md`](../../docs/STATS_PLAN.md).
