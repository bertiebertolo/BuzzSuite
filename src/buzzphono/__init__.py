"""BuzzPhono: phonotaxis analysis of BuzzWatch mosquito tracking data.

BuzzPhono reads the tracking output that BuzzWatch (the BuzzSuite tracker) writes for each
20-minute video chunk and asks one question in many ways: do the mosquitoes respond to the sound?
The headline measure is the speaker-zone resting fraction (of all resting mosquitoes, the fraction
resting in the speaker zone), compared between the stimulus-ON and stimulus-OFF windows of every
hour.

All analysis lives in this package. The command line (`buzzphono ...`, see :mod:`buzzphono.cli`)
and the Streamlit app (:mod:`buzzphono.app`) only call into it. The guide is
docs/BUZZPHONO_GUIDE.md.
"""

__version__ = "1.0.0.dev0"

__all__ = ["__version__"]
