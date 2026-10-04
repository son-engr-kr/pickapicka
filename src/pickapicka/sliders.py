"""What every slider value is stored as.

Sliders hold tenths: the finest step a fine drag (Option-drag in the editor)
reaches. A whole value stays an int, so an edit saved when sliders held whole
numbers normalizes, and hashes into the thumbnail cache, exactly as it did.
"""
from __future__ import annotations


def tenth(val: float) -> int | float:
    r = round(float(val), 1)
    return int(r) if r == int(r) else r
