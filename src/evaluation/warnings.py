"""Warning-level evaluation against annotated risk EVENTS (intervals).

Requires expert event annotations (none exist in this repository). Given
per-frame timestamps and a boolean "warning active" series:

  * an event counts as detected if a warning is active anywhere in
    [start - max_lead_s, end];
  * lead time = event start - onset of the first warning in that window
    (positive = warned before the event; negative = warned only after it began);
  * false alarm = a warning onset (rising edge) that does not fall in any
    event's [start - max_lead_s, end] window; reported per minute of video.
"""

from __future__ import annotations

import numpy as np


def warning_onsets(t: np.ndarray, active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, bool)
    rising = active & ~np.concatenate([[False], active[:-1]])
    return np.asarray(t)[rising]


def evaluate_warnings(t, active, events: list[tuple[float, float]], max_lead_s: float = 5.0) -> dict:
    t = np.asarray(t, float)
    active = np.asarray(active, bool)
    if len(t) == 0:
        raise ValueError("Empty time series")
    duration_min = max((t[-1] - t[0]) / 60.0, 1e-9)
    detected, leads = 0, []
    for start, end in events:
        win = (t >= start - max_lead_s) & (t <= end) & active
        if win.any():
            detected += 1
            # Onset of the warning episode active at the first warning time in the window.
            i = int(np.argmax(win))
            while i > 0 and active[i - 1] and t[i - 1] >= start - max_lead_s:
                i -= 1
            leads.append(float(start - t[i]))
    onsets = warning_onsets(t, active)
    covered = np.zeros(len(onsets), bool)
    for start, end in events:
        covered |= (onsets >= start - max_lead_s) & (onsets <= end)
    false_alarms = int((~covered).sum())
    return {
        "events": len(events),
        "events_detected": detected,
        "event_recall": detected / len(events) if events else None,
        "lead_time_s_mean": float(np.mean(leads)) if leads else None,
        "lead_time_s_median": float(np.median(leads)) if leads else None,
        "warning_onsets": int(len(onsets)),
        "false_alarms": false_alarms,
        "false_alarms_per_min": false_alarms / duration_min,
        "duration_min": duration_min,
    }
