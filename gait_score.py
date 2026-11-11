#!/usr/bin/env python3
"""
gait_score.py - gives a walking video's joint-angle data a score from 0 to 100.

WHAT IT DOES
    It reads OpenSim / OpenCap ".mot" files (the joint angles of someone walking)
    and gives three scores, each from 0 to 100:

        symmetry     - are the left and right legs doing the same thing?
        consistency  - is every stride about the same as the last one?
        movement     - how little wasted motion is there (bouncing, swaying, twisting)?

    The final score is the average of those three.

HOW THIS FILE IS ORGANISED (read it from top to bottom)
    1. Settings          - the numbers you can tweak (what counts as a good/bad value)
    2. Reading files     - load a .mot file into a table
    3. Small helpers     - tiny maths functions used everywhere
    4. Result classes    - containers that hold the answer
    5. Finding strides   - find each step of the walk in the data
    6. Collecting data   - measure every stride and put them all in one pile
    7. The metrics       - the actual symmetry / consistency / movement measurements
    8. Ground force      - optional extra measurements from force-plate data
    9. Putting it together - turn the metrics into scores
   10. Score vs steps chart
   11. Command line

HOW TO USE IT
    python gait_score.py walk1.mot
    python gait_score.py walk1.mot walk2.mot walk3.mot      (same person, trials combined)
    python gait_score.py walk1.mot --plot chart.png --json result.json
"""

import argparse
import json
import math
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, filtfilt, find_peaks, detrend

__version__ = "0.1.0"


class GaitError(ValueError):
    """Raised when a file can't be scored (wrong file, not walking, too short...)."""
    pass


# ==========================================================================
# 1. SETTINGS
# ==========================================================================

# Every measurement (called a "metric") is turned into a score like this:
#     value is as good as "best" (or better)   -> 100 points
#     value is as bad as "worst" (or worse)    -> 0 points
#     anything in between                      -> a straight line between 0 and 100
# "weight" says how much a metric counts inside its group (0.5 = half as much).
# In every case below, a LOWER value is better.
#
# These best/worst numbers are STARTING GUESSES based on typical healthy walking.
# They have not been tested on a large group of people, so change them if needed.

def make_spec(component, label, unit, best, worst, weight=1.0):
    """Bundle the settings of one metric into a dictionary."""
    return {"component": component, "label": label, "unit": unit,
            "best": best, "worst": worst, "weight": weight}


SPECS = {
    # ---- symmetry: left vs right ----
    "rom_symmetry_index":    make_spec("symmetry", "Joint ROM symmetry index", "%", 3, 30),
    "waveform_asymmetry":    make_spec("symmetry", "Left/right waveform difference", "% of ROM", 8, 45),
    "step_time_asymmetry":   make_spec("symmetry", "Step-time asymmetry", "%", 2, 20),
    "grf_peak_asymmetry":    make_spec("symmetry", "Peak vertical GRF asymmetry (GRF)", "%", 3, 25),
    "grf_impulse_asymmetry": make_spec("symmetry", "Vertical impulse asymmetry (GRF)", "%", 3, 25),
    "stance_time_asymmetry": make_spec("symmetry", "Stance-time asymmetry (GRF)", "%", 2, 15),

    # ---- consistency: stride to stride ----
    "stride_time_cv":        make_spec("consistency", "Stride-time variability (CV)", "%", 1.5, 8),
    "stride_length_cv":      make_spec("consistency", "Stride-length variability (CV)", "%", 2, 12),
    "waveform_variability":  make_spec("consistency", "Joint-angle waveform variability", "% of ROM", 6, 30),
    "grf_peak_cv":           make_spec("consistency", "Peak vertical GRF variability (GRF)", "%", 3, 15),

    # ---- movement: wasted motion ----
    "vertical_oscillation":  make_spec("movement", "Vertical pelvis oscillation", "cm", 5, 12),
    "lateral_sway":          make_spec("movement", "Lateral pelvis sway", "cm", 6, 16),
    "pelvis_list_rom":       make_spec("movement", "Pelvis obliquity range", "deg", 6, 16, 0.5),
    "pelvis_rotation_rom":   make_spec("movement", "Pelvis rotation range", "deg", 10, 30, 0.5),
    "lumbar_bending_rom":    make_spec("movement", "Lumbar side-bend range", "deg", 6, 20, 0.5),
    "lumbar_rotation_rom":   make_spec("movement", "Lumbar rotation range", "deg", 8, 30, 0.5),
    "hip_adduction_rom":     make_spec("movement", "Hip ab/adduction range", "deg", 12, 30),
}

# How much each of the three groups counts in the final score (all equal).
COMPONENT_WEIGHTS = {"symmetry": 1.0, "consistency": 1.0, "movement": 1.0}

# Other settings
LOWPASS_HZ = 6.0           # smoothing used ONLY to find the steps (not for measuring)
MIN_HIP_RANGE_DEG = 15.0   # if the hip moves less than this, it's not walking
MIN_STRIDE_S = 0.5         # a stride shorter than this (seconds) must be a detection error
MAX_STRIDE_S = 3.0         # a stride longer than this must be a detection error
DEFAULT_DROP_LAST = 1      # ignore the final step of each trial (feet-together stop)
MIN_STRIDES = 3            # need at least this many strides per leg
RECOMMENDED_STRIDES = 6    # below this many per leg the results are noisy
MIN_TRAVEL_M = 1.0         # the pelvis must move this far (metres) to count as walking overground
MIN_STRIDE_LENGTH_M = 0.4  # average stride length must be at least this (metres)
N_POINTS = 101             # every stride is stretched to 101 points so strides can be compared

# Matplotlib "backends" that can save pictures but can never open a window.
NON_WINDOW_BACKENDS = ["agg", "cairo", "pdf", "pgf", "ps", "svg", "template"]

# Joints that we compare between left and right legs.
# (names in the file look like "hip_flexion_l" and "hip_flexion_r")
WAVEFORM_JOINTS = ["hip_flexion", "knee_angle", "ankle_angle"]
ROM_JOINTS = ["hip_flexion", "knee_angle", "ankle_angle", "hip_adduction"]

# Body-centre columns (not left/right) that we measure the movement of.
MIDLINE_COLS = ["pelvis_ty", "pelvis_list", "pelvis_rotation", "lumbar_bending", "lumbar_rotation"]

# "l" = left leg, "r" = right leg. Used as a short code all through the file.
SIDES = ["l", "r"]

# Columns the file MUST contain.
REQUIRED_KIN_COLUMNS = ["time", "pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_list",
                        "pelvis_rotation", "lumbar_bending", "lumbar_rotation"]
for _joint in ROM_JOINTS:
    for _side in SIDES:
        REQUIRED_KIN_COLUMNS.append(_joint + "_" + _side)


# ==========================================================================
# 2. READING FILES
# ==========================================================================

def read_mot(path):
    """Read an OpenSim .mot file and return its numbers as a table (DataFrame)."""
    path = Path(path)

    # Read the whole file as lines of text.
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as error:
        raise GaitError("cannot read " + str(path) + ": " + str(error.strerror or error))

    # The file starts with a header that ends with a line saying "endheader".
    end_of_header = None
    for number, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            end_of_header = number
            break
    if end_of_header is None:
        raise GaitError(path.name + ": no 'endheader' line - not an OpenSim .mot/.sto file")

    # The header may say "inDegrees=yes" or "inDegrees=no". We need to know which.
    in_degrees = True
    for line in lines[:end_of_header]:
        key, _, value = line.strip().partition("=")
        if key.strip().lower() == "indegrees":
            in_degrees = value.strip().lower() != "no"

    # Everything after the header is a table of numbers separated by whitespace.
    data_text = "\n".join(lines[end_of_header + 1:])
    try:
        table = pd.read_csv(StringIO(data_text), sep=r"\s+")
    except Exception as error:
        raise GaitError(path.name + ": could not parse data block (" + str(error) + ")")

    table = table.apply(pd.to_numeric, errors="coerce")   # anything that isn't a number -> NaN
    table.attrs["in_degrees"] = in_degrees                # remember the unit for later
    return table


def crop_time(table, t_start, t_end):
    """Keep only the rows between t_start and t_end seconds (None = no limit)."""
    if t_start is not None:
        table = table[table["time"] >= t_start]
    if t_end is not None:
        table = table[table["time"] <= t_end]
    return table.reset_index(drop=True)


def load_kinematics(path, t_start=None, t_end=None):
    """Load a joint-angle file. Returns (table in degrees and metres, samples per second)."""
    path = Path(path)
    raw = read_mot(path)

    # Check that every column we need is there.
    missing = []
    for column in REQUIRED_KIN_COLUMNS:
        if column not in raw.columns:
            missing.append(column)
    if len(missing) > 0:
        raise GaitError(path.name + ": missing kinematic columns: " + ", ".join(missing) +
                        ". Expected an OpenSim inverse-kinematics .mot.")

    table = raw[REQUIRED_KIN_COLUMNS].copy()

    # If the file is in radians, convert the angle columns to degrees.
    # (Columns ending in _tx, _ty, _tz are positions in metres, so leave those alone.)
    if raw.attrs.get("in_degrees", True) is False:
        for column in table.columns:
            is_position = column.endswith("_tx") or column.endswith("_ty") or column.endswith("_tz")
            if column != "time" and not is_position:
                table[column] = np.rad2deg(table[column])

    table = crop_time(table.dropna(), t_start, t_end)
    if len(table) < 30:
        raise GaitError(path.name + ": only " + str(len(table)) + " usable rows after cropping")

    # Time must always go forward.
    gaps = np.diff(table["time"].to_numpy())
    if np.any(gaps <= 0):
        raise GaitError(path.name + ": time column is not strictly increasing")

    samples_per_second = 1.0 / float(np.median(gaps))
    return table, samples_per_second


def load_grf(path, t_start=None, t_end=None):
    """Load a ground-reaction-force file (only the columns we use)."""
    path = Path(path)
    raw = read_mot(path)
    needed = ["time", "ground_force_right_vy", "ground_force_left_vy"]
    missing = []
    for column in needed:
        if column not in raw.columns:
            missing.append(column)
    if len(missing) > 0:
        raise GaitError(path.name + ": missing GRF columns: " + ", ".join(missing))
    return crop_time(raw[needed].dropna(), t_start, t_end)


# ==========================================================================
# 3. SMALL HELPERS
# ==========================================================================

def lowpass(signal, samples_per_second, cutoff=LOWPASS_HZ):
    """Smooth a signal so that small jitters don't look like steps."""
    if cutoff >= 0.45 * samples_per_second or len(signal) < 30:
        return signal                      # too little data / too low a sample rate to filter
    b, a = butter(2, cutoff / (samples_per_second / 2))
    return filtfilt(b, a, signal)


def symmetry_index(a, b):
    """How different are two numbers, in percent? 0 = identical.
    Example: 10 and 12 -> difference 2, average 11 -> 18%."""
    average = 0.5 * (abs(a) + abs(b))
    if average < 1e-9:
        return float("nan")                # both zero: can't say
    return 100.0 * abs(a - b) / average


def coefficient_of_variation(values):
    """How much do the numbers vary, in percent of their average?
    (standard deviation divided by the mean, times 100). Needs at least 3 numbers."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or abs(values.mean()) < 1e-9:
        return float("nan")
    return 100.0 * values.std(ddof=1) / abs(values.mean())


def mean_ignoring_nan(values):
    """Average of a list, skipping NaN ("not a number") entries."""
    good = []
    for v in values:
        if not math.isnan(v):
            good.append(v)
    if len(good) == 0:
        return float("nan")
    return float(np.mean(good))


def root_mean_square(values):
    """Square every number, average them, take the square root.
    Unlike a plain average, one big value stands out instead of being hidden by small ones."""
    good = []
    for v in values:
        if not math.isnan(v):
            good.append(v)
    if len(good) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(good))))


def linear_score(value, best, worst):
    """Turn a measurement into 0-100 points (see the SETTINGS section)."""
    points = 100.0 * (worst - value) / (worst - best)
    if points > 100.0:
        points = 100.0
    if points < 0.0:
        points = 0.0
    return float(points)


# ==========================================================================
# 4. RESULT CLASSES
# ==========================================================================

class Metric:
    """One measurement and the score it earned."""

    def __init__(self, name, value, detail=None):
        settings = SPECS[name]
        self.name = name
        self.label = settings["label"]
        self.component = settings["component"]
        self.unit = settings["unit"]
        self.best = settings["best"]
        self.worst = settings["worst"]
        self.weight = settings["weight"]
        self.value = float(value)                                          # the measurement
        self.score = linear_score(self.value, self.best, self.worst)       # 0-100 points
        if detail is None:
            detail = {}
        self.detail = detail                                               # extra numbers (e.g. per joint)


def add_metric(metric_list, name, value, detail=None):
    """Add a metric to the list, unless the value is missing (NaN)."""
    if value is None or math.isnan(value):
        return
    metric_list.append(Metric(name, value, detail))


class Result:
    """The final answer: the score, its three parts, and all the metrics."""

    def __init__(self, score, components, metrics, inputs, info, warnings):
        self.score = score              # total, 0-100
        self.components = components    # {"symmetry": .., "consistency": .., "movement": ..}
        self.metrics = metrics          # list of Metric
        self.inputs = inputs            # which files were used
        self.info = info                # facts about the trial (number of strides etc.)
        self.warnings = warnings        # list of text warnings

    def to_dict(self):
        """The result as plain dictionaries/lists, ready to save as JSON."""

        def tidy(number):
            if isinstance(number, float) and math.isnan(number):
                return None
            return round(float(number), 3)

        info = {}
        for key, value in self.info.items():
            if isinstance(value, float):
                info[key] = tidy(value)
            else:
                info[key] = value

        components = {}
        for key, value in self.components.items():
            components[key] = tidy(value)

        metrics = []
        for m in self.metrics:
            detail = {}
            for key, value in m.detail.items():
                detail[key] = tidy(value)
            metrics.append({"name": m.name, "component": m.component, "value": tidy(m.value),
                            "unit": m.unit, "score": tidy(m.score), "weight": m.weight,
                            "detail": detail})

        return {"version": __version__, "score": tidy(self.score), "components": components,
                "inputs": self.inputs, "info": info, "metrics": metrics,
                "warnings": self.warnings}

    def report(self):
        """The result as readable text."""
        lines = []
        lines.append("Gait efficiency score: " + format(self.score, ".1f") + " / 100")
        lines.append("")
        for name, value in self.components.items():
            lines.append("  " + format(name.capitalize(), "<12") + " " + format(value, "5.1f"))

        i = self.info
        if i["overground"]:
            walking_type = "overground"
        else:
            walking_type = "in place / treadmill"
        lines.append("")
        lines.append(str(i["n_trials"]) + " trial(s): " + str(i["n_strides_left"]) + " left / " +
                     str(i["n_strides_right"]) + " right strides, stride time " +
                     format(i["mean_stride_time_s"], ".2f") + " s, final-step strides ignored: " +
                     str(i["drop_last"]) + ", " + walking_type + ", inputs: " + ", ".join(self.inputs))
        lines.append("")

        for component in self.components:
            lines.append(component.capitalize())
            for m in self.metrics:
                if m.component == component:
                    lines.append("  " + format(m.label, "<40") + " " + format(m.value, "8.2f") + " " +
                                 format(m.unit, "<9") + " -> " + format(m.score, "5.1f"))

        if len(self.warnings) > 0:
            lines.append("")
            lines.append("Warnings")
            for w in self.warnings:
                lines.append("  - " + w)
        return "\n".join(lines)


# ==========================================================================
# 5. FINDING STRIDES
# ==========================================================================

class Strides:
    """Where the strides are in one trial.
        peaks[side] : row numbers where the hip is bent the most (about when the heel lands)
        pairs[side] : list of (start_row, end_row) - one entry per stride, from one peak
                      to the next peak of the same leg."""

    def __init__(self, peaks, pairs):
        self.peaks = peaks
        self.pairs = pairs


def detect_strides(kin, samples_per_second, warnings, label="", drop_last=0):
    """Find the strides of both legs in one trial.

    A stride goes from one moment of maximum hip bend to the next one of the same leg.
    drop_last: ignore the strides ending at the last N peaks of the trial (both legs
    together). With 1, the stride into the final step (feet coming together) is not used."""
    time = kin["time"].to_numpy()
    prefix = ""
    if label != "":
        prefix = label + ": "

    peaks = {}
    pairs = {}
    for side in SIDES:
        if side == "l":
            leg_name = "left"
        else:
            leg_name = "right"

        # Smooth the hip angle so tiny wiggles aren't mistaken for steps.
        hip = lowpass(kin["hip_flexion_" + side].to_numpy(), samples_per_second)

        # How far does the hip move? (5th to 95th percentile ignores odd spikes.)
        hip_range = np.percentile(hip, 95) - np.percentile(hip, 5)
        if hip_range < MIN_HIP_RANGE_DEG:
            raise GaitError(prefix + "hip flexion range is only " + format(hip_range, ".1f") +
                            " deg on the " + leg_name + " - this doesn't look like walking "
                            "(is the file in radians without 'inDegrees=no'?).")

        # A "peak" must stick out by at least 40% of the hip range, and be 0.5 s from the next.
        min_gap = max(1, int(0.5 * samples_per_second))
        found, _ = find_peaks(hip, prominence=0.4 * hip_range, distance=min_gap)
        peaks[side] = found

        # Join each peak to the next one to make strides; skip ones with impossible durations.
        good_strides = []
        for k in range(len(found) - 1):
            start = found[k]
            end = found[k + 1]
            duration = time[end] - time[start]
            if MIN_STRIDE_S <= duration <= MAX_STRIDE_S:
                good_strides.append((start, end))
        skipped = (len(found) - 1) - len(good_strides)
        if skipped > 0:
            warnings.append(prefix + "dropped " + str(skipped) + " implausible " + leg_name +
                            " stride(s) (outside " + str(MIN_STRIDE_S) + "-" + str(MAX_STRIDE_S) + " s).")
        pairs[side] = good_strides

    # Ignore the last few steps of the trial, if asked.
    if drop_last > 0:
        # List every peak of both legs as (time, side, row) and sort by time.
        events = []
        for side in SIDES:
            for row in peaks[side]:
                events.append((time[row], side, row))
        events.sort()
        for event in events[-drop_last:]:
            side = event[1]
            row = event[2]
            kept = []
            for stride in pairs[side]:
                if stride[1] != row:           # keep strides that don't END at this peak
                    kept.append(stride)
            pairs[side] = kept

    return Strides(peaks, pairs)


# ==========================================================================
# 6. COLLECTING DATA FROM STRIDES
# ==========================================================================

def stretch_strides(signal, stride_list):
    """Resample every stride to N_POINTS values (0% to 100% of the stride) so strides of
    different lengths can be compared point by point.
    Returns a table: one row per stride, N_POINTS columns."""
    target = np.linspace(0, 1, N_POINTS)
    rows = []
    for start, end in stride_list:
        piece = signal[start:end + 1]
        original = np.linspace(0, 1, len(piece))
        rows.append(np.interp(target, original, piece))
    return np.array(rows)


def range_per_stride(signal, stride_list):
    """For each stride: highest value minus lowest value (the range of motion)."""
    ranges = []
    for start, end in stride_list:
        ranges.append(np.ptp(signal[start:end + 1]))
    return np.array(ranges)


def pelvis_travel(kin):
    """Work out how far the pelvis moved forward and how far it wandered sideways.
    The walking direction is found from the pelvis path itself, so it doesn't matter
    which way the person walked in the room."""
    tx = kin["pelvis_tx"].to_numpy()
    tz = kin["pelvis_tz"].to_numpy()
    total_distance = float(np.hypot(np.ptp(tx), np.ptp(tz)))

    if total_distance >= MIN_TRAVEL_M:
        # Walking across the room. Find the main direction of the path.
        points = np.column_stack([tx, tz])
        points = points - points.mean(0)
        _, _, directions = np.linalg.svd(points, full_matrices=False)
        forward = points @ directions[0]       # distance along the walking direction
        sideways = points @ directions[1]      # distance across the walking direction
        return {"overground": True, "progression": forward, "lateral": sideways}

    # Barely moved: treadmill or walking in place.
    return {"overground": False, "progression": tx - tx.mean(), "lateral": tz - tz.mean()}


class Pooled:
    """One big pile of per-stride measurements, collected from one or more trials.
    Trials are NEVER glued together in time. Each trial's strides are found separately,
    then the measurements from every stride are put in the same pile."""

    def __init__(self):
        # joint curves: key = (joint name, side), value = list of tables (one per trial)
        self.cycles = {}
        for joint in ROM_JOINTS:
            for side in SIDES:
                self.cycles[(joint, side)] = []

        self.stride_times = {"l": [], "r": []}   # duration of each stride (seconds)
        self.stride_lengths = []                 # distance the pelvis moved per stride (metres)

        self.midline_ranges = {}                 # range of motion per stride, for body-centre columns
        for column in MIDLINE_COLS:
            self.midline_ranges[column] = []

        self.lateral_ranges = []                 # sideways pelvis range per stride (metres)
        self.step_a = []                         # time from a right peak to the next left peak
        self.step_b = []                         # time from that left peak to the next right peak

        self.n_trials = 0
        self.n_overground = 0                    # how many trials were walking across a room
        self.duration_s = 0.0

    def curves(self, joint, side):
        """All stretched-out stride curves of one joint on one side, as one table."""
        pieces = self.cycles[(joint, side)]
        if len(pieces) == 0:
            return np.empty((0, N_POINTS))
        return np.vstack(pieces)


def extract_trial(kin, samples_per_second, pooled, warnings, label="", drop_last=0, strides=None):
    """Measure every stride in one trial and add the results to the pooled pile.
    (strides: already-found strides; if None they are detected here.)"""
    time = kin["time"].to_numpy()
    if strides is None:
        strides = detect_strides(kin, samples_per_second, warnings, label, drop_last)
    travel = pelvis_travel(kin)

    pooled.n_trials += 1
    if travel["overground"]:
        pooled.n_overground += 1
    pooled.duration_s += float(time[-1] - time[0])

    # Measurements for each leg's strides.
    for side in SIDES:
        stride_list = strides.pairs[side]
        if len(stride_list) == 0:
            continue

        for joint in ROM_JOINTS:
            angles = kin[joint + "_" + side].to_numpy()
            pooled.cycles[(joint, side)].append(stretch_strides(angles, stride_list))

        for start, end in stride_list:
            pooled.stride_times[side].append(float(time[end] - time[start]))
            if travel["overground"]:
                distance = abs(travel["progression"][end] - travel["progression"][start])
                pooled.stride_lengths.append(float(distance))

    # Measurements of the body centre. Both legs' strides are used as time windows.
    all_strides = strides.pairs["l"] + strides.pairs["r"]
    if len(all_strides) > 0:
        for column in MIDLINE_COLS:
            ranges = range_per_stride(kin[column].to_numpy(), all_strides)
            pooled.midline_ranges[column] += list(ranges)

        # Remove slow drifting off the walking line, then measure the sideways range.
        sideways = detrend(travel["lateral"], type="linear")
        ranges = range_per_stride(sideways, all_strides)
        pooled.lateral_ranges += list(ranges)

    # Step timing: for each right stride, find the one left peak that falls inside it.
    left_peak_times = time[strides.peaks["l"]]
    for right_start, right_end in strides.pairs["r"]:
        inside = left_peak_times[(left_peak_times > time[right_start]) &
                                 (left_peak_times < time[right_end])]
        if len(inside) == 1:
            pooled.step_a.append(float(inside[0] - time[right_start]))
            pooled.step_b.append(float(time[right_end] - inside[0]))


# ==========================================================================
# 7. THE METRICS
# ==========================================================================

def symmetry_metrics(pooled, warnings):
    """Left vs right. Returns a list of Metric."""
    metrics = []

    # --- Metric 1: is the range of motion the same on both sides? ---
    # Compare the typical (median) range of each joint, left vs right.
    per_joint = {}
    for joint in ROM_JOINTS:
        ranges = {}
        for side in SIDES:
            ranges[side] = float(np.median(np.ptp(pooled.curves(joint, side), axis=1)))
        per_joint[joint] = symmetry_index(ranges["l"], ranges["r"])
    add_metric(metrics, "rom_symmetry_index", root_mean_square(per_joint.values()), per_joint)

    # --- Metric 2: is the SHAPE of the joint movement the same on both sides? ---
    # Average all left strides into one curve, same for right, then see how far apart they are.
    per_joint = {}
    for joint in WAVEFORM_JOINTS:
        left_curve = pooled.curves(joint, "l").mean(0)
        right_curve = pooled.curves(joint, "r").mean(0)
        typical_range = 0.5 * (np.ptp(left_curve) + np.ptp(right_curve))
        if typical_range > 1e-6:
            gap = float(np.sqrt(np.mean((left_curve - right_curve) ** 2)))
            per_joint[joint] = 100.0 * gap / typical_range     # as % of the joint's range
        else:
            per_joint[joint] = float("nan")
    add_metric(metrics, "waveform_asymmetry", root_mean_square(per_joint.values()), per_joint)

    # --- Metric 3: is the time of the right->left step the same as left->right? ---
    if len(pooled.step_a) >= 2:
        right_to_left = float(np.mean(pooled.step_a))
        left_to_right = float(np.mean(pooled.step_b))
        add_metric(metrics, "step_time_asymmetry", symmetry_index(right_to_left, left_to_right),
                   {"right_to_left_step_s": right_to_left, "left_to_right_step_s": left_to_right})
    else:
        warnings.append("Could not pair left/right steps; step-time symmetry skipped.")

    return metrics


def consistency_metrics(pooled, warnings):
    """Is each stride like the others? Returns a list of Metric."""
    metrics = []

    # --- Metric 1: how much does the stride DURATION vary? (average of left and right) ---
    left_cv = coefficient_of_variation(pooled.stride_times["l"])
    right_cv = coefficient_of_variation(pooled.stride_times["r"])
    add_metric(metrics, "stride_time_cv", mean_ignoring_nan([left_cv, right_cv]),
               {"left_cv": left_cv, "right_cv": right_cv})

    # --- Metric 2: how much does the stride LENGTH vary? (only when walking across a room) ---
    all_overground = (pooled.n_overground == pooled.n_trials)
    if all_overground and np.mean(pooled.stride_lengths) >= MIN_STRIDE_LENGTH_M:
        add_metric(metrics, "stride_length_cv", coefficient_of_variation(pooled.stride_lengths),
                   {"mean_stride_length_m": float(np.mean(pooled.stride_lengths))})
    else:
        warnings.append("Pelvis does not travel far enough (treadmill / in-place?): stride-length "
                        "consistency skipped; stride-time and waveform consistency still scored.")

    # --- Metric 3: does the SHAPE of each joint's movement change from stride to stride? ---
    # At each of the 101 points of the stride, measure the spread across strides,
    # average those spreads, and express them as a % of the joint's range.
    values = {}
    for joint in WAVEFORM_JOINTS:
        for side in SIDES:
            curves = pooled.curves(joint, side)
            joint_range = np.ptp(curves.mean(0))
            key = joint + "_" + side
            if joint_range > 1e-6 and len(curves) > 1:
                spread = float(curves.std(axis=0, ddof=1).mean())
                values[key] = 100.0 * spread / joint_range
            else:
                values[key] = float("nan")
    add_metric(metrics, "waveform_variability", mean_ignoring_nan(values.values()), values)

    return metrics


def movement_metrics(pooled):
    """How much unnecessary movement is there? Returns a list of Metric.
    Each value is the typical (median) range of motion within one stride."""
    metrics = []

    # Vertical bounce of the pelvis (metres -> centimetres).
    add_metric(metrics, "vertical_oscillation", 100.0 * float(np.median(pooled.midline_ranges["pelvis_ty"])))

    # Side-to-side sway of the pelvis (metres -> centimetres).
    add_metric(metrics, "lateral_sway", 100.0 * float(np.median(pooled.lateral_ranges)))

    # Pelvis tilt/twist and lower-back bend/twist (degrees).
    add_metric(metrics, "pelvis_list_rom", float(np.median(pooled.midline_ranges["pelvis_list"])))
    add_metric(metrics, "pelvis_rotation_rom", float(np.median(pooled.midline_ranges["pelvis_rotation"])))
    add_metric(metrics, "lumbar_bending_rom", float(np.median(pooled.midline_ranges["lumbar_bending"])))
    add_metric(metrics, "lumbar_rotation_rom", float(np.median(pooled.midline_ranges["lumbar_rotation"])))

    # How far each leg swings sideways (hip abduction/adduction); average of the two legs.
    left = float(np.median(np.ptp(pooled.curves("hip_adduction", "l"), axis=1)))
    right = float(np.median(np.ptp(pooled.curves("hip_adduction", "r"), axis=1)))
    add_metric(metrics, "hip_adduction_rom", (left + right) / 2.0,
               {"left_deg": left, "right_deg": right})

    return metrics


# ==========================================================================
# 8. GROUND FORCE (OPTIONAL)
# ==========================================================================

def find_stances(force, time, threshold):
    """Find each time a foot is on the ground (force above the threshold).
    Only complete stances are returned: the foot must land AND leave inside the data."""
    on_ground = force > threshold
    changes = np.diff(on_ground.astype(int))
    landings = np.where(changes == 1)[0] + 1        # rows where force goes from low to high
    takeoffs = np.where(changes == -1)[0] + 1       # rows where force goes from high to low

    stances = []
    for landing in landings:
        later_takeoffs = takeoffs[takeoffs > landing]
        if len(later_takeoffs) == 0:
            continue                                 # data ended while the foot was still down
        takeoff = later_takeoffs[0]
        duration = time[takeoff] - time[landing]
        if 0.2 <= duration <= 1.5:                   # ignore impossible stance times
            stances.append({"dur": duration,
                            "peak": float(force[landing:takeoff].max()),
                            "impulse": float(trapezoid(force[landing:takeoff], time[landing:takeoff]))})
    return stances


def average_of(stances, key):
    values = []
    for stance in stances:
        values.append(stance[key])
    return float(np.mean(values))


def grf_metrics(grf, warnings):
    """Extra left/right metrics from vertical ground force. Returns a list of Metric."""
    time = grf["time"].to_numpy()
    right_force = grf["ground_force_right_vy"].to_numpy()
    left_force = grf["ground_force_left_vy"].to_numpy()

    # "Foot on the ground" = force above 5% of the typical peak force.
    threshold = 0.05 * np.percentile(np.maximum(right_force, left_force), 95)
    right = find_stances(right_force, time, threshold)
    left = find_stances(left_force, time, threshold)

    if min(len(right), len(left)) < MIN_STRIDES:
        warnings.append("GRF file has only " + str(len(left)) + " left / " + str(len(right)) +
                        " right complete stance phases (need " + str(MIN_STRIDES) +
                        " each): GRF metrics skipped and NOT included in the score.")
        return []

    metrics = []
    add_metric(metrics, "grf_peak_asymmetry",
               symmetry_index(average_of(left, "peak"), average_of(right, "peak")))
    add_metric(metrics, "grf_impulse_asymmetry",
               symmetry_index(average_of(left, "impulse"), average_of(right, "impulse")))
    add_metric(metrics, "stance_time_asymmetry",
               symmetry_index(average_of(left, "dur"), average_of(right, "dur")))

    left_peaks = []
    for stance in left:
        left_peaks.append(stance["peak"])
    right_peaks = []
    for stance in right:
        right_peaks.append(stance["peak"])
    add_metric(metrics, "grf_peak_cv",
               mean_ignoring_nan([coefficient_of_variation(left_peaks),
                                  coefficient_of_variation(right_peaks)]))
    return metrics


# ==========================================================================
# 9. PUTTING IT TOGETHER
# ==========================================================================

def analyze_trials(trials, grf=None, drop_last=DEFAULT_DROP_LAST, strides=None):
    """Score one or more trials of the SAME person.

    trials : list of (kinematics table, samples per second, name). Their strides are
             pooled, so several short trials can be combined.
    grf    : optional ground-force table.
    strides: (used by the chart code) strides that were already found."""
    warnings = []
    pooled = Pooled()

    # Step 1: measure all the strides of every trial.
    for number, trial in enumerate(trials):
        kin = trial[0]
        samples_per_second = trial[1]
        label = trial[2]
        if len(trials) == 1:
            label = ""                       # no need to name the file if there is only one
        found = None
        if strides is not None:
            found = strides[number]
        extract_trial(kin, samples_per_second, pooled, warnings, label, drop_last, found)

    # Step 2: make sure there are enough strides on each leg.
    for side in SIDES:
        if side == "l":
            leg_name = "left"
        else:
            leg_name = "right"
        count = len(pooled.stride_times[side])

        if count < MIN_STRIDES:
            where = ""
            if pooled.n_trials > 1:
                where = " across " + str(pooled.n_trials) + " trials"
            raise GaitError("Only " + str(count) + " complete " + leg_name + " strides detected" + where +
                            "; need at least " + str(MIN_STRIDES) + " (ideally " +
                            str(RECOMMENDED_STRIDES) + "+). Use longer walking trials, add more "
                            "trials of the same person, lower --drop-last, or this file is not a "
                            "walking trial.")
        if count < RECOMMENDED_STRIDES:
            warnings.append("Only " + str(count) + " " + leg_name + " strides - variability metrics "
                            "are noisy below " + str(RECOMMENDED_STRIDES) + ".")

    if pooled.n_trials > 1 and 0 < pooled.n_overground < pooled.n_trials:
        warnings.append("Mix of overground and treadmill trials - stride-length consistency skipped.")

    # Step 3: calculate every metric.
    metrics = []
    metrics += symmetry_metrics(pooled, warnings)
    metrics += consistency_metrics(pooled, warnings)
    metrics += movement_metrics(pooled)

    if pooled.n_trials == 1:
        inputs = ["kinematics"]
    else:
        inputs = ["kinematics x" + str(pooled.n_trials) + " trials"]

    if grf is not None:
        extra = grf_metrics(grf, warnings)
        if len(extra) > 0:
            metrics += extra
            inputs.append("grf")

    # Step 4: each group's score = weighted average of its metrics' scores.
    components = {}
    for component in COMPONENT_WEIGHTS:
        weighted_total = 0.0
        weight_total = 0.0
        for m in metrics:
            if m.component == component:
                weighted_total += m.weight * m.score
                weight_total += m.weight
        if weight_total == 0:
            raise GaitError("No valid metrics for component '" + component + "'.")
        components[component] = weighted_total / weight_total

    # Step 5: the total score = weighted average of the three groups.
    total_points = 0.0
    total_weight = 0.0
    for component in components:
        total_points += COMPONENT_WEIGHTS[component] * components[component]
        total_weight += COMPONENT_WEIGHTS[component]
    total = total_points / total_weight

    # Step 6: a few facts about the trial(s), for the report.
    all_stride_times = pooled.stride_times["l"] + pooled.stride_times["r"]
    rates = []
    for trial in trials:
        rates.append(trial[1])
    info = {"n_trials": pooled.n_trials,
            "n_strides_left": len(pooled.stride_times["l"]),
            "n_strides_right": len(pooled.stride_times["r"]),
            "mean_stride_time_s": float(np.mean(all_stride_times)),
            "overground": pooled.n_overground == pooled.n_trials,
            "sample_rate_hz": float(np.mean(rates)),
            "duration_s": float(pooled.duration_s),
            "drop_last": drop_last}

    return Result(total, components, metrics, inputs, info, warnings)


def analyze(kin, samples_per_second, grf=None, drop_last=DEFAULT_DROP_LAST):
    """Score a single trial that is already loaded as a table."""
    return analyze_trials([(kin, samples_per_second, "")], grf, drop_last)


def load_trials(kin_paths, t_start=None, t_end=None):
    """Load one or more kinematics files. Returns a list of (table, rate, file name)."""
    if isinstance(kin_paths, (str, Path)):
        kin_paths = [kin_paths]                 # a single file name -> make it a list
    trials = []
    for path in kin_paths:
        kin, samples_per_second = load_kinematics(path, t_start, t_end)
        trials.append((kin, samples_per_second, Path(path).name))
    return trials


def score_files(kin_paths, grf_path=None, t_start=None, t_end=None, drop_last=DEFAULT_DROP_LAST):
    """Score from file names. kin_paths is one file name, or a list of files of the same person."""
    if isinstance(kin_paths, (str, Path)):
        kin_paths = [kin_paths]
    if grf_path and len(kin_paths) > 1:
        raise GaitError("--grf can only be used with a single kinematics file.")
    trials = load_trials(kin_paths, t_start, t_end)
    grf = None
    if grf_path:
        grf = load_grf(grf_path, t_start, t_end)
    return analyze_trials(trials, grf, drop_last)


# ==========================================================================
# 10. SCORE VS NUMBER OF STEPS (CHART)
# ==========================================================================

def score_curve(kin_paths, t_start=None, t_end=None, drop_last=DEFAULT_DROP_LAST):
    """Score using only the first N steps, for N = 1, 2, 3, ... up to all of them.

    A "step" here is one heel strike (the end of one stride of either leg). Steps are
    added in time order: trial by trial, in the order the files are given.
    The last point equals the normal score. A point only exists once each leg has
    enough strides (MIN_STRIDES).

    Returns a list with one dictionary per point: steps, score, symmetry, consistency,
    movement, n_left, n_right, and reliable (True if both legs have at least
    RECOMMENDED_STRIDES strides)."""
    trials = load_trials(kin_paths, t_start, t_end)

    # Find the strides of every trial once.
    all_strides = []
    for kin, samples_per_second, label in trials:
        all_strides.append(detect_strides(kin, samples_per_second, [], label, drop_last))

    # Put every stride from every trial in time order: (trial number, end row, side).
    order = []
    for trial_number, found in enumerate(all_strides):
        for side in SIDES:
            for start, end in found.pairs[side]:
                order.append((trial_number, end, side))
    order.sort()

    curve = []
    for n_steps in range(1, len(order) + 1):
        keep = set(order[:n_steps])              # the first n_steps strides

        # Build a copy of the strides containing only those.
        reduced = []
        for trial_number, found in enumerate(all_strides):
            kept_pairs = {}
            for side in SIDES:
                kept_pairs[side] = []
                for start, end in found.pairs[side]:
                    if (trial_number, end, side) in keep:
                        kept_pairs[side].append((start, end))
            reduced.append(Strides(found.peaks, kept_pairs))

        try:
            result = analyze_trials(trials, None, drop_last, strides=reduced)
        except GaitError:
            continue                              # not enough strides on one leg yet

        left = result.info["n_strides_left"]
        right = result.info["n_strides_right"]
        point = {"steps": n_steps,
                 "score": result.score,
                 "symmetry": result.components["symmetry"],
                 "consistency": result.components["consistency"],
                 "movement": result.components["movement"],
                 "n_left": left,
                 "n_right": right,
                 "reliable": min(left, right) >= RECOMMENDED_STRIDES}
        curve.append(point)

    if len(curve) == 0:
        raise GaitError("Not enough steps to draw a curve: each leg needs at least " +
                        str(MIN_STRIDES) + " strides (after ignoring the final step).")
    return curve


def draw_line(ax, curve, key, label, color, line_width=2.0):
    """Draw one line on a chart. Hollow dots = noisy points (few strides)."""
    x_values = []
    y_values = []
    dot_colors = []
    for point in curve:
        x_values.append(point["steps"])
        y_values.append(point[key])
        if point["reliable"]:
            dot_colors.append(color)
        else:
            dot_colors.append("white")
    ax.plot(x_values, y_values, color=color, lw=line_width, label=label, zorder=2)
    ax.scatter(x_values, y_values, s=28, zorder=3, color=dot_colors, edgecolors=color, linewidths=1.4)


def style_axes(ax, title, x_min, x_max, show_x_label=True):
    """Titles, limits and grid for one chart."""
    from matplotlib.ticker import MaxNLocator
    ax.set_ylim(0, 100)
    ax.set_xlim(x_min - 1, x_max + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))     # whole numbers on the x axis
    ax.set_title(title, fontsize=11)
    if show_x_label:
        ax.set_xlabel("Number of steps included")
    else:
        ax.set_xlabel("")
    ax.set_ylabel("Score (0-100)")
    ax.grid(alpha=0.3)


def plot_score_curves(sessions, out=None, show=None, title=None, t_start=None, t_end=None,
                      drop_last=DEFAULT_DROP_LAST):
    """Draw score (y axis, 0-100) against number of steps (x axis).

    sessions: dictionary {name: file or list of files of that session's trials}
        - one session   -> one chart with the total score and its three parts
        - several       -> four small charts (total, symmetry, consistency, movement),
                           one line per session
    out : file name to save the picture (e.g. "chart.png"); None = just show it.
    Hollow dots mean a leg has fewer than RECOMMENDED_STRIDES strides (noisy)."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        raise GaitError("Plotting needs matplotlib: pip install matplotlib")

    # Work out the curve of every session.
    curves = {}
    for name, paths in sessions.items():
        curves[name] = score_curve(paths, t_start, t_end, drop_last)

    x_max = 0
    x_min = 10 ** 9
    for curve in curves.values():
        x_max = max(x_max, curve[-1]["steps"])
        x_min = min(x_min, curve[0]["steps"])

    if len(curves) == 1:
        # ---- one session: a single chart ----
        name = list(curves.keys())[0]
        curve = curves[name]
        fig, ax = plt.subplots(figsize=(8, 5))
        draw_line(ax, curve, "score", "Total", "black", 3.0)
        draw_line(ax, curve, "symmetry", "Symmetry", "tab:blue", 1.8)
        draw_line(ax, curve, "consistency", "Consistency", "tab:orange", 1.8)
        draw_line(ax, curve, "movement", "Movement", "tab:green", 1.8)
        if title is None:
            title = "Score vs number of steps - " + name
        style_axes(ax, title, x_min, x_max)
        ax.legend(loc="lower right")
    else:
        # ---- several sessions: four small charts, one line per session ----
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        panels = [("score", "Total score"), ("symmetry", "Symmetry"),
                  ("consistency", "Consistency"), ("movement", "Movement")]
        colors = plt.get_cmap("tab10")
        axes_list = axes.ravel()
        for number in range(4):
            ax = axes_list[number]
            key = panels[number][0]
            panel_title = panels[number][1]
            session_number = 0
            for name, curve in curves.items():
                draw_line(ax, curve, key, name, colors(session_number))
                session_number += 1
            is_bottom_row = number >= 2
            style_axes(ax, panel_title, x_min, x_max, show_x_label=is_bottom_row)
            ax.tick_params(labelbottom=True)
        axes_list[0].legend(loc="lower right", fontsize=9)
        if title:
            fig.suptitle(title)

    fig.text(0.5, 0.005, "Steps = heel strikes, added in time order. Hollow markers: fewer than " +
             str(RECOMMENDED_STRIDES) + " strides on a leg (noisy).", ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    if out:
        fig.savefig(out, dpi=150)

    want_window = (show is True) or (show is None and not out)
    if want_window:
        backend = matplotlib.get_backend()
        if backend.lower() not in NON_WINDOW_BACKENDS:
            plt.show()
        else:
            # This backend can only save pictures, so a window is impossible.
            # Save the chart as a file instead and say how to get a window.
            picture = out
            if not picture:
                picture = "score_vs_steps.png"
                fig.savefig(picture, dpi=150)
            print()
            print("The chart can't be shown in a window because the current Matplotlib backend")
            print("('" + backend + "') only saves pictures. The chart was saved as:")
            print("    " + str(Path(picture).resolve()))
            print("To see charts inside Spyder: Tools > Preferences > IPython console > Graphics,")
            print("set Backend to Inline (or Automatic), then restart the kernel")
            print("(Consoles > Restart kernel).")
    return fig


# ==========================================================================
# 11. COMMAND LINE
# ==========================================================================

def clean_file_name(text):
    """Remove spaces and quote marks around a file name (people often paste "C:\\file.mot")."""
    return text.strip().strip('"').strip("'").strip()


def split_file_names(text):
    """Turn what the user typed into a list of file names.
    - If there are commas, the commas separate the names (so names may contain spaces).
    - Otherwise a single existing file is used as it is, or names are split at spaces."""
    text = text.strip()
    if text == "":
        return []

    if "," in text:
        pieces = text.split(",")
    elif Path(clean_file_name(text)).is_file():
        pieces = [text]                        # one file whose name may contain spaces
    else:
        pieces = text.split()

    names = []
    for piece in pieces:
        name = clean_file_name(piece)
        if name != "":
            names.append(name)
    return names


def ask_for_files():
    """Ask the user to type the file names. Keeps asking until every file exists.
    Returns the list of names, or an empty list if the user just presses Enter."""
    print("Gait efficiency score")
    print("---------------------")
    print("Type the names of the .mot files to score, for ONE person / session.")
    print("Separate the names with commas, e.g.:   walk1.mot, walk2.mot, walk3.mot")
    print("(Files are looked for in: " + str(Path.cwd()) + ")")
    print("Press Enter with nothing typed to quit.")
    print()

    while True:
        try:
            text = input("File names: ")
        except EOFError:                       # no keyboard available (e.g. input piped in)
            return []

        names = split_file_names(text)
        if len(names) == 0:
            return []

        not_found = []
        for name in names:
            if not Path(name).is_file():
                not_found.append(name)
        if len(not_found) == 0:
            return names

        print("Could not find: " + ", ".join(not_found))
        print("Check the spelling, or type the full path, and try again.")
        print()


def ask_yes_no(question):
    """Ask a yes/no question. Anything starting with y counts as yes."""
    try:
        answer = input(question + " (y/n): ")
    except EOFError:
        return False
    return answer.strip().lower().startswith("y")


def main(argv=None):
    """Run from a terminal:  python gait_score.py file1.mot [file2.mot ...] [options]"""
    parser = argparse.ArgumentParser(description="Score walking efficiency from OpenSim/OpenCap .mot files.")
    parser.add_argument("kinematics", nargs="*",
                        help="inverse-kinematics .mot of a walking trial; pass several trials of the "
                             "SAME person to pool their strides. Leave empty to be asked for the "
                             "file names")
    parser.add_argument("--grf", help="ground reaction force .mot (single kinematics file only)")
    parser.add_argument("--t-start", type=float, help="crop start time (s) applied to every file")
    parser.add_argument("--t-end", type=float, help="crop end time (s) applied to every file")
    parser.add_argument("--drop-last", type=int, default=DEFAULT_DROP_LAST, metavar="N",
                        help="ignore the stride ending at the last N hip peaks of each trial, e.g. the "
                             "feet-together stopping step (default %(default)s; 0 keeps everything)")
    parser.add_argument("--plot", metavar="FILE.png",
                        help="also save a chart of the scores against number of steps")
    parser.add_argument("--json", metavar="FILE", help="also write the full result as JSON")
    args = parser.parse_args(argv)

    # No file names given (for example Spyder's Run button)? Ask for them.
    files = args.kinematics
    asked_interactively = (len(files) == 0)
    if asked_interactively:
        files = ask_for_files()
        if len(files) == 0:
            print("No files given, so nothing to score.")
            return 0

    # Score the files. If something is wrong with them, say so politely instead of crashing.
    try:
        result = score_files(files, args.grf, args.t_start, args.t_end, args.drop_last)
    except GaitError as error:
        print("Cannot score: " + str(error), file=sys.stderr)
        return 2

    print(result.report())

    if len(files) == 1:
        session_name = Path(files[0]).stem
    else:
        session_name = "session"

    if args.plot:
        # Save the score-vs-steps chart to a picture file (no window is opened).
        try:
            plot_score_curves({session_name: files}, out=args.plot, show=False,
                              t_start=args.t_start, t_end=args.t_end, drop_last=args.drop_last)
        except GaitError as error:
            print("Could not draw plot: " + str(error), file=sys.stderr)
            return 2
        print("Saved plot to " + args.plot)
    elif asked_interactively:
        # In interactive mode, offer to show the chart on screen.
        print()
        if ask_yes_no("Show the score-vs-steps chart?"):
            try:
                plot_score_curves({session_name: files}, show=True, t_start=args.t_start,
                                  t_end=args.t_end, drop_last=args.drop_last)
            except GaitError as error:
                print("Could not draw plot: " + str(error))

    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    exit_code = main()
    # With no command-line arguments (e.g. Spyder's Run button) just finish quietly;
    # sys.exit() would make Spyder print an alarming "SystemExit" message.
    if exit_code != 0 and len(sys.argv) > 1:
        sys.exit(exit_code)