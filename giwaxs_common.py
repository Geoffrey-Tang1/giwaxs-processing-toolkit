#!/usr/bin/env python3
"""
giwaxs_common.py
=================

Shared helpers for the GIWAXS processing agents:
  - giwaxs_2d1d_agent.py    (2D q-space image + 1D line-cut profiles)
  - giwaxs_polefigure_agent.py  (pole figures)

This module holds the pyFAI geometry construction, mask loading, file
discovery, interactive directory prompting, and plotting helpers that are
common to both agents so the geometry/calibration parameters stay
consistent between them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import warnings
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MultipleLocator, NullLocator

# matplotlib logs a "findfont: Font family '...' not found" warning for
# EVERY text element rendered with an unavailable font (once per axis
# label, tick, title, ...) -- on a busy server this floods the logs so
# badly that genuinely useful log output (build/deploy messages, our own
# print() diagnostics) gets pushed out of any bounded log viewer/download.
# We already handle "font not available" ourselves (see
# get_available_font_categories / resolve_font_family), so this warning
# is redundant noise here, not new information -- silence it specifically
# (not all matplotlib logging) so real problems still surface normally.
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# NumPy >= 2.0 renamed trapz -> trapezoid (and removed the old name in some
# builds); NumPy < 2.0 only has trapz. This shim works either way.
_trapz = getattr(np, "trapezoid", None) or np.trapz


class GiwaxsError(Exception):
    """Raised for user-facing input/configuration errors (bad file paths,
    missing required parameters, malformed data, etc.).

    Library functions in this module raise this instead of calling
    sys.exit() directly. sys.exit() raises SystemExit, which is NOT a
    subclass of Exception -- a normal "except Exception" handler (as used
    throughout giwaxs_app.py) does NOT catch it, so calling sys.exit()
    from code that's imported and called in-process by the Streamlit app
    (rather than run as its own standalone CLI process) would crash or
    hang the whole app session instead of showing a clean error message.

    Each CLI agent's main() catches GiwaxsError at the top level and
    converts it to sys.exit(str(exc)) itself, so command-line behaviour
    (clean one-line error message, non-zero exit code, no traceback) is
    unchanged from before this was introduced.
    """
    pass


def _raise_error(message: str):
    """Library code calls this instead of sys.exit() directly -- see
    GiwaxsError's docstring for why."""
    raise GiwaxsError(message)


# --------------------------------------------------------------------------- #
# Dependency import
# --------------------------------------------------------------------------- #
def import_pyfai_stack():
    """Import pyFAI / fabio pieces, with a friendly error if missing."""
    try:
        import fabio
        from pyFAI.integrator.fiber import FiberIntegrator
        from pyFAI.units import get_unit_fiber
        from pyFAI.detectors import Detector, detector_factory
    except ImportError as exc:
        _raise_error(
            "Missing dependency: {}\n"
            "Install requirements with:\n"
            "    pip install pyFAI fabio numpy matplotlib\n".format(exc)
        )
    return fabio, FiberIntegrator, get_unit_fiber, Detector, detector_factory


# --------------------------------------------------------------------------- #
# argparse value parsers
# --------------------------------------------------------------------------- #
def parse_range(text: str) -> Tuple[float, float]:
    try:
        a, b = text.split(",")
        return float(a), float(b)
    except Exception:
        raise argparse.ArgumentTypeError(
            "Expected two comma-separated numbers, e.g. '-0.5,2.4'"
        )


def parse_shape(text: str) -> Tuple[int, int]:
    try:
        h, w = text.split(",")
        return int(h), int(w)
    except Exception:
        raise argparse.ArgumentTypeError(
            "Expected 'height,width' in pixels, e.g. '1043,981'"
        )


def parse_fit_region(text: str) -> Tuple[float, float, str]:
    """Parse a peak-fitting window given as 'QMIN:QMAX' or 'QMIN:QMAX:LABEL'.

    Colon-separated (not comma-separated like parse_range) so that a label
    containing a comma -- e.g. '0.2:0.3:(100), lamellar' -- still parses.
    The label is optional and defaults to the numeric range, so a bare
    '1.6:1.8' is valid. q values are in inverse Angstrom.
    """
    parts = text.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "Expected 'QMIN:QMAX' or 'QMIN:QMAX:LABEL' in inverse Angstrom, "
            "e.g. '1.6:1.8:pi-pi stacking'"
        )
    try:
        q_min, q_max = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Could not read '{parts[0]}' and '{parts[1]}' as numbers -- "
            "expected 'QMIN:QMAX[:LABEL]', e.g. '1.6:1.8:pi-pi stacking'"
        )
    if q_max <= q_min:
        raise argparse.ArgumentTypeError(
            f"Fit region QMAX ({q_max}) must be greater than QMIN ({q_min})."
        )
    label = parts[2].strip() if len(parts) == 3 and parts[2].strip() else f"{q_min:g}-{q_max:g}"
    return q_min, q_max, label


# --------------------------------------------------------------------------- #
# Shared CLI arguments (geometry / calibration)
# --------------------------------------------------------------------------- #
def add_io_args(p: argparse.ArgumentParser):
    """Input/output path arguments. Left optional -- if not given on the
    command line, the agent will interactively prompt for them."""
    p.add_argument("--input", default=None,
                    help="Path to a single .tif/.tiff file, OR a directory "
                         "containing multiple .tif/.tiff files to batch-process. "
                         "If omitted, you will be prompted for it.")
    p.add_argument("--output-dir", default=None,
                    help="Destination directory for output images/data "
                         "(created automatically if it doesn't exist). "
                         "If omitted, you will be prompted for it.")


def add_geometry_args(p: argparse.ArgumentParser):
    """Beam centre / detector / grazing-incidence calibration arguments,
    shared by both agents so they stay consistent with each other.

    All of --beam-center-y/x, --distance, --wavelength/--energy,
    --rot1/2/3, --detector-name (or --pixel-size + --detector-shape) are
    OPTIONAL if --poni-file is given -- in that case the entire geometry
    (including the detector, so you never need to know its pixel size or
    shape yourself) is loaded directly from that file. Any of them given
    ALONGSIDE --poni-file are ignored (with a warning), since the file is
    treated as authoritative. Without --poni-file, beam centre / distance
    / wavelength-or-energy are required (validated at runtime, not by
    argparse, so the helpful "use --poni-file instead" message can be
    shown for a missing value rather than a generic argparse error).
    """
    p.add_argument("--poni-file", default=None,
                    help="Path to an existing pyFAI .poni file to load the "
                         "ENTIRE geometry from (distance, beam centre, "
                         "rotations, wavelength, and detector -- including "
                         "its pixel size and shape, so you don't need to "
                         "know those separately). Takes priority over all "
                         "other geometry arguments below if given. If you "
                         "also give --agbeh-file, this is used as the "
                         "initial guess for a fresh refinement rather than "
                         "the final answer.")

    p.add_argument("--beam-center-y", type=float, default=None,
                    help="Beam centre row position on the detector, in pixels "
                         "(equivalent to PONI1 / pixel size). Not needed if "
                         "--poni-file is given. If a calibration image is "
                         "also used (see add_calibration_args), this is only "
                         "the INITIAL GUESS for refinement; otherwise it is "
                         "used as-is.")
    p.add_argument("--beam-center-x", type=float, default=None,
                    help="Beam centre column position on the detector, in "
                         "pixels (equivalent to PONI2 / pixel size). Same "
                         "caveats as --beam-center-y.")
    p.add_argument("--distance", type=float, default=None,
                    help="Sample-to-detector distance, in metres. Same "
                         "caveats as --beam-center-y.")
    p.add_argument("--wavelength", type=float, default=None,
                    help="X-ray wavelength, in metres (e.g. 1.5406e-10 for "
                         "Cu-K-alpha). Provide this OR --energy. Not needed "
                         "if --poni-file is given.")
    p.add_argument("--energy", type=float, default=None,
                    help="X-ray photon energy, in keV. Alternative to --wavelength.")
    p.add_argument("--rot1", type=float, default=0.0,
                    help="Detector rotation 1, in radians (pyFAI convention). "
                         "Not needed if --poni-file is given.")
    p.add_argument("--rot2", type=float, default=0.0,
                    help="Detector rotation 2, in radians.")
    p.add_argument("--rot3", type=float, default=0.0,
                    help="Detector rotation 3, in radians.")

    p.add_argument("--detector-name", default=None,
                    help="Name of a built-in pyFAI detector (e.g. 'Pilatus1M', "
                         "'Eiger2_4M'). If given, overrides --pixel-size/"
                         "--detector-shape. Not needed if --poni-file is "
                         "given (the detector, including its shape, comes "
                         "from the file).")
    p.add_argument("--pixel-size", type=float, default=172e-6,
                    help="Detector pixel size, in metres (used if neither "
                         "--detector-name nor --poni-file is given). Default "
                         "is 172 micron (common Pilatus/Eiger pitch).")
    p.add_argument("--detector-shape", type=parse_shape, default=None,
                    help="Detector shape as 'height,width' in pixels "
                         "(used if neither --detector-name nor --poni-file "
                         "is given).")

    p.add_argument("--incident-angle", type=float, required=True,
                    help="Angle of incidence of the X-ray beam relative to the "
                         "sample surface, in degrees (used for the GI q-space "
                         "transform). This is NOT stored in a .poni file (it's "
                         "an experiment setting, not a detector geometry "
                         "parameter), so it's always required regardless of "
                         "--poni-file. Used as the fallback value for any "
                         "file if --incident-angle-from-filename is set but "
                         "no pattern is found in that file's name.")
    p.add_argument("--incident-angle-from-filename", action="store_true",
                    help="For each file, try to auto-detect its incident "
                         "angle from its filename using the '0p095'-style "
                         "convention (e.g. 'sample_0p095_1234.tif' -> 0.095 "
                         "degrees) -- useful for a batch/folder where "
                         "different frames used different angles. Falls "
                         "back to --incident-angle for any file where no "
                         "such pattern is found.")
    p.add_argument("--incident-angle-map", default=None,
                    help="Path to a JSON/CSV/Excel file mapping each "
                         "filename to its OWN incident angle (degrees), for "
                         "a batch where different files need different "
                         "angles but don't follow the '0p095' filename "
                         "convention. Takes priority over "
                         "--incident-angle-from-filename; any file not "
                         "listed falls back to that or --incident-angle.")

    p.add_argument("--mask", default=None,
                    help="Path to a mask image (any fabio-readable format, or "
                         ".npy). Nonzero/True pixels are excluded from integration.")

    p.add_argument("--npt", type=int, default=1000,
                    help="Number of integration bins used for pyFAI's 2D remap "
                         "and 1D integrations. The default is almost always "
                         "fine -- you don't need to change this.")


def add_calibration_args(p: argparse.ArgumentParser):
    """Optional AgBeh (or other calibrant) refinement arguments.

    If an calibration image is supplied (via --agbeh-file, or interactively
    when prompted), --beam-center-y/x and --distance are only used as the
    INITIAL GUESS for a proper pyFAI ring-fitting refinement (matching the
    calibration notebook this toolkit is based on); an approximate value is
    fine in that case. If no calibration image is used, those values are
    taken as the final, already-calibrated geometry as-is.
    """
    p.add_argument("--agbeh-file", default=None,
                    help="Path to an AgBeh (or other calibrant) TIFF image, "
                         "used to refine the beam centre / sample-detector "
                         "distance via pyFAI's ring-fitting calibration "
                         "before processing your data. If omitted (and "
                         "--no-calibration-prompt is not set), you will be "
                         "asked interactively whether you have one.")
    p.add_argument("--calibrant", default="AgBh",
                    help="Name of the calibrant standard used in "
                         "--agbeh-file, as recognised by pyFAI (e.g. 'AgBh', "
                         "'LaB6', 'CeO2', 'Si'). Only used if a calibration "
                         "image is given.")
    p.add_argument("--calib-max-rings", type=int, default=5,
                    help="Maximum number of calibrant rings to fit during "
                         "refinement.")
    p.add_argument("--calib-min-intensity", type=float, default=200.0,
                    help="Minimum peak intensity (Imin) used when detecting "
                         "calibrant ring control points.")
    p.add_argument("--save-calibrated-poni", default=None,
                    help="Optional path to save the refined geometry as a "
                         ".poni file after calibration, for reuse/inspection "
                         "later (e.g. in pyFAI-calib2 or another notebook).")
    p.add_argument("--no-calibration-prompt", action="store_true",
                    help="Do not interactively ask whether a calibration "
                         "image is available; only use one if --agbeh-file "
                         "is explicitly given. Useful for scripted/config-"
                         "driven runs where this has already been decided.")


# --------------------------------------------------------------------------- #
# Interactive directory prompting
# --------------------------------------------------------------------------- #
def strip_path_input(text: str) -> str:
    """Clean up a pasted path: strip whitespace and, if present, a single
    pair of matching surrounding quotes. This fixes a common Windows issue
    where "Copy as path" from Explorer copies the path WITH literal
    double-quote characters included (e.g. "C:\\data\\file.tif"), which
    would otherwise make os.path.exists() fail even though the path itself
    is correct.
    """
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def prompt_for_input_path(default: Optional[str] = None) -> str:
    """Interactively ask for the input file/directory; re-prompt until the
    path actually exists on disk."""
    while True:
        suffix = f" [{default}]" if default else ""
        text = strip_path_input(input(
            f"Enter path to the input TIFF file or directory of TIFFs{suffix}: "
        ))
        if not text and default:
            text = default
        if not text:
            print("  A path is required.")
            continue
        if not os.path.exists(text):
            print(f"  Path not found: {text}")
            continue
        return text


def prompt_for_output_dir(default: str = "./GIWAXS_output") -> str:
    """Interactively ask for the destination directory and create it."""
    text = strip_path_input(input(
        f"Enter destination directory for output files "
        f"(created automatically if it doesn't exist) [{default}]: "
    ))
    if not text:
        text = default
    os.makedirs(text, exist_ok=True)
    return text


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    resp = input(prompt + suffix).strip().lower()
    if not resp:
        return default
    return resp in ("y", "yes")


def prompt_float(prompt: str, default: Optional[float] = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        text = input(f"{prompt}{suffix}: ").strip()
        if not text and default is not None:
            return default
        try:
            return float(text)
        except ValueError:
            print("  Please enter a number.")


def prompt_for_calibration_setup(args) -> None:
    """If the user didn't already supply --agbeh-file (and hasn't opted out
    via --no-calibration-prompt), interactively ask whether they have an
    AgBeh (or other calibrant) image to refine the geometry with, and if so
    fill in args.agbeh_file / args.calibrant in place.
    """
    if args.agbeh_file or args.no_calibration_prompt:
        return
    if not prompt_yes_no(
        "\nDo you have an AgBeh (or other calibrant) image to refine the "
        "beam centre / sample-detector distance before processing? "
        "(Recommended for accurate results -- your --beam-center-y/x and "
        "--distance will only be used as the initial guess.)",
        default=False,
    ):
        return
    while True:
        path = strip_path_input(input("  Path to the calibration image (e.g. AgBeh .tif): "))
        if path and os.path.exists(path):
            break
        print(f"  File not found: {path}")
    args.agbeh_file = path
    calibrant = input(f"  Calibrant name [{args.calibrant}]: ").strip()
    if calibrant:
        args.calibrant = calibrant


def resolve_input_output(args) -> Tuple[str, str]:
    """Resolve --input/--output-dir from CLI args, prompting interactively
    for whichever one was not supplied. Ensures the output directory exists.
    """
    input_path = args.input if args.input else prompt_for_input_path()
    if not os.path.exists(input_path):
        _raise_error(f"Input path not found: {input_path}")

    output_dir = args.output_dir if args.output_dir else prompt_for_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    return input_path, output_dir


def resolve_tiff_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        tiff_files = sorted(
            glob.glob(os.path.join(input_path, "*.tif"))
            + glob.glob(os.path.join(input_path, "*.tiff"))
        )
        if not tiff_files:
            _raise_error(f"No .tif/.tiff files found in directory: {input_path}")
        return tiff_files
    elif os.path.isfile(input_path):
        return [input_path]
    else:
        _raise_error(f"Input path not found: {input_path}")


def load_pole_figure_q_map(path: str) -> Dict[str, List[float]]:
    """Load a per-file pole-figure target-q mapping, so different files in
    a batch can each get their own reflection(s) instead of sharing one
    fixed list. Keyed by basename (not full path) so it works regardless
    of which folder the files are actually processed from.

    JSON format:  {"sample_0001.tif": [1.673, 0.252], "sample_0002.tif": 1.68}
    CSV/Excel format: columns 'filename' and 'q_values' (space/comma
                  separated numbers in the q_values column), one row per file.
    """
    if not os.path.exists(path):
        _raise_error(f"Pole-figure q-map file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    result: Dict[str, List[float]] = {}

    def _parse_rows(rows):
        for row in rows:
            fname = str(row.get("filename") or row.get("Filename") or "").strip()
            qvals_raw = (row.get("q_values") if row.get("q_values") is not None else
                         row.get("Q_values") if row.get("Q_values") is not None else
                         row.get("q") if row.get("q") is not None else
                         row.get("target_q"))
            if not fname or qvals_raw is None or str(qvals_raw).strip() == "":
                continue
            try:
                qvals = [float(v) for v in str(qvals_raw).replace(",", " ").split()]
            except ValueError:
                _raise_error(f"Could not parse q values '{qvals_raw}' for '{fname}' in {path}")
            result[os.path.basename(fname)] = qvals

    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        for fname, qvals in data.items():
            if isinstance(qvals, (int, float)):
                qvals = [qvals]
            result[os.path.basename(fname)] = [float(v) for v in qvals]
    elif ext == ".csv":
        with open(path, newline="") as f:
            _parse_rows(csv.DictReader(f))
    elif ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError:
            _raise_error(
                "Reading an Excel (.xlsx/.xls) q-map requires the 'pandas' "
                "and 'openpyxl' packages. Install with:\n"
                "    pip install pandas openpyxl\n"
                "...or save your mapping as .json or .csv instead."
            )
        df = pd.read_excel(path, dtype=str)
        _parse_rows(df.to_dict("records"))
    else:
        _raise_error(f"Unsupported q-map file format: '{ext}' -- use .json, .csv, or .xlsx")

    if not result:
        _raise_error(f"No usable filename/q-value rows found in {path}. Expected "
                  f"a 'filename' column/key and a 'q_values' (or 'q') column/key.")

    return result


#: An incident angle above this is not grazing incidence at all, so a
#: filename token that large is something else -- a temperature, a contact
#: angle, a stage position. Used to tell the real angle apart from the
#: other 'NpM' tokens beamline filenames routinely carry.
MAX_PLAUSIBLE_INCIDENT_ANGLE_DEG = 5.0


def incident_angle_candidates_in_filename(filename: str) -> List[float]:
    """Every '<digits>p<digits>' token in the filename, as floats, in the
    order they appear. These are CANDIDATES, not answers -- see
    parse_incident_angle_from_filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    out: List[float] = []
    # Lookahead for the trailing separator, not a consuming group: an
    # underscore shared between two tokens can only be eaten once, so
    # '_35p0_0p1_' with a consuming '(?:_|$)' matches 35p0 and then can
    # no longer see the '_' that 0p1 needs in front of it. That single
    # character is why only the first value was ever found.
    for m in re.findall(r'(?:^|_)(\d+p\d+)(?=_|$)', base):
        try:
            out.append(float(m.replace('p', '.', 1)))
        except ValueError:
            continue
    return out


def parse_incident_angle_from_filename(
        filename: str,
        max_plausible_deg: float = MAX_PLAUSIBLE_INCIDENT_ANGLE_DEG) -> Optional[float]:
    """Extract the incident angle from a filename that encodes decimals
    with 'p', e.g. 'sample_0p095_1234.tif' -> 0.095.

    Real beamline filenames carry more than one such token. A name like
    'Q1_01_35p0_0p1_0627.tif' holds a contact angle (35.0) AND the
    incident angle (0.1), and simply taking the first match returns 35
    degrees -- which is not grazing incidence at all, and silently
    produces a badly distorted reciprocal-space map rather than any kind
    of error.

    So candidates are filtered by physics: a grazing-incidence angle is
    a fraction of a degree, never tens of degrees. Anything above
    max_plausible_deg is some other quantity and is discarded.

    Returns the angle when exactly one candidate survives; None when none
    does, and also None when SEVERAL do -- two plausible angles in one
    name is genuinely ambiguous, and guessing between them is how a whole
    batch ends up silently processed at the wrong angle. The caller falls
    back to the explicit value and says why.
    """
    plausible = [a for a in incident_angle_candidates_in_filename(filename)
                 if 0.0 < a <= max_plausible_deg]
    if len(plausible) == 1:
        return plausible[0]
    return None


def load_incident_angle_map(path: str) -> Dict[str, float]:
    """Load a per-file incident-angle mapping (JSON, CSV, or Excel),
    keyed by basename. Same file formats/conventions as
    load_pole_figure_q_map, but each file maps to a single angle (degrees)
    rather than a list of q values.

    JSON format: {"sample_0001.tif": 0.095, "sample_0002.tif": 0.1}
    CSV/Excel format: columns 'filename' and 'incident_angle' (or 'angle').
    """
    if not os.path.exists(path):
        _raise_error(f"Incident-angle-map file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    result: Dict[str, float] = {}

    def _parse_rows(rows):
        for row in rows:
            fname = str(row.get("filename") or row.get("Filename") or "").strip()
            angle_raw = (row.get("incident_angle") if row.get("incident_angle") is not None else
                         row.get("angle") if row.get("angle") is not None else
                         row.get("Incident_angle"))
            if not fname or angle_raw is None or str(angle_raw).strip() == "":
                continue
            try:
                result[os.path.basename(fname)] = float(angle_raw)
            except ValueError:
                _raise_error(f"Could not parse incident angle '{angle_raw}' for '{fname}' in {path}")

    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        for fname, angle in data.items():
            result[os.path.basename(fname)] = float(angle)
    elif ext == ".csv":
        with open(path, newline="") as f:
            _parse_rows(csv.DictReader(f))
    elif ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError:
            _raise_error(
                "Reading an Excel (.xlsx/.xls) angle-map requires the "
                "'pandas' and 'openpyxl' packages. Install with:\n"
                "    pip install pandas openpyxl\n"
                "...or save your mapping as .json or .csv instead."
            )
        df = pd.read_excel(path, dtype=str)
        _parse_rows(df.to_dict("records"))
    else:
        _raise_error(f"Unsupported angle-map file format: '{ext}' -- use .json, .csv, or .xlsx")

    if not result:
        _raise_error(f"No usable filename/angle rows found in {path}. Expected "
                  f"a 'filename' column/key and an 'incident_angle' (or "
                  f"'angle') column/key.")

    return result


def resolve_incident_angle_for_file(tiff_path: str, fallback_deg: float,
                                     use_filename: bool, verbose: bool = True,
                                     angle_map: Optional[Dict[str, float]] = None) -> float:
    """Resolve the incident angle (degrees) to use for a specific file, in
    priority order: an explicit per-file angle_map entry, then (if
    use_filename) a pattern parsed from the filename, then fallback_deg.
    """
    if angle_map:
        mapped = angle_map.get(os.path.basename(tiff_path))
        if mapped is not None:
            if verbose:
                print(f"  Incident angle from map: {mapped} deg")
            return mapped
        if verbose:
            print(f"  '{os.path.basename(tiff_path)}' not found in the "
                  f"incident-angle map -- ", end="")
    if not use_filename:
        if angle_map and verbose:
            print(f"using fallback --incident-angle={fallback_deg} for this file.")
        return fallback_deg
    parsed = parse_incident_angle_from_filename(tiff_path)
    if parsed is None:
        if verbose:
            name = os.path.basename(tiff_path)
            found = incident_angle_candidates_in_filename(tiff_path)
            plausible = [a for a in found
                         if 0.0 < a <= MAX_PLAUSIBLE_INCIDENT_ANGLE_DEG]
            if len(plausible) > 1:
                why = (f"it contains more than one value that could be the "
                       f"incident angle ({', '.join(f'{a:g}' for a in plausible)} "
                       f"deg) and guessing between them is not safe")
            elif found:
                why = (f"the only 'NpM' value(s) in it "
                       f"({', '.join(f'{a:g}' for a in found)}) are too large "
                       f"to be a grazing-incidence angle "
                       f"(> {MAX_PLAUSIBLE_INCIDENT_ANGLE_DEG:g} deg), so they "
                       f"are some other quantity")
            else:
                why = "it contains no 'NpM' value"
            print(f"  Could not read an incident angle from '{name}': {why} "
                  f"-- using {fallback_deg} deg for this file instead.")
        return fallback_deg
    if verbose:
        print(f"  Incident angle from filename: {parsed} deg")
    return parsed


# --------------------------------------------------------------------------- #
# Font styling presets
# --------------------------------------------------------------------------- #
#: Candidate font families grouped by category. Proprietary fonts (Arial,
#: Helvetica, Times New Roman) are only genuinely usable if the OS happens
#: to have them installed -- on a typical Linux server (e.g. Streamlit
#: Community Cloud) they're usually NOT present, and matplotlib silently
#: substitutes its own default font instead of raising an error, so
#: picking one from a UI dropdown would look like it did nothing. Each
#: category also includes an open, metrically-similar alternative that's
#: commonly actually installed on Linux (Liberation/TeX-Gyre families),
#: plus the generic CSS-style keyword as a guaranteed-to-work fallback.
_FONT_CATEGORIES_CANDIDATES = {
    "Sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans",
                   "TeX Gyre Heros", "sans-serif"],
    "Serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif",
              "TeX Gyre Termes", "serif"],
    "Monospace": ["DejaVu Sans Mono", "Liberation Mono", "monospace"],
}

#: Matplotlib always resolves these generic CSS-style family keywords to
#: SOME installed font (never raises, never silently no-ops) -- safe to
#: offer regardless of what's actually installed.
_ALWAYS_SAFE_FONT_KEYWORDS = {"sans-serif", "serif", "monospace"}

#: Familiar commercial font names (Arial, Helvetica, Times New Roman) are
#: almost never actually INSTALLED on a Linux server (they're proprietary),
#: so matplotlib would silently fall back to its default whenever one is
#: requested -- but they're names people recognize and expect to see in a
#: picker. Rather than hide them, keep them SELECTABLE (as long as their
#: metric-compatible open-source substitute is actually installed) and
#: transparently swap in the substitute at render time -- see
#: resolve_font_family(), used by style_context() so every plot benefits
#: without each caller needing to know about this.
FONT_SUBSTITUTIONS = {
    "Arial": "Liberation Sans",
    "Helvetica": "TeX Gyre Heros",
    "Times New Roman": "Liberation Serif",
}


def get_available_font_categories() -> Dict[str, List[str]]:
    """Filter _FONT_CATEGORIES_CANDIDATES down to fonts that will actually
    RENDER as something sensible in the CURRENT environment (checked via
    matplotlib's own font manager), so a UI picker built from this never
    offers a choice that would silently do nothing. A name in
    FONT_SUBSTITUTIONS (e.g. "Arial") counts as available if its
    substitute (e.g. "Liberation Sans") is installed, even though the
    literal requested name isn't -- resolve_font_family() does that swap
    at render time. Generic keywords (sans-serif/serif/monospace) are
    always kept since matplotlib always resolves those successfully.

    Rebuilds matplotlib's font manager from scratch (re-scanning the
    system's actual font directories) rather than trusting its cached
    font list -- matplotlib caches what it finds in a JSON file on first
    use, and if that cache was built BEFORE packages.txt's font packages
    were installed (e.g. an old cached container layer, or the cache
    simply predating this deploy), newly-installed system fonts wouldn't
    be picked up without this. A fresh scan costs well under a second and
    this only runs once at import time, so the cost is a non-issue.
    """
    import matplotlib.font_manager as fm
    fm.fontManager = fm.FontManager()  # re-scan system fonts, ignore stale cache
    installed = {f.name for f in fm.fontManager.ttflist}

    def _is_available(name: str) -> bool:
        if name in _ALWAYS_SAFE_FONT_KEYWORDS or name in installed:
            return True
        substitute = FONT_SUBSTITUTIONS.get(name)
        return substitute is not None and substitute in installed

    result = {}
    for category, candidates in _FONT_CATEGORIES_CANDIDATES.items():
        available = [f for f in candidates if _is_available(f)]
        if available:
            result[category] = available
    return result


def resolve_font_family(font_family: Optional[str]) -> Optional[str]:
    """Swap a familiar-but-likely-uninstalled name (Arial, Helvetica,
    Times New Roman) for its metric-compatible, actually-installed
    open-source substitute. Anything else (DejaVu Sans, Liberation Sans,
    the generic sans-serif/serif/monospace keywords, etc.) passes through
    unchanged. Centralizing this here (called from style_context) means
    every plotting function benefits automatically.
    """
    if font_family is None:
        return None
    return FONT_SUBSTITUTIONS.get(font_family, font_family)


#: Computed once at import time for convenience (gc.FONT_CATEGORIES is used
#: as a plain dict elsewhere) -- reflects whatever's installed wherever
#: this process happens to be running (local machine, Streamlit Cloud, etc).
FONT_CATEGORIES = get_available_font_categories()

#: Font size presets (points), for a friendlier picker than a raw slider.
#: "Custom" (mapped to None here) signals the caller to show a free-entry
#: number input instead of using a preset value.
FONT_SIZE_PRESETS = {
    "Small (8pt)": 8.0,
    "Normal (11pt)": 11.0,
    "Large (14pt)": 14.0,
    "Larger (16pt)": 16.0,
    "Extra large (18pt)": 18.0,
    "Huge (24pt)": 24.0,
    "Custom...": None,
}



# --------------------------------------------------------------------------- #
def load_poni_file(path: str) -> Dict[str, object]:
    """Load an entire geometry (distance, beam centre, rotations,
    wavelength, and detector -- including its pixel size and shape) from
    an existing pyFAI .poni file, so you never need to know those values
    separately or re-derive them by hand.

    Returns a dict with keys: dist, poni1, poni2, rot1, rot2, rot3,
    wavelength, detector (a ready-to-use pyFAI Detector instance).
    """
    if not os.path.exists(path):
        _raise_error(f"PONI file not found: {path}")
    from pyFAI.io.ponifile import PoniFile
    try:
        poni = PoniFile(data=path)
    except Exception as exc:
        _raise_error(f"Could not read PONI file '{path}': {exc}")
    if poni.detector is None:
        _raise_error(f"PONI file '{path}' does not specify a detector.")
    return {
        "dist": poni.dist,
        "poni1": poni.poni1,
        "poni2": poni.poni2,
        "rot1": poni.rot1 or 0.0,
        "rot2": poni.rot2 or 0.0,
        "rot3": poni.rot3 or 0.0,
        "wavelength": poni.wavelength,
        "detector": poni.detector,
    }


def resolve_wavelength(args) -> float:
    if args.wavelength is None and args.energy is None:
        _raise_error("You must provide either --wavelength (metres) or --energy "
                  "(keV) -- or use --poni-file to load it from an existing "
                  "calibration file.")
    if args.wavelength is not None:
        return args.wavelength
    hc_keV_m = 1.2398419843320025e-9  # h*c in keV*m
    return hc_keV_m / args.energy


def build_detector(args, Detector, detector_factory):
    if args.detector_name:
        return detector_factory(args.detector_name)
    if args.detector_shape is None:
        _raise_error("You must provide --detector-shape 'height,width' when "
                  "--detector-name is not given (or use --poni-file, which "
                  "already knows the detector's shape).")
    return Detector(
        pixel1=args.pixel_size,
        pixel2=args.pixel_size,
        max_shape=args.detector_shape,
    )


def plot_calibration_diagnostic(calib_img, cp, refined_geom, calibrant, out_path,
                                 title: Optional[str] = None, max_pixels: int = 400_000):
    """Visual sanity check for an AgBeh (or other calibrant) ring fit:
    the raw calibration image (log scale) with the extracted control
    points overlaid (colored by ring index) and the fitted ring positions
    (from the refined geometry) drawn as contour lines on top. If the fit
    is good, the contour lines should sit right on top of the real
    diffraction rings, and the coloured dots should form clean circles.

    For large (real detector-sized) images, this downsamples BEFORE
    rendering -- a full-resolution detector frame plus a full-resolution
    2-theta array plus matplotlib's own rendering overhead can add up to
    a meaningful amount of memory, which matters on memory-constrained
    deployments (e.g. Streamlit Community Cloud's free tier). This is a
    sanity-check plot, not a publication figure, so a coarser resolution
    doesn't lose anything that matters for judging the fit.
    """
    h, w = calib_img.shape
    factor = max(1, int(np.ceil(np.sqrt((h * w) / max_pixels))))

    if factor > 1:
        small_img = calib_img[::factor, ::factor]
    else:
        small_img = calib_img

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    positive = small_img[small_img > 0]
    vmin = max(np.percentile(positive, 1), 1) if positive.size else 1
    vmax = np.percentile(positive, 99.5) if positive.size else 1
    if vmax <= vmin:
        vmax = vmin * 10
    ax.imshow(small_img, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="inferno")

    pts = np.asarray(cp.getList())
    if pts.size:
        sc = ax.scatter(pts[:, 1] / factor, pts[:, 0] / factor, s=4, c=pts[:, 2],
                         cmap="tab10", label="detected points")
        plt.colorbar(sc, ax=ax, label="ring index", fraction=0.046, pad=0.04)

    try:
        tth_array = refined_geom.twoThetaArray(calib_img.shape)
        if factor > 1:
            tth_array = tth_array[::factor, ::factor]
        for tth_ring in calibrant.get_2th():
            if tth_ring is None:
                continue
            ax.contour(tth_array, levels=[tth_ring], colors="lime", linewidths=0.6)
        del tth_array  # this one can be large before downsampling; free it promptly
    except Exception:
        pass  # diagnostic overlay is best-effort; missing it isn't fatal

    ax.set_title(
        title or "Calibration fit check\n"
        "(dots = detected ring points, green lines = fitted ring positions "
        "-- they should overlap)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _radial_profile(values, radii, nbins, rmax, min_count=None):
    """Azimuthal mean of `values` binned by `radii`. Bins fed by too few
    pixels come back as NaN rather than as noise.

    min_count scales with how many pixels there actually are: a fixed
    threshold that is sensible at full resolution rejects almost every
    bin on a decimated copy, which silently turns the whole profile into
    NaN and makes any score computed from it meaningless.
    """
    b = (radii * (nbins / rmax)).astype(np.int32)
    keep = b < nbins
    total = np.bincount(b[keep], weights=values[keep], minlength=nbins)
    count = np.bincount(b[keep], minlength=nbins)
    if min_count is None:
        min_count = max(4, int(0.15 * keep.sum() / max(nbins, 1)))
    prof = np.where(count >= min_count, total / np.maximum(count, 1), np.nan)
    return prof, (np.arange(nbins) + 0.5) * rmax / nbins


def _ring_sharpness(vx, vy, vv, cx, cy, rmax, min_fraction=0.25):
    """How ring-like the image looks about (cx, cy).

    Azimuthally average about the candidate centre and take the total
    variation of that profile. Concentric rings only average coherently
    about their own centre: anywhere else they smear out and the profile
    flattens, so this peaks sharply at the true centre and needs no
    initial guess at all.

    The profile is lightly smoothed first, because total variation counts
    bin-to-bin noise just as happily as it counts real rings, and a
    candidate centre with sparser sampling would otherwise score on its
    noise. Candidates whose profile is mostly empty score -1 so they can
    never win.
    """
    r = np.hypot(vx - cx, vy - cy)
    m = r < rmax
    if m.sum() < 2000:
        return -1.0
    nbins = max(32, int(rmax))
    prof, _ = _radial_profile(vv[m], r[m], nbins, rmax)
    finite = np.isfinite(prof)
    if finite.sum() < min_fraction * nbins:
        return -1.0
    p = prof[finite]
    if p.size > 5:                      # 3-bin box smooth, NaN-free by now
        k = np.ones(3) / 3.0
        p = np.convolve(p, k, mode="valid")
    return float(np.abs(np.diff(p)).sum())


def find_beam_centre_from_rings(calib_img, search_margin: float = 0.25,
                                 coarse_step: int = 8):
    """Locate the beam centre of a calibrant image WITHOUT any geometry.

    Returns (col, row, sharpness). The centre may legitimately fall
    outside the detector -- an offset or grazing-incidence setup often
    puts the direct beam just off the active area -- so the search box
    extends beyond the image by `search_margin`.

    Multi-resolution: a coarse sweep on a heavily decimated copy, then a
    local pattern search at successively finer decimation. The coarse
    pass is what makes this affordable; a full-resolution sweep over the
    same box would be orders of magnitude more work for the same answer.
    """
    h, w = calib_img.shape
    good = np.isfinite(calib_img) & (calib_img > 0)
    logi = np.log10(np.where(good, calib_img, 1.0) + 1.0)
    rmax_full = float(np.hypot(h, w))

    cache = {}

    def points(step):
        if step not in cache:
            gs = good[::step, ::step]
            ys, xs = np.nonzero(gs)
            cache[step] = (xs.astype(np.float64), ys.astype(np.float64),
                            logi[::step, ::step][gs])
        return cache[step]

    # --- coarse sweep -----------------------------------------------------
    vx, vy, vv = points(coarse_step)
    mx, my = w * search_margin, h * search_margin
    best = (-1.0, 0.0, 0.0)
    for cy in np.linspace(-my, h + my, 40) / coarse_step:
        for cx in np.linspace(-mx, w + mx, 40) / coarse_step:
            s = _ring_sharpness(vx, vy, vv, cx, cy, rmax_full / coarse_step)
            if s > best[0]:
                best = (s, cx, cy)
    _, bx, by = best
    bx, by = bx * coarse_step, by * coarse_step

    # --- refine, halving the decimation as we go --------------------------
    step = coarse_step
    while True:
        vx, vy, vv = points(step)
        cx, cy = bx / step, by / step
        rmax = rmax_full / step
        s0 = _ring_sharpness(vx, vy, vv, cx, cy, rmax)
        delta = max(2.0, coarse_step * 1.5 / step)
        while delta >= 0.25:
            improved = False
            for dy in (-delta, 0.0, delta):
                for dx in (-delta, 0.0, delta):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    s = _ring_sharpness(vx, vy, vv, cx + dx, cy + dy, rmax)
                    if s > s0:
                        s0, cx, cy, improved = s, cx + dx, cy + dy, True
            if not improved:
                delta /= 2.0
        bx, by = cx * step, cy * step
        if step == 1:
            return float(bx), float(by), float(s0)
        step //= 2


def _fit_order_comb(radii, weights, max_peaks: int = 20,
                     rel_prominence: float = 0.02, tol: float = 0.1):
    """Given ring radii, find the fundamental spacing they are multiples of.

    Diffraction orders of a lamellar calibrant sit at very nearly integer
    multiples of the first-order radius, so sweeping a candidate
    fundamental and scoring how closely every peak lands on an integer
    multiple recovers the right spacing even when the first orders are
    hidden behind the beamstop and some detected peaks belong to other
    phases entirely.

    Two things this has to get right:

    * Only the strong peaks are evidence. A real frame throws up dozens of
      weak maxima at large radii; insisting that a majority of ALL of them
      fit a comb means no comb ever fits.
    * Half the true spacing always scores at least as well as the spacing
      itself, because every multiple of r1 is also a multiple of r1/2. So
      among candidates that explain the peaks about equally well, take the
      LARGEST -- otherwise the orders come out doubled and the distance
      comes out halved.
    """
    radii = np.asarray(radii, float)
    weights = np.asarray(weights, float)
    if len(radii) < 3:
        return None

    cut = rel_prominence * weights.max()
    sel = weights >= cut
    if sel.sum() >= 4:
        radii, weights = radii[sel], weights[sel]
    order = np.argsort(-weights)[:max_peaks]
    radii, weights = radii[order], weights[order]
    idx = np.argsort(radii)
    radii, weights = radii[idx], weights[idx]

    lo = max(1.0, 0.04 * radii.min())
    hi = 1.1 * radii.min()
    cands = np.linspace(lo, hi, 6000)
    scores = np.full(cands.size, -np.inf)
    for i, c in enumerate(cands):
        k = radii / c
        frac = np.abs(k - np.round(k))
        on = (frac < tol) & (np.round(k) >= 1)
        if on.sum() < max(3, int(0.5 * len(radii))):
            continue
        scores[i] = float((weights[on] * (1.0 - frac[on] / tol)).sum())
    if not np.isfinite(scores).any():
        return None
    best = scores.max()
    # among candidates that explain the peaks about as well, the largest
    # spacing is the fundamental; the smaller ones are its sub-harmonics
    near = cands[scores >= 0.95 * best]
    return float(near.max())


def estimate_geometry_from_rings(calib_img, pixel_size: float, wavelength: float,
                                  calibrant_dspacing: float,
                                  beam_centre=None) -> Dict[str, object]:
    """Derive the beam centre AND the sample-detector distance from a
    calibrant image alone -- no initial guess required.

    This exists because pyFAI's control-point extraction searches for ring
    points NEAR where the supplied geometry predicts them. Hand it a guess
    that is far off and it attaches the detected points to the wrong ring
    orders; the refinement then converges to a geometry that is perfectly
    self-consistent with those wrong labels and reports no error at all.
    Deriving the starting point from the data removes that failure mode
    instead of asking the user to avoid it.

    wavelength and calibrant_dspacing are both in metres and Angstrom
    respectively -- i.e. wavelength in m (as pyFAI uses) and d in A (as
    calibrant tables list).

    Returns a dict with beam_centre_col/row (pixels), dist (m),
    first_order_px, ring_radii_px, n_rings, rms_residual_px and sharpness.
    rms_residual_px is the headline number: on a good calibrant frame it
    is well under a pixel.
    """
    lam_A = wavelength * 1e10
    h, w = calib_img.shape

    if beam_centre is None:
        cx, cy, sharp = find_beam_centre_from_rings(calib_img)
    else:
        cx, cy = float(beam_centre[0]), float(beam_centre[1])
        good = np.isfinite(calib_img) & (calib_img > 0)
        ys, xs = np.nonzero(good)
        sharp = _ring_sharpness(xs.astype(float), ys.astype(float),
                                 np.log10(calib_img[good] + 1.0), cx, cy,
                                 float(np.hypot(h, w)))

    # --- radial profile about that centre ---------------------------------
    good = np.isfinite(calib_img) & (calib_img > 0)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - cx, yy - cy)[good]
    v = np.log10(calib_img[good] + 1.0)
    rmax = float(np.hypot(h, w))
    nbins = int(rmax)
    prof, rr = _radial_profile(v, r, nbins, rmax)
    finite = np.isfinite(prof)
    if finite.sum() < 50:
        raise GiwaxsError("Could not build a radial profile from this image -- "
                          "too few usable pixels.")

    # Background window must be MUCH wider than the ring spacing: a window
    # comparable to it tracks the rings themselves and the detrended
    # profile then shows spurious peaks half way between the real ones.
    from scipy.ndimage import uniform_filter1d
    filled = np.where(finite, prof, float(np.nanmedian(prof)))
    detrended = np.where(finite, prof - uniform_filter1d(filled, 201), -9.0)

    from scipy.signal import find_peaks
    pk, props = find_peaks(detrended, prominence=0.006, distance=15)
    radii = rr[pk]
    prom = props["prominences"]
    keep = radii > 0.02 * rmax           # ignore the beamstop halo
    radii, prom = radii[keep], prom[keep]
    if len(radii) < 3:
        raise GiwaxsError(
            "Fewer than three calibrant rings could be found in this image. "
            "Check that it really is a calibrant exposure and that the "
            "calibrant name is right."
        )

    fundamental = _fit_order_comb(radii, prom)
    if fundamental is None:
        raise GiwaxsError(
            "Could not work out the spacing between the calibrant rings. "
            "Check that this really is a calibrant exposure and that the "
            "calibrant name matches it."
        )

    # --- solve the distance from the ring positions ------------------------
    from scipy.optimize import least_squares

    def predict(dist_m, nn):
        s = np.clip(np.asarray(nn, float) * lam_A / (2.0 * calibrant_dspacing),
                    -1.0, 1.0)
        return dist_m * np.tan(2.0 * np.arcsin(s)) / pixel_size

    dist = fundamental * pixel_size / np.tan(
        2.0 * np.arcsin(np.clip(lam_A / (2.0 * calibrant_dspacing), -1, 1)))

    # Match PREDICTED ring positions to detected peaks, not the other way
    # round, and tighten the acceptance window each pass.
    #
    # Assigning an order to every detected peak is what goes wrong on a
    # real frame: a strong peak that is not this calibrant (another phase,
    # the substrate) sits near some order, gets claimed as that order, and
    # drags the distance by percent. Going the other way, each order may
    # claim at most one peak and only if it is close enough -- so as the
    # window shrinks, a near-miss impostor is dropped in favour of the
    # real ring once the distance is good enough to tell them apart.
    matched_n = np.array([], int)
    matched_r = np.array([], float)
    for tol_frac in (0.30, 0.12, 0.05, 0.02):
        n_max = max(2, int(rmax / max(fundamental, 1.0)) + 2)
        ns = np.arange(1, n_max + 1)
        pred = predict(dist, ns)
        inside = pred < rmax
        ns, pred = ns[inside], pred[inside]
        take_n, take_r = [], []
        for n, rp in zip(ns, pred):
            gap = np.abs(radii - rp)
            k = int(gap.argmin())
            if gap[k] <= tol_frac * fundamental:
                take_n.append(n)
                take_r.append(radii[k])
        if len(take_n) < 3:
            break
        matched_n = np.asarray(take_n, int)
        matched_r = np.asarray(take_r, float)
        fit = least_squares(
            lambda p: predict(p[0], matched_n) - matched_r, [dist])
        dist = float(fit.x[0])

    if matched_n.size < 3:
        raise GiwaxsError(
            "Could not match enough calibrant rings to fit the "
            "sample-detector distance."
        )
    resid = predict(dist, matched_n) - matched_r
    radii, orders = matched_r, matched_n

    return {
        "beam_centre_col": float(cx),
        "beam_centre_row": float(cy),
        "dist": dist,
        "first_order_px": float(predict(dist, np.array([1]))[0]),
        "ring_radii_px": radii,
        "ring_orders": orders,
        "n_rings": int(len(radii)),
        "rms_residual_px": float(np.sqrt((resid ** 2).mean())),
        "sharpness": float(sharp),
    }

def run_agbeh_calibration(calib_path: str, detector, wavelength: float,
                           dist_guess: float, poni1_guess: float, poni2_guess: float,
                           rot1_guess: float, rot2_guess: float, rot3_guess: float,
                           calibrant_name: str, max_rings: int, imin: float,
                           fabio_mod, diagnostic_path: Optional[str] = None,
                           auto_initial_guess: bool = True,
                           max_centre_mismatch_px: float = 25.0,
                           max_rms_residual_px: float = 3.0,
                           strict: bool = True) -> Dict[str, float]:
    """Refine the beam centre / sample-detector distance against a calibrant
    (e.g. AgBeh) image, mirroring the notebook's calibration workflow:
    extract ring control points with pyFAI's SingleGeometry, then run its
    geometry_refinement.curve_fit().

    dist_guess/poni1_guess/poni2_guess/rot*_guess are used as the initial
    geometry for the fit -- they only need to be approximately correct.

    If diagnostic_path is given, also saves a visual fit-quality check
    there (see plot_calibration_diagnostic) and includes its path in the
    returned dict as "diagnostic_path".

    Returns a dict with the refined dist/poni1/poni2/rot1/rot2/rot3 plus
    the chi-squared value before and after refinement.
    """
    from pyFAI.geometry import Geometry
    from pyFAI.goniometer import SingleGeometry
    from pyFAI.calibrant import get_calibrant

    if not os.path.exists(calib_path):
        _raise_error(f"Calibration image not found: {calib_path}")

    calib_img = fabio_mod.open(calib_path).data
    calibrant = get_calibrant(calibrant_name, wavelength=wavelength)

    pix1 = detector.pixel1
    pix2 = detector.pixel2
    notes = []

    # --- an estimate that owes nothing to the supplied guess ----------------
    # pyFAI's extract_cp looks for ring points NEAR where the supplied
    # geometry predicts them. A guess that is far off makes it attach the
    # points to the wrong ring orders, and the refinement then converges on
    # a geometry that is perfectly self-consistent with those wrong labels
    # and reports no error whatsoever. Deriving a starting point straight
    # from the rings removes that failure mode rather than asking the user
    # to avoid it, and doubles as the one check that can catch a converged
    # but wrong fit, precisely because it shares no assumption with it.
    estimate = None
    try:
        estimate = estimate_geometry_from_rings(
            calib_img, pix1, wavelength, calibrant.dspacing[0])
    except Exception as exc:                       # never fatal on its own
        notes.append(f"Could not estimate the geometry from the rings "
                     f"directly ({exc}); falling back to the supplied guess "
                     f"alone.")

    guess_replaced = False
    if estimate is not None and auto_initial_guess:
        gap = float(np.hypot(poni2_guess / pix2 - estimate["beam_centre_col"],
                              poni1_guess / pix1 - estimate["beam_centre_row"]))
        if gap > max_centre_mismatch_px:
            notes.append(
                f"The supplied beam centre is {gap:.0f} px from where the "
                f"rings in this image actually are, which is too far for the "
                f"ring finder to start from -- using the centre and distance "
                f"measured from the image instead."
            )
            poni1_guess = estimate["beam_centre_row"] * pix1
            poni2_guess = estimate["beam_centre_col"] * pix2
            dist_guess = estimate["dist"]
            guess_replaced = True

    initial_geom = Geometry(
        dist=dist_guess, poni1=poni1_guess, poni2=poni2_guess,
        rot1=rot1_guess, rot2=rot2_guess, rot3=rot3_guess,
        detector=detector, wavelength=wavelength,
    )

    sg = SingleGeometry(
        label=os.path.basename(calib_path),
        calibrant=calibrant,
        image=calib_img,
        detector=detector,
        geometry=initial_geom,
    )
    cp = sg.extract_cp(max_rings=max_rings, Imin=imin)
    if cp is None or len(cp.getList()) == 0:
        _raise_error(
            f"No calibrant ring control points were found in {calib_path} "
            f"(calibrant='{calibrant_name}', Imin={imin}). Try lowering "
            "--calib-min-intensity, check the calibrant name, or check that "
            "the initial beam-centre/distance guess is reasonably close."
        )

    gr = sg.geometry_refinement
    gr.data = np.array(cp.getList())
    init_chi2 = gr.chi2()
    gr.set_tolerance(50)
    gr.curve_fit(with_rot=False)
    final_chi2 = gr.chi2()

    cfg = gr.get_config()

    # --- how far off are the control points, in pixels? --------------------
    # chi2 is the number the refinement minimises, but its units mean
    # nothing to a person: 0.0034 looks small and was in fact a complete
    # failure, while 3.4e-6 was a good fit. The radial residual is in
    # detector pixels, so "41 px out" needs no interpretation and can be
    # given a fixed threshold.
    rms_residual_px = float("nan")
    try:
        pts = np.array(cp.getList())            # rows, cols, ring index
        rows, cols, ring = pts[:, 0], pts[:, 1], pts[:, 2].astype(int)
        tth_measured = gr.tth(rows, cols)
        tth_ring = np.array(calibrant.get_2th())
        valid = ring < len(tth_ring)
        dtth = tth_measured[valid] - tth_ring[ring[valid]]
        # angle -> pixels at the detector, exact for a flat normal detector
        scale = cfg["dist"] / (np.cos(tth_measured[valid]) ** 2) / pix1
        resid_px = dtth * scale
        rms_residual_px = float(np.sqrt((resid_px ** 2).mean()))
    except Exception as exc:
        notes.append(f"Could not compute the fit residual in pixels ({exc}).")

    # --- the independent cross-check ---------------------------------------
    centre_mismatch_px = float("nan")
    if estimate is not None:
        centre_mismatch_px = float(np.hypot(
            cfg["poni2"] / pix2 - estimate["beam_centre_col"],
            cfg["poni1"] / pix1 - estimate["beam_centre_row"]))

    problems = []
    if np.isfinite(rms_residual_px) and rms_residual_px > max_rms_residual_px:
        problems.append(
            f"the fitted rings miss the detected ring points by "
            f"{rms_residual_px:.1f} px on average (expected well under "
            f"{max_rms_residual_px:.0f} px)")
    if np.isfinite(centre_mismatch_px) and centre_mismatch_px > max_centre_mismatch_px:
        problems.append(
            f"the fitted beam centre is {centre_mismatch_px:.0f} px away from "
            f"where the rings in this image actually are "
            f"(fit: {cfg['poni2'] / pix2:.0f}, {cfg['poni1'] / pix1:.0f}; "
            f"image: {estimate['beam_centre_col']:.0f}, "
            f"{estimate['beam_centre_row']:.0f})")

    saved_diagnostic_path = None
    if diagnostic_path:
        try:
            plot_calibration_diagnostic(
                calib_img, cp, gr, calibrant, diagnostic_path,
                title=f"Calibration fit check: {os.path.basename(calib_path)}",
            )
            saved_diagnostic_path = diagnostic_path
        except Exception as exc:
            print(f"  (Could not generate calibration diagnostic plot: {exc})")

    if problems and strict:
        _raise_error(
            "This calibration did not work: " + "; and ".join(problems) + ".\n\n"
            "The usual cause is that the starting geometry describes a "
            "different measurement than this calibrant image -- for example a "
            ".poni saved from another experiment or another camera length. "
            "What the image itself says: beam centre "
            f"({(estimate or {}).get('beam_centre_col', float('nan')):.0f}, "
            f"{(estimate or {}).get('beam_centre_row', float('nan')):.0f}) px, "
            f"distance {(estimate or {}).get('dist', float('nan')):.4f} m. "
            "Enter those as the geometry, or pass strict=False to accept the "
            "fit anyway."
        )

    return {

        "dist": cfg["dist"],
        "poni1": cfg["poni1"],
        "poni2": cfg["poni2"],
        "rot1": cfg.get("rot1", rot1_guess),
        "rot2": cfg.get("rot2", rot2_guess),
        "rot3": cfg.get("rot3", rot3_guess),
        "init_chi2": init_chi2,
        "final_chi2": final_chi2,
        "n_control_points": len(cp.getList()),
        "diagnostic_path": saved_diagnostic_path,
        "rms_residual_px": rms_residual_px,
        "centre_mismatch_px": centre_mismatch_px,
        "guess_replaced": guess_replaced,
        "fit_ok": not problems,
        "problems": problems,
        "notes": notes,
        "image_centre_col": (estimate or {}).get("beam_centre_col"),
        "image_centre_row": (estimate or {}).get("beam_centre_row"),
        "image_dist": (estimate or {}).get("dist"),
    }


def save_refined_poni(path: str, dist: float, poni1: float, poni2: float,
                       rot1: float, rot2: float, rot3: float,
                       wavelength: float, detector):
    if os.path.isdir(path):
        _raise_error(
            f"--save-calibrated-poni must be a FILE path, not a directory: "
            f"'{path}'. Did you mean a file inside it, e.g. "
            f"'{os.path.join(path, 'calibrated.poni')}'?"
        )
    if not path.lower().endswith(".poni"):
        path = path + ".poni"

    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        _raise_error(f"Cannot save calibrated .poni file: directory does not "
                  f"exist: '{parent}'")

    from pyFAI.geometry import Geometry
    geom = Geometry(dist=dist, poni1=poni1, poni2=poni2,
                     rot1=rot1, rot2=rot2, rot3=rot3,
                     wavelength=wavelength, detector=detector)
    if os.path.exists(path):
        os.remove(path)
    geom.save(path)


def validate_geometry(detector, dist: float, poni1: float, poni2: float,
                      wavelength: float, image_shape=None,
                      strict: bool = True) -> List[str]:
    """Check that a geometry is physically usable, and that it belongs to
    the images it is about to be applied to.

    These are the checks whose absence lets a wrong answer through
    silently. pyFAI will happily build a coordinate map from a geometry
    that describes a different detector than the one that took the
    picture: nothing raises, every q value is simply wrong. The same goes
    for a wavelength left at a default from another beamline.

    Returns the list of problems found (empty when all is well). With
    strict=True a non-empty list is raised as a GiwaxsError instead,
    which is the right default -- these are contradictions, not missing
    optional settings, and continuing past them produces confident
    nonsense rather than a visible failure.
    """
    problems: List[str] = []

    det_shape = getattr(detector, "max_shape", None)
    if image_shape is not None and det_shape is not None:
        if tuple(image_shape) != tuple(det_shape):
            problems.append(
                f"The images are {image_shape[0]}x{image_shape[1]} pixels but "
                f"the geometry declares a "
                f"{getattr(detector, 'name', 'detector')} of "
                f"{det_shape[0]}x{det_shape[1]}. These must match -- a "
                f"geometry for a different detector maps every pixel to the "
                f"wrong q."
            )

    lam_A = wavelength * 1e10 if wavelength else 0.0
    if not (0.05 <= lam_A <= 20.0):
        problems.append(
            f"The wavelength is {lam_A:.4g} A ({wavelength:.4g} m), which is "
            f"outside the range any X-ray scattering measurement uses "
            f"(0.05-20 A). Check whether metres and Angstrom have been mixed up."
        )

    if not (1e-3 <= dist <= 1e2):
        problems.append(
            f"The sample-detector distance is {dist:.4g} m, which is not a "
            f"plausible value (expected roughly 0.01-20 m)."
        )

    if det_shape is not None:
        pix1 = getattr(detector, "pixel1", None) or 1.0
        pix2 = getattr(detector, "pixel2", None) or 1.0
        row, col = poni1 / pix1, poni2 / pix2
        # A beam centre just off the active area is normal for an offset or
        # grazing-incidence detector, so only a wild value is a problem.
        margin = 2.0
        if not (-margin * det_shape[0] <= row <= (1 + margin) * det_shape[0]) or \
           not (-margin * det_shape[1] <= col <= (1 + margin) * det_shape[1]):
            problems.append(
                f"The beam centre works out at row {row:.0f}, column "
                f"{col:.0f}, which is far outside a "
                f"{det_shape[0]}x{det_shape[1]} detector. Check the PONI "
                f"values and the pixel size."
            )

    if problems and strict:
        _raise_error(
            "This geometry cannot be used with these images:\n  - "
            + "\n  - ".join(problems)
        )
    return problems


def build_fiber_integrator(args, Detector, detector_factory, FiberIntegrator,
                            fabio=None, image_shape=None):
    """Construct the pyFAI FiberIntegrator (geometry + detector) from CLI
    args, optionally refining the beam centre / distance first against an
    AgBeh (or other calibrant) image if one was provided (via --agbeh-file
    or the interactive prompt).

    If --poni-file is given, the entire geometry (distance, beam centre,
    rotations, wavelength, detector) is loaded from it directly -- this is
    the most reliable path if you already have an accurate calibration,
    since it avoids re-deriving values by hand or re-running a fresh
    (potentially less accurate) AgBeh fit. Any of --beam-center-y/x,
    --distance, --wavelength/--energy, --rot1/2/3, --detector-name/
    --pixel-size/--detector-shape given ALONGSIDE --poni-file are ignored
    (with a warning), since the file is authoritative. If --agbeh-file is
    ALSO given, the loaded geometry is used as the initial guess for a
    fresh refinement rather than the final answer.

    When run interactively (not --no-calibration-prompt), a diagnostic
    plot of the fitted rings is generated after each calibration attempt
    and the user is asked to confirm the fit looks right before
    proceeding; if not, they can re-enter the initial guess and/or
    calibrant and try again.
    """
    poni_file = getattr(args, "poni_file", None)
    if poni_file:
        ignored = []
        for name, val in [("--beam-center-y", args.beam_center_y), ("--beam-center-x", args.beam_center_x),
                           ("--distance", args.distance), ("--wavelength", args.wavelength),
                           ("--energy", args.energy), ("--detector-name", args.detector_name),
                           ("--detector-shape", args.detector_shape)]:
            if val is not None:
                ignored.append(name)
        if ignored:
            print(f"NOTE: --poni-file was given, so {', '.join(ignored)} "
                  f"(also given) will be IGNORED -- the .poni file is used "
                  f"as the source of truth for geometry.\n")

        loaded = load_poni_file(poni_file)
        wavelength = loaded["wavelength"]
        detector = loaded["detector"]
        dist = loaded["dist"]
        poni1 = loaded["poni1"]
        poni2 = loaded["poni2"]
        rot1, rot2, rot3 = loaded["rot1"], loaded["rot2"], loaded["rot3"]
        print(f"Loaded geometry from {poni_file}:")
        print(f"  Detector: {detector.name if hasattr(detector, 'name') else detector}, "
              f"shape={detector.max_shape}, pixel size={detector.pixel1:.3g} m")
        print(f"  Beam centre: y={poni1 / detector.pixel1:.3f} px, "
              f"x={poni2 / detector.pixel2:.3f} px")
        print(f"  Distance: {dist:.6f} m, wavelength: {wavelength:.6g} m\n")
    else:
        missing = [name for name, val in [("--beam-center-y", args.beam_center_y),
                                           ("--beam-center-x", args.beam_center_x),
                                           ("--distance", args.distance)] if val is None]
        if missing:
            _raise_error(
                f"Missing required geometry argument(s): {', '.join(missing)}. "
                f"Either provide these directly, or use --poni-file to load "
                f"the whole geometry from an existing calibration file."
            )
        wavelength = resolve_wavelength(args)
        detector = build_detector(args, Detector, detector_factory)

        poni1 = args.beam_center_y * detector.pixel1
        poni2 = args.beam_center_x * detector.pixel2
        dist = args.distance
        rot1, rot2, rot3 = args.rot1, args.rot2, args.rot3

    if getattr(args, "agbeh_file", None):
        if fabio is None:
            import fabio as fabio  # local import so this module stays optional

        interactive = not getattr(args, "no_calibration_prompt", False)
        # Use whatever geometry was just resolved above (from --poni-file or
        # manual args) as the initial guess for refinement, converting the
        # PONI coordinates back to pixel units.
        guess_y = poni1 / detector.pixel1
        guess_x = poni2 / detector.pixel2
        guess_dist = dist
        calibrant_name = args.calibrant
        attempt = 0

        while True:
            attempt += 1
            guess_poni1 = guess_y * detector.pixel1
            guess_poni2 = guess_x * detector.pixel2
            print(f"\nRunning calibration refinement against: {args.agbeh_file} "
                  f"(attempt {attempt})")
            print(f"  Calibrant: {calibrant_name}, initial guess: "
                  f"beam centre = ({guess_y}, {guess_x}) px, distance = {guess_dist} m")

            diagnostic_path = os.path.join(
                tempfile_dir_for_diagnostics(),
                f"calibration_fit_check_attempt{attempt}.png"
            )
            result = run_agbeh_calibration(
                args.agbeh_file, detector, wavelength,
                guess_dist, guess_poni1, guess_poni2, rot1, rot2, rot3,
                calibrant_name, args.calib_max_rings, args.calib_min_intensity,
                fabio, diagnostic_path=diagnostic_path if interactive else None,
            )
            dist, poni1, poni2 = result["dist"], result["poni1"], result["poni2"]
            rot1, rot2, rot3 = result["rot1"], result["rot2"], result["rot3"]
            print(f"  Used {result['n_control_points']} ring control point(s).")
            print(f"  Chi2: {result['init_chi2']:.6g} -> {result['final_chi2']:.6g} "
                  "(lower is better)")
            print(f"  Refined beam centre: y={poni1 / detector.pixel1:.3f} px, "
                  f"x={poni2 / detector.pixel2:.3f} px")
            print(f"  Refined sample-detector distance: {dist:.6f} m")

            if not interactive:
                break  # scripted/automated mode -- trust the fit, no prompt

            if result.get("diagnostic_path"):
                diag_abs = os.path.abspath(result["diagnostic_path"])
                opened = open_file_externally(diag_abs)
                if opened:
                    print(f"\n  Opening calibration fit check image: {diag_abs}\n"
                          f"  (dots = detected ring points, green lines = fitted "
                          f"rings -- they should overlap closely.)")
                else:
                    print(f"\n  Calibration fit check image saved to:\n"
                          f"    {diag_abs}\n"
                          f"  Please open it: dots = detected ring points, green "
                          f"lines = fitted rings -- they should overlap closely.")

            if prompt_yes_no("\nDoes the calibration fit look correct?", default=True):
                break

            print("\nLet's try again with a new initial guess.")
            guess_y = prompt_float("  Beam centre Y (pixels)", default=guess_y)
            guess_x = prompt_float("  Beam centre X (pixels)", default=guess_x)
            guess_dist = prompt_float("  Sample-detector distance (m)", default=guess_dist)
            new_calibrant = input(f"  Calibrant name [{calibrant_name}]: ").strip()
            if new_calibrant:
                calibrant_name = new_calibrant

        if getattr(args, "save_calibrated_poni", None):
            save_refined_poni(args.save_calibrated_poni, dist, poni1, poni2,
                               rot1, rot2, rot3, wavelength, detector)
            print(f"  Saved refined geometry to: {os.path.abspath(args.save_calibrated_poni)}\n")
        else:
            print()

    # Last gate before any q is computed. image_shape is optional only so
    # that callers which genuinely have no image yet still work; every
    # caller that has one should pass it, because the detector-mismatch
    # check is the one that catches a .poni belonging to another setup.
    validate_geometry(detector, dist, poni1, poni2, wavelength,
                       image_shape=image_shape, strict=True)

    fi = FiberIntegrator(
        dist=dist,
        poni1=poni1,
        poni2=poni2,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        wavelength=wavelength,
        detector=detector,
    )
    return fi, detector


def tempfile_dir_for_diagnostics() -> str:
    """A per-run temp directory to hold calibration diagnostic plots
    (separate from the main output directory, since these are ephemeral
    sanity-check images rather than a final deliverable)."""
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "giwaxs_calibration_diagnostics")
    os.makedirs(path, exist_ok=True)
    return path


def open_file_externally(path: str) -> bool:
    """Best-effort attempt to open a file in the OS's default viewer, so a
    diagnostic image actually pops up on screen instead of just printing a
    path the user has to go find themselves. Returns True only if the
    viewer command actually reported success (not just "launched without
    raising a Python exception" -- subprocess.run doesn't raise just
    because e.g. xdg-open couldn't find an application, so the return
    code has to be checked explicitly). Caller should fall back to
    printing the path if this returns False.
    """
    import subprocess
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]  # raises on failure
            return True
        elif sys.platform == "darwin":
            result = subprocess.run(["open", path], check=False,
                                     capture_output=True)
        else:
            result = subprocess.run(["xdg-open", path], check=False,
                                     capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


def load_mask(args, fabio, shape) -> np.ndarray:
    if args.mask is None:
        return np.zeros(shape, dtype=bool)
    if args.mask.lower().endswith(".npy"):
        mask = np.load(args.mask)
    else:
        mask = fabio.open(args.mask).data
    mask = np.asarray(mask).astype(bool)
    if mask.shape != shape:
        _raise_error(f"Mask shape {mask.shape} does not match image shape {shape}.")
    return mask


def build_grazing_units(get_unit_fiber, incident_angle_deg: float):
    """Return the four grazing-incidence units used across both agents,
    with the incident angle already set."""
    unit_gi_ip = get_unit_fiber("qip_A^-1")
    unit_gi_oop = get_unit_fiber("qoop_A^-1")
    unit_gi_chi = get_unit_fiber("chigi_deg")
    unit_gi_qtot = get_unit_fiber("qtot_A^-1")
    incident_angle_rad = np.deg2rad(incident_angle_deg)
    for u in (unit_gi_ip, unit_gi_oop, unit_gi_chi, unit_gi_qtot):
        # pyFAI deprecated set_incident_angle() in 2025.10 in favour of the
        # plain property. The deprecated call still WORKS, but pyFAI's
        # deprecation decorator prints the notice plus a full stack trace to
        # stderr on EVERY call -- four units per file, so a normal working
        # session buries the log under thousands of lines and makes a real
        # traceback impossible to find. The property assignment is silent and
        # produces an identical result (verified on pyFAI 2026.5.0).
        # getattr/setattr on the class keeps the old setter as a fallback for
        # pyFAI < 2025.10, where the property doesn't exist yet.
        if isinstance(getattr(type(u), "incident_angle", None), property):
            u.incident_angle = incident_angle_rad
        else:
            u.set_incident_angle(incident_angle_rad)
    return unit_gi_ip, unit_gi_oop, unit_gi_chi, unit_gi_qtot


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Plot styling (shared by all plotting functions below)
# --------------------------------------------------------------------------- #
def style_context(font_family: Optional[str] = None, font_size: Optional[float] = None):
    """Return a matplotlib rc_context scoping font settings to a single plot
    call, rather than mutating global rcParams -- important for a
    long-running process (Streamlit app / API server) handling many
    requests with potentially different style choices concurrently.
    """
    rc: Dict[str, object] = {}
    if font_family:
        rc["font.family"] = resolve_font_family(font_family)
    if font_size:
        rc["font.size"] = font_size
        rc["axes.titlesize"] = font_size
        rc["axes.labelsize"] = font_size
        rc["xtick.labelsize"] = font_size * 0.9
        rc["ytick.labelsize"] = font_size * 0.9
        rc["legend.fontsize"] = font_size * 0.85
    return plt.rc_context(rc)


#: Colormaps offered in the UI/API -- all valid matplotlib names also work
#: even if not in this list (this is just a convenient curated subset).
COMMON_COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "turbo", "jet", "gist_heat", "hot", "afmhot",
    "gray", "bone", "coolwarm", "twilight",
]

#: Same colormaps grouped by category, for a friendlier two-step picker
#: (pick a category, then a colormap within it) instead of one long list.
COLORMAP_CATEGORIES = {
    "Perceptually uniform (recommended for scientific figures)":
        ["viridis", "plasma", "inferno", "magma", "cividis"],
    "High contrast / warm": ["turbo", "jet", "gist_heat", "hot", "afmhot"],
    "Single hue (grayscale-like)": ["gray", "bone"],
    "Diverging / cyclic": ["coolwarm", "twilight"],
}

#: A curated subset of fonts that are broadly available/bundled with
#: matplotlib's default fallback fonts, safe to offer in a UI dropdown.
#: Flattened, deduplicated view of FONT_CATEGORIES for places that just
#: want a simple list (e.g. CLI --help text) rather than the grouped dict.
COMMON_FONTS = list(dict.fromkeys(
    font for fonts in FONT_CATEGORIES.values() for font in fonts
))


#: Total byte budget for one rendered-PNG cache in the Streamlit app.
#: Sized so a full set of previews for a large batch stays cached (the
#: whole point of the cache -- unrelated reruns must not re-invoke
#: matplotlib), while a long session of style tweaking can't grow without
#: bound. See cache_png_bytes for why that mattered.
PLOT_CACHE_MAX_BYTES = 192 * 1024 * 1024


def cache_png_bytes(cache: Dict[object, bytes], key, png: bytes,
                     max_bytes: int = PLOT_CACHE_MAX_BYTES) -> bytes:
    """Insert a rendered PNG into `cache` and evict oldest-first until the
    cache's total size is within `max_bytes`. Returns the PNG.

    The cache key includes every style parameter that affects the output,
    so changing a colormap or a font does not overwrite the old entry --
    it ADDS one. Without eviction that is unbounded growth: a 14-file
    batch is ~140 MB of previews per style setting at the default 400 dpi,
    so a handful of tweaks in one session was enough to exhaust the
    memory of a Streamlit Community Cloud container and kill the process
    (which the user sees as the "Oh no." page, with no Python traceback
    anywhere in the logs, since the process is killed rather than raising).

    Eviction is plain insertion-order (dicts preserve it), which
    approximates LRU well here: the entries for the style you're currently
    looking at were inserted most recently, so the ones dropped are from
    settings you've moved on from. Evicting an entry only costs a
    re-render if you go back to that exact combination.
    """
    cache[key] = png
    if max_bytes is None or max_bytes <= 0:
        return png
    total = sum(len(v) for v in cache.values())
    while total > max_bytes and len(cache) > 1:
        oldest_key = next(iter(cache))
        total -= len(cache.pop(oldest_key))
    return png


def resolve_vmin_vmax(intensity: np.ndarray, vmin_percentile: float,
                       vmin: Optional[float] = None, vmax: Optional[float] = None,
                       vmax_percentile: float = 99.9, color_scale: str = "log"):
    """Resolve the colour-scale vmin/vmax: explicit values win if given,
    otherwise fall back to a percentile-of-nonzero-pixels heuristic for
    BOTH ends (not just the min) -- using the raw max as vmax tends to
    wash out contrast when a few hot/saturated pixels are present.

    This function is TOTAL: whatever it is handed, it returns a pair that
    LogNorm/Normalize will accept (finite, and vmin strictly less than
    vmax; both strictly positive when color_scale="log"). That matters
    because these values can come straight from a user-typed box or from
    an AI style suggestion, and matplotlib's failure mode for a bad pair
    is an unhandled `ValueError: Invalid vmin or vmax` raised deep inside
    colorbar drawing -- which in the Streamlit app killed the whole script
    run rather than surfacing as a message next to the offending widget.
    Non-positive values under a log scale are the common case: log10(0)
    is -inf, so a vmax of 0 (an empty number box) is enough to trigger it.

    Callers that want to TELL the user their input was unusable should
    check it themselves first (see validate_manual_color_scale) -- this
    function silently falls back rather than raising, so that rendering
    always succeeds.
    """
    is_log = color_scale == "log"
    floor = 1e-12  # smallest value with a finite log10, used as the log-scale ceiling-stop

    def _clean(v):
        """Drop a value that can't be used as given (None, NaN/inf, or
        non-positive under a log scale) so it falls back to automatic."""
        if v is None:
            return None
        v = float(v)
        if not np.isfinite(v):
            return None
        if is_log and v <= 0:
            return None
        return v

    vmin, vmax = _clean(vmin), _clean(vmax)

    finite = intensity[np.isfinite(intensity)]
    usable = finite[finite > 0] if is_log else finite
    if usable.size == 0:
        auto_vmin, auto_vmax = (floor, 1.0) if is_log else (0.0, 1.0)
    else:
        auto_vmin = float(np.percentile(usable, vmin_percentile))
        auto_vmax = float(np.percentile(usable, vmax_percentile))
        if is_log:
            auto_vmin = max(auto_vmin, floor)
        if auto_vmax <= auto_vmin:
            auto_vmax = auto_vmin * 10 if is_log else auto_vmin + 1.0

    v_lo = vmin if vmin is not None else auto_vmin
    v_hi = vmax if vmax is not None else auto_vmax

    # Final ordering guard: a reversed or degenerate pair (vmin >= vmax) is
    # meaningless for a colour scale and matplotlib rejects it, so widen the
    # top end rather than fail. Kept last so it also covers the case where
    # one end was explicit and the other came from the percentile heuristic.
    if v_hi <= v_lo:
        v_hi = v_lo * 10 if is_log else v_lo + 1.0

    return v_lo, v_hi


def validate_manual_color_scale(vmin, vmax, color_scale: str = "log") -> Optional[str]:
    """Check a user-entered explicit colour-scale pair and return a short
    human-readable reason if it can't be used as given, or None if it's
    fine. Purely advisory -- resolve_vmin_vmax will fall back safely
    either way; this exists so a UI can say WHY the picture doesn't match
    what was typed, instead of silently substituting different numbers.
    """
    for name, v in (("minimum", vmin), ("maximum", vmax)):
        if v is None:
            continue
        if not np.isfinite(float(v)):
            return f"The colour-scale {name} is not a finite number."
        if color_scale == "log" and float(v) <= 0:
            return (f"The colour-scale {name} ({float(v):g}) must be greater than zero "
                    f"on a log colour scale, since log(0) is undefined. Either enter a "
                    f"positive value or switch the colour scale to linear.")
    if vmin is not None and vmax is not None and float(vmax) <= float(vmin):
        return (f"The colour-scale maximum ({float(vmax):g}) must be greater than the "
                f"minimum ({float(vmin):g}).")
    return None


#: Axis label conventions for the 2D image -- both refer to the identical
#: quantities (in-plane / out-of-plane components of q); "xyz" (q_xy / q_z)
#: is this toolkit's default; "ip_oop" (q_ip / q_oop) is also available.
AXIS_LABELS = {
    "ip_oop": (r"$q_{ip}$ (Å$^{-1}$)", r"$q_{oop}$ (Å$^{-1}$)"),
    "xyz": (r"$q_{xy}$ (Å$^{-1}$)", r"$q_{z}$ (Å$^{-1}$)"),
}

#: Default output figure size (inches) and resolution for saved plots.
DEFAULT_FIGSIZE = (5, 4)
DEFAULT_DPI = 400


def add_edge_labels(fig, top: Optional[str] = None, bottom: Optional[str] = None,
                     left: Optional[str] = None, right: Optional[str] = None,
                     fontsize: Optional[float] = None,
                     top_rotation: float = 0.0, bottom_rotation: float = 0.0,
                     left_rotation: float = 90.0, right_rotation: float = 270.0):
    """Add optional text annotations along the four edges of a figure,
    OUTSIDE the plot axes -- e.g. a condition label like "25 C" or
    "Sample A" when building a multi-panel comparison across a series.
    Rotation is in degrees, counterclockwise from horizontal (matplotlib's
    usual text-rotation convention). Defaults: top/bottom horizontal (0),
    left rotated 90 (reads bottom-to-top), right rotated 270 i.e. -90
    (reads top-to-bottom) -- the conventional MIRRORED style used for
    axis labels on opposite sides of a plot, so both read comfortably
    outward. Override any of the four freely if you want a different
    convention. Any of the four labels may be left as None to omit it.
    Call this AFTER the plot's own layout is otherwise finalized but
    BEFORE saving -- it reserves a small margin via subplots_adjust so
    the labels don't overlap the axes or get clipped at the figure edge.
    """
    pad = 0.12 if fontsize is None else min(0.10 + fontsize / 300, 0.22)
    left_pad = pad if left else 0.0
    right_pad = pad if right else 0.0
    top_pad = pad if top else 0.0
    bottom_pad = pad if bottom else 0.0
    fig.subplots_adjust(
        left=fig.subplotpars.left + left_pad,
        right=fig.subplotpars.right - right_pad,
        top=fig.subplotpars.top - top_pad,
        bottom=fig.subplotpars.bottom + bottom_pad,
    )
    common = dict(fontsize=fontsize, transform=fig.transFigure)
    if top:
        fig.text(0.5, 0.99, top, ha="center", va="top", rotation=top_rotation, **common)
    if bottom:
        fig.text(0.5, 0.01, bottom, ha="center", va="bottom", rotation=bottom_rotation, **common)
    if left:
        fig.text(0.01, 0.5, left, ha="left", va="center", rotation=left_rotation, **common)
    if right:
        fig.text(0.99, 0.5, right, ha="right", va="center", rotation=right_rotation, **common)


def plot_2d_image(qx, qy, intensity, out_path=None, qlim_x=None, qlim_y=None,
                   vmin_percentile: float = 1.0, vmax_percentile: float = 99.9,
                   cmap: str = "viridis",
                   vmin: Optional[float] = None, vmax: Optional[float] = None,
                   font_family: Optional[str] = None, font_size: Optional[float] = None,
                   dpi: int = DEFAULT_DPI, figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
                   edge_label_top: Optional[str] = None, edge_label_bottom: Optional[str] = None,
                   edge_label_left: Optional[str] = None, edge_label_right: Optional[str] = None,
                   edge_label_rotations: Optional[Dict[str, float]] = None,
                   axis_label_style: str = "xyz", tick_spacing: float = 0.5,
                   tick_color: str = "white",
                   axes_linewidth: float = 1.6,
                   subtick_spacing: Optional[float] = None, color_scale: str = "log",
                   show_colorbar: bool = True):
    """2D GIWAXS q-space image -- deliberately has NO title (kept plain for
    publication/figure use); use an external caption/label if you need one.

    color_scale: "log" (default) shows the colorbar with scientific-notation
    ticks (10^n) on a logarithmic mapping -- the norm GIWAXS convention,
    since scattering intensity typically spans several orders of magnitude.
    "linear" instead uses evenly-spaced round-number ticks (100, 200, 300,
    ...) on a linear mapping -- easier to read exact values off, but weak
    features next to strong ones (e.g. higher-order peaks near the direct
    beam) will be much less visible than under a log scale.

    show_colorbar: draw the intensity bar on the right (default). Turn it
    off for a panel that will sit in a multi-panel figure sharing a single
    colorbar -- the bar is then removed entirely rather than hidden, so
    the plot fills the whole figure width and panels line up.

    subtick_spacing: minor-tick interval (1/A), OFF by default (None or 0
    both mean "no minor ticks", matching typical publication-figure
    conventions where they're an opt-in extra rather than the default).
    Minor ticks are unlabelled short marks between the major ticks; they
    don't need to evenly divide tick_spacing.

    If out_path is given, saves the figure there and returns None (closes
    the figure). If out_path is None, returns the (open) Figure instead --
    useful for interactive display (e.g. Streamlit's st.pyplot(fig)) without
    an extra disk round-trip. Caller is responsible for plt.close(fig) in
    that case.
    """
    if color_scale not in ("log", "linear"):
        raise GiwaxsError(f"Unknown color_scale '{color_scale}' -- use 'log' or 'linear'.")
    with style_context(font_family, font_size):
        # Without the colorbar the figure is a single axes, not a 1x2 grid
        # with the second one hidden: a hidden axes still reserves its
        # column, so the panel would keep a strip of blank space on the
        # right and the plot itself would stay narrower than the figure.
        # Panels destined for a multi-panel figure usually share ONE
        # colorbar, so that strip is exactly what has to go.
        if show_colorbar:
            fig, axes = plt.subplots(1, 2, width_ratios=[1, 0.05], figsize=figsize)
            ax, cax = axes[0], axes[1]
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            cax = None
        ax = [ax, cax]          # keep the rest of the body unchanged

        v_lo, v_hi = resolve_vmin_vmax(intensity, vmin_percentile, vmin, vmax, vmax_percentile,
                                        color_scale=color_scale)
        norm = LogNorm(vmin=v_lo, vmax=v_hi) if color_scale == "log" else Normalize(vmin=v_lo, vmax=v_hi)
        mesh = ax[0].pcolormesh(qx, qy, intensity, norm=norm, cmap=cmap)
        ax[0].set_facecolor("black")
        ax[0].set_aspect("equal")
        xlabel, ylabel = AXIS_LABELS.get(axis_label_style, AXIS_LABELS["xyz"])
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel(ylabel)
        if qlim_x is not None:
            ax[0].set_xlim(qlim_x)
        if qlim_y is not None:
            ax[0].set_ylim(qlim_y)
        if tick_spacing:
            ax[0].xaxis.set_major_locator(MultipleLocator(tick_spacing))
            ax[0].yaxis.set_major_locator(MultipleLocator(tick_spacing))
        if subtick_spacing:
            ax[0].xaxis.set_minor_locator(MultipleLocator(subtick_spacing))
            ax[0].yaxis.set_minor_locator(MultipleLocator(subtick_spacing))
        else:
            ax[0].xaxis.set_minor_locator(NullLocator())
            ax[0].yaxis.set_minor_locator(NullLocator())
        # The ticks point INTO the map, so they sit on the image rather than
        # on the page: on a dark log-scaled map a black mark is invisible,
        # which is why white is the default here and not elsewhere. Only the
        # marks are recoloured -- the labels stay outside on white paper.
        #
        # Frame and ticks are also drawn heavier than matplotlib's 0.8 pt
        # default. A hairline survives a screen but thins out in print and
        # all but vanishes once a panel is scaled down into a figure, which
        # is where these end up.
        ax[0].tick_params(axis="both", which="major", direction="in",
                           color=tick_color, width=axes_linewidth,
                           length=4.0 + 2.0 * axes_linewidth)
        ax[0].tick_params(axis="both", which="minor", direction="in",
                           color=tick_color, width=axes_linewidth * 0.75,
                           length=2.0 + 1.2 * axes_linewidth)
        for _spine in ax[0].spines.values():
            _spine.set_linewidth(axes_linewidth)
        # fig.colorbar, NOT plt.colorbar: the pyplot version routes through
        # gcf(), which is global state shared across Streamlit's script-run
        # threads, so under the app it can attach the colorbar to a DIFFERENT
        # figure than the one being built here (matplotlib says so out loud:
        # "Adding colorbar to a different Figure ... than ... fig.colorbar is
        # called on"). Going through the figure object keeps it bound to the
        # right one regardless of what else is being drawn concurrently.
        if show_colorbar:
            cbar = fig.colorbar(mesh, cax=ax[1], orientation="vertical")
            # Not tick_color: that is for marks lying ON the map. The
            # colour bar is its own axes with its own background.
            cbar.ax.tick_params(which="both", direction="in")
        fig.tight_layout()
        add_edge_labels(fig, top=edge_label_top, bottom=edge_label_bottom,
                         left=edge_label_left, right=edge_label_right, fontsize=font_size,
                         **(edge_label_rotations or {}))
        if out_path:
            # Without the colour bar the axes keeps its height-limited width
            # (aspect is equal, as a q-space map requires), so the space the
            # bar used to fill is simply left blank on the right. Trimming on
            # save turns that into an actual bare panel rather than a plot
            # with a gap where the bar was.
            fig.savefig(out_path, dpi=dpi,
                         **({} if show_colorbar else {"bbox_inches": "tight"}))
            plt.close(fig)
            return None
        return fig


def check_direct_beam_centred(qx, qy, intensity,
                               qz_band: float = 0.12,
                               qxy_window: float = 0.6,
                               tolerance: float = 0.05):
    """Is the direct beam where the geometry says q_xy = 0?

    A grazing-incidence map has the direct beam and the specular rod at
    q_xy = 0 by definition -- that is not a property of the sample, it is
    what q_xy means. So their position is a free check on the beam centre
    that uses the SAMPLE frame itself and needs no calibrant: if the
    bright low-q feature sits at q_xy = -0.2, the geometry's beam centre
    column is wrong by however many pixels that is, and every peak
    position read off the map is wrong with it.

    Worth having because a geometry can be wrong in a way nothing else
    notices: it may be internally consistent, describe the right detector
    and come from a real calibration -- just one taken with the detector
    somewhere else.

    Returns (ok, offset_q, message). offset_q is the intensity-weighted
    centre of the near-horizon band in 1/Angstrom; NaN when there is not
    enough signal to judge, in which case ok is True (silence beats a
    false alarm).
    """
    qx = np.asarray(qx, float)
    qy = np.asarray(qy, float)
    I = np.asarray(intensity, float)
    if I.ndim != 2 or qx.ndim != 1 or qy.ndim != 1:
        return True, float("nan"), ""

    rows = np.abs(qy) <= qz_band          # the near-horizon strip
    cols = np.abs(qx) <= qxy_window       # around where the beam should be
    if rows.sum() < 3 or cols.sum() < 10:
        return True, float("nan"), ""

    band = I[np.ix_(rows, cols)]
    w = np.where(np.isfinite(band) & (band > 0), band, 0.0).sum(axis=0)
    if w.sum() <= 0:
        return True, float("nan"), ""

    # Weight by intensity above the band's own median, so a broadly bright
    # strip does not drag the centroid; it is the PEAK we are locating.
    base = np.median(w[w > 0]) if (w > 0).any() else 0.0
    wt = np.clip(w - base, 0.0, None)
    if wt.sum() <= 0:
        return True, float("nan"), ""
    offset = float((qx[cols] * wt).sum() / wt.sum())

    if abs(offset) <= tolerance:
        return True, offset, ""
    return False, offset, (
        f"The direct beam sits at q_xy = {offset:+.3f} 1/A, but it is q_xy = 0 "
        f"by definition -- so the beam-centre COLUMN in this geometry is off. "
        f"Every q read off this map is shifted with it. Check that the "
        f"geometry belongs to these frames: a .poni or a calibration from a "
        f"run with the detector in a different position looks perfectly "
        f"valid and still does this."
    )


HORIZON_SEARCH_ROWS = (0, 450)
MAX_BEAM_ROW_MISMATCH_PX = 12.0


def find_horizon_row(image, search_rows=HORIZON_SEARCH_ROWS, n_blocks: int = 20,
                      min_block_frac: float = 0.60, agree_px: float = 6.0):
    """Locate the sample horizon in a grazing-incidence frame.

    The horizon is the one feature a GI frame is guaranteed to carry across
    the WHOLE detector width: it is the sample surface seen edge-on, so it
    is a straight line, not an arc, and above it the sample shadows the
    detector. That full-width straightness is what separates it from the
    two things it is otherwise easy to confuse it with -- a beamstop edge,
    which is confined to the stop's own columns, and a Debye arc, which
    curves away within a few hundred pixels.

    So: split the width into blocks, find the strongest rising edge in each
    INDEPENDENTLY, and keep the answer only if most blocks agree. Blocks
    that are mostly dead (beamstop, module gaps) abstain rather than vote.

    Returns (row, agreement_fraction, n_voting_blocks). `row` is None when
    the blocks do not agree. That is a refusal, not a fallback: a frame
    with no sample in the beam -- a calibrant exposure, say -- has no
    horizon to find, and inventing one would be worse than saying so.
    """
    image = np.asarray(image)
    h, w = image.shape
    lo = int(search_rows[0])
    hi = int(min(search_rows[1], h))
    I = np.asarray(image, dtype=float)
    I = np.where(np.isfinite(I) & (I > 0), I, np.nan)
    rows = np.arange(lo, hi)
    edges = np.linspace(0, w, n_blocks + 1).astype(int)

    votes = []
    for a, b in zip(edges[:-1], edges[1:]):
        block = I[lo:hi, a:b]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
            prof = np.nanmedian(block, axis=1)
        dead = ~np.isfinite(prof)
        if dead.mean() > 0.30:
            continue                       # beamstop / dead block abstains
        if dead.any():
            # Bridge module gaps rather than let their edges masquerade as
            # the horizon -- a gap is a step in the data too.
            idx = np.arange(prof.size)
            prof = np.interp(idx, idx[~dead], prof[~dead])
        k = 5
        sm = np.convolve(prof, np.ones(k) / k, mode="same")
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.gradient(np.log(np.clip(sm, 1e-6, None)))
        g[:k] = 0.0
        g[-k:] = 0.0
        votes.append(float(rows[int(np.argmax(g))]))

    if len(votes) < 4:
        return None, 0.0, len(votes)
    votes_arr = np.asarray(votes, dtype=float)
    med = float(np.median(votes_arr))
    agreeing = np.abs(votes_arr - med) <= agree_px
    frac = float(np.mean(agreeing))
    if frac < min_block_frac:
        return None, frac, len(votes)
    return float(np.median(votes_arr[agreeing])), frac, len(votes)


def beam_row_from_horizon(horizon_row: float, incident_angle_deg: float,
                           dist: float, pixel_size: float) -> float:
    """Back out the direct-beam row from the horizon row.

    Rays leaving along the sample surface travel at the incident angle
    above the direct beam, so the horizon sits that far up the detector:
        horizon_row = beam_row + dist * tan(alpha) / pixel_size
    """
    return float(horizon_row - dist * np.tan(np.deg2rad(incident_angle_deg)) / pixel_size)


def check_beam_row_against_horizon(image, incident_angle_deg: float, dist: float,
                                    pixel_size: float, declared_beam_row: float,
                                    tolerance_px: float = MAX_BEAM_ROW_MISMATCH_PX):
    """Check a geometry's beam-centre ROW against the frame's own horizon.

    This exists because the row is the badly-conditioned half of a
    grazing-incidence calibration. The beam centre sits near the detector
    edge, so every calibrant ring is cut off on that side and only its
    lower arc is ever recorded. Fitting a centre to an arc missing its top
    leaves the row nearly degenerate with the distance -- the fit reports a
    sub-pixel residual either way. The column, sampled on both sides, is
    over-determined and comes out right; the row can be tens of pixels out
    and nothing in the fit complains.

    The sample frame carries the answer the calibrant cannot give, so use
    it. Returns (ok, measured_row, message). `ok` is None -- neither pass
    nor fail -- when the frame has no usable horizon, since absence of
    evidence is not a disagreement.
    """
    horizon, frac, n_blocks = find_horizon_row(image)
    if horizon is None:
        return None, None, (
            f"No full-width horizon in this frame (only {frac:.0%} of "
            f"{n_blocks} column blocks agreed), so the beam-centre row "
            f"could not be cross-checked against the data."
        )
    measured = beam_row_from_horizon(horizon, incident_angle_deg, dist, pixel_size)
    delta = measured - declared_beam_row
    if abs(delta) <= tolerance_px:
        return True, measured, ""
    return False, measured, (
        f"This frame's own horizon puts the direct beam at row {measured:.1f}, "
        f"but the geometry declares row {declared_beam_row:.1f} -- a "
        f"{abs(delta):.0f} px disagreement. The horizon is the sample surface "
        f"seen edge-on and spans the full detector width ({frac:.0%} of "
        f"{n_blocks} column blocks agreed on it), so it is measured, not "
        f"assumed. A calibrant cannot settle this: with the beam centre near "
        f"the detector edge every ring is cut off, leaving the row nearly "
        f"free. Peak POSITIONS move very little, but the rings are smeared, "
        f"so peak widths and any coherence length read off them are wrong."
    )


def find_beamstop_mask(image, beam_col, rel_threshold: float = 0.30,
                        min_rows: int = 40, grow: int = 3):
    """Find the beamstop shadow so it can be masked out of the integration.

    This matters far more than it looks. A grazing-incidence beamstop is
    not a small disc over the direct beam: it is a long vertical finger
    covering the specular rod, which runs straight up the beam COLUMN --
    exactly where the out-of-plane sector (chi near 0) lives. Unmasked,
    those shadow pixels are not ignored; they are averaged in as genuine
    low intensity, so the out-of-plane cut is pulled down and broken up
    wherever the stop reaches. On the dataset this was written for that
    understated the out-of-plane lamellar peak by 2.5x and filled the
    region below it with ragged nonsense.

    A pixel is shadow when it records far less than the rest of its OWN
    row, which separates the stop from an overall dim frame. Only the run
    through the beam column is taken, so unrelated dark patches are left
    alone, and a run spanning a quarter of the width is treated as a
    detector module gap rather than a stop. If no run persists over
    `min_rows`, there is no beamstop and an empty mask is returned.

    Returns a boolean array, True where the beamstop shadows the detector.
    """
    I = np.asarray(image, dtype=float)
    h, w = I.shape
    c = int(round(beam_col))
    if not (0 <= c < w):
        return np.zeros((h, w), dtype=bool)

    left = I[:, max(0, c - 700):max(1, c - 200)]
    right = I[:, min(w, c + 200):min(w, c + 700)]
    ref_cols = np.concatenate([left, right], axis=1)
    if ref_cols.shape[1] < 20:
        return np.zeros((h, w), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(np.where(np.isfinite(ref_cols), ref_cols, np.nan), axis=1)
    ref = np.where(np.isfinite(ref), ref, 0.0)

    dark = I < (rel_threshold * ref[:, None])
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        if not dark[r, c]:
            continue
        lo = c
        while lo - 1 >= 0 and dark[r, lo - 1]:
            lo -= 1
        hi = c
        while hi + 1 < w and dark[r, hi + 1]:
            hi += 1
        if (hi - lo + 1) > 0.25 * w:
            continue                      # a whole dead row: module gap, not a stop
        mask[r, max(0, lo - grow):min(w, hi + grow + 1)] = True
    if int(mask.any(axis=1).sum()) < min_rows:
        return np.zeros((h, w), dtype=bool)
    return mask


def write_linecut_workbook(out_path, combined=None, sources=None, report=None,
                            extra_sheets=None, source_file=None):
    """Write line cuts to .xlsx, with the join's provenance beside them.

    A combined curve is two measurements reconciled into one, and a bare
    column of numbers carries no trace of that. Anyone who opens the
    file later -- including the person who made it -- cannot tell which
    part came from which cut, what scale factor was applied, or how well
    the two actually agreed. So the workbook keeps three things
    together: the combined curve, the cuts it was built from, and the
    numbers behind the join.

    combined:     (q, intensity) or None
    sources:      {"sheet name": (q, intensity)}
    report:       the dict stitch_cuts() returns
    extra_sheets: {"sheet name": (q, intensity)} for any other cuts
    """
    try:
        from openpyxl import Workbook
    except ImportError:
        _raise_error(
            "Writing .xlsx needs openpyxl, which is not installed. "
            "Install it with `pip install openpyxl`, or take the .txt line "
            "cuts saved alongside instead -- they carry the same numbers."
        )

    wb = Workbook()
    wb.remove(wb.active)

    def _add(name, q, intensity):
        ws = wb.create_sheet(title=name[:31])
        ws.append(["q (1/A)", "Intensity (a.u.)"])
        for a, b in zip(np.asarray(q).tolist(), np.asarray(intensity).tolist()):
            ws.append([float(a), float(b)])
        ws.freeze_panes = "A2"
        ws.column_dimensions["A"].width = 14
        ws.column_dimensions["B"].width = 18
        return ws

    if combined is not None:
        _add("combined", combined[0], combined[1])
    for name, (q, i) in (sources or {}).items():
        _add(name, q, i)
    for name, (q, i) in (extra_sheets or {}).items():
        _add(name, q, i)

    ws = wb.create_sheet(title="method")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 80
    rows = [("source file", source_file or "")]
    joined_by_scaling = False
    if isinstance(report, dict):
        rows.append(("combined curve produced",
                     "yes" if report.get("joined") else "NO"))
        rows.append(("method", report.get("primary", "")))
        ov = report.get("overlap")
        if ov:
            rows.append(("overlap q (1/A)", f"{ov[0]:.4f} - {ov[1]:.4f}"))
        if report.get("n_overlap"):
            rows.append(("shared points used", report["n_overlap"]))
        if report.get("scale") is not None:
            joined_by_scaling = True
            rows.append(("scale factor (low/high cut)", round(report["scale"], 5)))
        if report.get("shape_scatter") is not None:
            rows.append(("shape agreement (dex)", round(report["shape_scatter"], 5)))
        if report.get("message"):
            rows.append(("note", report["message"]))
    rows.append(("", ""))
    if joined_by_scaling:
        rows += [
            ("what the scale factor is",
             "The two cuts have different angular acceptance, so their "
             "absolute scales differ. One was multiplied onto the other over "
             "the overlap."),
            ("what shape agreement means",
             "Scatter of the two cuts' ratio about its median, in dex. A "
             "scale factor alone proves nothing -- two curves can always be "
             "matched at one point. Small scatter is what says they are the "
             "same curve. The join is refused above 0.12 dex."),
        ]
    else:
        rows += [
            ("how the window works",
             "One region, not two cuts reconciled. Its inner edge stays at "
             "the beamstop's outer edge, where data begins; its outer edge is "
             "whichever is wider, the fixed strip or an angular limit growing "
             "with q. Near the origin that is the strip, passing beside the "
             "beamstop; further out it is the sector, clear of the missing "
             "wedge."),
            ("why that matters",
             "There is no scale factor between differing acceptances and "
             "nothing to check agreement on, because nothing was joined. The "
             "two sheets beside this one are the conventional cuts, kept for "
             "comparison only."),
            ("abscissa",
             "Reported as |q|, not the bare q_z: the window sits at a "
             "non-zero q_xy offset, so this puts its peaks where an angular "
             "cut's are."),
        ]
    for a, b in rows:
        ws.append([a, b])
    for row in ws.iter_rows(min_col=2, max_col=2):
        for cell in row:
            cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")
    wb.save(out_path)
    return out_path


def plot_stitch_diagnostic(q_box, i_box, q_sector, i_sector,
                            q_combined=None, i_combined=None, report=None,
                            out_path=None, title=None, show_combined: bool = False,
                            switch_q: Optional[float] = None,
                            font_family: Optional[str] = None,
                            font_size: Optional[float] = None,
                            dpi: int = DEFAULT_DPI,
                            figsize: Tuple[float, float] = (8.0, 5.5),
                            q_range: Optional[Tuple[float, float]] = None):
    """Put the two source cuts on one axes, as a companion to the join.

    A joined curve hides its own provenance: nothing in it says which
    part came from where, or that two measurements were reconciled to
    make it. This puts the inputs back on the page, with the overlap
    they were matched over and the span they were crossfaded across, so
    the join can be judged rather than taken on trust. Worth looking at
    before quoting anything measured off the combined curve.

    The combined curve itself is left OFF by default, which is the whole
    point: it follows the strip almost exactly over most of the range
    and draws straight over it, hiding the very line a reader needs to
    see. It is a separate output anyway. Pass show_combined=True to
    overlay it.

    `report` is the dict stitch_cuts() returns; its scale and shape
    agreement are printed on the plot.
    """
    with style_context(font_family, font_size):
        fig, ax = plt.subplots(figsize=figsize)
        if switch_q:
            ax.axvline(switch_q, color="0.4", ls="--", lw=1.0,
                       label=f"window opens ({switch_q:.2f} 1/\u00c5)")
        ov = report.get("overlap") if isinstance(report, dict) else None
        if ov:
            lo, hi = float(ov[0]), float(ov[1])
            ax.axvspan(lo, hi, color="0.5", alpha=0.12, lw=0,
                       label="overlap used to match scales")
            span = 0.3
            start = 10 ** (np.log10(hi) - span * (np.log10(hi) - np.log10(lo)))
            ax.axvspan(start, hi, color="tab:orange", alpha=0.18, lw=0,
                       label="crossfade")
        ax.plot(q_box, i_box, lw=1.4, color="tab:blue", label="box cut (strip)")
        ax.plot(q_sector, i_sector, lw=1.4, color="tab:red",
                label="angular sector")
        if show_combined and q_combined is not None:
            ax.plot(q_combined, i_combined, lw=1.8, color="k", alpha=0.8,
                    label="combined")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if q_range:
            ax.set_xlim(*q_range)
        ax.set_xlabel(r"$q$ ($\AA^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.tick_params(axis="both", which="both", direction="in")
        ax.set_title(title or "The two cuts behind the combined profile")
        if isinstance(report, dict) and report.get("joined"):
            _sc = report.get("scale")
            _note = (f"scale {_sc:.3f}  ·  "
                     f"shape {report.get('shape_scatter', float('nan')):.3f} dex  ·  "
                     f"{report.get('n_overlap', 0)} shared points"
                     if _sc is not None else report.get("primary", ""))
            ax.text(0.99, 0.02, _note,
                    transform=ax.transAxes, fontsize=8, color="0.35",
                    ha="right", va="bottom")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)
            return None
        return fig


def adaptive_cut(intensity, qx, qy, inner: float = 0.025, outer: float = 0.05,
                  angle_deg: float = 8.0, as_qtotal: bool = True):
    """One out-of-plane cut spanning the whole q range, without a join.

    The two conventional cuts fail at opposite ends. An angular sector
    is narrower than the beamstop near the origin, so it returns nothing
    until it clears it. A fixed q_xy strip passes beside the beamstop
    but is swallowed by the missing wedge as that widens. Joining them
    works, but it costs a scale factor between two different angular
    acceptances and a check that the two really are the same curve.

    Neither is needed if the window is simply allowed to open. Hold its
    inner edge at the beamstop's outer edge, where data begins, and let
    the outer edge be whichever is wider: a fixed strip, or an angular
    limit that grows with q.

        |q_xy| from `inner` to max(`outer`, q_z * tan(`angle_deg`))

    Near the origin that is the strip, passing beside the beamstop down
    to q_z = 0. Further out it is the sector, wide enough to stay clear
    of the missing wedge. It is one region, one average, no scale factor
    and nothing to reconcile. On real frames it reproduced the joined
    curve to 0.0002-0.0004 in fitted peak position over 0.038-2.15 1/A.

    `as_qtotal` reports the abscissa as |q| rather than the bare q_z,
    since the window sits at a non-zero q_xy offset; that puts its peaks
    where an angular cut's are. The offset used is the mean |q_xy| of
    the bins actually averaged in each row, so it follows the window as
    it opens.

    Returns (q, intensity) with rows that had no measured bin dropped.
    """
    I = np.asarray(intensity, dtype=float)
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    if not (outer > inner >= 0):
        _raise_error(f"adaptive_cut: need outer > inner >= 0, got "
                     f"inner={inner}, outer={outer}.")
    t = np.tan(np.deg2rad(float(angle_deg)))
    aqx = np.abs(qx)
    qs, vs = [], []
    for r in range(I.shape[0]):
        qz = float(qy[r])
        if qz <= 0:
            continue
        hi = max(outer, qz * t)
        sel = (aqx >= inner) & (aqx <= hi)
        if not sel.any():
            continue
        row = I[r, sel]
        live = np.isfinite(row) & (row > 0)
        if not live.any():
            continue
        if as_qtotal:
            qs.append(float(np.hypot(qz, float(np.mean(aqx[sel][live])))))
        else:
            qs.append(qz)
        vs.append(float(row[live].mean()))
    if not qs:
        _raise_error(
            f"adaptive_cut: no measured bins anywhere in |q_xy| from {inner} "
            f"outward. Check the inner edge against the beamstop's actual "
            f"width, which the beamstop mask reports."
        )
    return np.asarray(qs), np.asarray(vs)


def stitch_cuts(q_a, i_a, q_b, i_b, max_shape_scatter: float = 0.12,
                 min_overlap_points: int = 20, primary: str = "low",
                 blend_span: float = 0.3):
    """Join two line cuts covering different q ranges into one curve.

    In the out-of-plane direction neither cut spans the range and they
    fail at opposite ends. An angular sector is narrower than the
    beamstop near the origin, so it begins only once it clears it. A
    fixed q_xy strip passes beside the beamstop but is swallowed by the
    missing wedge as that widens. Between those two failures is a wide
    band where both are measured, and that band is what makes a join
    possible at all.

    Which curve covers the low end and which the high end is worked out
    from the data, not assumed, so the arguments may be given either way
    round.

    `primary` picks whose absolute scale the result carries -- "low" for
    the curve reaching furthest down in q, "high" for the other. The
    other is multiplied onto it. `blend_span` is the fraction of the
    overlap, measured from its top, across which the two are crossfaded;
    below that the primary is used alone. The default 0.3 with
    primary="low" means the strip is used by itself over the lower 70%
    of the overlap and only hands over near the end of its reach, which
    is what "use the strip, and let the sector fill in what it cannot
    reach" means in practice. Pass blend_span=1.0 to crossfade across
    the whole overlap instead.

    A scale factor alone proves nothing -- two curves can always be made
    to agree at one point. What has to hold is that they have the same
    SHAPE where they overlap, or joining them manufactures a feature. So
    the scatter of their ratio about its median is measured, in dex, and
    the join is refused past `max_shape_scatter`. On real data the two
    out-of-plane cuts agreed to a scale factor of 0.95-0.97 with a
    scatter of 0.046-0.076 dex.

    Returns (q, intensity, report). `report` carries scale, overlap,
    shape_scatter, n_overlap, primary and a message. On refusal q and
    intensity are the primary curve unchanged, and the message says why.
    """
    q_a = np.asarray(q_a, dtype=float); i_a = np.asarray(i_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float); i_b = np.asarray(i_b, dtype=float)
    base = {"scale": None, "overlap": None, "shape_scatter": None,
            "n_overlap": 0, "primary": primary, "joined": False}

    for arr in (q_a, i_a, q_b, i_b):
        if arr.size < 2:
            return q_a, i_a, dict(base, message="One cut is empty; nothing to join.")

    # Orient by the data: which one reaches further down in q.
    if float(q_a.min()) <= float(q_b.min()):
        (q_lo_c, i_lo_c), (q_hi_c, i_hi_c) = (q_a, i_a), (q_b, i_b)
    else:
        (q_lo_c, i_lo_c), (q_hi_c, i_hi_c) = (q_b, i_b), (q_a, i_a)

    lo = max(float(q_lo_c.min()), float(q_hi_c.min()))
    hi = min(float(q_lo_c.max()), float(q_hi_c.max()))
    fallback_q, fallback_i = ((q_lo_c, i_lo_c) if primary == "low"
                              else (q_hi_c, i_hi_c))
    if not (hi > lo * 1.02):
        return fallback_q, fallback_i, dict(
            base, message=(
                f"The two cuts do not overlap (one covers {q_lo_c.min():.3f}-"
                f"{q_lo_c.max():.3f}, the other {q_hi_c.min():.3f}-"
                f"{q_hi_c.max():.3f} 1/A), so there is nothing to put them on a "
                f"common scale with. Widen the strip or the sector until they do."))

    grid = np.geomspace(lo * 1.02, hi * 0.98, 200)
    a = np.interp(grid, q_lo_c, i_lo_c)
    b = np.interp(grid, q_hi_c, i_hi_c)
    good = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if int(good.sum()) < min_overlap_points:
        return fallback_q, fallback_i, dict(
            base, n_overlap=int(good.sum()), overlap=(lo, hi),
            message=(f"Only {int(good.sum())} usable points in the overlap "
                     f"{lo:.3f}-{hi:.3f} 1/A, too few to fix a scale factor."))

    ratio = a[good] / b[good]                      # low-curve / high-curve
    scale = float(np.median(ratio))
    scatter = float(np.std(np.log10(ratio / scale)))
    base.update(scale=scale, overlap=(lo, hi), shape_scatter=scatter,
                n_overlap=int(good.sum()))
    if scatter > max_shape_scatter:
        return fallback_q, fallback_i, dict(
            base, message=(
                f"Refusing to join: across the overlap {lo:.3f}-{hi:.3f} 1/A "
                f"the two cuts differ in SHAPE, not just scale (ratio scatter "
                f"{scatter:.3f} dex against a limit of {max_shape_scatter:.3f}). "
                f"A single scale factor would not make them one curve, so "
                f"joining them would manufacture a feature. They are probably "
                f"not sampling the same scattering -- check the 2D map."))

    if primary == "low":
        i_lo_use, i_hi_use, applied = i_lo_c, i_hi_c * scale, scale
    else:
        i_lo_use, i_hi_use, applied = i_lo_c / scale, i_hi_c, 1.0 / scale

    q_all = np.unique(np.concatenate([q_lo_c, q_hi_c]))
    lo_i = np.interp(q_all, q_lo_c, i_lo_use, left=np.nan, right=np.nan)
    hi_i = np.interp(q_all, q_hi_c, i_hi_use, left=np.nan, right=np.nan)

    # Crossfade only across the top `blend_span` of the overlap, so the
    # primary is used alone for as much of its own reach as possible.
    span = float(np.clip(blend_span, 1e-3, 1.0))
    l10lo, l10hi = np.log10(lo), np.log10(hi)
    if primary == "low":
        start = l10hi - span * (l10hi - l10lo)     # hand over near the top
    else:
        start = l10lo
        l10hi = l10lo + span * (np.log10(hi) - l10lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (np.log10(q_all) - start) / max(l10hi - start, 1e-9)
    w = np.clip(w, 0.0, 1.0)                       # 0 = low curve, 1 = high
    out = np.where(np.isnan(lo_i), hi_i,
                   np.where(np.isnan(hi_i), lo_i, (1 - w) * lo_i + w * hi_i))
    keep = np.isfinite(out) & (out > 0)
    verdict = "clean" if scatter < 0.05 else "acceptable"
    who = ("the low-q cut" if primary == "low" else "the high-q cut")
    return q_all[keep], out[keep], dict(
        base, joined=True,
        message=(f"Joined over {lo:.3f}-{hi:.3f} 1/A from {int(good.sum())} "
                 f"shared points, keeping {who}'s scale; the other was "
                 f"multiplied by {applied:.3f}. Shape agreement {scatter:.3f} "
                 f"dex ({verdict}). The primary is used alone up to "
                 f"q = {10**start:.3f} 1/A and crossfaded to the other by "
                 f"{hi:.3f}."))

def box_cut_report(intensity, qx, qy, along: str = "qz",
                    across_range=(0.025, 0.05)):
    """Say where a box cut's data stops, and why.

    A strip does not simply fade out: it ends, and for one of two very
    different reasons. Running off the detector is a boundary of the
    measurement. Entering the missing wedge is a boundary of what a
    single grazing-incidence shot can reach at all -- and because the
    wedge widens with q, a strip survives only until the wedge opens
    past the strip's own q_xy. A strip at 0.025-0.05 therefore ends
    around q = 0.9 and carries no pi-pi peak, while one at 0.20-0.30
    reaches 2.1 but is useless at low q.

    Without this, the curve just stops and a fit over a window beyond
    the stop reports whatever the truncated data supports.

    Returns (q_stop, reason, message). q_stop is None when the strip
    never had data at all.
    """
    I = np.asarray(intensity, dtype=float)
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    lo, hi = float(across_range[0]), float(across_range[1])

    if along == "qz":
        sel = (np.abs(qx) >= lo) & (np.abs(qx) <= hi)
        band = I[:, sel]
        axis_q = qy
        rows_full = I
        across_full = qx
    else:
        sel = (np.abs(qy) >= lo) & (np.abs(qy) <= hi)
        band = I[sel, :].T
        axis_q = qx
        rows_full = I.T
        across_full = qy

    alive = (np.isfinite(band) & (band > 0)).any(axis=1) & (axis_q > 0)
    if not alive.any():
        return None, "empty", (
            f"The strip |q| = {lo:g}-{hi:g} 1/A contains no measured bins at "
            f"all. If it sits inside the beamstop shadow, move its inner edge "
            f"outward."
        )
    last = int(np.where(alive)[0].max())
    q_stop = float(axis_q[last])
    if last >= alive.size - 1:
        return q_stop, "end_of_map", (
            f"The strip runs to the edge of the map at q = {q_stop:.2f} 1/A."
        )

    # Look just past the stop: is there data further out on the across axis?
    probe = rows_full[last + 1]
    outer = np.abs(across_full) > hi
    has_outer = bool((np.isfinite(probe[outer]) & (probe[outer] > 0)).any())
    if has_outer:
        return q_stop, "missing_wedge", (
            f"This strip carries data only up to q = {q_stop:.2f} 1/A. Beyond "
            f"that it is inside the missing wedge, which widens with q until "
            f"it swallows the strip -- there is measured data further out in "
            f"q_xy at the same height, so this is the wedge and not the "
            f"detector edge. A peak above {q_stop:.2f} 1/A (pi-pi, typically "
            f"near 1.7) will not appear here: read it from the angular sector "
            f"cut instead, which is unaffected that far out. A wider strip "
            f"reaches higher q but drifts off any peak that sits on the axis."
        )
    return q_stop, "detector_edge", (
        f"The strip carries data up to q = {q_stop:.2f} 1/A, where it runs "
        f"off the detector."
    )


def box_cut(intensity, qx, qy, along: str = "qz",
             across_range=(0.03, 0.12), along_range=None,
             as_qtotal: bool = True):
    """Line cut from a fixed strip of the remapped map, not an angular wedge.

    A chi wedge opens linearly with q, so it integrates a narrow band at
    low q and a wide one at high q. That is the wrong shape for two
    common jobs at once. Near the origin the wedge is narrower than the
    beamstop, so it returns nothing at all below the q where it finally
    clears the stop. Far out it is wide enough to average across
    orientation, which smears the higher orders of a lamellar series --
    the very thing a (001)/(002)/(003) series is measured to resolve.

    A strip of constant width does neither. It passes beside the beamstop
    all the way down to q = 0, and its angular acceptance SHRINKS as q
    grows, so high orders come out sharper than the wedge gives. On the
    dataset this was written for, a q_xy strip of 0.03-0.12 reached
    q_z = 0.001 (against 0.245 for a +-8 deg wedge) while narrowing (002)
    from 0.074 to 0.067 and (003) from 0.075 to 0.071 -- where widening
    the wedge to +-20 deg instead blew (002) out to 0.101.

    This reads the already-remapped map, so it inherits whatever mask
    that map was built with -- which, for the 2D image, leaves the
    detector's module gaps unmasked so the picture stays continuous.
    That was worth checking rather than assuming, and it turns out not
    to matter: against the same cut taken from a properly masked map,
    the worst local deviation near a gap moved by 0.001-0.005 on real
    frames, and in both cases it sat at q = 0.53, a real feature, not at
    the gap. The remapping spreads each detector row across a little
    range of q_z depending on q_xy, so a gap's deficit is smeared rather
    than landing on one bin the way it does in an angular sector. Paying
    a second integrate2d per file to close a 0.5% effect is not worth
    it, but do not assume the same of any other map-derived quantity.

    along:  "qz" for an out-of-plane profile from a q_xy strip,
            "qxy" for an in-plane profile from a q_z strip.
    across_range:  (lo, hi) on the OTHER axis, as |q|, so a q_xy strip
            takes both sides of the beam and averages them.
    as_qtotal:  report the abscissa as |q| = sqrt(along^2 + mean_across^2)
            rather than the bare along-axis value. A strip sits at a fixed
            offset from the axis, so its peaks appear at a slightly lower
            along-axis value than the same peak in a wedge; this puts the
            two on the same scale. Set False to get the raw axis value.

    Returns (q, intensity) with empty rows dropped.
    """
    I = np.asarray(intensity, dtype=float)
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    lo, hi = float(across_range[0]), float(across_range[1])
    if hi <= lo:
        _raise_error(f"box_cut: across_range must be increasing, got {across_range}.")

    if along == "qz":
        sel = (np.abs(qx) >= lo) & (np.abs(qx) <= hi)
        band = I[:, sel]
        axis_q = qy
        across_vals = np.abs(qx[sel])
    elif along == "qxy":
        sel = (np.abs(qy) >= lo) & (np.abs(qy) <= hi)
        band = I[sel, :].T
        axis_q = qx
        across_vals = np.abs(qy[sel])
    else:
        _raise_error(f"box_cut: along must be 'qz' or 'qxy', got {along!r}.")

    if band.size == 0 or sel.sum() == 0:
        _raise_error(
            f"box_cut: no bins fall in |q| = {lo}-{hi} 1/A on the across "
            f"axis. Check the range against the map's actual extent."
        )

    good = np.isfinite(band) & (band > 0)
    n = good.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(n > 0, np.where(good, band, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)

    keep = np.isfinite(prof) & (axis_q > 0)
    if along_range is not None:
        keep &= (axis_q >= float(along_range[0])) & (axis_q <= float(along_range[1]))
    q_out = axis_q[keep]
    if as_qtotal:
        mean_across = float(np.mean(across_vals))
        q_out = np.hypot(q_out, mean_across)
    return q_out, prof[keep]


def linecut_drop_empty_bins(result, min_count_fraction: float = 0.5):
    """Drop bins that no pixel reached, and bins that most pixels missed.

    pyFAI reports an empty bin as intensity 0. That is a different claim
    from "nothing was measured here": on a log axis it rules a spike
    through the plot, in a saved .txt it cannot be told from a real zero
    reading, and a peak fit will happily try to pass through it.

    A PARTLY filled bin is the subtler and more damaging case. Where a
    line cut crosses a detector module gap, the bins at the gap's edges
    keep a handful of pixels and are reported as ordinary points -- but
    those survivors all sit at one extreme of the sector, not spread
    across it, so their mean is biased. They come out low, and a fit
    spanning the gap chases them. The gap is then not a clean hole in
    the curve but a pair of false points pulling the peak down.

    So a bin is also dropped when its pixel count falls below
    `min_count_fraction` of the local median count. That leaves a clean
    gap, which a fit can span without being dragged, instead of two
    plausible-looking wrong points. Set the fraction to 0 to keep every
    non-empty bin.

    Accepts a pyFAI result object or a plain (q, intensity) pair.
    """
    q = np.asarray(result[0], dtype=float)
    intensity = np.asarray(result[1], dtype=float)
    count = getattr(result, "count", None)
    if count is None:
        # Older pyFAI with no count array: an exact 0.0 after averaging is
        # overwhelmingly an empty bin rather than a measured zero, and
        # partial bins cannot be detected at all.
        keep = intensity != 0.0
    else:
        c = np.asarray(count, dtype=float)
        keep = c > 0
        if min_count_fraction > 0 and keep.any():
            # Local median count, over a window wide enough to span a gap
            # without tracking the slow fall-off in solid angle.
            w = max(21, int(0.05 * c.size) | 1)
            half = w // 2
            padded = np.pad(c, half, mode="edge")
            local = np.array([np.median(padded[j:j + w]) for j in range(c.size)])
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(local > 0, c / local, 1.0)
            keep &= frac >= min_count_fraction
    keep &= np.isfinite(q) & np.isfinite(intensity)
    return q[keep], intensity[keep]

def combine_masks(*masks, detector=None, shape=None):
    """OR together any masks that are present (None entries skipped).

    `detector` folds in the detector's own mask -- the module gaps and
    known bad pixels. Know what that choice does before making it,
    because pyFAI's create_mask() reads `if mask is None: mask =
    self.mask`: an explicit mask REPLACES the detector's rather than
    adding to it. So passing any mask without `detector` silently leaves
    the module gaps unmasked, and their zero counts are averaged into
    every bin that straddles one.

    Leaving them unmasked is nonetheless a reasonable default here, and
    it is what this toolkit does. Measured on a Pilatus 2M at npt = 900,
    against the same cut with the gaps masked: peak positions move by at
    most 0.25%, widths are unchanged, and the intensity ratio has median
    1.0000 with a 5th percentile of 1.0000 -- only the handful of bins
    sitting directly on a gap read low, by at most 8%. In exchange the
    2D map stays continuous instead of being ruled with black bands,
    which otherwise have to be interpolated back over for any figure --
    and an interpolator wide enough to close them cannot reliably be
    kept out of the missing wedge, which must stay visible.

    The beamstop is a different matter and is always masked: it is 500x
    attenuation over the out-of-plane sector, not a few percent.

    Returns an int8 array in pyFAI's convention (nonzero = excluded), or
    None when nothing is masked, so callers can pass it straight through.
    """
    out = None
    sources = list(masks)
    if detector is not None:
        sources.append(getattr(detector, "mask", None))
    for m in sources:
        if m is None:
            continue
        a = np.asarray(m)
        if a.size == 0:
            continue
        b = a.astype(bool)
        out = b if out is None else (out | b)
    if out is None or not out.any():
        return None
    return out.astype(np.int8)

def find_beam_col_from_symmetry(image, centre_guess=None, search_half_width: int = 60,
                                 row_band=None, max_lever: int = 400, row_bin: int = 8,
                                 min_contrast: float = 0.05):
    """Locate the beam-centre COLUMN from the frame's own mirror symmetry.

    A fibre-textured film has no preferred in-plane direction, so its
    scattering is symmetric about the plane holding the surface normal and
    the beam -- on the detector, a mirror line through the beam column.
    The column that makes the frame most nearly equal to its own
    reflection therefore IS the beam column, and no calibrant is involved.
    A calibrant frame works too, for a different reason: its rings are
    centro-symmetric. So this measures the half of the geometry a
    calibrant gets RIGHT, and is a cross-check rather than a replacement.

    Returns (col, contrast). `col` is None when the score curve is flat
    (no mirror symmetry in this frame) or when the best column lands on
    the edge of the search window (the true centre is outside it) --
    either way a refusal, never the window edge dressed up as an answer.
    """
    I = np.asarray(image, dtype=float)
    h, w = I.shape
    r0, r1 = row_band if row_band else (int(0.10 * h), int(0.65 * h))
    band = I[r0:r1, :]
    nb = (band.shape[0] // row_bin) * row_bin
    if nb < row_bin:
        return None, 0.0
    band = band[:nb].reshape(-1, row_bin, w).mean(axis=1)
    good = np.isfinite(band) & (band > 0)
    L = np.where(good, np.log10(np.clip(band, 1e-3, None)), np.nan)

    c0 = int(centre_guess) if centre_guess is not None else w // 2
    cands = np.arange(max(1, c0 - search_half_width), min(w - 1, c0 + search_half_width + 1))
    if cands.size < 5:
        return None, 0.0
    scores = np.full(cands.size, np.nan)
    for i, c in enumerate(cands):
        k = int(min(c, w - 1 - c, max_lever))
        if k < 50:
            continue
        left = L[:, c - k:c][:, ::-1]
        right = L[:, c + 1:c + 1 + k]
        m = np.isfinite(left) & np.isfinite(right)
        if m.sum() < 0.3 * left.size:
            continue
        scores[i] = float(np.abs(left[m] - right[m]).mean())
    if np.all(np.isnan(scores)):
        return None, 0.0

    i = int(np.nanargmin(scores))
    lo, hi = np.nanpercentile(scores, [0, 95])
    contrast = float((hi - lo) / hi) if hi > 0 else 0.0
    if contrast < min_contrast or i == 0 or i == cands.size - 1:
        return None, contrast
    off = 0.0
    if 0 < i < cands.size - 1 and np.all(np.isfinite(scores[i - 1:i + 2])):
        y0, y1, y2 = scores[i - 1:i + 2]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            off = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return float(cands[i] + off), contrast


def autodefine_beam_centre(image, incident_angle_deg: float, dist: float,
                            pixel_size: float, centre_guess_col=None):
    """Measure the beam centre and the horizon from one sample frame.

    The two halves come from two different symmetries, which is the point:
    neither leans on a calibrant, and neither can quietly stand in for the
    other if its own evidence is missing.

      column  <- the frame equals its own mirror image about q_xy = 0
      row     <- the horizon sits one incident angle above the direct beam

    Each is reported independently and either may come back None. A result
    with one half measured and the other missing is the honest outcome for
    a frame that only supports one of them, and is more useful than a
    complete answer with an invented half in it.

    Returns a dict: col, col_contrast, row, horizon_row, horizon_agreement,
    horizon_blocks, notes (a list of plain-language strings).
    """
    notes = []
    col, contrast = find_beam_col_from_symmetry(image, centre_guess=centre_guess_col)
    if col is None:
        notes.append(
            f"Beam COLUMN not measurable from this frame (mirror-symmetry "
            f"contrast {contrast:.2f})."
            + (f" The best match also ran to the edge of the +/-60 px search "
               f"window around {int(centre_guess_col)}, so the true column is "
               f"probably outside it." if centre_guess_col is not None else "")
        )
    else:
        notes.append(f"Beam column {col:.2f} px, from mirror symmetry "
                     f"(contrast {contrast:.2f}).")

    horizon, agreement, n_blocks = find_horizon_row(image)
    if horizon is None:
        row = None
        notes.append(
            f"No full-width horizon in this frame (only {agreement:.0%} of "
            f"{n_blocks} column blocks agreed), so the beam ROW could not be "
            f"measured. A frame with no sample in the beam -- a calibrant "
            f"exposure -- has no horizon to find."
        )
    else:
        row = beam_row_from_horizon(horizon, incident_angle_deg, dist, pixel_size)
        notes.append(
            f"Horizon at row {horizon:.1f} ({agreement:.0%} of {n_blocks} "
            f"column blocks agreed), so at an incident angle of "
            f"{incident_angle_deg:g} deg the beam row is {row:.1f} px."
        )
    return {
        "col": col, "col_contrast": contrast,
        "row": row, "horizon_row": horizon,
        "horizon_agreement": agreement, "horizon_blocks": n_blocks,
        "notes": notes,
    }


def add_angle_lines(ax, qip, qoop, angles: Tuple[float, float], color="cyan"):
    """Draw the two integration-sector boundary lines on a q-space plot."""
    a1, a2 = angles
    plot_angles = (a1 + 90, a2 + 90)

    qip_min, qip_max = np.min(qip), np.max(qip)
    qoop_min, qoop_max = np.min(qoop), np.max(qoop)
    if not (qip_min < 0 < qip_max and qoop_min < 0 < qoop_max):
        return

    for angle in plot_angles:
        rad = np.deg2rad(angle)
        grad = np.tan(rad)
        if angle % 180 == 0:
            x_vals = [0, qip_max if np.cos(rad) > 0 else qip_min]
            y_vals = [0, 0]
        elif (angle + 90) % 180 == 0:
            x_vals = [0, 0]
            y_vals = [0, qoop_max if np.sin(rad) > 0 else qoop_min]
        else:
            sign_x = np.sign(np.cos(rad))
            sign_y = np.sign(np.sin(rad))
            if sign_x > 0 and sign_y > 0:
                x = min(qip_max, qoop_max / grad)
            elif sign_x > 0 and sign_y < 0:
                x = min(qip_max, qoop_min / grad)
            elif sign_x < 0 and sign_y > 0:
                x = max(qip_min, qoop_max / grad)
            else:
                x = max(qip_min, qoop_min / grad)
            x_vals = np.array([0, x])
            y_vals = grad * x_vals
        ax.plot(x_vals, y_vals, color=color, linestyle="-", linewidth=1.0, alpha=0.7)


def plot_1d_linecut(q, intensity, out_path=None, angle_range=(0, 0), title=None,
                     line_color: Optional[str] = None,
                     font_family: Optional[str] = None, font_size: Optional[float] = None,
                     dpi: int = DEFAULT_DPI, figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
                     q_range: Tuple[float, float] = (0.15, 2.0), tick_spacing: float = 0.3,
                     subtick_spacing: Optional[float] = None,
                     edge_label_top: Optional[str] = None, edge_label_bottom: Optional[str] = None,
                     edge_label_left: Optional[str] = None, edge_label_right: Optional[str] = None,
                     edge_label_rotations: Optional[Dict[str, float]] = None):
    """1D line-cut plot: log-scaled q-axis, but with major tick MARKS placed
    at round linear-spaced values (0.3, 0.6, 0.9, ...) rather than the
    log-scale default (powers of ten / 1-2-5 pattern) -- matching the
    reference literature figure's tick labelling while still getting a
    log-x view of the data. Ticks will look visually non-uniform (that's
    an inherent property of a log axis), but the labelled values themselves
    are the clean round numbers from tick_spacing.

    subtick_spacing: minor-tick interval (1/A), OFF by default -- a log
    axis's default minor ticks are visually cluttered, so they're
    explicitly suppressed unless you opt in with a specific spacing here.
    """
    from matplotlib.ticker import ScalarFormatter, NullLocator
    with style_context(font_family, font_size):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(q, intensity, linewidth=1.0, color=line_color)
        ax.set_xlabel(r"q (Å$^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if q_range:
            ax.set_xlim(q_range)
        if tick_spacing:
            ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        if subtick_spacing:
            ax.xaxis.set_minor_locator(MultipleLocator(subtick_spacing))
            ax.tick_params(which="minor", length=3)
        else:
            ax.xaxis.set_minor_locator(NullLocator())  # avoid cluttered log-scale minor ticks
        ax.tick_params(axis="both", which="both", direction="in")
        ax.xaxis.set_major_formatter(ScalarFormatter())  # plain "0.3" not "3x10^-1"
        ax.set_title(title or f"Line profile: {angle_range[0]} to {angle_range[1]} deg")
        fig.tight_layout()
        add_edge_labels(fig, top=edge_label_top, bottom=edge_label_bottom,
                         left=edge_label_left, right=edge_label_right, fontsize=font_size,
                         **(edge_label_rotations or {}))
        if out_path:
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)
            return None
        return fig


# --------------------------------------------------------------------------- #
# Pole figure helpers
# --------------------------------------------------------------------------- #
def compute_chi_profile_at_q(fi, img_data, mask, target_q, dq, npt,
                              unit_gi_chi, unit_gi_qtot):
    """Azimuthal (chi) intensity profile in a narrow band around target_q.

    Returns (chi_axis_deg, intensity_profile).
    """
    res2d = fi.integrate2d_grazing_incidence(
        img_data, npt_ip=npt, npt_oop=npt,
        unit_ip=unit_gi_chi, unit_oop=unit_gi_qtot, mask=mask,
    )
    I2d, chi_axis, q_axis = res2d[0:3]
    q_sel = (q_axis >= target_q - dq) & (q_axis <= target_q + dq)
    if not np.any(q_sel):
        raise ValueError(
            f"No integrated data found within q = {target_q} +/- {dq} 1/A. "
            "Check the target q / dq values and your detector's q-space range."
        )
    with np.errstate(invalid="ignore"):
        profile = np.nanmean(I2d[q_sel, :], axis=0)
    return chi_axis, profile


def plot_pole_figure(chi_axis, profile, out_path, target_q, dq, title=None,
                      herman_s=None, cmap: str = "viridis",
                      vmin: Optional[float] = None, vmax: Optional[float] = None,
                      font_family: Optional[str] = None, font_size: Optional[float] = None,
                      show_colorbar: bool = True):
    """Fiber-texture pole figure: chi (tilt from surface normal) is radial,
    phi is angular and assumed uniform (revolved) since a single frame
    cannot resolve azimuthal (phi) texture.

    show_colorbar: draw the intensity bar (default). Off gives a bare
    pole figure, for a layout that labels the scale elsewhere.
    """
    with style_context(font_family, font_size):
        tilt = np.abs(chi_axis)
        order = np.argsort(tilt)
        tilt_sorted = tilt[order]
        profile_sorted = profile[order]

        n_phi = 181
        theta = np.linspace(0, 2 * np.pi, n_phi)
        R, THETA = np.meshgrid(tilt_sorted, theta)
        Z = np.tile(profile_sorted, (n_phi, 1))

        v_lo, v_hi = resolve_vmin_vmax(profile_sorted, 1.0, vmin, vmax)

        fig = plt.figure(figsize=(7, 6.5), dpi=150)
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location("N")
        mesh = ax.pcolormesh(THETA, R, Z, cmap=cmap,
                              norm=LogNorm(vmin=v_lo, vmax=v_hi))
        ax.set_rmax(90)
        ax.set_rticks([0, 30, 60, 90])
        ax.set_rlabel_position(135)

        title_text = title or (
            f"Pole figure (fiber-texture approx.), q = {target_q:.3f} "
            f"+/- {dq:.3f} 1/Å\n(radial = tilt from surface normal, 0deg center "
            f"/ 90deg edge = in-plane;\nphi assumed isotropic -- single-frame "
            f"approximation)"
        )
        if herman_s is not None:
            title_text += f"\nHerman's orientation factor S = {herman_s:.3f}"
        ax.set_title(title_text, fontsize=(font_size * 0.8) if font_size else 9)

        if show_colorbar:
            # fig.colorbar, not plt.* -- see plot_2d_image
            cb = fig.colorbar(mesh, ax=ax, pad=0.12)
            cb.set_label("Intensity (a.u.)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def plot_chi_intensity_profile(chi_axis, profile, out_path=None, target_q=0.0, dq=0.0,
                                title=None, herman_s=None, chi_range=(-90, 90),
                                series_label=None, extra_series=None,
                                line_color: Optional[str] = None,
                                font_family: Optional[str] = None,
                                font_size: Optional[float] = None,
                                dpi: int = DEFAULT_DPI,
                                figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
                                tick_spacing: float = 20.0,
                                edge_label_top: Optional[str] = None, edge_label_bottom: Optional[str] = None,
                                edge_label_left: Optional[str] = None, edge_label_right: Optional[str] = None,
                                edge_label_rotations: Optional[Dict[str, float]] = None):
    """Intensity-vs-chi 'pole figure' in the Cartesian/log-y style commonly
    used in the GIWAXS literature (e.g. for tracking a peak's orientation
    across an annealing series): chi (tilt from the surface normal, signed,
    NOT folded to |chi|) on the x-axis, intensity on a log y-axis.

    extra_series, if given, is a list of (label, chi_axis, profile) tuples
    to overlay on the same axes (e.g. the same reflection across multiple
    temperatures/times/samples) -- matching the multi-series style of that
    reference figure. Each auto-cycles through matplotlib's default color
    cycle unless line_color is given, in which case ALL series share that
    one color (only really sensible for a single series).

    If out_path is given, saves and closes the figure (returns None).
    Otherwise returns the open Figure for interactive display.
    """
    with style_context(font_family, font_size):
        fig, ax = plt.subplots(figsize=figsize)

        def _plot_one(chi, prof, label, color):
            order = np.argsort(chi)
            c = chi[order]
            p = prof[order]
            valid = np.isfinite(p) & (p > 0)
            ax.plot(c[valid], p[valid], marker='.', markersize=2, linestyle='none',
                    alpha=0.6, label=label, color=color)

        _plot_one(chi_axis, profile, series_label or "this frame", line_color)
        if extra_series:
            for label, chi_i, prof_i in extra_series:
                _plot_one(np.asarray(chi_i), np.asarray(prof_i), label, line_color)

        ax.set_yscale("log")
        ax.set_xlim(chi_range)
        if tick_spacing:
            ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        ax.tick_params(axis="both", which="both", direction="in")
        ax.set_xlabel(r"$\chi$ (°)")
        ax.set_ylabel("Intensity (a.u.)")

        title_text = title or f"q = {target_q:.3f} +/- {dq:.3f} 1/Å"
        if herman_s is not None:
            title_text += f"   (Herman's S = {herman_s:.3f})"
        ax.set_title(title_text, fontsize=(font_size * 0.9) if font_size else 10)

        if extra_series or series_label:
            ax.legend(markerscale=3, ncol=2)

        fig.tight_layout()
        add_edge_labels(fig, top=edge_label_top, bottom=edge_label_bottom,
                         left=edge_label_left, right=edge_label_right, fontsize=font_size,
                         **(edge_label_rotations or {}))
        if out_path:
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)
            return None
        return fig


def compute_herman_orientation(chi_axis, profile, chi_max: float = 90.0):
    """Compute Herman's orientation factor S from a chi (tilt) intensity
    profile, i.e. the same profile used to build the fiber-texture pole
    figure for a given reflection.

        S = (3<cos^2(chi)> - 1) / 2

    where chi is measured from the surface normal (chi=0 -> perpendicular
    to substrate, chi=90 -> parallel to substrate) and <cos^2(chi)> is a
    solid-angle-weighted average (weight = I(chi) * sin(chi)), matching the
    same fiber-symmetry (uniform phi) assumption used elsewhere.

    S ranges from -0.5 (crystallites/chains lying flat, parallel to the
    substrate) to +1 (perfectly perpendicular to the substrate); S = 0
    indicates a completely random orientation.

    IMPORTANT LIMITATION: GIWAXS pole figures always have a "missing
    wedge" (e.g. near the beamstop / horizon) where no intensity was
    measured. This function excludes non-finite / out-of-range points from
    the weighted average rather than assuming zero signal there, which
    would otherwise bias S. It also reports the fraction of the 0-chi_max
    range that was actually covered by valid data, so you can judge how
    trustworthy a given S value is -- low coverage should be treated with
    caution (or the missing region modelled/extrapolated separately for a
    fully quantitative result).

    Returns
    -------
    S : float
    mean_cos2_chi : float
    coverage_fraction : float
        Fraction of the [0, chi_max] range spanned by the valid data points
        used (a simple completeness indicator, not a measure of internal gaps).
    """
    tilt = np.abs(chi_axis)
    order = np.argsort(tilt)
    tilt_sorted = tilt[order]
    profile_sorted = profile[order]

    valid = np.isfinite(profile_sorted) & (tilt_sorted <= chi_max) & (profile_sorted >= 0)
    tilt_valid = tilt_sorted[valid]
    profile_valid = profile_sorted[valid]

    if tilt_valid.size < 2:
        raise ValueError(
            "Not enough valid (finite, in-range) data points across the "
            "tilt range to compute Herman's orientation factor."
        )

    weights = profile_valid * np.sin(np.deg2rad(tilt_valid))
    cos2_chi = np.cos(np.deg2rad(tilt_valid)) ** 2

    denominator = _trapz(weights, tilt_valid)
    if denominator <= 0:
        raise ValueError(
            "Total weighted intensity is zero -- cannot compute <cos^2 chi>. "
            "Check that the target q actually contains signal."
        )
    numerator = _trapz(weights * cos2_chi, tilt_valid)

    mean_cos2_chi = numerator / denominator
    S = (3 * mean_cos2_chi - 1) / 2

    coverage_fraction = (
        (tilt_valid.max() - tilt_valid.min()) / chi_max if chi_max > 0 else float("nan")
    )
    return S, mean_cos2_chi, coverage_fraction


def append_herman_summary_row(summary_path: str, row: Dict[str, object]):
    """Append one row to a CSV summary of Herman's orientation factor
    results, creating the file with a header if it doesn't exist yet."""
    import csv
    fieldnames = ["filename", "target_q", "dq", "S", "mean_cos2_chi", "coverage_fraction"]
    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------- #
# Peak fitting (line-cut peak shape -> q0, FWHM, d-spacing, coherence length)
# --------------------------------------------------------------------------- #
_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.3548


def fit_peak(q, intensity, q_min: float, q_max: float,
             shape: str = "gaussian", scherrer_k: float = 0.9,
             label: str = "") -> Dict[str, object]:
    """Fit a single diffraction peak within [q_min, q_max] (1/Angstrom) to
    a peak profile plus a linear background, via nonlinear least squares.

    shape: "gaussian", "lorentzian", or "pseudo_voigt" (a linear blend of
    the two, with the blend fraction eta also fit).

    Returns a dict with:
      q0                 fitted peak centre (1/A)
      fwhm                fitted full-width-at-half-max (1/A)
      d_spacing           2*pi/q0 (Angstrom) -- real-space periodicity
      coherence_length    2*pi*K/FWHM (Angstrom), the Scherrer equation in
                           q-space (K=0.9 by default) -- see e.g. Savikhin
                           & Toney, "Fundamentals of Organic Semiconductor
                           Device Physics" / Handbook of Organic Materials
                           for Electronic and Photonic Devices, for this
                           q-space formulation, or countless GIWAXS papers'
                           SI sections using the identical CL = 2*pi*K/FWHM.
      peak_intensity       fitted peak height ABOVE the local background
      peak_area            integrated area of the background-subtracted peak
      r_squared            goodness of fit over the fit window (1 = perfect)
      eta                  Gaussian/Lorentzian blend fraction (pseudo_voigt only)
      fit_curve_q/I         dense arrays (peak + background) for overlay plotting
      background_curve_I    the background component alone, same q as fit_curve_q
      label, q_range, shape, scherrer_k   echoed back for convenience

    Raises GiwaxsError (not a bare exception) if the window has too few
    points or the fit fails to converge, so this is safe to call from the
    Streamlit app's normal `except Exception` handling without special-casing.
    """
    from scipy.optimize import curve_fit

    if shape not in ("gaussian", "lorentzian", "pseudo_voigt", "voigt"):
        raise GiwaxsError(
            f"Unknown peak shape '{shape}' -- use 'gaussian', 'lorentzian', or 'pseudo_voigt'."
        )
    is_voigt = shape in ("pseudo_voigt", "voigt")

    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    mask = (q >= q_min) & (q <= q_max) & np.isfinite(q) & np.isfinite(intensity)
    q_fit = q[mask]
    I_fit = intensity[mask]
    if len(q_fit) < 6:
        raise GiwaxsError(
            f"Not enough data points in range [{q_min:.4g}, {q_max:.4g}] 1/A "
            f"to fit a peak (found {len(q_fit)}, need at least 6). Check the "
            f"range actually falls within this line cut's q coverage."
        )
    order = np.argsort(q_fit)  # defensive -- should already be sorted
    q_fit, I_fit = q_fit[order], I_fit[order]

    # A perfectly flat window (every intensity identical -- in practice all
    # zeros) carries no peak information, but curve_fit still "converges" on
    # it and returns a meaningless result with R^2 = nan, because the total
    # sum of squares is zero. Reject it explicitly instead, so the caller
    # gets a clear reason rather than a silent nan. The usual cause is a
    # fit window that falls outside this particular sector's detector
    # coverage (e.g. a high-q window against a narrow out-of-plane sector).
    if float(np.ptp(I_fit)) == 0.0:
        raise GiwaxsError(
            f"No intensity variation in range [{q_min:.4g}, {q_max:.4g}] 1/A"
            f"{f' ({label})' if label else ''} -- every point has the same value "
            f"({I_fit[0]:.4g}). This q window most likely falls outside this "
            f"line cut's actual detector coverage, so there is no peak to fit."
        )

    q_span = q_max - q_min
    q_center = float(q_fit.mean())  # background defined relative to this for numerical stability
    edge_n = max(1, len(q_fit) // 8)
    bg_left, bg_right = float(np.median(I_fit[:edge_n])), float(np.median(I_fit[-edge_n:]))
    bg0 = (bg_left + bg_right) / 2
    slope0 = (bg_right - bg_left) / (q_fit[-1] - q_fit[0]) if q_fit[-1] != q_fit[0] else 0.0

    def background(qv, bg, slope):
        return bg + slope * (qv - q_center)

    peak_idx = int(np.argmax(I_fit - background(q_fit, bg0, slope0)))
    q0_0 = float(q_fit[peak_idx])
    amp0 = max(float(I_fit[peak_idx] - background(q0_0, bg0, slope0)), 1e-6)
    fwhm0 = q_span / 3

    def _gaussian(qv, amp, q0, fwhm):
        sigma = abs(fwhm) / _FWHM_TO_SIGMA
        return amp * np.exp(-0.5 * ((qv - q0) / sigma) ** 2)

    def _lorentzian(qv, amp, q0, fwhm):
        gamma = abs(fwhm) / 2
        return amp * gamma ** 2 / ((qv - q0) ** 2 + gamma ** 2)

    if shape == "gaussian":
        peak_fn = _gaussian
    elif shape == "lorentzian":
        peak_fn = _lorentzian
    else:  # pseudo_voigt
        def peak_fn(qv, amp, q0, fwhm, eta):
            return amp * (eta * _lorentzian(qv, 1.0, q0, fwhm)
                           + (1 - eta) * _gaussian(qv, 1.0, q0, fwhm))

    if is_voigt:
        def model(qv, amp, q0, fwhm, eta, bg, slope):
            return peak_fn(qv, amp, q0, fwhm, eta) + background(qv, bg, slope)
        p0 = [amp0, q0_0, fwhm0, 0.5, bg0, slope0]
        bounds = ([0, q_min, 1e-6, 0, -np.inf, -np.inf],
                  [np.inf, q_max, q_span * 3, 1, np.inf, np.inf])
    else:
        def model(qv, amp, q0, fwhm, bg, slope):
            return peak_fn(qv, amp, q0, fwhm) + background(qv, bg, slope)
        p0 = [amp0, q0_0, fwhm0, bg0, slope0]
        bounds = ([0, q_min, 1e-6, -np.inf, -np.inf],
                  [np.inf, q_max, q_span * 3, np.inf, np.inf])

    try:
        popt, _ = curve_fit(model, q_fit, I_fit, p0=p0, bounds=bounds, maxfev=10000)
    except Exception as exc:
        raise GiwaxsError(
            f"Peak fit did not converge for range [{q_min:.4g}, {q_max:.4g}] "
            f"1/A{f' ({label})' if label else ''}: {exc}"
        )

    if is_voigt:
        amp, q0, fwhm, eta, bg, slope = popt
    else:
        amp, q0, fwhm, bg, slope = popt
        eta = None
    fwhm = abs(fwhm)

    I_model = model(q_fit, *popt)
    ss_res = float(np.sum((I_fit - I_model) ** 2))
    ss_tot = float(np.sum((I_fit - I_fit.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    d_spacing = 2 * np.pi / q0 if q0 > 0 else float("nan")
    coherence_length = 2 * np.pi * scherrer_k / fwhm if fwhm > 0 else float("nan")

    q_dense = np.linspace(q_min, q_max, 2000)
    peak_dense = peak_fn(q_dense, amp, q0, fwhm, eta) if is_voigt else peak_fn(q_dense, amp, q0, fwhm)
    peak_area = float(_trapz(peak_dense, q_dense))
    bg_dense = background(q_dense, bg, slope)

    return {
        "label": label, "q_range": (q_min, q_max), "shape": shape,
        "q0": float(q0), "fwhm": float(fwhm), "d_spacing": float(d_spacing),
        "coherence_length": float(coherence_length),
        "peak_intensity": float(amp), "peak_area": peak_area,
        "eta": float(eta) if eta is not None else None,
        "background_level": float(bg), "background_slope": float(slope),
        "r_squared": float(r_squared), "scherrer_k": scherrer_k,
        "fit_curve_q": q_dense, "fit_curve_I": peak_dense + bg_dense,
        "background_curve_I": bg_dense,
    }


def fit_multiple_peaks(q, intensity, regions, shape: str = "gaussian",
                        scherrer_k: float = 0.9) -> List[Dict[str, object]]:
    """Fit each (q_min, q_max, label) tuple in `regions` independently via
    fit_peak(). Returns a list of result dicts (see fit_peak's docstring),
    in the same order as `regions`. A region that fails to fit raises
    GiwaxsError with that region's label/range in the message (from
    fit_peak itself) -- callers doing batch/UI work typically want to
    catch per-region so one bad region doesn't lose the others; this
    function itself does NOT catch, so it fails fast for simple/scripted use.
    """
    return [
        fit_peak(q, intensity, q_min, q_max, shape=shape, scherrer_k=scherrer_k, label=label)
        for (q_min, q_max, label) in regions
    ]


PEAK_FIT_CSV_FIELDS = [
    "filename", "sector", "label", "q_min", "q_max", "shape",
    "q0_invA", "d_spacing_A", "fwhm_invA", "coherence_length_A", "scherrer_k",
    "peak_intensity", "peak_area", "eta", "r_squared", "error",
]


def peak_fit_csv_row(fit: Dict[str, object], filename: str, sector: str) -> Dict[str, object]:
    """Flatten one fit_peak() result dict into a CSV-writable row matching
    PEAK_FIT_CSV_FIELDS. The bulky overlay-plot arrays (fit_curve_q etc.)
    are dropped; only scalar results are kept."""
    q_min, q_max = fit["q_range"]
    return {
        "filename": filename, "sector": sector, "label": fit.get("label", ""),
        "q_min": q_min, "q_max": q_max, "shape": fit.get("shape", ""),
        "q0_invA": fit["q0"], "d_spacing_A": fit["d_spacing"],
        "fwhm_invA": fit["fwhm"], "coherence_length_A": fit["coherence_length"],
        "scherrer_k": fit.get("scherrer_k", ""),
        "peak_intensity": fit["peak_intensity"], "peak_area": fit["peak_area"],
        "eta": fit.get("eta"), "r_squared": fit["r_squared"], "error": "",
    }


def peak_fit_csv_error_row(filename: str, sector: str, label: str,
                            q_min: float, q_max: float, shape: str,
                            message: str) -> Dict[str, object]:
    """A row recording that one (file, sector, region) fit FAILED, so a
    failed region is visible in the CSV rather than silently absent."""
    row = {k: "" for k in PEAK_FIT_CSV_FIELDS}
    row.update({"filename": filename, "sector": sector, "label": label,
                "q_min": q_min, "q_max": q_max, "shape": shape, "error": message})
    return row


def write_peak_fit_csv(csv_path: str, rows: List[Dict[str, object]]):
    """Write all peak-fit result rows to one combined CSV (overwriting any
    previous run's file, unlike the Herman summary's append-mode helper --
    a fit run is reproducible from its inputs, so appending would just
    accumulate stale duplicates across re-runs)."""
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PEAK_FIT_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_linecut_with_fits(q, intensity, fits: List[Dict[str, object]], out_path=None,
                            title: Optional[str] = None, line_color: Optional[str] = None,
                            font_family: Optional[str] = None, font_size: Optional[float] = None,
                            dpi: int = DEFAULT_DPI, figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
                            q_range: Optional[Tuple[float, float]] = None,
                            tick_spacing: Optional[float] = 0.3,
                            subtick_spacing: Optional[float] = None,
                            edge_label_top: Optional[str] = None, edge_label_bottom: Optional[str] = None,
                            edge_label_left: Optional[str] = None, edge_label_right: Optional[str] = None,
                            edge_label_rotations: Optional[Dict[str, float]] = None):
    """Plot a 1D line-cut curve with one or more fitted peaks (each a dict
    from fit_peak/fit_multiple_peaks) overlaid: the fitted model curve
    drawn on top of the raw data, the fit window lightly shaded, and the
    fitted peak centre marked with a vertical line -- so fit quality can
    be judged visually at a glance. `fits` may be an empty list (just
    shows the raw data, e.g. before any fitting has been done yet).
    subtick_spacing: minor-tick interval (1/A), off by default.
    """
    with style_context(font_family, font_size):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(q, intensity, linewidth=1.0, color=line_color or "#1f77b4",
                label="data", zorder=2, alpha=0.85)

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
            "color", ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
        )
        for i, fit in enumerate(fits):
            c = color_cycle[i % len(color_cycle)]
            q_min, q_max = fit["q_range"]
            label = fit.get("label") or f"peak {i + 1}"
            ax.axvspan(q_min, q_max, alpha=0.08, color=c, zorder=0)
            ax.plot(fit["fit_curve_q"], fit["fit_curve_I"], color=c, linewidth=1.6,
                     linestyle="--", zorder=3, label=f"{label} fit")
            ax.axvline(fit["q0"], color=c, linewidth=0.7, linestyle=":", alpha=0.7, zorder=1)

        ax.set_xlabel(r"q (Å$^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_yscale("log")
        if q_range:
            ax.set_xlim(q_range)
        if tick_spacing:
            ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        if subtick_spacing:
            ax.xaxis.set_minor_locator(MultipleLocator(subtick_spacing))
            ax.tick_params(which="minor", length=3)
        ax.tick_params(axis="both", which="both", direction="in")
        if fits:
            ax.legend(fontsize=8, loc="best", framealpha=0.85)
        if title:
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        add_edge_labels(fig, top=edge_label_top, bottom=edge_label_bottom,
                         left=edge_label_left, right=edge_label_right, fontsize=font_size,
                         **(edge_label_rotations or {}))
        if out_path:
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)
            return None
        return fig
