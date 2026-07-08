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
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

# NumPy >= 2.0 renamed trapz -> trapezoid (and removed the old name in some
# builds); NumPy < 2.0 only has trapz. This shim works either way.
_trapz = getattr(np, "trapezoid", None) or np.trapz


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
        sys.exit(
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
        sys.exit(f"Input path not found: {input_path}")

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
            sys.exit(f"No .tif/.tiff files found in directory: {input_path}")
        return tiff_files
    elif os.path.isfile(input_path):
        return [input_path]
    else:
        sys.exit(f"Input path not found: {input_path}")


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
        sys.exit(f"Pole-figure q-map file not found: {path}")
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
                sys.exit(f"Could not parse q values '{qvals_raw}' for '{fname}' in {path}")
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
            sys.exit(
                "Reading an Excel (.xlsx/.xls) q-map requires the 'pandas' "
                "and 'openpyxl' packages. Install with:\n"
                "    pip install pandas openpyxl\n"
                "...or save your mapping as .json or .csv instead."
            )
        df = pd.read_excel(path, dtype=str)
        _parse_rows(df.to_dict("records"))
    else:
        sys.exit(f"Unsupported q-map file format: '{ext}' -- use .json, .csv, or .xlsx")

    if not result:
        sys.exit(f"No usable filename/q-value rows found in {path}. Expected "
                  f"a 'filename' column/key and a 'q_values' (or 'q') column/key.")

    return result


def parse_incident_angle_from_filename(filename: str) -> Optional[float]:
    """Try to extract an incident angle from a filename using the common
    beamline convention of encoding a decimal value with 'p' in place of
    the decimal point, as its own underscore-separated token -- e.g.
    'sample_0p095_1234.tif' -> 0.095, 'sample_0p1_scan.tif' -> 0.1.

    Returns None if no such token is found (caller should fall back to a
    manually-specified default in that case, since this is a best-effort
    heuristic that can occasionally false-match an unrelated numeric token
    if a filename happens to contain another '<digits>p<digits>' pattern).
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    matches = re.findall(r'(?:^|_)(\d+p\d+)(?:_|$)', base)
    for m in matches:
        try:
            return float(m.replace('p', '.', 1))
        except ValueError:
            continue
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
        sys.exit(f"Incident-angle-map file not found: {path}")
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
                sys.exit(f"Could not parse incident angle '{angle_raw}' for '{fname}' in {path}")

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
            sys.exit(
                "Reading an Excel (.xlsx/.xls) angle-map requires the "
                "'pandas' and 'openpyxl' packages. Install with:\n"
                "    pip install pandas openpyxl\n"
                "...or save your mapping as .json or .csv instead."
            )
        df = pd.read_excel(path, dtype=str)
        _parse_rows(df.to_dict("records"))
    else:
        sys.exit(f"Unsupported angle-map file format: '{ext}' -- use .json, .csv, or .xlsx")

    if not result:
        sys.exit(f"No usable filename/angle rows found in {path}. Expected "
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
            print(f"  Could not parse an incident angle from the filename "
                  f"'{os.path.basename(tiff_path)}' -- using fallback "
                  f"--incident-angle={fallback_deg} for this file.")
        return fallback_deg
    if verbose:
        print(f"  Incident angle from filename: {parsed} deg")
    return parsed


# --------------------------------------------------------------------------- #
# Font styling presets
# --------------------------------------------------------------------------- #
#: Font families grouped by category, for a friendlier picker than one long list.
FONT_CATEGORIES = {
    "Sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "Serif": ["Times New Roman", "serif"],
    "Monospace": ["monospace"],
}

#: Font size presets (points), for a friendlier picker than a raw slider.
#: "Custom" (mapped to None here) signals the caller to show a free-entry
#: number input instead of using a preset value.
FONT_SIZE_PRESETS = {
    "Small (8pt)": 8.0,
    "Normal (11pt)": 11.0,
    "Large (14pt)": 14.0,
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
        sys.exit(f"PONI file not found: {path}")
    from pyFAI.io.ponifile import PoniFile
    try:
        poni = PoniFile(data=path)
    except Exception as exc:
        sys.exit(f"Could not read PONI file '{path}': {exc}")
    if poni.detector is None:
        sys.exit(f"PONI file '{path}' does not specify a detector.")
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
        sys.exit("You must provide either --wavelength (metres) or --energy "
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
        sys.exit("You must provide --detector-shape 'height,width' when "
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


def run_agbeh_calibration(calib_path: str, detector, wavelength: float,
                           dist_guess: float, poni1_guess: float, poni2_guess: float,
                           rot1_guess: float, rot2_guess: float, rot3_guess: float,
                           calibrant_name: str, max_rings: int, imin: float,
                           fabio_mod, diagnostic_path: Optional[str] = None) -> Dict[str, float]:
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
        sys.exit(f"Calibration image not found: {calib_path}")

    calib_img = fabio_mod.open(calib_path).data
    calibrant = get_calibrant(calibrant_name, wavelength=wavelength)

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
        sys.exit(
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
    }


def save_refined_poni(path: str, dist: float, poni1: float, poni2: float,
                       rot1: float, rot2: float, rot3: float,
                       wavelength: float, detector):
    if os.path.isdir(path):
        sys.exit(
            f"--save-calibrated-poni must be a FILE path, not a directory: "
            f"'{path}'. Did you mean a file inside it, e.g. "
            f"'{os.path.join(path, 'calibrated.poni')}'?"
        )
    if not path.lower().endswith(".poni"):
        path = path + ".poni"

    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        sys.exit(f"Cannot save calibrated .poni file: directory does not "
                  f"exist: '{parent}'")

    from pyFAI.geometry import Geometry
    geom = Geometry(dist=dist, poni1=poni1, poni2=poni2,
                     rot1=rot1, rot2=rot2, rot3=rot3,
                     wavelength=wavelength, detector=detector)
    if os.path.exists(path):
        os.remove(path)
    geom.save(path)


def build_fiber_integrator(args, Detector, detector_factory, FiberIntegrator, fabio=None):
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
            sys.exit(
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
        sys.exit(f"Mask shape {mask.shape} does not match image shape {shape}.")
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
        rc["font.family"] = font_family
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
COMMON_FONTS = [
    "DejaVu Sans", "Arial", "Helvetica", "sans-serif",
    "Times New Roman", "serif", "monospace",
]


def resolve_vmin_vmax(intensity: np.ndarray, vmin_percentile: float,
                       vmin: Optional[float] = None, vmax: Optional[float] = None,
                       vmax_percentile: float = 99.9):
    """Resolve the LogNorm vmin/vmax: explicit values win if given,
    otherwise fall back to a percentile-of-nonzero-pixels heuristic for
    BOTH ends (not just the min) -- using the raw max as vmax tends to
    wash out contrast when a few hot/saturated pixels are present.
    """
    if vmin is not None and vmax is not None:
        return max(vmin, 1e-12), vmax
    nonzero = intensity[np.isfinite(intensity) & (intensity > 0)]
    if nonzero.size == 0:
        auto_vmin, auto_vmax = 1e-6, 1.0
    else:
        auto_vmin = max(np.percentile(nonzero, vmin_percentile), 1e-12)
        auto_vmax = np.percentile(nonzero, vmax_percentile)
        if auto_vmax <= auto_vmin:
            auto_vmax = auto_vmin * 10
    return (vmin if vmin is not None else auto_vmin,
            vmax if vmax is not None else auto_vmax)


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


def plot_2d_image(qx, qy, intensity, out_path=None, qlim_x=None, qlim_y=None,
                   vmin_percentile: float = 1.0, vmax_percentile: float = 99.9,
                   cmap: str = "viridis",
                   vmin: Optional[float] = None, vmax: Optional[float] = None,
                   font_family: Optional[str] = None, font_size: Optional[float] = None,
                   dpi: int = DEFAULT_DPI, figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
                   axis_label_style: str = "xyz", tick_spacing: float = 0.5):
    """2D GIWAXS q-space image -- deliberately has NO title (kept plain for
    publication/figure use); use an external caption/label if you need one.

    If out_path is given, saves the figure there and returns None (closes
    the figure). If out_path is None, returns the (open) Figure instead --
    useful for interactive display (e.g. Streamlit's st.pyplot(fig)) without
    an extra disk round-trip. Caller is responsible for plt.close(fig) in
    that case.
    """
    with style_context(font_family, font_size):
        fig, ax = plt.subplots(1, 2, width_ratios=[1, 0.05], figsize=figsize)

        v_lo, v_hi = resolve_vmin_vmax(intensity, vmin_percentile, vmin, vmax, vmax_percentile)
        norm = LogNorm(vmin=v_lo, vmax=v_hi)
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
        plt.colorbar(mesh, cax=ax[1], orientation="vertical")
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)
            return None
        return fig


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
                     q_range: Tuple[float, float] = (0.15, 2.0), tick_spacing: float = 0.3):
    """1D line-cut plot: log-scaled q-axis, but with major tick MARKS placed
    at round linear-spaced values (0.3, 0.6, 0.9, ...) rather than the
    log-scale default (powers of ten / 1-2-5 pattern) -- matching the
    reference literature figure's tick labelling while still getting a
    log-x view of the data. Ticks will look visually non-uniform (that's
    an inherent property of a log axis), but the labelled values themselves
    are the clean round numbers from tick_spacing.
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
            ax.xaxis.set_minor_locator(NullLocator())  # avoid cluttered log-scale minor ticks
        ax.xaxis.set_major_formatter(ScalarFormatter())  # plain "0.3" not "3x10^-1"
        ax.set_title(title or f"Line profile: {angle_range[0]} to {angle_range[1]} deg")
        fig.tight_layout()
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
                      font_family: Optional[str] = None, font_size: Optional[float] = None):
    """Fiber-texture pole figure: chi (tilt from surface normal) is radial,
    phi is angular and assumed uniform (revolved) since a single frame
    cannot resolve azimuthal (phi) texture.
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

        cb = plt.colorbar(mesh, ax=ax, pad=0.12)
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
                                tick_spacing: float = 20.0):
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
        ax.set_xlabel(r"$\chi$ (°)")
        ax.set_ylabel("Intensity (a.u.)")

        title_text = title or f"q = {target_q:.3f} +/- {dq:.3f} 1/Å"
        if herman_s is not None:
            title_text += f"   (Herman's S = {herman_s:.3f})"
        ax.set_title(title_text, fontsize=(font_size * 0.9) if font_size else 10)

        if extra_series or series_label:
            ax.legend(markerscale=3, ncol=2)

        fig.tight_layout()
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
