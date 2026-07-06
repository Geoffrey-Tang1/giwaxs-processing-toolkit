#!/usr/bin/env python3
"""
giwaxs_platform.py
====================

A small orchestration "platform" that ties together the two standalone
GIWAXS agents:

  - giwaxs_2d1d_agent.py       (2D q-space image + 1D line-cut profiles)
  - giwaxs_polefigure_agent.py (fiber-texture pole figures)

so you only have to enter the shared calibration/geometry information
(beam centre, sample-detector distance, wavelength, detector, incident
angle, mask, input/output paths, ...) ONCE, then choose which processing
step(s) to run on the same dataset. Both agents write into a shared
destination directory.

Two ways to use it
------------------
1. Interactive wizard (default): just run the script with no arguments
   and answer the prompts. At the end you can optionally save your
   answers to a JSON config file for instant reuse next time.

       python giwaxs_platform.py

2. Config file (fully scripted / repeatable / batchable):

       python giwaxs_platform.py --config my_sample.json

   A config file can be produced by option 1 ("save this configuration?"),
   or written by hand -- see `example_config()` below / --write-example.

Requirements
------------
    pip install pyFAI fabio numpy matplotlib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import giwaxs_2d1d_agent
import giwaxs_polefigure_agent
import giwaxs_common as gc


CONFIG_VERSION = 1


# --------------------------------------------------------------------------- #
# Small interactive-prompt helpers
# --------------------------------------------------------------------------- #
def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    resp = input(prompt + suffix).strip().lower()
    if not resp:
        return default
    return resp in ("y", "yes")


def ask_float(prompt: str, default: Optional[float] = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        text = input(f"{prompt}{suffix}: ").strip()
        if not text and default is not None:
            return default
        try:
            return float(text)
        except ValueError:
            print("  Please enter a number.")


def ask_int(prompt: str, default: Optional[int] = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        text = input(f"{prompt}{suffix}: ").strip()
        if not text and default is not None:
            return default
        try:
            return int(text)
        except ValueError:
            print("  Please enter an integer.")


def ask_choice(prompt: str, options: List[str], default_index: int = 0) -> int:
    """Show a numbered menu and return the (0-indexed) chosen option."""
    print(prompt)
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    while True:
        text = input(f"Choice [{default_index + 1}]: ").strip()
        if not text:
            return default_index
        try:
            choice = int(text)
            if 1 <= choice <= len(options):
                return choice - 1
        except ValueError:
            pass
        print(f"  Please enter a number from 1 to {len(options)}.")


def ask_str(prompt: str, default: Optional[str] = None, allow_empty: bool = False) -> Optional[str]:
    suffix = f" [{default}]" if default is not None else ""
    text = gc.strip_path_input(input(f"{prompt}{suffix}: "))
    if not text:
        if default is not None:
            return default
        if allow_empty:
            return None
    return text


def ask_float_list(prompt: str, default: Optional[List[float]] = None) -> List[float]:
    suffix = f" [{', '.join(str(v) for v in default)}]" if default else ""
    while True:
        text = input(f"{prompt}{suffix}: ").strip()
        if not text:
            return list(default) if default else []
        try:
            return [float(v) for v in text.replace(",", " ").split()]
        except ValueError:
            print("  Could not parse -- enter numbers separated by spaces or commas.")


def build_per_file_map_interactively(tiff_files: List[str], value_label: str,
                                      ask_fn, out_path: str) -> str:
    """List each file's basename and prompt for a per-file value (via
    ask_fn(prompt, default) -> value, so Enter repeats the previous file's
    value -- convenient when most files share the same setting), then
    write the resulting {filename: value} mapping to out_path (JSON) and
    return that path for use as e.g. --pole-figure-q-map / --incident-angle-map.
    """
    print(f"\nFound {len(tiff_files)} file(s) in the input folder. Enter a "
          f"value for each (press Enter to repeat the previous file's value):")
    mapping: Dict[str, Any] = {}
    last_value = None
    for f in tiff_files:
        fname = os.path.basename(f)
        value = ask_fn(f"  {value_label} for '{fname}'", last_value)
        mapping[fname] = value
        last_value = value
    with open(out_path, "w") as fh:
        json.dump(mapping, fh, indent=2)
    print(f"\nSaved per-file mapping to: {os.path.abspath(out_path)}\n")
    return out_path


def ask_range_list(prompt: str) -> List[Tuple[float, float]]:
    """Ask for zero or more 'a,b' angle-range pairs, one per line, blank to stop."""
    print(prompt)
    ranges: List[Tuple[float, float]] = []
    while True:
        text = input("  Enter 'start,end' in degrees (blank to stop): ").strip()
        if not text:
            break
        try:
            a, b = [float(v) for v in text.split(",")]
        except ValueError:
            print("  Could not parse -- expected two comma-separated numbers.")
            continue
        ranges.append((a, b))
    return ranges





# --------------------------------------------------------------------------- #
# Interactive wizard
# --------------------------------------------------------------------------- #
def run_wizard() -> Dict[str, Any]:
    print("=" * 70)
    print("GIWAXS Processing Platform -- setup wizard")
    print("=" * 70)

    cfg: Dict[str, Any] = {"version": CONFIG_VERSION}

    # --- Input / output ------------------------------------------------------
    cfg["input"] = gc.prompt_for_input_path()
    cfg["output_dir"] = gc.prompt_for_output_dir()
    # Resolved once here and reused below wherever a per-file list is needed
    # (incident angle, pole-figure q values) -- avoids re-scanning the
    # directory each time and lets us show real filenames in those prompts.
    input_files = gc.resolve_tiff_files(cfg["input"])

    # --- Beam centre / geometry ----------------------------------------------
    print("\n--- Geometry ---")
    if ask_yes_no(
        "Do you have an existing .poni calibration file? (Recommended if "
        "you have one -- loads the ENTIRE geometry, including the "
        "detector's shape and pixel size, so you don't need to know or "
        "re-enter those separately.)", default=False
    ):
        while True:
            poni_path = ask_str("Path to the .poni file")
            if poni_path and os.path.exists(poni_path):
                break
            print(f"  File not found: {poni_path}")
        cfg["poni_file"] = poni_path
        # These are all loaded from the .poni file; leave unset here.
        cfg["beam_center_y"] = None
        cfg["beam_center_x"] = None
        cfg["distance"] = None
        cfg["wavelength"] = None
        cfg["energy"] = None
        cfg["rot1"] = cfg["rot2"] = cfg["rot3"] = 0.0
        cfg["detector_name"] = None
        cfg["pixel_size"] = None
        cfg["detector_shape"] = None
    else:
        cfg["poni_file"] = None

    print("\n--- Calibration ---")
    if ask_yes_no(
        "Do you have an AgBeh (or other calibrant) image to refine the beam "
        "centre / sample-detector distance? (Recommended if you don't "
        "already have a .poni file above, or want to refine it further.)",
        default=False
    ):
        while True:
            path = ask_str("Path to the calibration image (e.g. AgBeh .tif)")
            if path and os.path.exists(path):
                break
            print(f"  File not found: {path}")
        cfg["agbeh_file"] = path
        cfg["calibrant"] = "AgBh"  # not prompted -- AgBeh is by far the most common; edit the saved config if you need a different calibrant
        cfg["calib_max_rings"] = ask_int("Maximum number of rings to fit", default=5)
        cfg["calib_min_intensity"] = 200.0  # not prompted -- sensible default for most detectors/exposures
        if ask_yes_no("Save the refined geometry as a .poni file for reuse?", default=False):
            cfg["save_calibrated_poni"] = ask_str(
                "Path to save the refined .poni file", default="calibrated.poni"
            )
        else:
            cfg["save_calibrated_poni"] = None
        if not cfg["poni_file"]:
            print(
                "\n  Since the beam centre / distance below will be refined "
                "automatically, approximate values are fine."
            )
    else:
        cfg["agbeh_file"] = None
        cfg["calibrant"] = "AgBh"
        cfg["calib_max_rings"] = 5
        cfg["calib_min_intensity"] = 200.0
        cfg["save_calibrated_poni"] = None

    if not cfg["poni_file"]:
        guess_note = " (initial guess -- will be refined)" if cfg["agbeh_file"] else ""
        cfg["beam_center_y"] = ask_float(f"Beam centre Y (pixels){guess_note}")
        cfg["beam_center_x"] = ask_float(f"Beam centre X (pixels){guess_note}")
        cfg["distance"] = ask_float(f"Sample-to-detector distance (metres){guess_note}")

        if ask_yes_no("Specify wavelength directly (metres)? (No = specify energy in keV instead)", default=True):
            cfg["wavelength"] = ask_float("Wavelength (metres, e.g. 1.5406e-10 for Cu-K-alpha)")
            cfg["energy"] = None
        else:
            cfg["wavelength"] = None
            cfg["energy"] = ask_float("Photon energy (keV)")

        cfg["rot1"] = ask_float("Detector rotation rot1 (radians)", default=0.0)
        cfg["rot2"] = ask_float("Detector rotation rot2 (radians)", default=0.0)
        cfg["rot3"] = ask_float("Detector rotation rot3 (radians)", default=0.0)

        if ask_yes_no("Use a named pyFAI detector (e.g. 'Pilatus1M')?", default=False):
            cfg["detector_name"] = ask_str("Detector name")
            cfg["pixel_size"] = None
            cfg["detector_shape"] = None
        else:
            cfg["detector_name"] = None
            cfg["pixel_size"] = ask_float("Pixel size (metres)", default=172e-6)
            h = ask_int("Detector shape - height (pixels)")
            w = ask_int("Detector shape - width (pixels)")
            cfg["detector_shape"] = [h, w]
    elif cfg["agbeh_file"]:
        # Have a .poni AND want a fresh AgBeh refinement -- still need to
        # know which detector/wavelength to hand to the refinement, but
        # since --poni-file supplies that entirely, nothing more to ask.
        pass

    print("\n--- Incident angle ---")
    angle_choice = ask_choice(
        "How should the incident angle be determined?",
        [
            "Same fixed value for every file",
            "Auto-detect from filename (the '0p095' convention, e.g. 'sample_0p095_1234.tif' -> 0.095 deg)",
            "Type a different value for each file",
            "Provide a path to a JSON/CSV/Excel per-file mapping",
        ],
    )
    cfg["incident_angle_from_filename"] = False
    cfg["incident_angle_map"] = None
    if angle_choice == 0:
        cfg["incident_angle"] = ask_float("Angle of incidence (degrees)", default=0.1)
    elif angle_choice == 1:
        cfg["incident_angle_from_filename"] = True
        cfg["incident_angle"] = ask_float(
            "Fallback angle (degrees) for any file where no pattern is found "
            "in its name", default=0.1
        )
    elif angle_choice == 2:
        cfg["incident_angle"] = 0.1
        cfg["incident_angle_map"] = build_per_file_map_interactively(
            input_files, "Incident angle (degrees)",
            lambda p, d: ask_float(p, default=d if d is not None else 0.1),
            os.path.join(cfg["output_dir"], "incident_angle_map.json"),
        )
    else:
        while True:
            path = ask_str("Path to the incident-angle map file (.json/.csv/.xlsx)")
            if path and os.path.exists(path):
                break
            print(f"  File not found: {path}")
        cfg["incident_angle_map"] = path
        cfg["incident_angle"] = ask_float(
            "Fallback angle (degrees) for any file NOT listed in the map", default=0.1
        )

    if ask_yes_no("Use a mask file?", default=False):
        cfg["mask"] = ask_str("Path to mask file")
    else:
        cfg["mask"] = None

    with_advanced = ask_yes_no(
        "Configure advanced options (integration resolution)? "
        "(The default is almost always fine -- most people can skip this.)",
        default=False
    )
    cfg["npt"] = ask_int("Number of integration bins (npt)", default=1000) if with_advanced else 1000

    # --- Which steps to run ---------------------------------------------------
    print("\n--- Processing steps ---")
    cfg["run_2d1d"] = ask_yes_no("Run the 2D image + 1D line-cut agent?", default=True)
    cfg["run_pole_figure"] = ask_yes_no("Run the pole figure agent?", default=True)

    if cfg["run_2d1d"]:
        print("\n--- 2D/1D agent options ---")
        cfg["qip_plot_range"] = list(ask_float_list(
            "2D image q_ip plot range 'min max' in 1/A (blank = -0.5 2.4)"
        ) or [-0.5, 2.4])
        cfg["qoop_plot_range"] = list(ask_float_list(
            "2D image q_oop plot range 'min max' in 1/A (blank = -0.25 2.75)"
        ) or [-0.25, 2.75])
        cfg["vmin_percentile"] = ask_float("Colour-scale minimum percentile", default=2.0)
        cfg["vmax_percentile"] = ask_float("Colour-scale maximum percentile", default=99.9)
        print(
            "\nDefault line-cut sectors are always processed:\n"
            "  * In-plane      : (-90, -80) deg\n"
            "  * Out-of-plane  : ( -8,   8) deg"
        )
        cfg["extra_ranges"] = ask_range_list(
            "You can add extra angular sectors now (they'll run without "
            "further prompting):"
        )
    else:
        cfg["qip_plot_range"] = [-0.5, 2.4]
        cfg["qoop_plot_range"] = [-0.25, 2.75]
        cfg["vmin_percentile"] = 2.0
        cfg["vmax_percentile"] = 99.9
        cfg["extra_ranges"] = []

    if cfg["run_pole_figure"]:
        print("\n--- Pole figure agent options ---")
        q_choice = ask_choice(
            "How should target q value(s) be determined for the pole figure(s)?",
            [
                "Same shared q value(s) for every file",
                "Type different q value(s) for each file",
                "Provide a path to a JSON/CSV/Excel per-file mapping",
            ],
        )
        cfg["pole_figure_q_map"] = None
        if q_choice == 0:
            cfg["pole_figure_q"] = ask_float_list(
                "Target q value(s) for pole figure(s), in 1/A (space/comma separated)"
            )
            while not cfg["pole_figure_q"]:
                print("  At least one q value is required for the pole figure step.")
                cfg["pole_figure_q"] = ask_float_list("Target q value(s), in 1/A")
        elif q_choice == 1:
            cfg["pole_figure_q"] = []
            cfg["pole_figure_q_map"] = build_per_file_map_interactively(
                input_files, "Target q value(s), 1/A (space/comma separated, "
                             "blank to skip this file)",
                lambda p, d: ask_float_list(p, default=d),
                os.path.join(cfg["output_dir"], "pole_figure_q_map.json"),
            )
        else:
            while True:
                path = ask_str("Path to the per-file q-map file (.json/.csv/.xlsx)")
                if path and os.path.exists(path):
                    break
                print(f"  File not found: {path}")
            cfg["pole_figure_q_map"] = path
            cfg["pole_figure_q"] = ask_float_list(
                "Fallback q value(s) for any file NOT listed in the map "
                "(blank = skip files not in the map)"
            )
        cfg["pole_figure_dq"] = ask_float("Q window half-width (dq, 1/A)", default=0.05)
        cfg["compute_herman"] = ask_yes_no(
            "Also compute Herman's orientation factor S = (3<cos^2 chi> - 1) / 2 "
            "for each pole figure?", default=True
        )
        if cfg["compute_herman"]:
            cfg["herman_chi_max"] = ask_float(
                "Maximum tilt angle chi (deg from surface normal) to integrate "
                "over for S (standard convention: 90)", default=90.0
            )
        else:
            cfg["herman_chi_max"] = 90.0
    else:
        cfg["pole_figure_q"] = []
        cfg["pole_figure_q_map"] = None
        cfg["pole_figure_dq"] = 0.05
        cfg["compute_herman"] = True
        cfg["herman_chi_max"] = 90.0

    # --- Plot styling (shared by both agents) ---------------------------------
    if cfg["run_2d1d"] or cfg["run_pole_figure"]:
        print("\n--- Plot styling ---")
        if ask_yes_no("Customize plot styling (colormap, colour-scale range, "
                      "line colour, fonts)?", default=False):
            cfg["cmap"] = ask_str(
                f"Colormap (e.g. {', '.join(gc.COMMON_COLORMAPS[:6])}, ...)",
                default="viridis"
            )
            if ask_yes_no("Set an explicit colour-scale min/max (instead of "
                          "automatic percentile-based)?", default=False):
                cfg["vmin"] = ask_float("Colour-scale minimum (intensity units)")
                cfg["vmax"] = ask_float("Colour-scale maximum (intensity units)")
            else:
                cfg["vmin"] = None
                cfg["vmax"] = None
            cfg["line_color"] = ask_str(
                "Line/marker colour for 1D plots (name like 'red' or hex "
                "like '#1f77b4', blank = matplotlib default)",
                default="", allow_empty=True
            ) or None
            cfg["sector_line_color"] = ask_str(
                "Sector boundary line colour on the 2D image", default="cyan"
            )
            cfg["font_family"] = ask_str(
                f"Font family (e.g. {', '.join(gc.COMMON_FONTS[:4])}, blank = default)",
                default="", allow_empty=True
            ) or None
            font_size_text = ask_str("Base font size in points (blank = default)",
                                      default="", allow_empty=True)
            cfg["font_size"] = float(font_size_text) if font_size_text else None
        else:
            cfg["cmap"] = "viridis"
            cfg["vmin"] = None
            cfg["vmax"] = None
            cfg["line_color"] = None
            cfg["sector_line_color"] = "cyan"
            cfg["font_family"] = None
            cfg["font_size"] = None
    else:
        cfg["cmap"] = "viridis"
        cfg["vmin"] = None
        cfg["vmax"] = None
        cfg["line_color"] = None
        cfg["sector_line_color"] = "cyan"
        cfg["font_family"] = None
        cfg["font_size"] = None

    # --- Save config for reuse -------------------------------------------------
    if ask_yes_no("\nSave this configuration to a file for reuse next time?", default=True):
        path = ask_str("Path to save config JSON", default="giwaxs_config.json")
        save_config(cfg, path)
        print(f"  Saved config to: {os.path.abspath(path)}")
        print(f"  Reuse it with:  python {os.path.basename(__file__)} --config {path}")

    return cfg


# --------------------------------------------------------------------------- #
# Config load/save
# --------------------------------------------------------------------------- #
def save_config(cfg: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        sys.exit(f"Config file not found: {path}")
    with open(path) as f:
        cfg = json.load(f)
    required = ["input", "output_dir", "beam_center_y", "beam_center_x", "distance", "incident_angle"]
    missing = [k for k in required if k not in cfg]
    if missing:
        sys.exit(f"Config file is missing required key(s): {missing}")
    return cfg


def example_config() -> Dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "input": "/path/to/tiff_folder",
        "output_dir": "/path/to/output",
        "poni_file": None,
        "agbeh_file": None,
        "calibrant": "AgBh",
        "calib_max_rings": 5,
        "calib_min_intensity": 200.0,
        "save_calibrated_poni": None,
        "beam_center_y": 145,
        "beam_center_x": 1088,
        "distance": 0.65,
        "wavelength": 1.5406e-10,
        "energy": None,
        "rot1": 0.0, "rot2": 0.0, "rot3": 0.0,
        "detector_name": None,
        "pixel_size": 172e-6,
        "detector_shape": [1043, 981],
        "incident_angle": 0.1,
        "incident_angle_from_filename": False,
        "incident_angle_map": None,
        "mask": None,
        "npt": 1000,
        "run_2d1d": True,
        "run_pole_figure": True,
        "qip_plot_range": [-0.5, 2.4],
        "qoop_plot_range": [-0.25, 2.75],
        "vmin_percentile": 2.0,
        "vmax_percentile": 99.9,
        "extra_ranges": [[-55, -45]],
        "pole_figure_q": [1.70, 0.30],
        "pole_figure_q_map": None,
        "pole_figure_dq": 0.05,
        "compute_herman": True,
        "herman_chi_max": 90.0,
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
        "line_color": None,
        "sector_line_color": "cyan",
        "font_family": None,
        "font_size": None,
        "dpi": 400,
        "axis_labels": "xyz",
    }


# --------------------------------------------------------------------------- #
# Building argv lists for each agent, from the shared config
# --------------------------------------------------------------------------- #
def build_common_argv(cfg: Dict[str, Any]) -> List[str]:
    argv = [
        f"--input={cfg['input']}",
        f"--output-dir={cfg['output_dir']}",
        f"--incident-angle={cfg['incident_angle']}",
        f"--npt={cfg.get('npt', 1000)}",
    ]
    # The platform already asked WHETHER to calibrate up-front (or read it
    # from the config), so the sub-agent should never re-ask that same
    # question -- but if this is an interactive wizard run and calibration
    # IS being used, the sub-agent's own diagnostic-popup + "does this look
    # right?" confirmation loop should still run (a different, valuable
    # prompt). Only suppress everything when: this is a scripted/config
    # run, OR no calibration was chosen at all (nothing to confirm, and
    # suppressing avoids a redundant "do you have a calibration file?"
    # re-ask of a question the wizard already asked and got "no" for).
    if not (cfg.get("_interactive_calibration") and cfg.get("agbeh_file")):
        argv.append("--no-calibration-prompt")
    if cfg.get("incident_angle_from_filename"):
        argv.append("--incident-angle-from-filename")
    if cfg.get("incident_angle_map"):
        argv.append(f"--incident-angle-map={cfg['incident_angle_map']}")
    if cfg.get("agbeh_file"):
        argv.append(f"--agbeh-file={cfg['agbeh_file']}")
        argv.append(f"--calibrant={cfg.get('calibrant', 'AgBh')}")
        argv.append(f"--calib-max-rings={cfg.get('calib_max_rings', 5)}")
        argv.append(f"--calib-min-intensity={cfg.get('calib_min_intensity', 200.0)}")
        if cfg.get("save_calibrated_poni"):
            argv.append(f"--save-calibrated-poni={cfg['save_calibrated_poni']}")

    if cfg.get("poni_file"):
        # The .poni file supplies the ENTIRE geometry (beam centre,
        # distance, rotations, wavelength, detector) -- none of those
        # need to be (or should be) passed separately.
        argv.append(f"--poni-file={cfg['poni_file']}")
    else:
        if cfg.get("beam_center_y") is None or cfg.get("beam_center_x") is None or cfg.get("distance") is None:
            sys.exit("Config must specify 'beam_center_y', 'beam_center_x', and "
                      "'distance' -- or a 'poni_file' to load geometry from instead.")
        argv.append(f"--beam-center-y={cfg['beam_center_y']}")
        argv.append(f"--beam-center-x={cfg['beam_center_x']}")
        argv.append(f"--distance={cfg['distance']}")
        argv.append(f"--rot1={cfg.get('rot1', 0.0)}")
        argv.append(f"--rot2={cfg.get('rot2', 0.0)}")
        argv.append(f"--rot3={cfg.get('rot3', 0.0)}")

        if cfg.get("wavelength") is not None:
            argv.append(f"--wavelength={cfg['wavelength']}")
        elif cfg.get("energy") is not None:
            argv.append(f"--energy={cfg['energy']}")
        else:
            sys.exit("Config must specify either 'wavelength' or 'energy' "
                      "(or a 'poni_file' to load geometry from instead).")

        if cfg.get("detector_name"):
            argv.append(f"--detector-name={cfg['detector_name']}")
        else:
            argv.append(f"--pixel-size={cfg.get('pixel_size', 172e-6)}")
            shape = cfg.get("detector_shape")
            if not shape:
                sys.exit("Config must specify 'detector_shape' [height, width] "
                          "when 'detector_name' is not given (or use 'poni_file').")
            argv.append(f"--detector-shape={shape[0]},{shape[1]}")

    if cfg.get("mask"):
        argv.append(f"--mask={cfg['mask']}")

    # --- Shared styling (font family/size apply identically to both agents) ---
    if cfg.get("font_family"):
        argv.append(f"--font-family={cfg['font_family']}")
    if cfg.get("font_size"):
        argv.append(f"--font-size={cfg['font_size']}")

    return argv


def build_2d1d_argv(cfg: Dict[str, Any]) -> List[str]:
    argv = build_common_argv(cfg)
    qip = cfg.get("qip_plot_range", [-0.5, 2.4])
    qoop = cfg.get("qoop_plot_range", [-0.25, 2.75])
    linecut_q = cfg.get("linecut_q_range", [0.15, 2.0])
    argv += [
        f"--qip-plot-range={qip[0]},{qip[1]}",
        f"--qoop-plot-range={qoop[0]},{qoop[1]}",
        f"--vmin-percentile={cfg.get('vmin_percentile', 1.0)}",
        f"--vmax-percentile={cfg.get('vmax_percentile', 99.9)}",
        f"--cmap={cfg.get('cmap', 'viridis')}",
        f"--sector-line-color={cfg.get('sector_line_color', 'cyan')}",
        f"--axis-labels={cfg.get('axis_labels', 'xyz')}",
        f"--tick-spacing={cfg.get('tick_spacing', 0.5)}",
        f"--linecut-q-range={linecut_q[0]},{linecut_q[1]}",
        f"--linecut-tick-spacing={cfg.get('linecut_tick_spacing', 0.3)}",
    ]
    if cfg.get("dpi"):
        argv.append(f"--dpi={cfg['dpi']}")
    if cfg.get("vmin") is not None:
        argv.append(f"--vmin={cfg['vmin']}")
    if cfg.get("vmax") is not None:
        argv.append(f"--vmax={cfg['vmax']}")
    if cfg.get("line_color"):
        argv.append(f"--line-color={cfg['line_color']}")
    # Each sector is passed as its own '--extra-ranges=a,b' token (the flag
    # uses action='append'), which is argparse-safe for negative angles --
    # e.g. '-55,-45' would otherwise be misread as a stray option.
    for a, b in cfg.get("extra_ranges", []):
        argv.append(f"--extra-ranges={a},{b}")
    # All angular sectors are already fixed via config, so avoid re-prompting.
    argv.append("--non-interactive")
    return argv


def build_pole_figure_argv(cfg: Dict[str, Any]) -> List[str]:
    argv = build_common_argv(cfg)
    if cfg.get("line_color"):
        argv.append(f"--line-color={cfg['line_color']}")
    if cfg.get("dpi"):
        argv.append(f"--dpi={cfg['dpi']}")
    q_values = cfg.get("pole_figure_q", [])
    q_map_path = cfg.get("pole_figure_q_map")
    if not q_values and not q_map_path:
        sys.exit("Config must specify at least one 'pole_figure_q' value "
                  "(or a 'pole_figure_q_map' file) to run the pole figure step.")
    if q_map_path:
        argv.append(f"--pole-figure-q-map={q_map_path}")
    # pole-figure-q uses nargs='+'; q magnitudes are always non-negative so
    # plain space-separated values are safe here. Still allowed alongside a
    # q-map, as the fallback for any file the map doesn't cover.
    if q_values:
        argv.append("--pole-figure-q")
        for q in q_values:
            argv.append(str(q))
    argv.append(f"--pole-figure-dq={cfg.get('pole_figure_dq', 0.05)}")
    if not cfg.get("compute_herman", True):
        argv.append("--no-herman")
    argv.append(f"--chi-max={cfg.get('herman_chi_max', 90.0)}")
    return argv


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Platform that runs the GIWAXS 2D/1D agent and the pole "
                    "figure agent together from a single shared configuration.",
    )
    p.add_argument("--config", default=None,
                    help="Path to a JSON config file. If omitted, an "
                         "interactive setup wizard is run instead.")
    p.add_argument("--write-example", metavar="PATH", default=None,
                    help="Write an example config JSON to PATH and exit "
                         "(does not run any processing).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    if args.write_example:
        save_config(example_config(), args.write_example)
        print(f"Wrote example config to: {os.path.abspath(args.write_example)}")
        print("Edit the values, then run:")
        print(f"    python {os.path.basename(__file__)} --config {args.write_example}")
        return

    used_wizard = args.config is None
    cfg = load_config(args.config) if args.config else run_wizard()
    # Only an interactive wizard run should let the sub-agents' own
    # calibration confirmation loop (diagnostic popup + "does this look
    # right?" prompt) actually run -- a --config-driven run is assumed to
    # be scripted/automated and must never block on an interactive prompt.
    cfg["_interactive_calibration"] = used_wizard

    ran_anything = False

    if cfg.get("run_2d1d", True):
        print("\n" + "=" * 70)
        print("Running 2D image + 1D line-cut agent...")
        print("=" * 70)
        giwaxs_2d1d_agent.main(build_2d1d_argv(cfg))
        ran_anything = True

    if cfg.get("run_pole_figure", True):
        print("\n" + "=" * 70)
        print("Running pole figure agent...")
        print("=" * 70)
        giwaxs_polefigure_agent.main(build_pole_figure_argv(cfg))
        ran_anything = True

    if not ran_anything:
        print("Nothing to do: both 'run_2d1d' and 'run_pole_figure' are false "
              "in the configuration.")
        return

    print("\n" + "=" * 70)
    print(f"All requested steps complete. Outputs in: {os.path.abspath(cfg['output_dir'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
