#!/usr/bin/env python3
"""
giwaxs_polefigure_agent.py
============================

Generates pole figures (intensity-vs-chi, Cartesian log-y style) from raw
GIWAXS TIFF detector images, for one or more target reflections (q values).

IMPORTANT ASSUMPTION: because a single GIWAXS frame only samples one
azimuthal (phi) sample orientation, this agent builds pole figures under
the FIBER-TEXTURE approximation -- i.e. it assumes the film has no
preferred in-plane / azimuthal orientation (rotationally symmetric about
the surface normal). This is the standard approach used for single-frame
GIWAXS data on spin-coated / blade-coated films.

(If you have a true phi-rotation series -- i.e. multiple GIWAXS frames
taken at different in-plane sample rotations -- a full, non-approximated
pole figure could be built instead; that is not implemented here.)

For the standard 2D q-space image and 1D line-cut profiles, use the
companion script `giwaxs_2d1d_agent.py`.

Requirements
------------
    pip install pyFAI fabio numpy matplotlib

Usage
-----
    python giwaxs_polefigure_agent.py \\
        --beam-center-y 145 --beam-center-x 1088 \\
        --distance 0.65 \\
        --wavelength 1.5406e-10 \\
        --pixel-size 172e-6 --detector-shape 1043,981 \\
        --incident-angle 0.1 \\
        --pole-figure-q 1.70 0.30 --pole-figure-dq 0.03

(If --input / --output-dir / --pole-figure-q are omitted, you'll be
prompted for them.)

Run `python giwaxs_polefigure_agent.py --help` for the full parameter list.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import giwaxs_common as gc


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate fiber-texture-approximation pole figures from "
                    "raw GIWAXS TIFF images using pyFAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    gc.add_io_args(p)
    gc.add_geometry_args(p)
    gc.add_calibration_args(p)

    p.add_argument("--pole-figure-q", type=float, nargs="+", default=None,
                    help="Target |q| value(s), in inverse Angstrom, of the "
                         "reflection(s) to build pole figures for. Used as "
                         "the fallback for any file not listed in "
                         "--pole-figure-q-map (if given); if omitted "
                         "entirely and no map is given, you will be "
                         "prompted for them.")
    p.add_argument("--pole-figure-q-map", default=None,
                    help="Path to a JSON or CSV file mapping each filename "
                         "to its OWN target q value(s), for a batch where "
                         "different files need different reflections. JSON: "
                         "{\"file1.tif\": [1.673, 0.252], \"file2.tif\": 1.68}. "
                         "CSV: columns 'filename' and 'q_values' (space/"
                         "comma-separated numbers). Any file not listed "
                         "falls back to --pole-figure-q.")
    p.add_argument("--pole-figure-dq", type=float, default=0.05,
                    help="Half-width (in inverse Angstrom) of the q window "
                         "averaged over when extracting the chi profile for "
                         "each pole figure.")

    p.add_argument("--no-herman", action="store_true",
                    help="Skip computing Herman's orientation factor S "
                         "(computed by default alongside each pole figure, "
                         "since it uses the same chi-intensity profile).")
    p.add_argument("--chi-max", type=float, default=90.0,
                    help="Maximum tilt angle (degrees, measured from the "
                         "surface normal) integrated over when computing "
                         "Herman's orientation factor. The standard "
                         "convention (0 = perpendicular to substrate, "
                         "90 = parallel to substrate) uses 90.")

    p.add_argument("--chi-plot-range", type=gc.parse_range, default=(-90, 90),
                    help="X-axis range (degrees) for the chi-intensity plot, "
                         "as 'min,max'.")
    p.add_argument("--chi-tick-spacing", type=float, default=20.0,
                    help="Major tick spacing (in degrees) for the chi-intensity "
                         "plot's x-axis.")

    # --- Style options (line colour, fonts) --------------------------------
    p.add_argument("--line-color", default=None,
                    help="Marker/line colour for the pole figure plot, "
                         "as any matplotlib colour spec (name like 'red', "
                         "hex like '#1f77b4', etc.). Default: matplotlib's "
                         "default color cycle.")
    p.add_argument("--font-family", default=None,
                    help=f"Font family for all plot text. Common choices: "
                         f"{', '.join(gc.COMMON_FONTS)}.")
    p.add_argument("--font-size", type=float, default=None,
                    help="Base font size (points) for all plot text.")
    p.add_argument("--dpi", type=int, default=400,
                    help="Resolution (dots per inch) for saved PNG files.")

    return p.parse_args(argv)


def prompt_for_q_values() -> List[float]:
    while True:
        text = input(
            "Enter target q value(s) for the pole figure(s), in 1/Å "
            "(space or comma separated, e.g. '1.70 0.30'): "
        ).strip()
        if not text:
            print("  At least one q value is required.")
            continue
        try:
            values = [float(v) for v in text.replace(",", " ").split()]
        except ValueError:
            print("  Could not parse one or more values -- please try again.")
            continue
        if values:
            return values


# --------------------------------------------------------------------------- #
# Core per-file processing
# --------------------------------------------------------------------------- #
def process_file(tiff_path: str, fi, get_unit_fiber, mask, args,
                  out_dir, fabio, target_qs: List[float], summary_path: Optional[str],
                  angle_map=None):
    base = os.path.splitext(os.path.basename(tiff_path))[0]
    print(f"\nProcessing: {tiff_path}")

    incident_angle_deg = gc.resolve_incident_angle_for_file(
        tiff_path, args.incident_angle, args.incident_angle_from_filename,
        angle_map=angle_map,
    )
    _, _, unit_gi_chi, unit_gi_qtot = gc.build_grazing_units(get_unit_fiber, incident_angle_deg)

    img_data = fabio.open(tiff_path).data

    for target_q in target_qs:
        chi_axis, profile = gc.compute_chi_profile_at_q(
            fi, img_data, mask, target_q, args.pole_figure_dq,
            args.npt, unit_gi_chi, unit_gi_qtot,
        )
        q_tag = f"{target_q:.3f}".replace(".", "p").replace("-", "m")

        herman_s = None
        if not args.no_herman:
            try:
                herman_s, mean_cos2_chi, coverage = gc.compute_herman_orientation(
                    chi_axis, profile, chi_max=args.chi_max
                )
                print(
                    f"  Herman's orientation factor S = {herman_s:.4f} "
                    f"(<cos^2 chi> = {mean_cos2_chi:.4f}, angular coverage = "
                    f"{coverage * 100:.0f}% of 0-{args.chi_max:.0f} deg) "
                    f"at q = {target_q} 1/A"
                )
                if coverage < 0.5:
                    print(
                        "  NOTE: low angular coverage -- a large missing "
                        "wedge was excluded from this calculation, so "
                        "treat this S value with caution."
                    )
                if summary_path:
                    gc.append_herman_summary_row(summary_path, {
                        "filename": base,
                        "target_q": target_q,
                        "dq": args.pole_figure_dq,
                        "S": f"{herman_s:.6f}",
                        "mean_cos2_chi": f"{mean_cos2_chi:.6f}",
                        "coverage_fraction": f"{coverage:.4f}",
                    })
            except ValueError as exc:
                print(f"  Could not compute Herman's orientation factor: {exc}")

        pf_path = os.path.join(out_dir, f"{base}_polefigure_q{q_tag}.png")
        gc.plot_chi_intensity_profile(chi_axis, profile, pf_path, target_q,
                                       args.pole_figure_dq, title=f"{base}",
                                       herman_s=herman_s, chi_range=args.chi_plot_range,
                                       line_color=args.line_color,
                                       font_family=args.font_family, font_size=args.font_size,
                                       dpi=args.dpi, tick_spacing=args.chi_tick_spacing)
        print(f"  Saved pole figure at q={target_q} 1/A -> {pf_path}")

        pf_data_path = os.path.join(out_dir, f"{base}_polefigure_q{q_tag}_chi_profile.txt")
        import numpy as np
        header = "Chi(deg)\tIntensity(a.u.)"
        if herman_s is not None:
            header = f"Herman_S={herman_s:.6f}\n{header}"
        np.savetxt(pf_data_path, np.c_[chi_axis, profile], header=header)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    fabio, FiberIntegrator, get_unit_fiber, Detector, detector_factory = gc.import_pyfai_stack()

    input_path, output_dir = gc.resolve_input_output(args)
    tiff_files = gc.resolve_tiff_files(input_path)

    pf_out_dir = os.path.join(output_dir, "pole_figures")
    os.makedirs(pf_out_dir, exist_ok=True)

    q_map = gc.load_pole_figure_q_map(args.pole_figure_q_map) if args.pole_figure_q_map else {}
    angle_map = gc.load_incident_angle_map(args.incident_angle_map) if args.incident_angle_map else None

    # The shared/fallback q list is only strictly required if there's no
    # per-file map, or the map doesn't cover every file.
    default_target_qs = args.pole_figure_q
    if default_target_qs is None and not q_map:
        default_target_qs = prompt_for_q_values()
    elif default_target_qs is None:
        default_target_qs = []  # fine as long as the map covers every file

    print(
        "\nNOTE: Pole figure(s) are generated assuming FIBER TEXTURE "
        "(no preferred in-plane / azimuthal orientation). This is the "
        "standard approximation for single-frame GIWAXS data (no "
        "phi-rotation series). The measured tilt (chi) profile is revolved "
        "uniformly about phi.\n"
    )
    if q_map:
        print(f"Using per-file target q values from: {args.pole_figure_q_map} "
              f"({len(q_map)} file(s) listed; any file not listed uses "
              f"--pole-figure-q={default_target_qs} as fallback)\n")

    gc.prompt_for_calibration_setup(args)

    fi, detector = gc.build_fiber_integrator(args, Detector, detector_factory, FiberIntegrator, fabio=fabio)

    first_shape = fabio.open(tiff_files[0]).data.shape
    mask = gc.load_mask(args, fabio, first_shape)

    summary_path = None
    if not args.no_herman:
        summary_path = os.path.join(pf_out_dir, "herman_orientation_summary.csv")

    for tiff_path in tiff_files:
        file_target_qs = q_map.get(os.path.basename(tiff_path), default_target_qs)
        if not file_target_qs:
            print(f"\nSkipping {tiff_path}: no target q value(s) specified "
                  f"(not in --pole-figure-q-map and no --pole-figure-q fallback given).")
            continue
        process_file(tiff_path, fi, get_unit_fiber, mask, args,
                     pf_out_dir, fabio, file_target_qs, summary_path,
                     angle_map=angle_map)

    print(f"\nDone. Pole figures written to: {os.path.abspath(pf_out_dir)}")
    if summary_path and os.path.exists(summary_path):
        print(f"Herman's orientation factor summary: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
