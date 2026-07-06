#!/usr/bin/env python3
"""
giwaxs_2d1d_agent.py
=====================

Processes raw GIWAXS TIFF detector images into:
  1. A 2D (q_ip, q_oop) reciprocal-space image, and
  2. 1D line-cut intensity profiles for two default angular sectors:
        * (-90, -80) deg  -> in-plane cut
        * ( -8,   8) deg  -> out-of-plane cut
     (0 deg = out-of-plane direction; angles measured clockwise, matching
     the `ip_range` convention of pyFAI's FiberIntegrator.)

After the defaults are processed you will be asked (interactively) whether
you'd like to process any additional angular sectors as well.

The agent will prompt you for the input path and destination directory if
they are not supplied on the command line; the destination directory is
created automatically if it doesn't already exist.

For pole figures, use the companion script `giwaxs_polefigure_agent.py`.

Requirements
------------
    pip install pyFAI fabio numpy matplotlib

Usage
-----
    python giwaxs_2d1d_agent.py \\
        --beam-center-y 145 --beam-center-x 1088 \\
        --distance 0.65 \\
        --wavelength 1.5406e-10 \\
        --pixel-size 172e-6 --detector-shape 1043,981 \\
        --incident-angle 0.1

(If --input / --output-dir are omitted, you'll be prompted for them.)

Run `python giwaxs_2d1d_agent.py --help` for the full parameter list.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Optional

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

import giwaxs_common as gc


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process raw GIWAXS TIFF images into 2D q-space images "
                    "and 1D line-cut profiles using pyFAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    gc.add_io_args(p)
    gc.add_geometry_args(p)
    gc.add_calibration_args(p)

    p.add_argument("--qip-plot-range", type=gc.parse_range, default=(-0.5, 2.4),
                    help="X axis (q_ip) plot limits for the 2D image, as "
                         "'min,max' in inverse Angstrom.")
    p.add_argument("--qoop-plot-range", type=gc.parse_range, default=(-0.25, 2.75),
                    help="Y axis (q_oop) plot limits for the 2D image, as "
                         "'min,max' in inverse Angstrom.")
    p.add_argument("--vmin-percentile", type=float, default=1.0,
                    help="Percentile (of nonzero pixels) used as the log-scale "
                         "colour minimum for the 2D image (ignored if --vmin "
                         "is given explicitly). Raised from a bare minimum so "
                         "the image isn't washed out by a few near-zero pixels.")
    p.add_argument("--vmax-percentile", type=float, default=99.9,
                    help="Percentile (of nonzero pixels) used as the log-scale "
                         "colour maximum for the 2D image (ignored if --vmax "
                         "is given explicitly). Lowered from the raw pixel max "
                         "so a few hot/saturated pixels don't wash out contrast "
                         "everywhere else.")

    # --- Style options (colormap, colour-scale range, line style, fonts) ---
    p.add_argument("--cmap", default="viridis",
                    help=f"Matplotlib colormap for the 2D image. Common choices: "
                         f"{', '.join(gc.COMMON_COLORMAPS)} (any valid matplotlib "
                         f"colormap name also works).")
    p.add_argument("--vmin", type=float, default=None,
                    help="Explicit colour-scale minimum (intensity units) for "
                         "the 2D image, overriding --vmin-percentile.")
    p.add_argument("--vmax", type=float, default=None,
                    help="Explicit colour-scale maximum (intensity units) for "
                         "the 2D image, overriding --vmax-percentile.")
    p.add_argument("--line-color", default=None,
                    help="Line colour for the 1D line-cut plots, as any "
                         "matplotlib colour spec (name like 'red', hex like "
                         "'#1f77b4', etc.). Default: matplotlib's default blue.")
    p.add_argument("--sector-line-color", default="cyan",
                    help="Colour of the sector boundary lines overlaid on the "
                         "2D image for each line-cut sector.")
    p.add_argument("--font-family", default=None,
                    help=f"Font family for all plot text. Common choices: "
                         f"{', '.join(gc.COMMON_FONTS)}.")
    p.add_argument("--font-size", type=float, default=None,
                    help="Base font size (points) for all plot text.")
    p.add_argument("--dpi", type=int, default=400,
                    help="Resolution (dots per inch) for saved PNG files.")
    p.add_argument("--axis-labels", choices=["ip_oop", "xyz"], default="xyz",
                    help="Axis label convention for the 2D image: 'xyz' "
                         "(q_xy / q_z, this toolkit's default) or 'ip_oop' "
                         "(q_ip / q_oop, an equally common convention in the "
                         "literature -- purely cosmetic, same underlying data).")
    p.add_argument("--tick-spacing", type=float, default=0.5,
                    help="Major tick spacing (in inverse Angstrom) for both "
                         "axes of the 2D image.")
    p.add_argument("--linecut-q-range", type=gc.parse_range, default=(0.15, 2.0),
                    help="X-axis (q) range for line-cut plots, as 'min,max' "
                         "in inverse Angstrom.")
    p.add_argument("--linecut-tick-spacing", type=float, default=0.3,
                    help="Major tick spacing (in inverse Angstrom) for the "
                         "line-cut plots' q-axis.")

    p.add_argument("--extra-ranges", type=gc.parse_range, action="append", default=None,
                    help="Optional additional angular sector to integrate "
                         "non-interactively, as 'angle1,angle2' in degrees "
                         "(0 deg = out-of-plane, clockwise convention). "
                         "Repeat this flag for multiple sectors -- always use "
                         "the '=' form so negative angles parse correctly, "
                         "e.g. --extra-ranges=-55,-45 --extra-ranges=30,40")
    p.add_argument("--non-interactive", action="store_true",
                    help="Do not prompt for additional angular sectors; only "
                         "process the two default sectors (and any given via "
                         "--extra-ranges).")

    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Interactive prompt for extra angular ranges
# --------------------------------------------------------------------------- #
def ask_for_extra_ranges() -> List[Tuple[float, float]]:
    print("\nDefault line-cut sectors already processed:")
    print("  * In-plane      : (-90, -80) deg")
    print("  * Out-of-plane  : ( -8,   8) deg")
    ranges: List[Tuple[float, float]] = []
    while True:
        resp = input(
            "\nWould you like to process another angular sector? [y/N]: "
        ).strip().lower()
        if resp not in ("y", "yes"):
            break
        try:
            a1 = float(input("  Start angle (deg, 0 = out-of-plane direction): ").strip())
            a2 = float(input("  End angle (deg): ").strip())
        except ValueError:
            print("  Could not parse that as a number -- skipping this entry.")
            continue
        ranges.append((a1, a2))
        print(f"  Added sector: ({a1}, {a2}) deg")
    return ranges


# --------------------------------------------------------------------------- #
# Core per-file processing
# --------------------------------------------------------------------------- #
def process_file(tiff_path: str, fi, get_unit_fiber, mask, args, out_dirs, fabio, all_ranges,
                  angle_map=None):
    base = os.path.splitext(os.path.basename(tiff_path))[0]
    print(f"\nProcessing: {tiff_path}")

    incident_angle_deg = gc.resolve_incident_angle_for_file(
        tiff_path, args.incident_angle, args.incident_angle_from_filename,
        angle_map=angle_map,
    )
    unit_gi_ip, unit_gi_oop, unit_gi_chi, unit_gi_qtot = gc.build_grazing_units(
        get_unit_fiber, incident_angle_deg
    )

    img_data = fabio.open(tiff_path).data

    # --- 2D remap into (q_ip, q_oop) space ------------------------------------
    res2d = fi.integrate2d_grazing_incidence(
        img_data,
        npt_ip=args.npt, npt_oop=args.npt,
        unit_ip=unit_gi_ip, unit_oop=unit_gi_oop,
        mask=mask,
    )
    res_I, res_qx, res_qy = res2d[0:3]
    res_qx = -np.flip(res_qx)
    res_I = np.flip(res_I, axis=1)

    img_out_path = os.path.join(out_dirs["images"], f"{base}_2D_GIWAXS.png")
    gc.plot_2d_image(
        res_qx, res_qy, res_I, img_out_path,
        qlim_x=args.qip_plot_range, qlim_y=args.qoop_plot_range,
        vmin_percentile=args.vmin_percentile, vmax_percentile=args.vmax_percentile,
        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
        font_family=args.font_family, font_size=args.font_size,
        dpi=args.dpi, axis_label_style=args.axis_labels,
        tick_spacing=args.tick_spacing,
    )
    print(f"  Saved 2D image: {img_out_path}")

    # --- 1D line cuts ------------------------------------------------------
    incident_angle_rad = np.deg2rad(incident_angle_deg)

    for angles in all_ranges:
        res1d = fi.integrate1d_grazing_incidence(
            data=img_data,
            incident_angle=incident_angle_rad,
            unit_ip=unit_gi_chi, unit_oop=unit_gi_qtot,
            npt_oop=args.npt, npt_ip=args.npt,
            ip_range=angles,
            mask=mask,
        )
        q, intensity = res1d

        tag = f"{angles[0]}_to_{angles[1]}_deg".replace("-", "m")
        data_out_path = os.path.join(out_dirs["linecuts"], f"{base}_lineprofile_{tag}.txt")
        np.savetxt(data_out_path, np.c_[q, intensity], header="Q(1/A)\tIntensity(a.u.)")

        plot_out_path = os.path.join(out_dirs["linecuts"], f"{base}_lineprofile_{tag}.png")
        gc.plot_1d_linecut(q, intensity, plot_out_path, angles, title=f"{base}: {angles} deg",
                            line_color=args.line_color, font_family=args.font_family,
                            font_size=args.font_size, dpi=args.dpi,
                            q_range=args.linecut_q_range, tick_spacing=args.linecut_tick_spacing)

        # Overlay the sector on a copy of the 2D image for reference.
        overlay_path = os.path.join(out_dirs["images"], f"{base}_sector_{tag}.png")
        xlabel, ylabel = gc.AXIS_LABELS.get(args.axis_labels, gc.AXIS_LABELS["ip_oop"])
        with gc.style_context(args.font_family, args.font_size):
            fig, ax = plt.subplots(1, 2, width_ratios=[1, 0.05], figsize=gc.DEFAULT_FIGSIZE)
            v_lo, v_hi = gc.resolve_vmin_vmax(res_I, args.vmin_percentile, args.vmin, args.vmax,
                                               args.vmax_percentile)
            mesh = ax[0].pcolormesh(res_qx, res_qy, res_I, norm=LogNorm(vmin=v_lo, vmax=v_hi),
                                     cmap=args.cmap)
            ax[0].set_facecolor("black")
            ax[0].set_aspect("equal")
            ax[0].set_xlim(args.qip_plot_range)
            ax[0].set_ylim(args.qoop_plot_range)
            ax[0].xaxis.set_major_locator(MultipleLocator(args.tick_spacing))
            ax[0].yaxis.set_major_locator(MultipleLocator(args.tick_spacing))
            ax[0].set_xlabel(xlabel)
            ax[0].set_ylabel(ylabel)
            gc.add_angle_lines(ax[0], res_qx, res_qy, angles, color=args.sector_line_color)
            plt.colorbar(mesh, cax=ax[1], orientation="vertical")
            fig.suptitle(f"{base}: sector {angles} deg")
            fig.tight_layout()
            fig.savefig(overlay_path, dpi=args.dpi)
            plt.close(fig)

        print(f"  Saved line cut ({angles[0]}, {angles[1]}) deg -> {data_out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    fabio, FiberIntegrator, get_unit_fiber, Detector, detector_factory = gc.import_pyfai_stack()

    input_path, output_dir = gc.resolve_input_output(args)
    tiff_files = gc.resolve_tiff_files(input_path)

    angle_map = gc.load_incident_angle_map(args.incident_angle_map) if args.incident_angle_map else None

    out_dirs = {
        "images": os.path.join(output_dir, "images"),
        "linecuts": os.path.join(output_dir, "linecuts"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # NOTE: --non-interactive only controls the extra-line-cut-sectors
    # prompt below; calibration prompting/confirmation is controlled
    # independently by --no-calibration-prompt, since AgBeh fit
    # confirmation is valuable even in an otherwise-scripted run.
    gc.prompt_for_calibration_setup(args)

    fi, detector = gc.build_fiber_integrator(args, Detector, detector_factory, FiberIntegrator, fabio=fabio)

    first_shape = fabio.open(tiff_files[0]).data.shape
    mask = gc.load_mask(args, fabio, first_shape)

    # Determine the full list of angular sectors up-front (defaults + CLI extras
    # + interactively-requested extras), so the prompt only happens once.
    cli_extra_ranges = args.extra_ranges or []
    all_ranges = [(-90, -80), (-8, 8)] + list(cli_extra_ranges)
    if not args.non_interactive:
        all_ranges += ask_for_extra_ranges()

    for tiff_path in tiff_files:
        process_file(
            tiff_path, fi, get_unit_fiber,
            mask, args, out_dirs, fabio, all_ranges,
            angle_map=angle_map,
        )

    print(f"\nDone. Outputs written to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
