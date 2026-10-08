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

Optionally, peaks can be fitted on the resulting line cuts in the same run
by adding one --fit-region per peak (see --help for the full description):

    python giwaxs_2d1d_agent.py ... \\
        --fit-region "0.20:0.32:(100) lamellar" \\
        --fit-region "1.55:1.80:pi-pi stacking"

which additionally writes a `peakfits/` directory of overlay plots and one
combined `peak_fit_results.csv` (q0, d-spacing, FWHM, coherence length via
the q-space Scherrer equation, peak intensity/area, R^2).

Run `python giwaxs_2d1d_agent.py --help` for the full parameter list.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
from matplotlib.colors import LogNorm, Normalize
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
    p.add_argument("--vmin-percentile", type=float, default=30.0,
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
    p.add_argument("--color-scale", choices=["log", "linear"], default="log",
                    help="Colour mapping for the 2D image: 'log' (default) shows "
                         "scientific-notation colorbar ticks (10^n); 'linear' shows "
                         "evenly-spaced round-number ticks (100, 200, 300, ...).")
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
    p.add_argument("--no-auto-beamstop-mask", action="store_true",
                    help="Do not detect and mask the beamstop shadow. By "
                         "default it is masked, because a grazing-incidence "
                         "beamstop is a long finger up the beam column -- "
                         "exactly where the out-of-plane sector lies -- and "
                         "an unmasked shadow is averaged in as real low "
                         "intensity rather than ignored.")
    p.add_argument("--autodefine-centre", action="store_true",
                    help="Measure the beam centre from the FIRST data file "
                         "instead of trusting the .poni or the calibration "
                         "for it: the column from the frame's mirror "
                         "symmetry about q_xy = 0, the row from the horizon. "
                         "Either half that cannot be measured is left as the "
                         "geometry had it, and the reason is printed. Use "
                         "this when the calibrant was taken in a different "
                         "detector position -- a case a calibrant fit cannot "
                         "detect, since it reports a fine residual anyway.")
    p.add_argument("--mask-detector-gaps", action="store_true",
                    help="Also mask the detector's module gaps. Off by "
                         "default: pyFAI replaces its own detector mask with "
                         "whatever mask it is given, so leaving them unmasked "
                         "keeps the 2D map continuous instead of ruled with "
                         "black bands. Measured cost -- peak positions move at "
                         "most 0.25%%, widths unchanged, and only the few bins "
                         "sitting directly on a gap read low, by up to 8%%. "
                         "The beamstop is always masked either way.")
    p.add_argument("--no-colorbar", dest="show_colorbar", action="store_false",
                    help="Leave the intensity colour bar off the saved image. "
                         "Useful when the figure will carry its scale "
                         "elsewhere, or for a bare panel.")
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
    p.add_argument("--subtick-spacing", type=float, default=None,
                    help="Minor tick spacing (in inverse Angstrom) for both "
                         "axes of the 2D image. Off (no minor ticks) unless "
                         "given -- this is an opt-in extra, not shown by default.")
    p.add_argument("--linecut-q-range", type=gc.parse_range, default=(0.15, 2.0),
                    help="X-axis (q) range for line-cut plots, as 'min,max' "
                         "in inverse Angstrom.")
    p.add_argument("--linecut-tick-spacing", type=float, default=0.3,
                    help="Major tick spacing (in inverse Angstrom) for the "
                         "line-cut plots' q-axis.")
    p.add_argument("--linecut-subtick-spacing", type=float, default=None,
                    help="Minor tick spacing (in inverse Angstrom) for the "
                         "line-cut plots' q-axis. Off (no minor ticks) unless "
                         "given -- an opt-in extra, not shown by default.")

    # --- Peak fitting (optional; off unless at least one --fit-region given) ---
    p.add_argument("--box-cut", type=gc.parse_range, action="append", default=None,
                    dest="box_cuts", metavar="QXY_LO,QXY_HI",
                    help="Add an out-of-plane profile taken from a fixed q_xy "
                         "strip instead of an angular wedge. Repeat for more "
                         "strips. A wedge opens linearly with q, so near the "
                         "origin it is narrower than the beamstop and returns "
                         "nothing until it clears it -- on this detector the "
                         "+-8 deg wedge starts only at q = 0.21, which cuts the "
                         "low-q flank off a lamellar (100) sitting at 0.25. A "
                         "strip passes beside the stop all the way down. Start "
                         "with --box-cut 0.025,0.05: 0.025 is the stop's outer "
                         "edge, and a strip that narrow keeps about three "
                         "quarters of the peak intensity an oriented reflection "
                         "puts on the axis. Wider strips drift off the peak -- "
                         "0.03,0.12 kept only 13-38%% of it on real data.")
    p.add_argument("--stitch-oop", action="store_true",
                    help="Also write ONE out-of-plane profile covering the "
                         "whole q range, by joining each --box-cut strip to "
                         "the (-8, 8) deg sector. Neither covers it alone and "
                         "they fail at opposite ends -- the sector is narrower "
                         "than the beamstop near the origin, the strip is "
                         "swallowed by the missing wedge further out -- but "
                         "they overlap widely in between. The strip is put on "
                         "the sector's scale using the median ratio over that "
                         "overlap and crossfaded across it. The join is "
                         "REFUSED if the two differ in shape there rather than "
                         "only in scale, since a scale factor would then be "
                         "manufacturing a feature. Requires --box-cut.")
    p.add_argument("--box-cut-inplane", type=gc.parse_range, action="append", default=None,
                    dest="box_cuts_ip", metavar="QZ_LO,QZ_HI",
                    help="Add an IN-plane profile along q_xy taken from a fixed "
                         "q_z strip. Repeat for more strips. Use this when the "
                         "in-plane reflections are rods standing up from the "
                         "horizon: the chi -90..-80 sector drifts away from the "
                         "horizon as q grows -- at q = 0.74 its far edge is "
                         "already at q_z = 0.13 -- so it integrates along the "
                         "rod and lifts the background, while a strip hugs the "
                         "horizon at every q. On real data that moved the (300) "
                         "fit from R2 = 0.61 to 0.93. Start with "
                         "--box-cut-inplane 0.03,0.09.")
    p.add_argument("--fit-region", type=gc.parse_fit_region, action="append", default=None,
                    dest="fit_regions", metavar="QMIN:QMAX[:LABEL]",
                    help="Fit a diffraction peak within this q window (inverse "
                         "Angstrom) on every line cut produced by this run. "
                         "Colon-separated so labels may contain commas, e.g. "
                         "--fit-region 1.6:1.8:pi-pi stacking. Repeat the flag "
                         "for multiple peaks. Outputs an overlay PNG per "
                         "(file, sector) into a 'peakfits' subdirectory plus one "
                         "combined peak_fit_results.csv. NOTE: every region is "
                         "fitted against every sector -- a region that only has "
                         "real signal in one sector will still return a "
                         "numerically valid but physically meaningless fit "
                         "elsewhere, so check R^2 and q0 against expectations.")
    p.add_argument("--fit-shape", choices=["gaussian", "lorentzian", "pseudo_voigt"],
                    default="gaussian",
                    help="Peak profile fitted to each --fit-region (on top of a "
                         "linear background).")
    p.add_argument("--fit-scherrer-k", type=float, default=0.9,
                    help="Scherrer shape factor K used for the coherence length "
                         "L_c = 2*pi*K/FWHM, with FWHM taken directly in q-space "
                         "(the standard GIWAXS convention, not the 2-theta form).")

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
def fit_peaks_for_linecut(q, intensity, args, base: str, tag: str, sector_label: str,
                           out_dirs, fit_rows):
    """Fit every --fit-region against one line cut, save an overlay PNG, and
    append one CSV row per region to `fit_rows` (in place).

    Each region is fitted independently and failures are caught per region,
    so one non-convergent window doesn't lose the other regions' results --
    a failed region is recorded as an error row rather than dropped, so it
    stays visible in the combined CSV.
    """
    fits = []
    for q_min, q_max, label in args.fit_regions:
        try:
            fit = gc.fit_peak(q, intensity, q_min, q_max, shape=args.fit_shape,
                               scherrer_k=args.fit_scherrer_k, label=label)
        except Exception as exc:
            print(f"    Peak fit FAILED [{label}] {q_min}-{q_max} 1/A: {exc}")
            fit_rows.append(gc.peak_fit_csv_error_row(
                base, sector_label, label, q_min, q_max, args.fit_shape, str(exc)))
            continue
        fits.append(fit)
        fit_rows.append(gc.peak_fit_csv_row(fit, base, sector_label))
        print(f"    Peak [{label}]: q0={fit['q0']:.4f} 1/A, "
              f"d={fit['d_spacing']:.2f} A, FWHM={fit['fwhm']:.4f} 1/A, "
              f"L_c={fit['coherence_length']:.1f} A, R^2={fit['r_squared']:.4f}")

    if not fits:
        return
    overlay_path = os.path.join(out_dirs["peakfits"], f"{base}_peakfit_{tag}.png")
    gc.plot_linecut_with_fits(
        q, intensity, fits, overlay_path,
        title=f"{base}: {sector_label} peak fits",
        line_color=args.line_color, font_family=args.font_family,
        font_size=args.font_size, dpi=args.dpi,
        q_range=args.linecut_q_range, tick_spacing=args.linecut_tick_spacing,
        subtick_spacing=args.linecut_subtick_spacing,
    )
    print(f"    Saved peak-fit overlay: {overlay_path}")


def process_file(tiff_path: str, fi, get_unit_fiber, mask, args, out_dirs, fabio, all_ranges,
                  angle_map=None, fit_rows=None):
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

    # The beamstop is per-frame: detect it here, not once up front, so a
    # batch in which the stop moved is still handled correctly.
    bs = None
    if not args.no_auto_beamstop_mask:
        bs = gc.find_beamstop_mask(img_data, fi.poni2 / fi.detector.pixel2)
        if bs.any():
            print(f"  Beamstop masked: {int(bs.sum())} px "
                  f"({100 * bs.mean():.2f}% of the detector)")
    # Two masks, because the picture and the numbers want different things.
    # A module gap left unmasked averages in as zero intensity, which is a
    # quantitative error in a line cut but only a cosmetic one in the image
    # -- and masking it there rules the map with black bands. pyFAI is
    # called separately for the two, so they need not agree.
    # (Always through combine_masks with the detector where wanted: handing
    # pyFAI any explicit mask REPLACES its detector mask rather than adding
    # to it.)
    image_mask = gc.combine_masks(
        mask, bs, detector=fi.detector if args.mask_detector_gaps else None)
    cut_mask = gc.combine_masks(mask, bs, detector=fi.detector)

    # --- 2D remap into (q_ip, q_oop) space ------------------------------------
    res2d = fi.integrate2d_grazing_incidence(
        img_data,
        npt_ip=args.npt, npt_oop=args.npt,
        unit_ip=unit_gi_ip, unit_oop=unit_gi_oop,
        mask=image_mask,
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
        tick_spacing=args.tick_spacing, subtick_spacing=args.subtick_spacing,
        color_scale=args.color_scale, show_colorbar=args.show_colorbar,
    )
    print(f"  Saved 2D image: {img_out_path}")

    ok, _off, msg = gc.check_direct_beam_centred(res_qx, res_qy, res_I)
    if not ok:
        print(f"  WARNING: {msg}")

    # The check above pins the beam-centre COLUMN. It cannot see the ROW,
    # which is the badly conditioned half of a grazing-incidence
    # calibration -- so ask the frame's own horizon about that.
    row_ok, _row, row_msg = gc.check_beam_row_against_horizon(
        img_data, incident_angle_deg, fi.dist, fi.detector.pixel1,
        fi.poni1 / fi.detector.pixel1)
    if row_ok is False:
        print(f"  WARNING: {row_msg}")

    # --- 1D line cuts ------------------------------------------------------
    incident_angle_rad = np.deg2rad(incident_angle_deg)
    oop_sector = None

    for angles in all_ranges:
        res1d = fi.integrate1d_grazing_incidence(
            data=img_data,
            incident_angle=incident_angle_rad,
            unit_ip=unit_gi_chi, unit_oop=unit_gi_qtot,
            npt_oop=args.npt, npt_ip=args.npt,
            ip_range=angles,
            mask=cut_mask,
        )
        q, intensity = gc.linecut_drop_empty_bins(res1d)
        if tuple(angles) == (-8, 8):
            oop_sector = (q, intensity)

        tag = f"{angles[0]}_to_{angles[1]}_deg".replace("-", "m")
        data_out_path = os.path.join(out_dirs["linecuts"], f"{base}_lineprofile_{tag}.txt")
        np.savetxt(data_out_path, np.c_[q, intensity], header="Q(1/A)\tIntensity(a.u.)")

        plot_out_path = os.path.join(out_dirs["linecuts"], f"{base}_lineprofile_{tag}.png")
        gc.plot_1d_linecut(q, intensity, plot_out_path, angles, title=f"{base}: {angles} deg",
                            line_color=args.line_color, font_family=args.font_family,
                            font_size=args.font_size, dpi=args.dpi,
                            q_range=args.linecut_q_range, tick_spacing=args.linecut_tick_spacing,
                            subtick_spacing=args.linecut_subtick_spacing)

        # Overlay the sector on a copy of the 2D image for reference.
        overlay_path = os.path.join(out_dirs["images"], f"{base}_sector_{tag}.png")
        xlabel, ylabel = gc.AXIS_LABELS.get(args.axis_labels, gc.AXIS_LABELS["ip_oop"])
        with gc.style_context(args.font_family, args.font_size):
            if args.show_colorbar:
                fig, _axes = plt.subplots(1, 2, width_ratios=[1, 0.05],
                                           figsize=gc.DEFAULT_FIGSIZE)
                ax = [_axes[0], _axes[1]]
            else:
                _a = plt.figure(figsize=gc.DEFAULT_FIGSIZE).add_subplot(111)
                fig, ax = _a.figure, [_a, None]
            v_lo, v_hi = gc.resolve_vmin_vmax(res_I, args.vmin_percentile, args.vmin, args.vmax,
                                               args.vmax_percentile, color_scale=args.color_scale)
            sector_norm = (LogNorm(vmin=v_lo, vmax=v_hi) if args.color_scale == "log"
                            else Normalize(vmin=v_lo, vmax=v_hi))
            mesh = ax[0].pcolormesh(res_qx, res_qy, res_I, norm=sector_norm,
                                     cmap=args.cmap)
            ax[0].set_facecolor("black")
            ax[0].set_aspect("equal")
            ax[0].set_xlim(args.qip_plot_range)
            ax[0].set_ylim(args.qoop_plot_range)
            ax[0].xaxis.set_major_locator(MultipleLocator(args.tick_spacing))
            ax[0].yaxis.set_major_locator(MultipleLocator(args.tick_spacing))
            if args.subtick_spacing:
                ax[0].xaxis.set_minor_locator(MultipleLocator(args.subtick_spacing))
                ax[0].yaxis.set_minor_locator(MultipleLocator(args.subtick_spacing))
                ax[0].tick_params(which="minor", length=3)
            ax[0].tick_params(axis="both", which="both", direction="in")
            ax[0].set_xlabel(xlabel)
            ax[0].set_ylabel(ylabel)
            gc.add_angle_lines(ax[0], res_qx, res_qy, angles, color=args.sector_line_color)
            if args.show_colorbar:
                cbar = fig.colorbar(mesh, cax=ax[1], orientation="vertical")
                cbar.ax.tick_params(which="both", direction="in")
            fig.suptitle(f"{base}: sector {angles} deg")
            fig.tight_layout()
            fig.savefig(overlay_path, dpi=args.dpi)
            plt.close(fig)

        print(f"  Saved line cut ({angles[0]}, {angles[1]}) deg -> {data_out_path}")

        # --- Optional peak fitting on this line cut ------------------------
        # Done here, on the in-memory (q, intensity) arrays, rather than as a
        # separate pass that re-reads the saved .txt files -- same numbers,
        # no re-processing of the raw TIFF.
        if args.fit_regions:
            fit_peaks_for_linecut(
                q, intensity, args, base, tag,
                sector_label=f"({angles[0]}, {angles[1]}) deg",
                out_dirs=out_dirs, fit_rows=fit_rows if fit_rows is not None else [],
            )

    # --- Box cuts: an out-of-plane profile from a fixed q_xy strip ---------
    # Taken from the already-remapped map rather than a fresh integration,
    # so it inherits exactly the same geometry, mask and binning as the 2D
    # image above -- there is nothing for the two to disagree about.
    for qxy_lo, qxy_hi in (args.box_cuts or []):
        q, intensity = gc.box_cut(res_I, res_qx, res_qy, along="qz",
                                   across_range=(qxy_lo, qxy_hi))
        tag = f"boxcut_qxy_{qxy_lo:g}_to_{qxy_hi:g}".replace(".", "p")
        label = f"box cut, q_xy {qxy_lo:g}-{qxy_hi:g} 1/A"

        data_out_path = os.path.join(out_dirs["linecuts"], f"{base}_lineprofile_{tag}.txt")
        np.savetxt(data_out_path, np.c_[q, intensity], header="Q(1/A)\tIntensity(a.u.)")

        plot_out_path = os.path.join(out_dirs["linecuts"], f"{base}_lineprofile_{tag}.png")
        gc.plot_1d_linecut(q, intensity, plot_out_path, (qxy_lo, qxy_hi),
                            title=f"{base}: {label}",
                            line_color=args.line_color, font_family=args.font_family,
                            font_size=args.font_size, dpi=args.dpi,
                            q_range=args.linecut_q_range,
                            tick_spacing=args.linecut_tick_spacing,
                            subtick_spacing=args.linecut_subtick_spacing)

        overlay_path = os.path.join(out_dirs["images"], f"{base}_sector_{tag}.png")
        xlabel, ylabel = gc.AXIS_LABELS.get(args.axis_labels, gc.AXIS_LABELS["ip_oop"])
        with gc.style_context(args.font_family, args.font_size):
            fig, ax = plt.subplots(figsize=gc.DEFAULT_FIGSIZE)
            v_lo, v_hi = gc.resolve_vmin_vmax(res_I, args.vmin_percentile, args.vmin,
                                               args.vmax, args.vmax_percentile,
                                               color_scale=args.color_scale)
            norm = (LogNorm(vmin=v_lo, vmax=v_hi) if args.color_scale == "log"
                     else Normalize(vmin=v_lo, vmax=v_hi))
            ax.pcolormesh(res_qx, res_qy, res_I, norm=norm, cmap=args.cmap)
            ax.set_facecolor("black")
            ax.set_aspect("equal")
            ax.set_xlim(args.qip_plot_range)
            ax.set_ylim(args.qoop_plot_range)
            for sign in (-1.0, 1.0):
                ax.axvspan(sign * qxy_lo, sign * qxy_hi,
                           color=args.sector_line_color, alpha=0.30, lw=0)
            ax.tick_params(axis="both", which="both", direction="in")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.suptitle(f"{base}: {label}")
            fig.tight_layout()
            fig.savefig(overlay_path, dpi=args.dpi)
            plt.close(fig)

        print(f"  Saved box cut q_xy {qxy_lo:g}-{qxy_hi:g} -> {data_out_path}")
        _qstop, _why, _msg = gc.box_cut_report(
            res_I, res_qx, res_qy, along="qz", across_range=(qxy_lo, qxy_hi))
        if _why in ("missing_wedge", "empty"):
            print(f"    NOTE: {_msg}")

        if args.stitch_oop and oop_sector is not None:
            # One region rather than two cuts reconciled: the window's inner
            # edge stays at the beamstop while its outer edge opens with q,
            # so there is no scale factor between differing acceptances and
            # nothing to check agreement on.
            try:
                sq, si = gc.adaptive_cut(res_I, res_qx, res_qy,
                                          inner=qxy_lo, outer=qxy_hi,
                                          angle_deg=8.0)
                rep = {"joined": True, "overlap": None, "scale": None,
                       "shape_scatter": None, "n_overlap": 0,
                       "primary": "adaptive window",
                       "message": (
                           f"One adaptive window: |q_xy| from {qxy_lo:g} out to "
                           f"max({qxy_hi:g}, q_z*tan 8 deg). No join, so no "
                           f"scale factor and nothing to reconcile.")}
            except gc.GiwaxsError as exc:
                rep = {"joined": False, "overlap": None, "scale": None,
                       "shape_scatter": None, "n_overlap": 0,
                       "primary": "adaptive window", "message": str(exc)}
                sq = si = None
            if rep["joined"]:
                stag = f"oop_combined_qxy_{qxy_lo:g}_to_{qxy_hi:g}".replace(".", "p")
                slabel = (f"combined out-of-plane (sector + strip "
                          f"{qxy_lo:g}-{qxy_hi:g})")
                spath = os.path.join(out_dirs["linecuts"],
                                      f"{base}_lineprofile_{stag}.txt")
                np.savetxt(spath, np.c_[sq, si], header="Q(1/A)\tIntensity(a.u.)")
                gc.plot_1d_linecut(sq, si, os.path.join(
                    out_dirs["linecuts"], f"{base}_lineprofile_{stag}.png"),
                    (qxy_lo, qxy_hi), title=f"{base}: {slabel}",
                    line_color=args.line_color, font_family=args.font_family,
                    font_size=args.font_size, dpi=args.dpi,
                    q_range=args.linecut_q_range,
                    tick_spacing=args.linecut_tick_spacing,
                    subtick_spacing=args.linecut_subtick_spacing)
                diag_path = os.path.join(
                    out_dirs["linecuts"], f"{base}_lineprofile_{stag}_sources.png")
                gc.plot_stitch_diagnostic(
                    q, intensity, oop_sector[0], oop_sector[1], sq, si, rep,
                    switch_q=qxy_hi / np.tan(np.deg2rad(8.0)),
                    out_path=diag_path,
                    title=f"{base}: box cut vs angular sector (out-of-plane)",
                    font_family=args.font_family, font_size=args.font_size,
                    dpi=args.dpi)
                print(f"  Saved combined out-of-plane profile "
                      f"({sq.min():.3f}-{sq.max():.3f} 1/A) -> {spath}")
                print(f"  Saved the companion two-cut comparison -> {diag_path}")
                xlsx_path = os.path.join(
                    out_dirs["linecuts"], f"{base}_lineprofile_{stag}.xlsx")
                try:
                    gc.write_linecut_workbook(
                        xlsx_path, combined=(sq, si),
                        sources={"box cut (strip)": (q, intensity),
                                 "angular sector": (oop_sector[0], oop_sector[1])},
                        report=rep, source_file=os.path.basename(tiff_path))
                    print(f"  Saved the workbook -> {xlsx_path}")
                except gc.GiwaxsError as exc:
                    print(f"  No .xlsx written: {exc}")
                print(f"    {rep['message']}")
                if args.fit_regions:
                    fit_peaks_for_linecut(
                        sq, si, args, base, stag, sector_label=slabel,
                        out_dirs=out_dirs,
                        fit_rows=fit_rows if fit_rows is not None else [])
            else:
                print(f"  Combined out-of-plane profile NOT written. "
                      f"{rep['message']}")

        if args.fit_regions:
            fit_peaks_for_linecut(
                q, intensity, args, base, tag, sector_label=label,
                out_dirs=out_dirs, fit_rows=fit_rows if fit_rows is not None else [],
            )

    # --- In-plane box cuts: a profile along q_xy from a fixed q_z strip ----
    for qz_lo, qz_hi in (args.box_cuts_ip or []):
        q, intensity = gc.box_cut(res_I, res_qx, res_qy, along="qxy",
                                   across_range=(qz_lo, qz_hi))
        tag = f"boxcut_ip_qz_{qz_lo:g}_to_{qz_hi:g}".replace(".", "p")
        label = f"in-plane box cut, q_z {qz_lo:g}-{qz_hi:g} 1/A"

        data_out_path = os.path.join(out_dirs["linecuts"], f"{base}_lineprofile_{tag}.txt")
        np.savetxt(data_out_path, np.c_[q, intensity], header="Q(1/A)\tIntensity(a.u.)")

        plot_out_path = os.path.join(out_dirs["linecuts"], f"{base}_lineprofile_{tag}.png")
        gc.plot_1d_linecut(q, intensity, plot_out_path, (qz_lo, qz_hi),
                            title=f"{base}: {label}",
                            line_color=args.line_color, font_family=args.font_family,
                            font_size=args.font_size, dpi=args.dpi,
                            q_range=args.linecut_q_range,
                            tick_spacing=args.linecut_tick_spacing,
                            subtick_spacing=args.linecut_subtick_spacing)

        overlay_path = os.path.join(out_dirs["images"], f"{base}_sector_{tag}.png")
        xlabel, ylabel = gc.AXIS_LABELS.get(args.axis_labels, gc.AXIS_LABELS["ip_oop"])
        with gc.style_context(args.font_family, args.font_size):
            fig, ax = plt.subplots(figsize=gc.DEFAULT_FIGSIZE)
            v_lo, v_hi = gc.resolve_vmin_vmax(res_I, args.vmin_percentile, args.vmin,
                                               args.vmax, args.vmax_percentile,
                                               color_scale=args.color_scale)
            norm = (LogNorm(vmin=v_lo, vmax=v_hi) if args.color_scale == "log"
                     else Normalize(vmin=v_lo, vmax=v_hi))
            ax.pcolormesh(res_qx, res_qy, res_I, norm=norm, cmap=args.cmap)
            ax.set_facecolor("black")
            ax.set_aspect("equal")
            ax.set_xlim(args.qip_plot_range)
            ax.set_ylim(args.qoop_plot_range)
            for sign in (-1.0, 1.0):
                ax.axhspan(sign * qz_lo, sign * qz_hi,
                           color=args.sector_line_color, alpha=0.30, lw=0)
            ax.tick_params(axis="both", which="both", direction="in")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.suptitle(f"{base}: {label}")
            fig.tight_layout()
            fig.savefig(overlay_path, dpi=args.dpi)
            plt.close(fig)

        print(f"  Saved in-plane box cut q_z {qz_lo:g}-{qz_hi:g} -> {data_out_path}")
        _qstop, _why, _msg = gc.box_cut_report(
            res_I, res_qx, res_qy, along="qxy", across_range=(qz_lo, qz_hi))
        if _why in ("missing_wedge", "empty"):
            print(f"    NOTE: {_msg}")

        if args.fit_regions:
            fit_peaks_for_linecut(
                q, intensity, args, base, tag, sector_label=label,
                out_dirs=out_dirs, fit_rows=fit_rows if fit_rows is not None else [],
            )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None):
    """Thin wrapper around _main_impl that converts a GiwaxsError (raised
    by shared giwaxs_common.py logic for bad input/config) into a clean
    sys.exit(), exactly matching this script's pre-existing CLI error
    behaviour. This wrapper (rather than a bare `if __name__ == "__main__"`
    guard) is what actually matters here: giwaxs_platform.py calls this
    main() function directly, in-process, so the try/except needs to be
    inside main() itself to protect that call path too, not just direct
    command-line invocation.
    """
    try:
        _main_impl(argv)
    except gc.GiwaxsError as exc:
        sys.exit(str(exc))


def _main_impl(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    fabio, FiberIntegrator, get_unit_fiber, Detector, detector_factory = gc.import_pyfai_stack()

    input_path, output_dir = gc.resolve_input_output(args)
    tiff_files = gc.resolve_tiff_files(input_path)

    angle_map = gc.load_incident_angle_map(args.incident_angle_map) if args.incident_angle_map else None

    out_dirs = {
        "images": os.path.join(output_dir, "images"),
        "linecuts": os.path.join(output_dir, "linecuts"),
    }
    # Only created when peak fitting was actually requested, so a plain run
    # doesn't leave an empty directory behind.
    if args.fit_regions:
        out_dirs["peakfits"] = os.path.join(output_dir, "peakfits")
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # NOTE: --non-interactive only controls the extra-line-cut-sectors
    # prompt below; calibration prompting/confirmation is controlled
    # independently by --no-calibration-prompt, since AgBeh fit
    # confirmation is valuable even in an otherwise-scripted run.
    gc.prompt_for_calibration_setup(args)

    first_shape = fabio.open(tiff_files[0]).data.shape
    fi, detector = gc.build_fiber_integrator(
        args, Detector, detector_factory, FiberIntegrator,
        fabio=fabio, image_shape=first_shape)

    mask = gc.load_mask(args, fabio, first_shape)

    # Determine the full list of angular sectors up-front (defaults + CLI extras
    # + interactively-requested extras), so the prompt only happens once.
    cli_extra_ranges = args.extra_ranges or []
    all_ranges = [(-90, -80), (-8, 8)] + list(cli_extra_ranges)
    if not args.non_interactive:
        all_ranges += ask_for_extra_ranges()

    if args.autodefine_centre:
        first = tiff_files[0]
        first_angle = gc.resolve_incident_angle_for_file(
            first, args.incident_angle, args.incident_angle_from_filename,
            angle_map=angle_map)
        meas = gc.autodefine_beam_centre(
            fabio.open(first).data, first_angle, fi.dist, fi.detector.pixel1,
            centre_guess_col=fi.poni2 / fi.detector.pixel2)
        print(f"\nAuto-defining the beam centre from {os.path.basename(first)} "
              f"(incident angle {first_angle:g} deg):")
        for note in meas["notes"]:
            print(f"  {note}")
        if meas["col"] is not None:
            print(f"  Beam centre X: {fi.poni2 / fi.detector.pixel2:.2f} -> "
                  f"{meas['col']:.2f} px")
            fi.poni2 = meas["col"] * fi.detector.pixel2
        if meas["row"] is not None:
            print(f"  Beam centre Y: {fi.poni1 / fi.detector.pixel1:.2f} -> "
                  f"{meas['row']:.2f} px")
            fi.poni1 = meas["row"] * fi.detector.pixel1
        if meas["col"] is None and meas["row"] is None:
            print("  Nothing measurable in that frame -- geometry left unchanged.")
        print()

    fit_rows: List[dict] = []
    for tiff_path in tiff_files:
        process_file(
            tiff_path, fi, get_unit_fiber,
            mask, args, out_dirs, fabio, all_ranges,
            angle_map=angle_map, fit_rows=fit_rows,
        )

    if args.fit_regions:
        csv_path = os.path.join(output_dir, "peak_fit_results.csv")
        gc.write_peak_fit_csv(csv_path, fit_rows)
        n_ok = sum(1 for r in fit_rows if not r.get("error"))
        print(f"\nPeak fitting: {n_ok}/{len(fit_rows)} region fits succeeded "
              f"-> {csv_path}")

    print(f"\nDone. Outputs written to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
