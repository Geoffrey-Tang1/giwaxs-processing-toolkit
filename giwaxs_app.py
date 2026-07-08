#!/usr/bin/env python3
"""
giwaxs_app.py
==============

Interactive Streamlit app for the GIWAXS processing toolkit: upload a
TIFF (or several), set geometry/calibration parameters, and get a live
preview of the 2D image and cartesian pole figure -- with real widgets
(color pickers, dropdowns, sliders) for colormap, colour-scale range,
line colour, font family, and font size, all updating instantly without
re-running the (slower) pyFAI integration each time.

Optionally, describe the look you want in plain language (e.g. "make the
line red and increase the font size, use the plasma colormap") and have
Claude translate that into the actual widget values for you (requires
your own Anthropic API key).

Run with:
    streamlit run giwaxs_app.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

import giwaxs_common as gc


st.set_page_config(page_title="GIWAXS Processing Toolkit", layout="wide")


# --------------------------------------------------------------------------- #
# Session-state defaults (so widgets and the AI assistant can both set these
# without conflicting -- widgets are always created with key=... only, never
# both key= and value=, so whichever was last written to session_state wins)
# --------------------------------------------------------------------------- #
STYLE_DEFAULTS = {
    "cmap": "viridis",
    "use_manual_scale": False,
    "vmin": 100.0,
    "vmax": 100000.0,
    "vmin_percentile": 2.0,
    "vmax_percentile": 99.9,
    "line_color": "#1f77b4",
    "sector_line_color": "#00ffff",
    "font_family": "DejaVu Sans",
    "font_size": 11.0,
    "dpi": 400,
    "axis_labels": "xyz",
}
# Each tab gets its OWN independent copy of every style setting (keys
# prefixed "2d_"/"pf_") -- style_widgets() is defined once but instantiated
# inside BOTH tabs, and Streamlit runs the code in every tab on every
# script run (not just the visible one), so two widgets sharing one
# unprefixed key would collide (StreamlitDuplicateElementKey).
for prefix in ("2d_", "pf_"):
    for k, v in STYLE_DEFAULTS.items():
        st.session_state.setdefault(prefix + k, v)
st.session_state.setdefault("processed_2d", None)   # cached heavy-computation results
st.session_state.setdefault("processed_pf", None)
st.session_state.setdefault("calibration_confirmed", False)
st.session_state.setdefault("calibration_diagnostic_path", None)

# Apply any calibration result from the PREVIOUS run now, before the
# Geometry section's widgets (beam_center_y/x, distance, rot1-3) are
# instantiated below -- Streamlit forbids setting a widget's
# session_state value after that widget has already rendered in the
# current run, so the "Calibrate now" button (see the Calibration
# section further down) can't update these directly; instead it stashes
# the new values here and triggers a rerun, and THIS block is what
# actually applies them, at a point in the script that's safely before
# those widgets exist yet.
_pending_calib = st.session_state.pop("_pending_calibration_update", None)
if _pending_calib:
    for _k, _v in _pending_calib.items():
        st.session_state[_k] = _v


# --------------------------------------------------------------------------- #
# AI style assistant (optional -- needs an Anthropic API key)
# --------------------------------------------------------------------------- #
AI_SYSTEM_PROMPT = """You translate a plot-styling request into JSON parameters.
Return ONLY a JSON object (no prose, no markdown fences) with any of these
keys you can confidently infer, omitting any you cannot:
- "cmap": a valid matplotlib colormap name (e.g. viridis, plasma, inferno, magma, cividis, turbo, jet, gray, hot, coolwarm)
- "line_color": a matplotlib color spec -- a CSS/X11 name (e.g. "red", "darkorange") or hex code (e.g. "#ff0000")
- "sector_line_color": same format as line_color, for the sector overlay lines on the 2D image
- "font_family": one of "DejaVu Sans", "Arial", "Times New Roman", "serif", "sans-serif", "monospace"
- "font_size": a number between 6 and 30
- "vmin": a positive number (colour-scale minimum, only if the user gave/implied a specific intensity value)
- "vmax": a positive number (colour-scale maximum, only if the user gave/implied a specific intensity value)
Example: user says "make the line red and bump up the font size" ->
{"line_color": "red", "font_size": 16}
"""


def call_ai_style_assistant(request_text: str, api_key: str) -> dict:
    """Call the Anthropic API to translate a natural-language styling
    request into concrete parameter values. Raises on any failure --
    caller should catch and show a friendly error."""
    import anthropic  # imported lazily so the app works without it installed
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=AI_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": request_text}],
    )
    text = "".join(block.text for block in response.content if hasattr(block, "text"))
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def render_ai_assistant():
    with st.expander("✨ AI style assistant (optional)"):
        st.caption(
            "Describe the look you want in plain language, and Claude will "
            "set the widgets below for you. Requires your own Anthropic API "
            "key (never stored or sent anywhere except api.anthropic.com)."
        )
        api_key = st.text_input(
            "Anthropic API key", type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            key="anthropic_api_key",
        )
        request_text = st.text_input(
            "Styling request",
            placeholder="e.g. 'use a warm colormap, red line, bigger font'",
            key="ai_style_request",
        )
        if st.button("Apply with AI", key="ai_apply_button"):
            if not api_key:
                st.error("Please enter your Anthropic API key first.")
            elif not request_text:
                st.error("Please describe what you'd like changed.")
            else:
                try:
                    with st.spinner("Asking Claude..."):
                        result = call_ai_style_assistant(request_text, api_key)
                except ImportError:
                    st.error("The 'anthropic' package isn't installed. "
                             "Run: pip install anthropic")
                except Exception as exc:  # noqa: BLE001 -- surfaced to the user directly
                    st.error(f"AI request failed: {exc}")
                else:
                    applied = []
                    for key in ("cmap", "line_color", "sector_line_color",
                                "font_family", "font_size", "vmin", "vmax"):
                        if key in result:
                            for prefix in ("2d_", "pf_"):
                                st.session_state[prefix + key] = result[key]
                            applied.append(f"{key} = {result[key]}")
                    if applied:
                        if "vmin" in result or "vmax" in result:
                            for prefix in ("2d_", "pf_"):
                                st.session_state[prefix + "use_manual_scale"] = True
                        st.success("Applied to both tabs: " + ", ".join(applied))
                        st.rerun()
                    else:
                        st.warning("Claude didn't return any recognized style keys.")


# --------------------------------------------------------------------------- #
# Sidebar: input files + geometry + calibration
# --------------------------------------------------------------------------- #
def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".tif"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


# --------------------------------------------------------------------------- #
# Config-file loading (prefills the sidebar so you don't have to type every
# field by hand -- see docs/EXAMPLE_INPUTS.md / example_config.json)
# --------------------------------------------------------------------------- #
GEOMETRY_DEFAULTS = {
    "beam_center_y": 145.0,
    "beam_center_x": 1088.0,
    "distance": 0.65,
    "wavelength": 1.5406e-10,
    "energy": None,
    "rot1": 0.0, "rot2": 0.0, "rot3": 0.0,
    "detector_name": "Pilatus2M",
    "pixel_size": 172e-6,
    "detector_shape": [1679, 1475],
    "incident_angle": 0.1,
    "npt": 1000,
}
for k, v in GEOMETRY_DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("use_named_detector", True)
st.session_state.setdefault("wavelength_mode", "Wavelength (m)")


def load_config_into_session(cfg: dict):
    for key in ("beam_center_y", "beam_center_x", "distance", "rot1", "rot2",
                "rot3", "incident_angle", "npt"):
        if key in cfg and cfg[key] is not None:
            st.session_state[key] = cfg[key]
    if cfg.get("wavelength") is not None:
        st.session_state["wavelength"] = cfg["wavelength"]
        st.session_state["wavelength_mode"] = "Wavelength (m)"
    elif cfg.get("energy") is not None:
        st.session_state["energy"] = cfg["energy"]
        st.session_state["wavelength_mode"] = "Energy (keV)"
    if cfg.get("detector_name"):
        st.session_state["detector_name"] = cfg["detector_name"]
        st.session_state["use_named_detector"] = True
    elif cfg.get("detector_shape"):
        st.session_state["use_named_detector"] = False
        st.session_state["pixel_size"] = cfg.get("pixel_size", 172e-6)
        st.session_state["detector_shape"] = cfg["detector_shape"]


with st.sidebar:
    st.header("0. Load example / saved config (optional)")
    st.caption(
        "Not sure what to type in the fields below? Upload a config JSON "
        "(see example_config.json in the repo) to fill everything in "
        "automatically -- you can still edit any field afterward."
    )
    config_upload = st.file_uploader("Config JSON", type=["json"], key="config_upload")
    if config_upload is not None and st.session_state.get("_loaded_config_name") != config_upload.name:
        try:
            loaded_cfg = json.loads(config_upload.getvalue())
            load_config_into_session(loaded_cfg)
            st.session_state["_loaded_config_name"] = config_upload.name
            st.success(f"Loaded {config_upload.name} -- fields below are now filled in.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not read config file: {exc}")

    st.header("1. Input")
    uploaded_files = st.file_uploader(
        "GIWAXS TIFF file(s)", type=["tif", "tiff"], accept_multiple_files=True
    )

    st.header("2. Geometry")
    use_poni_file = st.checkbox(
        "Load geometry from an existing .poni file", key="use_poni_file",
        help="Loads beam centre, distance, rotations, wavelength, AND the "
             "detector (including its shape and pixel size) all at once -- "
             "recommended if you already have an accurate calibration.",
    )
    poni_upload = None
    if use_poni_file:
        poni_upload = st.file_uploader("PONI file", type=["poni"], key="poni_upload")
        st.caption(
            "Beam centre, distance, rotations, wavelength, and the detector "
            "(including its shape) are all loaded from this file -- nothing "
            "else below is needed for geometry. You can still refine it "
            "further in the Calibration section below if you also have an "
            "AgBeh image."
        )
        beam_center_y = beam_center_x = distance = None
        wavelength = energy = None
        rot1 = rot2 = rot3 = 0.0
        detector_name = pixel_size = detector_shape = None
    else:
        beam_center_y = st.number_input("Beam centre Y (px)", format="%.4f", key="beam_center_y")
        beam_center_x = st.number_input("Beam centre X (px)", format="%.4f", key="beam_center_x")
        distance = st.number_input("Sample-detector distance (m)", format="%.6f", key="distance")

        wavelength_mode = st.radio("Specify beam energy as", ["Wavelength (m)", "Energy (keV)"],
                                    horizontal=True, key="wavelength_mode")
        if wavelength_mode == "Wavelength (m)":
            wavelength = st.number_input("Wavelength (m)", format="%.6e", key="wavelength")
            energy = None
        else:
            energy = st.number_input("Energy (keV)", format="%.4f", key="energy")
            wavelength = None

        use_named_detector = st.checkbox("Use a named pyFAI detector", key="use_named_detector")
        if use_named_detector:
            detector_name = st.text_input("Detector name", key="detector_name")
            pixel_size, detector_shape = None, None
        else:
            detector_name = None
            pixel_size = st.number_input("Pixel size (m)", format="%.6e", key="pixel_size")
            shape_h = st.number_input("Detector height (px)", step=1,
                                       value=int(st.session_state["detector_shape"][0]))
            shape_w = st.number_input("Detector width (px)", step=1,
                                       value=int(st.session_state["detector_shape"][1]))
            detector_shape = (int(shape_h), int(shape_w))

        with st.expander("Detector rotations (advanced -- almost never needed)"):
            rot1 = st.number_input("rot1 (rad)", format="%.6f", key="rot1")
            rot2 = st.number_input("rot2 (rad)", format="%.6f", key="rot2")
            rot3 = st.number_input("rot3 (rad)", format="%.6f", key="rot3")

    st.header("3. Calibration (optional)")
    use_calibration = st.checkbox("Refine geometry with an AgBeh (or other calibrant) image",
                                   key="use_calibration")
    agbeh_upload = None
    if use_calibration:
        agbeh_upload = st.file_uploader("Calibration image", type=["tif", "tiff"], key="agbeh_upload")
        calibrant_name = st.text_input("Calibrant name", value="AgBh", key="calibrant_name")
        calib_max_rings = st.number_input("Max rings to fit", min_value=1, max_value=20, value=5,
                                           key="calib_max_rings")
        calib_min_intensity = st.number_input(
            "Min peak intensity (Imin)", value=200.0, key="calib_min_intensity",
            help="Ring-detection intensity threshold: pixels below this are "
                 "treated as background noise and ignored. Too low picks up "
                 "noise as false ring points; too high can miss real but "
                 "weaker (often higher-order) rings.",
        )
        st.caption(
            "This runs as its OWN step, separate from processing your data below -- "
            "click Calibrate, check the fit, and only proceed once you're happy with it."
        )

        if st.button("🔧 Calibrate now", type="primary", key="calibrate_button"):
            if agbeh_upload is None:
                st.error("Please upload a calibration image first.")
            else:
                try:
                    fabio, _, _, Detector, detector_factory = gc.import_pyfai_stack()
                    if use_poni_file:
                        if poni_upload is None:
                            raise ValueError("Please upload a .poni file above first.")
                        loaded = gc.load_poni_file(save_upload_to_temp(poni_upload))
                        detector = loaded["detector"]
                        wl = loaded["wavelength"]
                        guess_poni1, guess_poni2 = loaded["poni1"], loaded["poni2"]
                        guess_dist = loaded["dist"]
                        guess_r1, guess_r2 = loaded["rot1"], loaded["rot2"]
                        guess_r3 = loaded["rot3"]
                    else:
                        class Args:
                            pass
                        cargs = Args()
                        cargs.detector_name = detector_name
                        cargs.pixel_size = pixel_size
                        cargs.detector_shape = detector_shape
                        cargs.wavelength = wavelength
                        cargs.energy = energy
                        detector = gc.build_detector(cargs, Detector, detector_factory)
                        wl = gc.resolve_wavelength(cargs)
                        guess_poni1 = beam_center_y * detector.pixel1
                        guess_poni2 = beam_center_x * detector.pixel2
                        guess_dist = distance
                        guess_r1, guess_r2, guess_r3 = rot1, rot2, rot3

                    calib_path = save_upload_to_temp(agbeh_upload)
                    diagnostic_path = os.path.join(tempfile.mkdtemp(), "calibration_fit_check.png")
                    result = gc.run_agbeh_calibration(
                        calib_path, detector, wl, guess_dist, guess_poni1, guess_poni2,
                        guess_r1, guess_r2, guess_r3, calibrant_name, int(calib_max_rings),
                        calib_min_intensity, fabio, diagnostic_path=diagnostic_path,
                    )
                    # NOTE: can't directly assign st.session_state["beam_center_y"]
                    # etc. here -- those widgets already rendered earlier in
                    # THIS run (Geometry is section 2, above Calibration's
                    # section 3), and Streamlit forbids modifying a widget's
                    # session_state value after it's been instantiated in the
                    # same run. Instead, stash the new values under a
                    # differently-named key and apply them at the very top of
                    # the script on the NEXT run (before those widgets are
                    # instantiated again) -- see the "pending calibration
                    # update" block near the top of this file.
                    st.session_state["_pending_calibration_update"] = {
                        "beam_center_y": result["poni1"] / detector.pixel1,
                        "beam_center_x": result["poni2"] / detector.pixel2,
                        "distance": result["dist"],
                        "rot1": result["rot1"],
                        "rot2": result["rot2"],
                        "rot3": result["rot3"],
                    }
                    st.session_state["calibration_confirmed"] = True
                    st.session_state["calibration_diagnostic_path"] = result.get("diagnostic_path")
                    st.session_state["calibration_chi2"] = (result["init_chi2"], result["final_chi2"])
                    st.session_state["calibration_n_points"] = result["n_control_points"]
                    st.rerun()
                except Exception as exc:
                    st.error(f"Calibration failed: {exc}")

        if st.session_state.get("calibration_diagnostic_path"):
            chi2_before, chi2_after = st.session_state["calibration_chi2"]
            st.image(
                st.session_state["calibration_diagnostic_path"],
                caption=f"Fit check ({st.session_state['calibration_n_points']} ring points, "
                        f"chi2: {chi2_before:.4g} -> {chi2_after:.4g}). Dots = detected ring "
                        f"points, green lines = fitted rings -- should overlap closely.",
                use_container_width=True,
            )
            if st.session_state.get("calibration_confirmed"):
                st.success(
                    "Calibration applied -- Beam centre/Distance above now show the "
                    "refined values. If the fit doesn't actually look right, adjust "
                    "the guess and click Calibrate again; otherwise proceed to "
                    "process your data below."
                )
    else:
        st.session_state["calibration_confirmed"] = False

    incident_angle = st.number_input("Incident angle (deg)", format="%.4f", key="incident_angle")
    incident_angle_from_filename = st.checkbox(
        "Auto-detect each file's incident angle from its filename "
        "(the '0p095'-style convention, e.g. 'sample_0p095_1234.tif' -> "
        "0.095 deg) -- falls back to the value above if no pattern is found",
        key="incident_angle_from_filename",
    )

    mask_upload = st.file_uploader("Mask file (optional)", type=["tif", "tiff", "npy"], key="mask_upload")

    with st.expander("Advanced options (the defaults are almost always fine)"):
        npt = st.number_input(
            "Integration bins (npt)", step=100, key="npt",
            help="Resolution of the re-gridded q-space image/profiles. "
                 "Higher = finer but slower. You usually don't need to touch this.",
        )


def build_geometry():
    """Build the FiberIntegrator (geometry only -- independent of incident
    angle) from the current sidebar widgets.
    Returns (fi, get_unit_fiber, mask, fabio, error). Grazing-incidence
    units (which DO depend on incident angle) are built separately, per
    file, via units_for_file() below -- since different files in a batch
    may need different incident angles.
    """
    fabio, FiberIntegrator, get_unit_fiber, Detector, detector_factory = gc.import_pyfai_stack()

    if use_calibration and not st.session_state.get("calibration_confirmed"):
        return None, None, None, fabio, (
            "You've enabled AgBeh calibration but haven't run it yet -- "
            "click 'Calibrate now' in the sidebar and check the fit before "
            "processing your data (otherwise you'd be processing with an "
            "un-refined initial guess)."
        )

    try:
        if use_poni_file:
            if poni_upload is None:
                return None, None, None, fabio, "Please upload a .poni file."
            poni_path = save_upload_to_temp(poni_upload)
            loaded = gc.load_poni_file(poni_path)
            detector = loaded["detector"]
            wl = loaded["wavelength"]
            dist = loaded["dist"]
            poni1 = loaded["poni1"]
            poni2 = loaded["poni2"]
            r1, r2, r3 = loaded["rot1"], loaded["rot2"], loaded["rot3"]
            st.sidebar.success(
                f"Loaded from .poni: beam centre = "
                f"({poni1/detector.pixel1:.2f}, {poni2/detector.pixel2:.2f}) px, "
                f"distance = {dist:.6f} m, detector shape = {detector.max_shape}"
            )
        else:
            class Args:
                pass
            args = Args()
            args.beam_center_y = beam_center_y
            args.beam_center_x = beam_center_x
            args.distance = distance
            args.wavelength = wavelength
            args.energy = energy
            args.rot1, args.rot2, args.rot3 = rot1, rot2, rot3
            args.detector_name = detector_name
            args.pixel_size = pixel_size
            args.detector_shape = detector_shape

            detector = gc.build_detector(args, Detector, detector_factory)
            wl = gc.resolve_wavelength(args)

            poni1 = beam_center_y * detector.pixel1
            poni2 = beam_center_x * detector.pixel2
            dist = distance
            r1, r2, r3 = rot1, rot2, rot3

        # NOTE: calibration is a separate, explicit step now (the "Calibrate
        # now" button above) -- Process never re-runs it. Whatever
        # beam_center_y/x, distance, rot1-3 are currently set to (whether
        # typed manually, loaded from a .poni, or refined via Calibrate)
        # are used directly as-is.

        fi = FiberIntegrator(dist=dist, poni1=poni1, poni2=poni2,
                              rot1=r1, rot2=r2, rot3=r3, wavelength=wl, detector=detector)

        mask = None
        if mask_upload is not None:
            mask_path = save_upload_to_temp(mask_upload)

            class MaskArgs:
                pass
            margs = MaskArgs()
            margs.mask = mask_path
            # shape resolved once a file is loaded, in the caller
            mask = margs

        return fi, get_unit_fiber, mask, fabio, None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, fabio, str(exc)


def units_for_file(get_unit_fiber, filename: str, verbose: bool = False):
    """Resolve the incident angle for this specific file (auto-detected
    from its filename if the sidebar checkbox is on, else the sidebar's
    fixed value) and build the grazing-incidence units for it."""
    angle_deg = gc.resolve_incident_angle_for_file(
        filename, incident_angle, incident_angle_from_filename, verbose=verbose
    )
    return gc.build_grazing_units(get_unit_fiber, angle_deg), angle_deg


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.title("GIWAXS Processing Toolkit")
render_ai_assistant()

tab_2d, tab_pf = st.tabs(["2D image + line cuts", "Pole figure (cartesian)"])

# --------------------------------------------------------------------------- #
# Shared style widgets (used by both tabs where relevant)
# --------------------------------------------------------------------------- #
def style_widgets(show_cmap: bool, show_sector_color: bool, key_prefix: str):
    p = key_prefix  # short alias, this function's keys get VERY repetitive otherwise
    cols = st.columns(4)
    with cols[0]:
        if show_cmap:
            category = st.selectbox("Colormap category", list(gc.COLORMAP_CATEGORIES.keys()),
                                     key=f"{p}_cmap_category")
            options = gc.COLORMAP_CATEGORIES[category]
            if st.session_state[f"{p}_cmap"] not in options:
                st.session_state[f"{p}_cmap"] = options[0]
            st.selectbox("Colormap", options, key=f"{p}_cmap")
    with cols[1]:
        st.color_picker("Line colour", key=f"{p}_line_color")
    with cols[2]:
        font_category = st.selectbox("Font category", list(gc.FONT_CATEGORIES.keys()),
                                      key=f"{p}_font_category")
        font_options = gc.FONT_CATEGORIES[font_category]
        if st.session_state[f"{p}_font_family"] not in font_options:
            st.session_state[f"{p}_font_family"] = font_options[0]
        st.selectbox("Font family", font_options, key=f"{p}_font_family")
    with cols[3]:
        preset_label = st.selectbox("Font size", list(gc.FONT_SIZE_PRESETS.keys()),
                                     key=f"{p}_font_size_preset")
        preset_value = gc.FONT_SIZE_PRESETS[preset_label]
        if preset_value is None:
            st.number_input("Custom size (pt)", min_value=4.0, max_value=48.0,
                             key=f"{p}_font_size", format="%.1f")
        else:
            st.session_state[f"{p}_font_size"] = preset_value
            st.caption(f"{preset_value:.0f}pt")

    cols2 = st.columns(3)
    with cols2[0]:
        if show_sector_color:
            st.color_picker("Sector line colour", key=f"{p}_sector_line_color")
    with cols2[1]:
        st.slider("Output resolution (DPI)", 72, 600, key=f"{p}_dpi", step=1)
    with cols2[2]:
        if show_cmap:  # axis labels only meaningful for the 2D q-space image
            st.selectbox(
                "Axis labels", ["ip_oop", "xyz"], key=f"{p}_axis_labels",
                format_func=lambda v: "q_ip / q_oop" if v == "ip_oop" else "q_xy / q_z",
            )

    st.checkbox("Set explicit colour-scale range (instead of automatic percentile)",
                key=f"{p}_use_manual_scale")
    if st.session_state[f"{p}_use_manual_scale"]:
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Colour-scale min", key=f"{p}_vmin", format="%.4g")
        with c2:
            st.number_input("Colour-scale max", key=f"{p}_vmax", format="%.4g")
    else:
        pc1, pc2 = st.columns(2)
        with pc1:
            st.slider("Colour-scale minimum percentile", 0.0, 10.0, key=f"{p}_vmin_percentile")
        with pc2:
            st.slider("Colour-scale maximum percentile", 90.0, 100.0, key=f"{p}_vmax_percentile")


# --------------------------------------------------------------------------- #
# Tab 1: 2D image + line cuts
# --------------------------------------------------------------------------- #
with tab_2d:
    st.subheader("Style")
    style_widgets(show_cmap=True, show_sector_color=True, key_prefix="2d")

    st.subheader("Line-cut sectors")
    st.caption("Defaults: in-plane (-90,-80) deg, out-of-plane (-8,8) deg.")
    extra_sector_text = st.text_input(
        "Extra sectors (comma-separated 'start:end' pairs, e.g. '-55:-45, 30:40')",
        key="extra_sectors_2d",
    )

    qip_range = st.slider("q_ip plot range (1/Å)", -3.0, 3.0, (-0.5, 2.4), key="qip_range")
    qoop_range = st.slider("q_oop plot range (1/Å)", -1.0, 4.0, (-0.25, 2.75), key="qoop_range")

    if st.button("Process 2D image + line cuts", type="primary") and uploaded_files:
        fi, get_unit_fiber, mask_args, fabio, err = build_geometry()
        if err:
            st.error(f"Geometry error: {err}")
        else:
            results = []
            for uf in uploaded_files:
                tmp_path = save_upload_to_temp(uf)
                img = fabio.open(tmp_path).data
                mask = np.zeros(img.shape, dtype=bool)
                if mask_args is not None:
                    mask = gc.load_mask(mask_args, fabio, img.shape)

                (unit_ip, unit_oop, unit_chi, unit_qtot), angle_deg = units_for_file(
                    get_unit_fiber, uf.name, verbose=False
                )
                if incident_angle_from_filename:
                    st.caption(f"{uf.name}: using incident angle = {angle_deg} deg")

                res2d = fi.integrate2d_grazing_incidence(
                    img, npt_ip=int(npt), npt_oop=int(npt),
                    unit_ip=unit_ip, unit_oop=unit_oop, mask=mask,
                )
                res_I, res_qx, res_qy = res2d[0:3]
                res_qx = -np.flip(res_qx)
                res_I = np.flip(res_I, axis=1)

                sectors = [(-90, -80), (-8, 8)]
                for pair in extra_sector_text.split(","):
                    pair = pair.strip()
                    if not pair:
                        continue
                    try:
                        a, b = [float(v) for v in pair.split(":")]
                        sectors.append((a, b))
                    except ValueError:
                        st.warning(f"Could not parse sector '{pair}', skipping.")

                linecuts = []
                incident_angle_rad = np.deg2rad(angle_deg)
                for angles in sectors:
                    q, intensity = fi.integrate1d_grazing_incidence(
                        data=img, incident_angle=incident_angle_rad,
                        unit_ip=unit_chi, unit_oop=unit_qtot,
                        npt_oop=int(npt), npt_ip=int(npt),
                        ip_range=angles, mask=mask,
                    )
                    linecuts.append((angles, q, intensity))

                results.append({
                    "name": os.path.splitext(uf.name)[0],
                    "res_I": res_I, "res_qx": res_qx, "res_qy": res_qy,
                    "linecuts": linecuts,
                })
            st.session_state["processed_2d"] = results

    if st.session_state["processed_2d"]:
        for res in st.session_state["processed_2d"]:
            st.markdown(f"#### {res['name']}")
            fig2d = gc.plot_2d_image(
                res["res_qx"], res["res_qy"], res["res_I"],
                out_path=None, qlim_x=qip_range, qlim_y=qoop_range,
                vmin_percentile=st.session_state["2d_vmin_percentile"],
                vmax_percentile=st.session_state.get("2d_vmax_percentile", 99.9),
                cmap=st.session_state["2d_cmap"],
                vmin=st.session_state["2d_vmin"] if st.session_state["2d_use_manual_scale"] else None,
                vmax=st.session_state["2d_vmax"] if st.session_state["2d_use_manual_scale"] else None,
                font_family=st.session_state["2d_font_family"],
                font_size=st.session_state["2d_font_size"],
                axis_label_style=st.session_state["2d_axis_labels"],
            )
            c1, c2 = st.columns([2, 1])
            with c1:
                st.pyplot(fig2d)
            buf = io.BytesIO()
            fig2d.savefig(buf, format="png", dpi=st.session_state["2d_dpi"])
            import matplotlib.pyplot as plt
            plt.close(fig2d)
            c2.download_button("Download 2D image PNG", buf.getvalue(),
                                file_name=f"{res['name']}_2D_GIWAXS.png", mime="image/png",
                                key=f"dl2d_{res['name']}")

            for angles, q, intensity in res["linecuts"]:
                fig1d = gc.plot_1d_linecut(
                    q, intensity, out_path=None, angle_range=angles,
                    title=f"{res['name']}: {angles} deg",
                    line_color=st.session_state["2d_line_color"],
                    font_family=st.session_state["2d_font_family"],
                    font_size=st.session_state["2d_font_size"],
                )
                lc1, lc2 = st.columns([2, 1])
                with lc1:
                    st.pyplot(fig1d)
                buf2 = io.BytesIO()
                fig1d.savefig(buf2, format="png", dpi=st.session_state["2d_dpi"])
                plt.close(fig1d)
                tag = f"{angles[0]}_{angles[1]}".replace("-", "m").replace(".", "p")
                lc2.download_button(
                    f"Download line cut {angles} PNG", buf2.getvalue(),
                    file_name=f"{res['name']}_lineprofile_{tag}.png", mime="image/png",
                    key=f"dl1d_{res['name']}_{tag}",
                )

                linecut_df = pd.DataFrame({"Q (1/A)": q, "Intensity (a.u.)": intensity})

                txt_buf = io.StringIO()
                np.savetxt(txt_buf, np.c_[q, intensity], header="Q(1/A)\tIntensity(a.u.)")
                lc2.download_button(
                    f"Download line cut {angles} data (.txt)", txt_buf.getvalue(),
                    file_name=f"{res['name']}_lineprofile_{tag}.txt", mime="text/plain",
                    key=f"dltxt_{res['name']}_{tag}",
                )
                lc2.download_button(
                    f"Download line cut {angles} data (.csv)",
                    linecut_df.to_csv(index=False),
                    file_name=f"{res['name']}_lineprofile_{tag}.csv", mime="text/csv",
                    key=f"dlcsv_{res['name']}_{tag}",
                )

                with st.expander(f"View data table -- {res['name']}: {angles} deg"):
                    st.dataframe(linecut_df, use_container_width=True, height=250)

# --------------------------------------------------------------------------- #
# Tab 2: pole figure (cartesian only)
# --------------------------------------------------------------------------- #
with tab_pf:
    st.subheader("Style")
    style_widgets(show_cmap=False, show_sector_color=False, key_prefix="pf")

    st.subheader("Target q value(s) per file")
    st.caption(
        "Different files can each get their own reflection(s) -- edit the "
        "'target q' column below (space/comma-separated for multiple "
        "reflections per file, e.g. '1.673, 0.252'), or upload a filled-in "
        "mapping to pre-fill it automatically."
    )
    qmap_upload = st.file_uploader(
        "Optional: upload a mapping file to pre-fill the table (.json, .csv, or .xlsx)",
        type=["json", "csv", "xlsx", "xls"], key="qmap_upload",
    )
    if qmap_upload is not None and st.session_state.get("_qmap_upload_name") != qmap_upload.name:
        try:
            qmap_path = save_upload_to_temp(qmap_upload)
            loaded_map = gc.load_pole_figure_q_map(qmap_path)
            st.session_state["q_table_rows"] = [
                {"filename": fname, "target_q": " ".join(str(v) for v in qvals)}
                for fname, qvals in loaded_map.items()
            ]
            st.session_state["_qmap_upload_name"] = qmap_upload.name
            st.success(f"Loaded {len(loaded_map)} file(s) from {qmap_upload.name} into the table below.")
            st.rerun()
        except gc.GiwaxsError as exc:
            st.error(f"Could not read mapping file: {exc}")

    if uploaded_files:

        # Build/refresh the table to match the currently uploaded files,
        # preserving any values already typed for filenames still present.
        prior = {row["filename"]: row["target_q"]
                 for row in st.session_state.get("q_table_rows", [])}
        table_rows = [
            {"filename": uf.name, "target_q": prior.get(uf.name, "1.673")}
            for uf in uploaded_files
        ]
        edited = st.data_editor(
            pd.DataFrame(table_rows),
            key="q_table_editor",
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "filename": st.column_config.TextColumn("Filename", disabled=True),
                "target_q": st.column_config.TextColumn(
                    "Target q (1/Å)", help="Space/comma-separated for multiple reflections"
                ),
            },
            hide_index=True,
        )
        st.session_state["q_table_rows"] = edited.to_dict("records")

        with st.expander("Fill every row at once (optional)"):
            bulk_q = st.text_input("Value to apply to all files", value="1.673", key="bulk_q_value")
            if st.button("Apply to all rows"):
                st.session_state["q_table_rows"] = [
                    {"filename": r["filename"], "target_q": bulk_q} for r in table_rows
                ]
                st.rerun()
    else:
        st.info("Upload file(s) in the sidebar to fill in this table.")

    dq = st.number_input("Q window half-width (dq, 1/Å)", value=0.05, format="%.4f")
    chi_plot_range = st.slider("Chi plot range (deg)", -180.0, 180.0, (-90.0, 90.0), key="chi_plot_range_pf")
    compute_herman = st.checkbox("Compute Herman's orientation factor S", value=True)
    chi_max = st.number_input("Chi max for Herman's S (deg)", value=90.0)

    if st.button("Process pole figure(s)", type="primary") and uploaded_files:
        # Parse each file's own q value(s) from the table.
        q_per_file = {}
        parse_ok = True
        for row in st.session_state.get("q_table_rows", []):
            try:
                q_per_file[row["filename"]] = [
                    float(v) for v in str(row["target_q"]).replace(",", " ").split()
                ]
            except ValueError:
                st.error(f"Could not parse target q value(s) for '{row['filename']}': "
                         f"'{row['target_q']}'")
                parse_ok = False

        if parse_ok:
            fi, get_unit_fiber, mask_args, fabio, err = build_geometry()
            if err:
                st.error(f"Geometry error: {err}")
            else:
                results = []
                for uf in uploaded_files:
                    target_qs = q_per_file.get(uf.name, [])
                    if not target_qs:
                        st.warning(f"Skipping {uf.name}: no target q value(s) given.")
                        continue

                    tmp_path = save_upload_to_temp(uf)
                    img = fabio.open(tmp_path).data
                    mask = np.zeros(img.shape, dtype=bool)
                    if mask_args is not None:
                        mask = gc.load_mask(mask_args, fabio, img.shape)

                    (_, _, unit_chi, unit_qtot), angle_deg = units_for_file(
                        get_unit_fiber, uf.name, verbose=False
                    )
                    if incident_angle_from_filename:
                        st.caption(f"{uf.name}: using incident angle = {angle_deg} deg")

                    per_q = []
                    for target_q in target_qs:
                        try:
                            chi_axis, profile = gc.compute_chi_profile_at_q(
                                fi, img, mask, target_q, dq, int(npt), unit_chi, unit_qtot,
                            )
                        except ValueError as exc:
                            st.warning(f"{uf.name} @ q={target_q}: {exc}")
                            continue
                        herman_s = None
                        if compute_herman:
                            try:
                                herman_s, mean_cos2, coverage = gc.compute_herman_orientation(
                                    chi_axis, profile, chi_max=chi_max
                                )
                                if coverage < 0.5:
                                    st.warning(
                                        f"{uf.name} @ q={target_q}: low angular coverage "
                                        f"({coverage*100:.0f}%) -- treat S with caution."
                                    )
                            except ValueError as exc:
                                st.warning(f"Could not compute Herman's S: {exc}")
                        per_q.append((target_q, chi_axis, profile, herman_s))
                    results.append({"name": os.path.splitext(uf.name)[0], "per_q": per_q})
                st.session_state["processed_pf"] = results

    if st.session_state["processed_pf"]:
        import matplotlib.pyplot as plt
        for res in st.session_state["processed_pf"]:
            st.markdown(f"#### {res['name']}")
            for target_q, chi_axis, profile, herman_s in res["per_q"]:
                fig = gc.plot_chi_intensity_profile(
                    chi_axis, profile, out_path=None, target_q=target_q, dq=dq,
                    title=res["name"], herman_s=herman_s, chi_range=chi_plot_range,
                    line_color=st.session_state["pf_line_color"],
                    font_family=st.session_state["pf_font_family"],
                    font_size=st.session_state["pf_font_size"],
                )
                pc1, pc2 = st.columns([2, 1])
                with pc1:
                    st.pyplot(fig)
                if herman_s is not None:
                    pc2.metric("Herman's S", f"{herman_s:.3f}")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=st.session_state["pf_dpi"])
                plt.close(fig)
                q_tag = f"{target_q:.3f}".replace(".", "p")
                pc2.download_button(
                    "Download PNG", buf.getvalue(),
                    file_name=f"{res['name']}_polefigure_q{q_tag}.png", mime="image/png",
                    key=f"dlpf_{res['name']}_{q_tag}",
                )
                txt_buf = io.StringIO()
                header = "Chi(deg)\tIntensity(a.u.)"
                if herman_s is not None:
                    header = f"Herman_S={herman_s:.6f}\n{header}"
                np.savetxt(txt_buf, np.c_[chi_axis, profile], header=header)
                pc2.download_button(
                    "Download data (.txt)", txt_buf.getvalue(),
                    file_name=f"{res['name']}_polefigure_q{q_tag}_chi_profile.txt", mime="text/plain",
                    key=f"dlpftxt_{res['name']}_{q_tag}",
                )
