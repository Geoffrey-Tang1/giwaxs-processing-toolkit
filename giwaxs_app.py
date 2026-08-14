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
import zipfile
import json
import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import giwaxs_common as gc  # noqa: E402 -- sets the Agg backend before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402 -- must come after giwaxs_common


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
    "edge_top": "", "edge_bottom": "", "edge_left": "", "edge_right": "",
}
# Each tab gets its OWN independent copy of every style setting (keys
# prefixed "2d_"/"pf_") -- style_widgets() is defined once but instantiated
# inside BOTH tabs, and Streamlit runs the code in every tab on every
# script run (not just the visible one), so two widgets sharing one
# unprefixed key would collide (StreamlitDuplicateElementKey).
for prefix in ("2d_", "pf_"):
    for k, v in STYLE_DEFAULTS.items():
        st.session_state.setdefault(prefix + k, v)
# Tick spacing defaults differ per tab/plot-type (matching each one's own
# established default), so these can't go through the uniform loop above.
st.session_state.setdefault("2d_tick_spacing", 0.5)            # 2D image, 1/A
st.session_state.setdefault("2d_linecut_tick_spacing", 0.3)    # line cuts, 1/A
st.session_state.setdefault("2d_subtick_spacing", 0.0)         # 2D image minor ticks, 1/A -- 0 = off
st.session_state.setdefault("2d_color_scale", "log")           # 2D image colorbar mapping
st.session_state.setdefault("pf_tick_spacing", 20.0)           # pole figure chi axis, deg
st.session_state.setdefault("processed_2d", None)   # cached heavy-computation results
st.session_state.setdefault("processed_pf", None)
st.session_state.setdefault("calibration_confirmed", False)
st.session_state.setdefault("calibration_diagnostic_path", None)
st.session_state.setdefault("_2d_zip_bytes", None)
st.session_state.setdefault("_pf_zip_bytes", None)
st.session_state.setdefault("_2d_plot_png_cache", {})
st.session_state.setdefault("_pf_plot_png_cache", {})

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

# Same pattern as above, for the symbol keyboard: a button click can't
# write directly to a text_input's key once that widget has already been
# instantiated this run, so it stores (target_key, symbol) here instead
# and reruns; this block applies it before any widgets exist yet.
_pending_symbol = st.session_state.pop("_pending_symbol_append", None)
if _pending_symbol:
    _target_key, _sym = _pending_symbol
    st.session_state[_target_key] = st.session_state.get(_target_key, "") + _sym

# Same pattern again, for the per-tab edge-label AI box: it can set several
# keys at once (text + rotation for one or more edges), so this is a dict
# rather than a single append.
_pending_edge_ai = st.session_state.pop("_pending_edge_ai_update", None)
if _pending_edge_ai:
    for _k, _v in _pending_edge_ai.items():
        st.session_state[_k] = _v


# --------------------------------------------------------------------------- #
# AI provider configuration -- centralized, backend-configured (via Streamlit
# secrets or environment variables), NOT entered by each user in the UI.
#
# To enable AI features, whoever DEPLOYS this app adds ONE of these to the
# app's Secrets (Streamlit Cloud: app settings -> Secrets; locally: a
# .streamlit/secrets.toml file) or as an environment variable:
#   ANTHROPIC_API_KEY = "sk-ant-..."     (Claude)
#   OPENAI_API_KEY    = "sk-..."         (GPT)
#   GOOGLE_API_KEY     = "AI..."          (Gemini)
# Checked in that order; the first one found is used for every AI feature
# in the app for every visitor -- individual users never enter a key.
#
# Model names below are a moving target (all three vendors ship new models
# every few months) -- these are reasonable choices as of when this was
# written, but update them freely if a vendor's current lineup has moved on;
# each can also be overridden without a code change via an optional secret/
# env var of the same name (e.g. ANTHROPIC_MODEL, OPENAI_MODEL, GOOGLE_MODEL).
# --------------------------------------------------------------------------- #
AI_PROVIDERS = [
    {"name": "Claude (Anthropic)", "key_name": "ANTHROPIC_API_KEY",
     "model_name": "ANTHROPIC_MODEL", "default_model": "claude-sonnet-4-6"},
    {"name": "GPT (OpenAI)", "key_name": "OPENAI_API_KEY",
     "model_name": "OPENAI_MODEL", "default_model": "gpt-5.5"},
    {"name": "Gemini (Google)", "key_name": "GOOGLE_API_KEY",
     "model_name": "GOOGLE_MODEL", "default_model": "gemini-2.5-flash"},
]
AI_PROVIDER_NAMES = [p["name"] for p in AI_PROVIDERS]


def _get_secret_or_env(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # no secrets.toml at all -- st.secrets access can raise in that case
    return os.environ.get(name)


def get_configured_ai_provider() -> Optional[dict]:
    """Return the config dict (from AI_PROVIDERS) for whichever provider is
    available, with its resolved api_key and model filled in -- or None if
    nothing is configured. Priority order:
      1. What the user entered THIS session in the AI settings box below
         (render_ai_settings) -- this is the normal path.
      2. A backend-configured key (Streamlit secrets or an environment
         variable) for the SAME provider the user selected, if they didn't
         type a key themselves -- lets whoever deployed the app pre-fill a
         shared key for convenience without forcing everyone to paste one.
      3. If the user hasn't touched the provider selector at all, fall back
         to scanning secrets/env for ANY configured provider.
    """
    selected_name = st.session_state.get("ai_selected_provider")
    user_key = st.session_state.get("ai_user_api_key", "").strip()
    selected_cfg = next((p for p in AI_PROVIDERS if p["name"] == selected_name), None)

    if selected_cfg and user_key:
        model = _get_secret_or_env(selected_cfg["model_name"]) or selected_cfg["default_model"]
        return {**selected_cfg, "api_key": user_key, "model": model}

    if selected_cfg:
        backend_key = _get_secret_or_env(selected_cfg["key_name"])
        if backend_key:
            model = _get_secret_or_env(selected_cfg["model_name"]) or selected_cfg["default_model"]
            return {**selected_cfg, "api_key": backend_key, "model": model}
        return None  # a provider IS selected but no key anywhere -- don't silently fall through to a different one

    for provider in AI_PROVIDERS:
        backend_key = _get_secret_or_env(provider["key_name"])
        if backend_key:
            model = _get_secret_or_env(provider["model_name"]) or provider["default_model"]
            return {**provider, "api_key": backend_key, "model": model}
    return None


def call_ai(system_prompt: str, user_message: str, max_tokens: int = 1000) -> str:
    """Call whichever AI provider is configured (see get_configured_ai_provider),
    returning the raw text response. Raises RuntimeError if nothing is
    configured, or whatever the provider's own SDK raises on failure --
    caller should catch and show a friendly error either way.
    """
    provider = get_configured_ai_provider()
    if provider is None:
        raise RuntimeError(
            "No API key is set. Enter one in the AI settings box above "
            "(or ask whoever deployed this app to pre-configure one)."
        )
    name, api_key, model = provider["name"], provider["api_key"], provider["model"]

    if name.startswith("Claude"):
        import anthropic  # imported lazily so the app works without it installed
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model, max_tokens=max_tokens, system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(b.text for b in response.content if hasattr(b, "text"))

    elif name.startswith("GPT"):
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_message}],
        )
        return response.choices[0].message.content or ""

    elif name.startswith("Gemini"):
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config={"system_instruction": system_prompt, "max_output_tokens": max_tokens},
        )
        return response.text or ""

    else:
        raise RuntimeError(f"Unknown AI provider '{name}'.")


def render_ai_settings():
    """ONE shared place (rendered once, near the top of the page) to pick
    an AI provider and paste an API key -- used by BOTH the style
    assistant and peak-fitting AI features below, so you only enter a key
    once per session rather than once per feature. Never stored beyond
    this browser session/st.session_state (not written to disk, not sent
    anywhere except the selected provider's own API).
    """
    with st.expander("🔑 AI settings (needed for the AI style assistant and peak-fitting AI features)"):
        st.caption(
            "Pick a provider and paste your own API key -- used only for "
            "this session's requests to that provider, never stored or "
            "sent anywhere else. Leave blank if the app deployer has "
            "already pre-configured one for you (this box will say so below)."
        )
        cols = st.columns(2)
        with cols[0]:
            st.selectbox("Provider", AI_PROVIDER_NAMES, key="ai_selected_provider")
        with cols[1]:
            st.text_input("API key", type="password", key="ai_user_api_key")

        provider = get_configured_ai_provider()
        if provider is None:
            selected = st.session_state.get("ai_selected_provider", AI_PROVIDER_NAMES[0])
            st.warning(
                f"No key available for {selected} yet -- paste one above, "
                f"pick a different provider you have a key for, or ask the "
                f"app deployer to pre-configure one."
            )
        else:
            source = "you entered" if st.session_state.get("ai_user_api_key", "").strip() else "pre-configured by the deployer"
            st.success(f"Ready -- using {provider['name']} (key {source}).")


def ai_availability_notice(feature_label: str = "this feature"):
    """Show either a quiet 'using <provider>' caption, or a reminder that
    an API key needs to be set -- purely informational. Does NOT hide or
    gate the rest of the calling section: the prompt box and button should
    always be visible, and attempting a request without a key will fail
    with a clear, specific error at that point (via call_ai's own
    RuntimeError) -- that's enough; there's no need to hide the box itself.
    """
    provider = get_configured_ai_provider()
    if provider is None:
        st.caption(f"⚠️ No API key set yet -- set one in \"🔑 AI settings\" "
                   f"near the top of the page to use {feature_label}.")
    else:
        st.caption(f"Using {provider['name']}.")


# --------------------------------------------------------------------------- #
# AI style assistant (optional -- uses whichever provider is configured above)
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
- "edge_top", "edge_bottom", "edge_left", "edge_right": short text labels to place OUTSIDE the plot along that
  edge (e.g. a condition like "25 C" or a sample name) -- only if the user is clearly asking to add/change one
- "edge_top_rotation", "edge_bottom_rotation", "edge_left_rotation", "edge_right_rotation": rotation in degrees,
  counterclockwise from horizontal, for the corresponding edge label -- only if the user explicitly asks for a
  specific angle or orientation for that label. Defaults if not asked about: top/bottom 0 (horizontal), left 90
  (reads bottom-to-top), right 270 (reads top-to-bottom) -- don't set these unless the user wants something
  DIFFERENT from that default.
Example: user says "make the line red and bump up the font size" ->
{"line_color": "red", "font_size": 16}
Example: user says "add a label on top saying Annealing Series, and tilt the right-side label 45 degrees" ->
{"edge_top": "Annealing Series", "edge_right_rotation": 45}
"""


def call_ai_style_assistant(request_text: str) -> dict:
    """Translate a natural-language styling request into concrete parameter
    values, via whichever AI provider is configured. Raises on any failure
    -- caller should catch and show a friendly error."""
    text = call_ai(AI_SYSTEM_PROMPT, request_text, max_tokens=300).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def render_ai_assistant():
    with st.expander("✨ AI style assistant (optional)"):
        st.caption("Describe the look you want in plain language, and it'll be applied to the widgets below.")
        ai_availability_notice("the AI style assistant")
        request_text = st.text_input(
            "Styling request",
            placeholder="e.g. 'use a warm colormap, red line, bigger font'",
            key="ai_style_request",
        )
        if st.button("Apply with AI", key="ai_apply_button"):
            if not request_text:
                st.error("Please describe what you'd like changed.")
            else:
                try:
                    with st.spinner("Asking the AI..."):
                        result = call_ai_style_assistant(request_text)
                except ImportError as exc:
                    st.error(f"A required package isn't installed: {exc}")
                except Exception as exc:  # noqa: BLE001 -- surfaced to the user directly
                    st.error(f"AI request failed: {exc}")
                else:
                    applied = []
                    for key in ("cmap", "line_color", "sector_line_color",
                                "font_family", "font_size", "vmin", "vmax",
                                "edge_top", "edge_bottom", "edge_left", "edge_right",
                                "edge_top_rotation", "edge_bottom_rotation",
                                "edge_left_rotation", "edge_right_rotation"):
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
                        st.warning("The AI didn't return any recognized style keys.")


# --------------------------------------------------------------------------- #
# AI peak-region assistant (optional -- uses whichever provider is configured above)
# --------------------------------------------------------------------------- #
AI_PEAKFIT_SYSTEM_PROMPT = """You translate a peak-fitting request into a JSON list of fitting regions.
Return ONLY a JSON array (no prose, no markdown fences) of objects, each with:
- "q_min": lower bound of the q-range to fit, a positive number in inverse Angstrom
- "q_max": upper bound of the q-range to fit, a positive number greater than q_min
- "label": a short label for this peak -- a Miller index like "(100)" or "(010)" if the user gave or implied one, otherwise a short descriptive name (e.g. "pi-pi stacking"). Use whatever the user said; invent a short sensible label only if they gave none.
Example: user says "fit 0.25 to 0.35 as (010), and 1.6-1.8 as pi-pi stacking" ->
[{"q_min": 0.25, "q_max": 0.35, "label": "(010)"}, {"q_min": 1.6, "q_max": 1.8, "label": "pi-pi stacking"}]
If the user's request is ambiguous about exact bounds (e.g. "fit the peak near 1.67"), use your best
judgement for a reasonable window (e.g. +/- 0.1-0.15 1/A) around the stated position.
"""


def call_ai_peak_regions(request_text: str) -> List[Tuple[float, float, str]]:
    """Translate a natural-language peak-fitting request into a list of
    (q_min, q_max, label) fitting regions, via whichever AI provider is
    configured. Raises on any failure -- caller should catch and show a
    friendly error. Malformed individual regions (q_max <= q_min,
    non-numeric, etc.) are silently dropped rather than raising, so one
    bad region in a longer request doesn't lose the rest.
    """
    text = call_ai(AI_PEAKFIT_SYSTEM_PROMPT, request_text, max_tokens=1000).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    raw_regions = json.loads(text)
    regions = []
    for r in raw_regions:
        try:
            q_min, q_max = float(r["q_min"]), float(r["q_max"])
            if q_max <= q_min:
                continue
            regions.append((q_min, q_max, str(r.get("label", ""))))
        except (KeyError, TypeError, ValueError):
            continue
    return regions


def peakfit_ai_prompt_widget(target_list_key: str, widget_key_suffix: str):
    """Render a natural-language-prompt + button that appends AI-parsed
    (q_min, q_max, label) regions to st.session_state[target_list_key].
    Reused for both the batch (all line cuts) and per-line-cut refit
    prompts. Always visible, with a small reminder if no key is set yet
    (see ai_availability_notice) -- attempting to use it without a key
    fails with a clear, specific error rather than hiding the box entirely.
    """
    ai_availability_notice("peak-fitting AI assistance")
    request_text = st.text_input(
        "Describe the peak(s) to fit",
        placeholder="e.g. 'fit 0.25-0.35 as (010) and 1.6-1.8 as pi-pi stacking'",
        key=f"peakfit_ai_request_{widget_key_suffix}",
    )
    if st.button("Add region(s) with AI", key=f"peakfit_ai_button_{widget_key_suffix}"):
        if not request_text:
            st.error("Please describe the peak(s) you want fit.")
        else:
            try:
                with st.spinner("Asking the AI..."):
                    new_regions = call_ai_peak_regions(request_text)
            except ImportError as exc:
                st.error(f"A required package isn't installed: {exc}")
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user directly
                st.error(f"AI request failed: {exc}")
            else:
                if not new_regions:
                    st.warning("The AI didn't return any usable regions -- try rephrasing.")
                else:
                    st.session_state[target_list_key].extend(new_regions)
                    st.success(f"Added {len(new_regions)} region(s).")
                    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar: input files + geometry + calibration
# --------------------------------------------------------------------------- #
def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".tif"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


def build_2d_results_zip(qip_range, qoop_range) -> bytes:
    """Bundle every currently-processed 2D image + line cut (across the
    whole batch) into one ZIP, built in-memory using whatever style
    settings are CURRENTLY selected (so it matches what's on screen).
    Figures are cheap to regenerate on demand (the expensive pyFAI work
    is already cached in session_state).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for res in st.session_state["processed_2d"]:
            name = res["name"]
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
                tick_spacing=st.session_state["2d_tick_spacing"],
                subtick_spacing=st.session_state["2d_subtick_spacing"] or None,
                color_scale=st.session_state["2d_color_scale"],
                edge_label_top=st.session_state["2d_edge_top"] or None,
                edge_label_bottom=st.session_state["2d_edge_bottom"] or None,
                edge_label_left=st.session_state["2d_edge_left"] or None,
                edge_label_right=st.session_state["2d_edge_right"] or None,
                edge_label_rotations=edge_rotations_dict("2d"),
            )
            img_buf = io.BytesIO()
            fig2d.savefig(img_buf, format="png", dpi=st.session_state["2d_dpi"])
            plt.close(fig2d)
            zf.writestr(f"{name}/{name}_2D_GIWAXS.png", img_buf.getvalue())

            for angles, q, intensity in res["linecuts"]:
                tag = f"{angles[0]}_{angles[1]}".replace("-", "m").replace(".", "p")
                fig1d = gc.plot_1d_linecut(
                    q, intensity, out_path=None, angle_range=angles,
                    title=f"{name}: {angles} deg",
                    line_color=st.session_state["2d_line_color"],
                    font_family=st.session_state["2d_font_family"],
                    font_size=st.session_state["2d_font_size"],
                    tick_spacing=st.session_state["2d_linecut_tick_spacing"],
                    edge_label_top=st.session_state["2d_edge_top"] or None,
                    edge_label_bottom=st.session_state["2d_edge_bottom"] or None,
                    edge_label_left=st.session_state["2d_edge_left"] or None,
                    edge_label_right=st.session_state["2d_edge_right"] or None,
                    edge_label_rotations=edge_rotations_dict("2d"),
                )
                lc_buf = io.BytesIO()
                fig1d.savefig(lc_buf, format="png", dpi=st.session_state["2d_dpi"])
                plt.close(fig1d)
                zf.writestr(f"{name}/{name}_lineprofile_{tag}.png", lc_buf.getvalue())

                txt_buf = io.StringIO()
                np.savetxt(txt_buf, np.c_[q, intensity], header="Q(1/A)\tIntensity(a.u.)")
                zf.writestr(f"{name}/{name}_lineprofile_{tag}.txt", txt_buf.getvalue())

                linecut_df = pd.DataFrame({"Q (1/A)": q, "Intensity (a.u.)": intensity})
                zf.writestr(f"{name}/{name}_lineprofile_{tag}.csv", linecut_df.to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()


def build_pf_results_zip(dq, chi_plot_range) -> bytes:
    """Same idea as build_2d_results_zip, for every currently-processed
    pole figure (across the whole batch and every target q)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for res in st.session_state["processed_pf"]:
            name = res["name"]
            for target_q, chi_axis, profile, herman_s in res["per_q"]:
                q_tag = f"{target_q:.3f}".replace(".", "p")
                fig = gc.plot_chi_intensity_profile(
                    chi_axis, profile, out_path=None, target_q=target_q, dq=dq,
                    title=name, herman_s=herman_s, chi_range=chi_plot_range,
                    line_color=st.session_state["pf_line_color"],
                    font_family=st.session_state["pf_font_family"],
                    font_size=st.session_state["pf_font_size"],
                    tick_spacing=st.session_state["pf_tick_spacing"],
                    edge_label_top=st.session_state["pf_edge_top"] or None,
                    edge_label_bottom=st.session_state["pf_edge_bottom"] or None,
                    edge_label_left=st.session_state["pf_edge_left"] or None,
                    edge_label_right=st.session_state["pf_edge_right"] or None,
                    edge_label_rotations=edge_rotations_dict("pf"),
                )
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", dpi=st.session_state["pf_dpi"])
                plt.close(fig)
                zf.writestr(f"{name}/{name}_polefigure_q{q_tag}.png", img_buf.getvalue())

                txt_buf = io.StringIO()
                header = "Chi(deg)\tIntensity(a.u.)"
                if herman_s is not None:
                    header = f"Herman_S={herman_s:.6f}\n{header}"
                np.savetxt(txt_buf, np.c_[chi_axis, profile], header=header)
                zf.writestr(f"{name}/{name}_polefigure_q{q_tag}_chi_profile.txt", txt_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()


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
        "GIWAXS TIFF file(s)", type=["tif", "tiff"], accept_multiple_files=True,
        key="giwaxs_files_upload",
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
                width='stretch',
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
render_ai_settings()
render_ai_assistant()

# --------------------------------------------------------------------------- #
# Symbol keyboard (Greek letters, units, common scientific symbols) --
# reusable anywhere a text field could use characters that aren't on a
# normal keyboard. Streamlit buttons can only append to the END of a
# field (no true cursor-position insertion), which is a real but
# acceptable limitation given the platform.
# --------------------------------------------------------------------------- #
SYMBOL_KEYBOARD_CHARS = [
    "α", "β", "γ", "δ", "ε", "θ", "λ", "μ",
    "π", "ρ", "σ", "τ", "φ", "χ", "ψ", "ω",
    "Δ", "Σ", "Φ", "Ψ", "Ω", "°", "Å", "±",
    "×", "·", "→", "∥", "⊥", "≈", "≤", "≥",
    "⁻¹", "⁻²", "²", "³", "₀", "₁", "₂", "₃",
]


def symbol_keyboard(target_key: str, widget_key_suffix: str):
    """Render a compact grid of buttons for common Greek letters and
    scientific symbols/units; clicking one appends it to
    st.session_state[target_key]. Can't write directly to target_key here
    (its widget was already instantiated earlier this run) -- stores a
    pending append instead and reruns; applied at the top of the script,
    before any widgets exist yet (see _pending_symbol_append above).
    """
    n_cols = 8
    rows = [SYMBOL_KEYBOARD_CHARS[i:i + n_cols] for i in range(0, len(SYMBOL_KEYBOARD_CHARS), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, sym in zip(cols, row):
            with col:
                if st.button(sym, key=f"symkey_{widget_key_suffix}_{sym}", width="stretch"):
                    st.session_state["_pending_symbol_append"] = (target_key, sym)
                    st.rerun()


def edge_rotations_dict(key_prefix: str) -> dict:
    """Read the 4 edge-label rotation angles from session_state as the
    kwargs dict add_edge_labels/plot_*'s edge_label_rotations expects."""
    p = key_prefix
    return {
        "top_rotation": st.session_state[f"{p}_edge_top_rotation"],
        "bottom_rotation": st.session_state[f"{p}_edge_bottom_rotation"],
        "left_rotation": st.session_state[f"{p}_edge_left_rotation"],
        "right_rotation": st.session_state[f"{p}_edge_right_rotation"],
    }


def edge_rotations_cache_tuple(key_prefix: str) -> tuple:
    """Same 4 values as edge_rotations_dict, as a tuple for cache keys."""
    d = edge_rotations_dict(key_prefix)
    return (d["top_rotation"], d["bottom_rotation"], d["left_rotation"], d["right_rotation"])


def edge_label_inputs(key_prefix: str):
    """Render the four optional edge-label text inputs (top/bottom/left/
    right), plus ONE shared symbol keyboard and rotation control below
    them that applies to whichever field is picked with the segmented
    control -- not a dropdown, not a separate popover per field. Shared
    across the 2D and pole-figure tabs via key_prefix.
    """
    p = key_prefix
    default_rotation = {"Top": 0.0, "Bottom": 0.0, "Left": 90.0, "Right": 270.0}
    st.write("**Edge labels (optional)** -- e.g. a condition like temperature")
    ec1, ec2, ec3, ec4 = st.columns(4)
    for col, label in zip((ec1, ec2, ec3, ec4), ("Top", "Bottom", "Left", "Right")):
        with col:
            field_key = f"{p}_edge_{label.lower()}"
            rot_key = f"{field_key}_rotation"
            st.session_state.setdefault(rot_key, default_rotation[label])
            st.text_input(label, key=field_key)

    st.write("Edit label:")
    target_label = st.segmented_control(
        "Edit label", ("Top", "Bottom", "Left", "Right"), default="Top",
        key=f"{p}_edge_target", label_visibility="collapsed",
    ) or "Top"
    target_key = f"{p}_edge_{target_label.lower()}"
    target_rot_key = f"{target_key}_rotation"

    st.number_input(
        "Rotation angle (degrees, counterclockwise from horizontal)",
        step=15.0, key=target_rot_key,
        help=f"Applies to the {target_label} label. Default for {target_label}: "
             f"{default_rotation[target_label]:g}°.",
    )
    with st.expander(f"Symbol keyboard (adds to the {target_label} label)"):
        symbol_keyboard(target_key, f"{p}_edge_shared")

    with st.expander("✨ Describe edge labels with AI (optional)"):
        ai_availability_notice("AI edge-label assistance")
        edge_ai_request = st.text_input(
            "Describe the edge label(s) you want",
            placeholder="e.g. 'put Annealing Temperature on top, and tilt the right label 45 degrees'",
            key=f"{p}_edge_ai_request",
        )
        if st.button("Apply with AI", key=f"{p}_edge_ai_button"):
            if not edge_ai_request:
                st.error("Please describe what you'd like changed.")
            else:
                try:
                    with st.spinner("Asking the AI..."):
                        edge_result = call_ai_style_assistant(edge_ai_request)
                except ImportError as exc:
                    st.error(f"A required package isn't installed: {exc}")
                except Exception as exc:  # noqa: BLE001 -- surfaced to the user directly
                    st.error(f"AI request failed: {exc}")
                else:
                    edge_keys = ("edge_top", "edge_bottom", "edge_left", "edge_right",
                                 "edge_top_rotation", "edge_bottom_rotation",
                                 "edge_left_rotation", "edge_right_rotation")
                    updates = {f"{p}_{k}": edge_result[k] for k in edge_keys if k in edge_result}
                    if updates:
                        # Can't write straight to session_state here -- this
                        # tab's edge-label widgets were already instantiated
                        # earlier this run. Defer via the pending-update
                        # pattern (applied at the top of the script, before
                        # any widgets exist, on the next run) instead.
                        st.session_state["_pending_edge_ai_update"] = updates
                        st.success("Applied: " + ", ".join(f"{k}={v}" for k, v in updates.items()))
                        st.rerun()
                    else:
                        st.warning("The AI didn't return any recognized edge-label keys.")


tab_2d, tab_peakfit, tab_pf = st.tabs(
    ["2D image + line cuts", "Peak fitting", "Pole figure (cartesian)"]
)

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

    cols3 = st.columns(3)
    if show_cmap:
        with cols3[0]:
            st.number_input("2D image tick spacing (1/Å)", min_value=0.01, step=0.05,
                             key=f"{p}_tick_spacing", format="%.2f")
        with cols3[1]:
            st.number_input("Line cut tick spacing (1/Å)", min_value=0.01, step=0.05,
                             key=f"{p}_linecut_tick_spacing", format="%.2f")
        with cols3[2]:
            st.number_input(
                "2D image subticks (1/Å)", min_value=0.0, step=0.05,
                key=f"{p}_subtick_spacing", format="%.2f",
                help="Minor tick spacing between the major ticks. 0 = off "
                     "(no subticks), which is the default.",
            )
    else:
        with cols3[0]:
            st.number_input("Chi axis tick spacing (deg)", min_value=1.0, step=5.0,
                             key=f"{p}_tick_spacing", format="%.1f")

    if show_cmap:
        st.selectbox(
            "Colour-scale type", ["log", "linear"], key=f"{p}_color_scale",
            format_func=lambda v: "Logarithmic (10ⁿ ticks)" if v == "log" else "Linear (evenly-spaced ticks)",
            help="Log (default) is the usual GIWAXS convention -- intensity "
                 "spans orders of magnitude, so weak higher-order peaks stay "
                 "visible next to the strong direct beam. Linear makes exact "
                 "colorbar values easier to read but compresses weak features.",
        )

    st.checkbox("Set explicit colour-scale range (instead of automatic percentile)",
                key=f"{p}_use_manual_scale")
    if st.session_state[f"{p}_use_manual_scale"]:
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Colour-scale min", key=f"{p}_vmin", format="%.4g")
        with c2:
            st.number_input("Colour-scale max", key=f"{p}_vmax", format="%.4g")
        # A non-positive value here is fatal to a log colour scale (log(0) is
        # -inf), and an empty number box reads back as 0 -- so this is easy to
        # hit by accident. Rendering no longer crashes on it (resolve_vmin_vmax
        # falls back), but silently substituting different numbers than the
        # ones on screen would be confusing, so say what happened and why.
        scale_problem = gc.validate_manual_color_scale(
            st.session_state[f"{p}_vmin"], st.session_state[f"{p}_vmax"],
            color_scale=st.session_state.get(f"{p}_color_scale", "log"),
        )
        if scale_problem:
            st.warning(f"{scale_problem} Using the automatic range instead for now.")
    else:
        pc1, pc2 = st.columns(2)
        with pc1:
            st.slider("Colour-scale minimum percentile", 0.0, 10.0, key=f"{p}_vmin_percentile")
        with pc2:
            st.slider("Colour-scale maximum percentile", 90.0, 100.0, key=f"{p}_vmax_percentile")

    edge_label_inputs(p)


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
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            n_files = len(uploaded_files)
            for i, uf in enumerate(uploaded_files):
                status_text.text(f"Processing {uf.name} ({i + 1}/{n_files})...")
                try:
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
                        # float32, not pyFAI's float64: this array is kept in
                        # session_state for the whole session and is only ever
                        # used for the 2D image (pcolormesh + intensity
                        # percentiles), where 7 significant digits is far more
                        # than a detector's dynamic range needs. Halves the
                        # per-file resident cost (~7.7 -> ~3.9 MB at npt=1000).
                        # The line cuts below stay float64 -- those feed peak
                        # fitting, where the extra precision is free to keep.
                        "res_I": res_I.astype(np.float32, copy=False),
                        "res_qx": res_qx, "res_qy": res_qy,
                        "linecuts": linecuts,
                    })
                except Exception as exc:
                    # One bad file (unexpected format, geometry mismatch, a
                    # pyFAI-internal error, ...) shouldn't take down the
                    # whole batch that's already been waiting on this run.
                    st.error(f"Failed to process {uf.name}: {exc}")
                progress_bar.progress((i + 1) / n_files)
            status_text.text(f"Done -- processed {n_files} file(s).")
            st.session_state["processed_2d"] = results
            st.session_state["_2d_zip_bytes"] = None  # invalidate any stale cached zip
            st.session_state["_2d_plot_png_cache"] = {}  # invalidate any stale cached images

    if st.session_state["processed_2d"]:
        if st.button("Prepare ZIP of ALL 2D images + line cuts for download", key="prep_2d_zip"):
            with st.spinner("Building ZIP..."):
                st.session_state["_2d_zip_bytes"] = build_2d_results_zip(qip_range, qoop_range)
        if st.session_state.get("_2d_zip_bytes"):
            st.download_button(
                "⬇️ Download ALL 2D images + line cuts (ZIP)",
                st.session_state["_2d_zip_bytes"],
                file_name="giwaxs_2d_results.zip", mime="application/zip",
                key="dl_all_2d_zip",
            )
        d2_plot_cache = st.session_state.setdefault("_2d_plot_png_cache", {})
        for res in st.session_state["processed_2d"]:
            st.markdown(f"#### {res['name']}")
            img_cache_key = (
                res["name"], qip_range, qoop_range,
                st.session_state["2d_vmin_percentile"],
                st.session_state.get("2d_vmax_percentile", 99.9),
                st.session_state["2d_cmap"],
                st.session_state["2d_vmin"] if st.session_state["2d_use_manual_scale"] else None,
                st.session_state["2d_vmax"] if st.session_state["2d_use_manual_scale"] else None,
                st.session_state["2d_font_family"], st.session_state["2d_font_size"],
                st.session_state["2d_axis_labels"], st.session_state["2d_dpi"],
                st.session_state["2d_tick_spacing"], st.session_state["2d_subtick_spacing"],
                st.session_state["2d_color_scale"],
                st.session_state["2d_edge_top"], st.session_state["2d_edge_bottom"],
                st.session_state["2d_edge_left"], st.session_state["2d_edge_right"],
                edge_rotations_cache_tuple("2d"),
            )
            if img_cache_key not in d2_plot_cache:
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
                    tick_spacing=st.session_state["2d_tick_spacing"],
                    subtick_spacing=st.session_state["2d_subtick_spacing"] or None,
                    color_scale=st.session_state["2d_color_scale"],
                    edge_label_top=st.session_state["2d_edge_top"] or None,
                    edge_label_bottom=st.session_state["2d_edge_bottom"] or None,
                    edge_label_left=st.session_state["2d_edge_left"] or None,
                    edge_label_right=st.session_state["2d_edge_right"] or None,
                    edge_label_rotations=edge_rotations_dict("2d"),
                )
                buf = io.BytesIO()
                fig2d.savefig(buf, format="png", dpi=st.session_state["2d_dpi"])
                plt.close(fig2d)
                gc.cache_png_bytes(d2_plot_cache, img_cache_key, buf.getvalue())
            png_bytes_2d = d2_plot_cache[img_cache_key]

            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(png_bytes_2d)
            c2.download_button("Download 2D image PNG", png_bytes_2d,
                                file_name=f"{res['name']}_2D_GIWAXS.png", mime="image/png",
                                key=f"dl2d_{res['name']}")

            for angles, q, intensity in res["linecuts"]:
                lc_cache_key = (
                    res["name"], angles, st.session_state["2d_line_color"],
                    st.session_state["2d_font_family"], st.session_state["2d_font_size"],
                    st.session_state["2d_dpi"], st.session_state["2d_linecut_tick_spacing"],
                    st.session_state["2d_edge_top"], st.session_state["2d_edge_bottom"],
                    st.session_state["2d_edge_left"], st.session_state["2d_edge_right"],
                    edge_rotations_cache_tuple("2d"),
                )
                if lc_cache_key not in d2_plot_cache:
                    fig1d = gc.plot_1d_linecut(
                        q, intensity, out_path=None, angle_range=angles,
                        title=f"{res['name']}: {angles} deg",
                        line_color=st.session_state["2d_line_color"],
                        font_family=st.session_state["2d_font_family"],
                        font_size=st.session_state["2d_font_size"],
                        tick_spacing=st.session_state["2d_linecut_tick_spacing"],
                        edge_label_top=st.session_state["2d_edge_top"] or None,
                        edge_label_bottom=st.session_state["2d_edge_bottom"] or None,
                        edge_label_left=st.session_state["2d_edge_left"] or None,
                        edge_label_right=st.session_state["2d_edge_right"] or None,
                        edge_label_rotations=edge_rotations_dict("2d"),
                    )
                    buf2 = io.BytesIO()
                    fig1d.savefig(buf2, format="png", dpi=st.session_state["2d_dpi"])
                    plt.close(fig1d)
                    gc.cache_png_bytes(d2_plot_cache, lc_cache_key, buf2.getvalue())
                png_bytes_lc = d2_plot_cache[lc_cache_key]

                lc1, lc2 = st.columns([2, 1])
                with lc1:
                    st.image(png_bytes_lc)
                tag = f"{angles[0]}_{angles[1]}".replace("-", "m").replace(".", "p")
                lc2.download_button(
                    f"Download line cut {angles} PNG", png_bytes_lc,
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
                    st.dataframe(linecut_df, width='stretch', height=250)

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
            width='stretch',
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
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                n_files = len(uploaded_files)
                for i, uf in enumerate(uploaded_files):
                    status_text.text(f"Processing {uf.name} ({i + 1}/{n_files})...")
                    target_qs = q_per_file.get(uf.name, [])
                    if not target_qs:
                        st.warning(f"Skipping {uf.name}: no target q value(s) given.")
                        progress_bar.progress((i + 1) / n_files)
                        continue

                    try:
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
                            except Exception as exc:
                                # Broad catch (not just ValueError) -- pyFAI's own
                                # C-extension calls can raise other exception
                                # types too, and one bad file/q value should
                                # skip gracefully rather than crash the WHOLE
                                # batch that's already been waiting on this run.
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
                                except Exception as exc:
                                    st.warning(f"Could not compute Herman's S: {exc}")
                            per_q.append((target_q, chi_axis, profile, herman_s))
                        results.append({"name": os.path.splitext(uf.name)[0], "per_q": per_q})
                    except Exception as exc:
                        st.error(f"Failed to process {uf.name}: {exc}")
                    progress_bar.progress((i + 1) / n_files)
                status_text.text(f"Done -- processed {n_files} file(s).")
                st.session_state["processed_pf"] = results
                st.session_state["_pf_zip_bytes"] = None  # invalidate any stale cached zip
                st.session_state["_pf_plot_png_cache"] = {}  # invalidate any stale cached images

    if st.session_state["processed_pf"]:
        # Two-step (prepare, then download) rather than regenerating the
        # ZIP inline on every rerun: with many files, rebuilding it from
        # scratch on EVERY page interaction (even ones unrelated to pole
        # figures entirely, since Streamlit reruns the whole script on any
        # widget change) is expensive enough to feel like the app hung.
        if st.button("Prepare ZIP of ALL pole figures for download", key="prep_pf_zip"):
            with st.spinner("Building ZIP..."):
                st.session_state["_pf_zip_bytes"] = build_pf_results_zip(dq, chi_plot_range)
        if st.session_state.get("_pf_zip_bytes"):
            st.download_button(
                "⬇️ Download ALL pole figures (ZIP)",
                st.session_state["_pf_zip_bytes"],
                file_name="giwaxs_pole_figure_results.zip", mime="application/zip",
                key="dl_all_pf_zip",
            )
        pf_plot_cache = st.session_state.setdefault("_pf_plot_png_cache", {})
        for res in st.session_state["processed_pf"]:
            st.markdown(f"#### {res['name']}")
            for target_q, chi_axis, profile, herman_s in res["per_q"]:
                cache_key = (
                    res["name"], target_q, dq, chi_plot_range,
                    st.session_state["pf_line_color"], st.session_state["pf_font_family"],
                    st.session_state["pf_font_size"], st.session_state["pf_dpi"],
                    st.session_state["pf_tick_spacing"],
                    st.session_state["pf_edge_top"], st.session_state["pf_edge_bottom"],
                    st.session_state["pf_edge_left"], st.session_state["pf_edge_right"],
                    edge_rotations_cache_tuple("pf"),
                )
                if cache_key not in pf_plot_cache:
                    # Cache miss -- data OR style actually changed for this
                    # specific plot, so (and only so) actually re-render it.
                    # An unrelated rerun (e.g. editing a different row in the
                    # q-value table) hits the cache instead of re-invoking
                    # matplotlib for every already-rendered plot every time.
                    fig = gc.plot_chi_intensity_profile(
                        chi_axis, profile, out_path=None, target_q=target_q, dq=dq,
                        title=res["name"], herman_s=herman_s, chi_range=chi_plot_range,
                        line_color=st.session_state["pf_line_color"],
                        font_family=st.session_state["pf_font_family"],
                        font_size=st.session_state["pf_font_size"],
                        tick_spacing=st.session_state["pf_tick_spacing"],
                        edge_label_top=st.session_state["pf_edge_top"] or None,
                        edge_label_bottom=st.session_state["pf_edge_bottom"] or None,
                        edge_label_left=st.session_state["pf_edge_left"] or None,
                        edge_label_right=st.session_state["pf_edge_right"] or None,
                        edge_label_rotations=edge_rotations_dict("pf"),
                    )
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=st.session_state["pf_dpi"])
                    plt.close(fig)
                    gc.cache_png_bytes(pf_plot_cache, cache_key, buf.getvalue())
                png_bytes = pf_plot_cache[cache_key]

                pc1, pc2 = st.columns([2, 1])
                with pc1:
                    st.image(png_bytes)
                if herman_s is not None:
                    pc2.metric("Herman's S", f"{herman_s:.3f}")
                q_tag = f"{target_q:.3f}".replace(".", "p")
                pc2.download_button(
                    "Download PNG", png_bytes,
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


def _sanitize_tag(text: str) -> str:
    """Turn an arbitrary label/line-cut key into a safe filename fragment."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


with tab_peakfit:
    st.header("Peak fitting")

    st.session_state.setdefault("peakfit_regions", [])
    st.session_state.setdefault("peakfit_results", {})
    st.session_state.setdefault("_peakfit_plot_cache", {})

    linecut_lookup = {}
    for res in st.session_state["processed_2d"] or []:
        for angles, q, intensity in res["linecuts"]:
            linecut_lookup[f"{res['name']} :: {angles} deg"] = (q, intensity)

    st.subheader("1. Peak shape")
    shape_col, k_col = st.columns(2)
    with shape_col:
        peak_shape = st.selectbox("Peak shape", ["gaussian", "lorentzian", "pseudo_voigt"],
                                   key="peakfit_shape")
    with k_col:
        scherrer_k = st.number_input(
            "Scherrer shape factor K", value=0.9, min_value=0.1, max_value=2.0, step=0.05,
            key="peakfit_scherrer_k",
            help="Coherence length = 2*pi*K / FWHM. 0.9 is the most common literature default.",
        )

    st.subheader("2. Define fitting regions")
    st.caption(
        "Add regions manually, or describe them in plain language below -- "
        "both add to the same list, which is used for \"Fit ALL\" below. You "
        "can set these up before processing any data in the first tab."
    )
    rc1, rc2, rc3, rc4 = st.columns([1, 1, 1.2, 0.7])
    with rc1:
        new_qmin = st.number_input("q min (1/Å)", value=0.20, format="%.4f", key="peakfit_new_qmin")
    with rc2:
        new_qmax = st.number_input("q max (1/Å)", value=0.40, format="%.4f", key="peakfit_new_qmax")
    with rc3:
        new_label = st.text_input("Label", value="(100)", key="peakfit_new_label")
    with rc4:
        st.markdown("<div style='height: 1.7em'></div>", unsafe_allow_html=True)
        if st.button("+ Add region", key="peakfit_add_region"):
            if new_qmax <= new_qmin:
                st.error("q max must be greater than q min.")
            else:
                st.session_state["peakfit_regions"].append((new_qmin, new_qmax, new_label))
                st.rerun()

    with st.expander("✨ Or describe peak(s) in plain language (AI, optional)"):
        peakfit_ai_prompt_widget("peakfit_regions", "batch")

    if st.session_state["peakfit_regions"]:
        st.write("Current regions:")
        for i, (qmin, qmax, label) in enumerate(st.session_state["peakfit_regions"]):
            rcol1, rcol2 = st.columns([5, 1])
            rcol1.write(f"`{label}`: {qmin:.4f} - {qmax:.4f} 1/Å")
            if rcol2.button("Remove", key=f"peakfit_remove_region_{i}"):
                st.session_state["peakfit_regions"].pop(i)
                st.rerun()
    else:
        st.caption("No regions defined yet.")

    st.subheader("3. Select line cuts to fit")
    if not linecut_lookup:
        st.info(
            "No line cuts available yet -- process some 2D images + line cuts "
            "in the first tab, then come back here to select them. (The "
            "regions you've defined above will still be here when you do.)"
        )
    selected_keys = st.multiselect(
        "Line cuts", list(linecut_lookup.keys()),
        default=list(linecut_lookup.keys()), key="peakfit_selected_linecuts",
    )

    st.divider()
    if st.button("🔬 Fit ALL selected line cuts", type="primary", key="peakfit_fit_all"):
        if not st.session_state["peakfit_regions"]:
            st.error("Add at least one fitting region first.")
        elif not selected_keys:
            st.error("Select at least one line cut to fit.")
        else:
            results = {}
            n_ok = 0
            for lc_key in selected_keys:
                q, intensity = linecut_lookup[lc_key]
                fit_list = []
                for q_min, q_max, label in st.session_state["peakfit_regions"]:
                    try:
                        fit_list.append(gc.fit_peak(
                            q, intensity, q_min, q_max,
                            shape=peak_shape, scherrer_k=scherrer_k, label=label,
                        ))
                        n_ok += 1
                    except gc.GiwaxsError as exc:
                        st.warning(f"{lc_key} -- {label}: {exc}")
                results[lc_key] = fit_list
                # Seed this line cut's OWN region list (for the per-line-cut
                # refit box below) from the batch list, first time only --
                # after that it's independently editable.
                st.session_state.setdefault(
                    f"peakfit_regions__{lc_key}", list(st.session_state["peakfit_regions"])
                )
            st.session_state["peakfit_results"] = results
            st.session_state["_peakfit_plot_cache"] = {}  # invalidate stale cached plots
            st.success(f"Fit {n_ok} peak(s) across {len(results)} line cut(s).")

    if st.session_state["peakfit_results"]:
        st.divider()
        st.subheader("Results")
        all_rows = []
        plot_cache = st.session_state["_peakfit_plot_cache"]

        for lc_key, fit_list in st.session_state["peakfit_results"].items():
            if lc_key not in linecut_lookup:
                continue  # stale entry from before a reprocess in tab 1
            q, intensity = linecut_lookup[lc_key]
            tag = _sanitize_tag(lc_key)
            st.markdown(f"#### {lc_key}")

            fit_signature = tuple(
                (round(f["q0"], 6), round(f["fwhm"], 6), f["label"], f["shape"])
                for f in fit_list
            )
            cache_key = (
                lc_key, fit_signature, st.session_state["2d_line_color"],
                st.session_state["2d_font_family"], st.session_state["2d_font_size"],
                st.session_state["2d_dpi"], st.session_state["2d_linecut_tick_spacing"],
            )
            if cache_key not in plot_cache:
                fig = gc.plot_linecut_with_fits(
                    q, intensity, fit_list, out_path=None, title=lc_key,
                    line_color=st.session_state["2d_line_color"],
                    font_family=st.session_state["2d_font_family"],
                    font_size=st.session_state["2d_font_size"],
                    tick_spacing=st.session_state["2d_linecut_tick_spacing"],
                )
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=st.session_state["2d_dpi"])
                plt.close(fig)
                gc.cache_png_bytes(plot_cache, cache_key, buf.getvalue())
            png_bytes = plot_cache[cache_key]

            pf1, pf2 = st.columns([2, 1])
            with pf1:
                st.image(png_bytes)
            pf2.download_button(
                "Download plot PNG", png_bytes, file_name=f"{tag}_peakfit.png",
                mime="image/png", key=f"peakfit_dlplot_{tag}",
            )

            with pf2.expander("Refit just this one"):
                st.session_state.setdefault(
                    f"peakfit_regions__{lc_key}", list(st.session_state["peakfit_regions"])
                )
                for j, (qmin, qmax, label) in enumerate(st.session_state[f"peakfit_regions__{lc_key}"]):
                    st.caption(f"`{label}`: {qmin:.4f} - {qmax:.4f} 1/Å")
                peakfit_ai_prompt_widget(f"peakfit_regions__{lc_key}", f"single_{tag}")
                if st.button("Refit this line cut", key=f"peakfit_refit_{tag}"):
                    new_fit_list = []
                    for q_min, q_max, label in st.session_state[f"peakfit_regions__{lc_key}"]:
                        try:
                            new_fit_list.append(gc.fit_peak(
                                q, intensity, q_min, q_max,
                                shape=peak_shape, scherrer_k=scherrer_k, label=label,
                            ))
                        except gc.GiwaxsError as exc:
                            st.warning(f"{label}: {exc}")
                    st.session_state["peakfit_results"][lc_key] = new_fit_list
                    st.rerun()

            if fit_list:
                table_rows = [{
                    "Label": f["label"], "q [1/Å]": round(f["q0"], 5),
                    "d-spacing [Å]": round(f["d_spacing"], 4),
                    "FWHM [1/Å]": round(f["fwhm"], 5),
                    "L_c [Å]": round(f["coherence_length"], 2),
                    "Peak Intensity": round(f["peak_intensity"], 2),
                    "Peak Area": round(f["peak_area"], 2),
                    "R\u00b2": round(f["r_squared"], 4),
                } for f in fit_list]
                df = pd.DataFrame(table_rows)
                st.dataframe(df, width='stretch')
                st.download_button(
                    "Download table (.csv)", df.to_csv(index=False),
                    file_name=f"{tag}_peakfit.csv", mime="text/csv",
                    key=f"peakfit_dltable_{tag}",
                )
                for row in table_rows:
                    row_with_source = {"Line cut": lc_key, **row}
                    all_rows.append(row_with_source)
            else:
                st.caption("No successfully fit peaks for this line cut.")

        if all_rows:
            st.divider()
            combined_df = pd.DataFrame(all_rows)
            st.download_button(
                "⬇️ Download ALL peak fit results (CSV)",
                combined_df.to_csv(index=False),
                file_name="giwaxs_peak_fit_results.csv", mime="text/csv",
                key="peakfit_dl_all",
            )
