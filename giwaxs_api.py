#!/usr/bin/env python3
"""
giwaxs_api.py
==============

A small FastAPI service that wraps the GIWAXS processing pipeline so it
can be called programmatically -- from a custom frontend, a notebook,
another script, or an AI agent that wants to drive the pipeline directly
via HTTP instead of the command line.

Each processing request runs the existing, already-tested CLI agents
(giwaxs_2d1d_agent.py / giwaxs_polefigure_agent.py) as an isolated
subprocess (not an in-process function call) -- this is deliberate: it
means a bad request (bad geometry, corrupt file, etc.) can never crash the
API server itself, since each run is fully sandboxed in its own process
with its own temp directory.

Run with:
    uvicorn giwaxs_api:app --reload --port 8000

Then see the interactive docs at http://localhost:8000/docs

Endpoints
---------
GET  /health                    -- liveness check
GET  /example-config            -- returns an example config (see giwaxs_platform.example_config)
POST /process                   -- upload TIFF(s) [+ optional calibration image] + JSON params,
                                    get back a ZIP of all output files
POST /ai-style                  -- translate a natural-language styling request into
                                    concrete parameter values (proxies to the Anthropic API,
                                    using an API key YOU supply in the request -- never stored)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import giwaxs_platform as plat

app = FastAPI(
    title="GIWAXS Processing API",
    description="Programmatic access to the GIWAXS 2D/1D + pole figure pipeline.",
    version="1.0",
)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Request schema -- mirrors the platform's JSON config, minus input/output
# paths (the API manages those itself via temp directories + uploads).
# --------------------------------------------------------------------------- #
class ProcessParams(BaseModel):
    # Geometry -- all optional if a .poni file is uploaded (see poni_file
    # in the /process endpoint), which supplies the entire geometry
    # including the detector's shape and pixel size.
    poni_file: Optional[str] = None  # set internally after the file upload is saved
    beam_center_y: Optional[float] = None
    beam_center_x: Optional[float] = None
    distance: Optional[float] = None
    wavelength: Optional[float] = None
    energy: Optional[float] = None
    rot1: float = 0.0
    rot2: float = 0.0
    rot3: float = 0.0
    detector_name: Optional[str] = None
    pixel_size: Optional[float] = 172e-6
    detector_shape: Optional[List[int]] = None
    incident_angle: float
    incident_angle_from_filename: bool = False
    npt: int = 1000

    # Calibration (agbeh_file is handled as a separate uploaded file, not here)
    calibrant: str = "AgBh"
    calib_max_rings: int = 5
    calib_min_intensity: float = 200.0

    # Which steps to run
    run_2d1d: bool = True
    run_pole_figure: bool = True

    # 2D/1D options
    qip_plot_range: List[float] = [-0.5, 2.4]
    qoop_plot_range: List[float] = [-0.25, 2.75]
    vmin_percentile: float = 2.0
    vmax_percentile: float = 99.9
    extra_ranges: List[List[float]] = []

    # Pole figure options (cartesian only)
    pole_figure_q: List[float] = []
    pole_figure_q_map: Optional[str] = None  # set internally after the file upload is saved
    pole_figure_dq: float = 0.05
    chi_plot_range: List[float] = [-90, 90]
    compute_herman: bool = True
    herman_chi_max: float = 90.0

    # Style
    cmap: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    line_color: Optional[str] = None
    sector_line_color: str = "cyan"
    font_family: Optional[str] = None
    font_size: Optional[float] = None
    dpi: Optional[int] = 400
    axis_labels: str = "xyz"  # or "ip_oop" for q_ip / q_oop labels


class AIStyleRequest(BaseModel):
    request_text: str
    api_key: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/example-config")
def example_config():
    return plat.example_config()


@app.post("/process")
async def process(
    params: str = Form(..., description="JSON-encoded ProcessParams object"),
    files: List[UploadFile] = File(..., description="One or more GIWAXS TIFF files"),
    agbeh_file: Optional[UploadFile] = File(None, description="Optional AgBeh/calibrant image"),
    poni_file: Optional[UploadFile] = File(
        None, description="Optional .poni file supplying the entire geometry "
                           "(beam centre, distance, rotations, wavelength, detector) -- "
                           "if given, beam_center_y/x, distance, wavelength/energy, "
                           "detector_name/pixel_size/detector_shape in params are ignored."
    ),
    qmap_file: Optional[UploadFile] = File(
        None, description="Optional per-file pole-figure q-map (.json or .csv) -- "
                           "see docs/FIELD_REFERENCE.md for the format"
    ),
):
    try:
        parsed = ProcessParams(**json.loads(params))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid params: {exc}")

    if not parsed.run_2d1d and not parsed.run_pole_figure:
        raise HTTPException(status_code=400, detail="At least one of run_2d1d / run_pole_figure must be true.")
    if parsed.run_pole_figure and not parsed.pole_figure_q and qmap_file is None:
        raise HTTPException(status_code=400,
                             detail="pole_figure_q (or a qmap_file upload) is required when run_pole_figure is true.")
    if poni_file is None and (parsed.beam_center_y is None or parsed.beam_center_x is None or parsed.distance is None):
        raise HTTPException(
            status_code=400,
            detail="beam_center_y, beam_center_x, and distance are required in params "
                   "unless a poni_file is uploaded.",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        for f in files:
            dest = os.path.join(input_dir, f.filename)
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)

        agbeh_path = None
        if agbeh_file is not None:
            agbeh_path = os.path.join(tmp_dir, agbeh_file.filename)
            with open(agbeh_path, "wb") as out:
                shutil.copyfileobj(agbeh_file.file, out)

        poni_path = None
        if poni_file is not None:
            poni_path = os.path.join(tmp_dir, poni_file.filename)
            with open(poni_path, "wb") as out:
                shutil.copyfileobj(poni_file.file, out)

        qmap_path = None
        if qmap_file is not None:
            qmap_path = os.path.join(tmp_dir, qmap_file.filename)
            with open(qmap_path, "wb") as out:
                shutil.copyfileobj(qmap_file.file, out)

        cfg = parsed.dict()
        cfg["input"] = input_dir
        cfg["output_dir"] = output_dir
        cfg["agbeh_file"] = agbeh_path
        cfg["poni_file"] = poni_path
        cfg["pole_figure_q_map"] = qmap_path
        cfg["save_calibrated_poni"] = None
        cfg["mask"] = None

        config_path = os.path.join(tmp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f)

        # Run as an isolated subprocess -- see module docstring for why.
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_DIR, "giwaxs_platform.py"), "--config", config_path],
            capture_output=True, text=True, cwd=REPO_DIR, timeout=600,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Processing failed.",
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                },
            )

        # Zip the output directory and return it.
        zip_path = os.path.join(tmp_dir, "results.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, filenames in os.walk(output_dir):
                for fname in filenames:
                    full = os.path.join(root, fname)
                    arcname = os.path.relpath(full, output_dir)
                    zf.write(full, arcname)

        # Copy the zip out of the TemporaryDirectory before it's cleaned up
        # (FileResponse streams lazily, so the file must still exist then).
        persistent_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        persistent_zip.close()
        shutil.copy(zip_path, persistent_zip.name)

    return FileResponse(
        persistent_zip.name, media_type="application/zip",
        filename="giwaxs_results.zip",
        background=None,  # caller/OS reclaims the temp file; acceptable for a small utility API
    )


@app.post("/ai-style")
def ai_style(req: AIStyleRequest):
    """Translate a natural-language styling request into concrete plot
    parameters (cmap, line_color, font_family, font_size, vmin/vmax) using
    the Anthropic API. The API key is used only for this one request and
    is never logged or persisted.
    """
    try:
        import anthropic
    except ImportError:
        raise HTTPException(status_code=500, detail="The 'anthropic' package is not installed on the server.")

    system_prompt = (
        "You translate a plot-styling request into JSON parameters. "
        "Return ONLY a JSON object (no prose, no markdown fences) with any of "
        "these keys you can confidently infer, omitting any you cannot: "
        '"cmap" (a valid matplotlib colormap name), "line_color" (a matplotlib '
        'color spec), "sector_line_color" (same format), "font_family" (one of '
        '"DejaVu Sans", "Arial", "Times New Roman", "serif", "sans-serif", '
        '"monospace"), "font_size" (a number 6-30), "vmin", "vmax" (positive '
        "numbers, only if implied)."
    )
    try:
        client = anthropic.Anthropic(api_key=req.api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": req.request_text}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        style = json.loads(text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI request failed: {exc}")

    return JSONResponse(style)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
