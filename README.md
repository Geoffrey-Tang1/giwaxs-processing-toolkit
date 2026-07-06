# GIWAXS Processing Toolkit

Reduce raw GIWAXS TIFF images into 2D q-space maps, 1D line-cut profiles,
and fiber-texture pole figures with Herman's orientation factor -- built
on [pyFAI](https://pyfai.readthedocs.io/). Includes AgBeh (or other
calibrant) ring-fitting calibration, batch processing of whole scan
series, an interactive setup wizard, and a scriptable JSON-config mode.

New here? -> **[Full step-by-step tutorial](docs/TUTORIAL.md)** -- and if
you're not sure what any specific field means, see
**[docs/FIELD_REFERENCE.md](docs/FIELD_REFERENCE.md)** for a plain-language
explanation of every input.

Originally adapted from a GIWAXS reduction/calibration workflow developed
for pyFAI + FFmpeg-based data reduction (see `docs/` for background); this
toolkit packages that workflow into standalone, reusable command-line
tools plus a guided setup wizard.

## Install

```bash
git clone https://github.com/Geoffrey-Tang1/giwaxs-processing-toolkit.git
cd giwaxs-processing-toolkit

# macOS/Linux
chmod +x run_giwaxs_platform.command
./run_giwaxs_platform.command

# Windows: double-click run_giwaxs_platform.bat
```
First run creates an isolated `.venv` and installs dependencies
automatically (no admin rights needed). See
**[docs/TUTORIAL.md](docs/TUTORIAL.md)** for the full walkthrough of every
wizard prompt, or the sections below for a quicker reference.

---

## What's included

| File                              | Purpose                                                        |
|------------------------------------|-----------------------------------------------------------------|
| `giwaxs_common.py`                 | Shared geometry / calibration / plotting code used by everything below. |
| `giwaxs_2d1d_agent.py`             | 2D q-space image + 1D line-cut agent (standalone, runnable).     |
| `giwaxs_polefigure_agent.py`       | Pole figure (cartesian) + Herman's orientation factor agent.    |
| `giwaxs_platform.py`               | Orchestrator: runs both agents from one shared setup/config.     |
| `giwaxs_app.py`                    | Streamlit app: upload, process, and interactively style results. |
| `giwaxs_api.py`                    | FastAPI service: same pipeline, callable over HTTP.              |
| `requirements.txt`                 | Core Python dependencies.                                        |
| `requirements-app.txt`             | Extra dependencies for `giwaxs_app.py` / `giwaxs_api.py`.        |
| `run_giwaxs_platform.command`      | Double-click launcher for macOS/Linux.                          |
| `run_giwaxs_platform.bat`          | Double-click launcher for Windows.                              |

All four `.py` files must stay together in the same folder (they import
each other by filename).

---

## Quick start

### Option A -- Double-click launcher (recommended, no terminal needed after setup)

Sets up an isolated Python environment automatically on first run --
**no admin rights required**, works on a locked-down institutional
computer.

**macOS:**
1. Put all the files above into one folder.
2. Open Terminal, `cd` into that folder once, and run:
   ```bash
   chmod +x run_giwaxs_platform.command
   ```
   (Only needed the very first time -- macOS strips the "executable" flag
   from downloaded files.)
3. From then on, just **double-click `run_giwaxs_platform.command`**.
   - First double-click: creates a local `.venv` folder and installs
     `numpy`, `matplotlib`, `pyFAI`, `fabio` into it (takes a minute or two).
   - Every double-click after that: launches straight into the wizard.
   - If macOS blocks it ("unidentified developer"): right-click the file
     -> **Open** -> **Open** again to confirm once.

**Windows:** double-click `run_giwaxs_platform.bat` (same first-run/later-run behaviour).

### Option B -- Manual terminal setup

```bash
cd /path/to/this/folder
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux; Windows: .venv\Scripts\activate
pip install -r requirements.txt
python giwaxs_platform.py
```
Leave the environment later with `deactivate`.

### Running a saved config (no prompts -- good for repeat/scripted processing)

```bash
python giwaxs_platform.py --write-example my_sample.json   # generate a template
# edit my_sample.json with your beam centre / paths / q values / etc.
python giwaxs_platform.py --config my_sample.json
```

### Running an individual agent directly

```bash
python giwaxs_2d1d_agent.py --beam-center-y 145 --beam-center-x 1088 \
    --distance 0.65 --wavelength 1.5406e-10 --pixel-size 172e-6 \
    --detector-shape 1043,981 --incident-angle 0.1

python giwaxs_polefigure_agent.py --beam-center-y 145 --beam-center-x 1088 \
    --distance 0.65 --wavelength 1.5406e-10 --pixel-size 172e-6 \
    --detector-shape 1043,981 --incident-angle 0.1 \
    --pole-figure-q 1.70 0.30 --pole-figure-dq 0.03
```
(Omit `--input`/`--output-dir` and either agent will prompt you for them.)

Run `--help` on any script for the full parameter list.

---

## Batch processing

Point `--input` (or a config's `"input"` field) at a **folder** instead of
a single file, and every `.tif`/`.tiff` inside it is processed
automatically:

```bash
python giwaxs_polefigure_agent.py --input /path/to/scan_series/ \
    --pole-figure-q 1.673 0.2521 ...
```

- **2D/1D agent**: each frame gets its own 2D image + line-cut files, named after the source file.
- **Pole figure agent**: each frame gets its own pole figure(s), **and every frame x every q value is appended as a row to one shared `herman_orientation_summary.csv`** -- ideal for tracking orientation across a temperature/time series (columns: filename, target_q, dq, S, mean_cos2_chi, coverage_fraction).
- **Platform**: if an AgBeh calibration image is given, calibration runs once and the refined geometry is applied to every frame in the batch.

Current limitation: `--pole-figure-dq` is a single value shared across all `--pole-figure-q` targets in one run. If different reflections need different integration widths, run the command multiple times (once per q) rather than passing a list.

---

## Loading geometry from an existing .poni file (recommended if you have one)

If you already have an accurate calibration for this specific measurement
session, load it directly with `--poni-file path/to/file.poni` (or the
`poni_file` config key, or the app's "Load geometry from an existing
.poni file" checkbox, or the API's `poni_file` upload) -- this loads the
**entire** geometry (distance, beam centre, rotations, wavelength, AND
the detector including its shape/pixel size) in one go, so you never
need to know or re-enter those separately.

**Important:** a `.poni` file (or a beam centre/distance) is only valid
for the specific measurement session it was calibrated against. Reusing
one from a *different* sample or day, without a fresh AgBeh refinement
to confirm it still matches, will silently shift your entire q-space
mapping if the detector or beam position moved between measurements --
this is a common cause of a 2D image that looks subtly (or not so
subtly) wrong even though nothing in the software itself is broken. If
in doubt, use `--agbeh-file` alongside `--poni-file` to refine it fresh
for the current session (the loaded file is used as the initial guess
in that case).

---

## AgBeh (or other calibrant) calibration

Both agents can refine the beam centre / sample-detector distance against
a calibrant image (AgBeh, LaB6, CeO2, Si, etc.) before processing your
data -- reproducing the ring-fitting calibration from the original
notebook (pyFAI's `SingleGeometry` + `geometry_refinement.curve_fit()`),
rather than requiring you to already know the exact geometry.

- **Interactively**: if you don't pass `--agbeh-file`, you'll be asked
  whether you have a calibration image, and for its path and calibrant name.
- **On the command line**: `--agbeh-file path/to/AgBeh.tif --calibrant AgBh`
- **Via the platform wizard/config**: answered in the "Calibration"
  section, with `agbeh_file` / `calibrant` / `calib_max_rings` /
  `calib_min_intensity` / `save_calibrated_poni` config keys.

When a calibration image is used, `--beam-center-y/x` and `--distance`
only need to be an **approximate initial guess** -- refinement corrects
them. Without one, those values are used exactly as given. The refined
geometry (and chi-squared improvement) prints to the console, and can be
saved to a `.poni` file with `--save-calibrated-poni` for reuse.

**Fit confirmation (interactive mode only):** after each attempt, a
diagnostic image is opened automatically in your system's default image
viewer (falling back to printing its file path if that's not possible)
showing the detected ring points (coloured dots) overlaid with the
fitted ring positions (green contour lines) -- if the fit is good, the
lines sit right on top of the real diffraction rings. You'll be asked to
confirm; if not, you can re-enter the initial guess and/or calibrant
name and it retries. This runs when using the CLI agents directly, or
the platform's **interactive wizard** (`python giwaxs_platform.py` with
no `--config`). It's skipped entirely for **scripted/config-driven**
runs (`python giwaxs_platform.py --config ...`, and the API), since
those must never block on a prompt -- pass `--no-calibration-prompt`
directly to a CLI agent to get the same non-blocking behaviour there.
In the Streamlit app, the diagnostic image is shown inline on the page
instead of a popup; if it doesn't look right, just adjust the beam
centre/distance guess in the sidebar and click "Process" again.

---

## Pole figures and Herman's orientation factor

For each target q (reflection), the pole figure agent extracts the
azimuthal (chi) intensity profile and computes:

```
S = (3<cos^2(chi)> - 1) / 2
```

using a sin(chi)-weighted average (not a naive bin average -- each tilt
ring represents a different amount of solid angle) over the accessible
[0, chi_max] range, correctly excluding the detector's missing wedge from
the calculation rather than treating it as zero signal. It also reports
an angular *coverage fraction*; below 50% coverage a warning is printed,
since a large missing wedge makes S less trustworthy.

**Two plot styles** (`--pole-figure-style polar|cartesian|both`, default `both`):
- **polar**: radial rose plot (chi = radius, revolved uniformly around phi
  under the fiber-texture/uniaxial assumption).
- **cartesian**: intensity vs. signed chi on a log y-axis -- the style
  common in the literature for tracking a peak's orientation across a
  series (e.g. an annealing temperature series).

**Important assumption (fiber texture)**: a single GIWAXS frame only
samples one azimuthal (phi) sample orientation, so pole figures here
assume the film has no preferred in-plane orientation (rotationally
symmetric about the surface normal) -- standard for spin-coated /
blade-coated films. If you have a true phi-rotation series (multiple
frames at different in-plane sample rotations), a full non-approximated
pole figure isn't implemented here.

**Detector-edge caveat**: if your beam centre is off-centre on the
detector, the accessible chi range can be asymmetric (e.g. reaching the
full 90 deg on one side but running off the physical detector edge at a
smaller angle on the other, for a given q). S still computes correctly by
folding |chi| across whichever side has data, but you don't get an
independent two-sided confirmation of that fold in that case. Worth a
quick check for your specific q/geometry if it matters for your analysis.

---

---

## Interactive app + API

Beyond the command-line agents, two more ways to use this toolkit:

### Streamlit app (`giwaxs_app.py`) -- widgets for everything

```bash
pip install -r requirements.txt -r requirements-app.txt
streamlit run giwaxs_app.py
```

**Not sure what to type into the sidebar?** Use `example_config.json`
(included in this repo) -- it has the real, verified-working geometry
from a test sample we processed together. Upload it in the sidebar's
"0. Load example / saved config" box and every field fills in
automatically; you can still edit anything afterward. This is also a
handy starting template: duplicate it, swap in your own beam
centre/distance/wavelength, and reuse.

| Sidebar field | Example value | Where it comes from |
|---|---|---|
| Beam centre Y / X (px) | 142.71 / 1087.92 | AgBeh ring-fit calibration output |
| Distance (m) | 0.645410 | AgBeh ring-fit calibration output |
| Wavelength (m) | 8.26561e-11 | Beamline PONI file (≈15 keV) |
| Detector name | Pilatus2M | Matched from the TIFF's pixel dimensions |
| Incident angle (deg) | 0.095 | From this frame's filename |

Once loaded, use the **Style** widgets (colormap dropdown, colour-scale
range, line colour pickers, font family dropdown, font size slider) to
adjust the look -- changes apply instantly to the already-computed data
without re-running the (slower) pyFAI integration. An optional **AI
style assistant** lets you type something like *"use a warm colormap,
red line, bigger font"* and have Claude translate that into the actual
widget values (needs your own Anthropic API key, entered directly in the
app -- never stored).

### FastAPI service (`giwaxs_api.py`) -- for programmatic / app integration

```bash
pip install -r requirements.txt -r requirements-app.txt
uvicorn giwaxs_api:app --reload --port 8000
```

Then see the interactive docs at `http://localhost:8000/docs`. Key endpoints:

- `POST /process` -- upload TIFF(s) + a JSON body of parameters (mirrors
  the platform's config schema), get back a ZIP of all output files.
  Each request runs in an isolated subprocess, so a bad request can't
  crash the server.
- `POST /ai-style` -- send a natural-language styling request + your
  Anthropic API key, get back structured style parameters (`cmap`,
  `line_color`, `font_family`, `font_size`, etc.) -- useful if you're
  building your own frontend and want the same AI-assist behaviour.
- `GET /example-config` -- a template parameter set to start from.

### Style parameters (available in the CLI agents, the app, and the API)

| Parameter | CLI flag | Applies to |
|---|---|---|
| Colormap | `--cmap` | 2D image only |
| Colour-scale range | `--vmin` / `--vmax` (or `--vmin-percentile` for automatic) | 2D image only |
| Line colour | `--line-color` | 1D line cuts, pole figure |
| Sector overlay colour | `--sector-line-color` | 2D image sector overlays |
| Font family | `--font-family` | all plots |
| Font size | `--font-size` | all plots |

---

## Pole figures are cartesian-only

The pole figure agent produces **only** the Cartesian intensity-vs-chi
plot (log y-axis, signed chi) -- the style common in the literature for
tracking a peak's orientation across a series (e.g. an annealing
temperature series). An earlier polar rose-plot style existed but has
been removed from the pipeline based on user preference; the underlying
`plot_pole_figure()` function still exists in `giwaxs_common.py` if you
want to call it directly for your own purposes.

---

## Understanding chi (in case you edit the code)

The chi used throughout (`chigi_deg` in pyFAI) is *not* a raw pixel
azimuthal angle -- it's computed by: (1) building the standard lab-frame
scattering vector per pixel from your geometry, (2) rotating it into the
sample's own frame using your incident angle (and tilt angle, if
nonzero), (3) splitting into signed in-plane (`q_ip`) and out-of-plane
(`q_oop`) components, then (4) `chi = atan2(q_ip, q_oop)`. Because the
incident angle enters in step (2), it genuinely changes chi's numeric
value for a given pixel -- it's not just a display convention, so getting
`--incident-angle` right matters for correctness, not just for plot labels.

---

## Building a true standalone app (advanced, optional)

To hand this to someone with **no Python installed at all**, bundle it
with PyInstaller (heavier-weight, some trial and error possible with
pyFAI's compiled extensions):

```bash
source .venv/bin/activate
pip install pyinstaller

pyinstaller --onefile --name GIWAXSPlatform \
    --collect-all pyFAI --collect-all fabio \
    --add-data "giwaxs_common.py:." \
    --add-data "giwaxs_2d1d_agent.py:." \
    --add-data "giwaxs_polefigure_agent.py:." \
    giwaxs_platform.py
```
Result: `dist/GIWAXSPlatform` (or `.exe` on Windows) -- runs standalone,
no Python/pip needed on the target machine. Test thoroughly; adjust
`--collect-all` if you hit "file not found" errors for calibrant/detector
data at runtime.

---

## Troubleshooting

- **"python3: command not found"** -- install Python 3.10+ from
  [python.org](https://www.python.org/downloads/) (or `brew install python`
  on macOS); the launchers assume a system Python exists.
- **pyFAI/fabio fail to install** -- make sure you're using the venv's pip
  (the launcher scripts do this automatically); a stray system Python
  without build tools is the most common cause.
- **"No .tif/.tiff files found"** -- check the input path points at the
  folder containing your images, not a parent folder.
- **Pole figure step errors with "No integrated data found near q=..."**
  -- the q value is outside your detector geometry's accessible range;
  check the 1D line-cut output from the 2D/1D agent (or run a
  full-azimuthal 1D integration) to find real peaks first.
- **Herman's S seems off / low coverage warning** -- check whether your
  beam centre is far off-centre on the detector, which can cause a large
  or asymmetric missing wedge at higher q; see "Detector-edge caveat" above.

---

## License

MIT -- see [LICENSE](LICENSE).
