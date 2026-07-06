# GIWAXS Processing Toolkit -- Step-by-Step Tutorial

This walks through the whole workflow from a fresh download to interpreting
your results, using the exact prompts you'll actually see. If you just want
the short version, see the "Quick start" section of the main `README.md`.

---

## 1. Prerequisites

- Python 3.10 or newer. Check with:
  ```
  python3 --version        # macOS/Linux
  python --version         # Windows
  ```
  If missing, install from [python.org](https://www.python.org/downloads/)
  (Windows: tick "Add Python to PATH" during install).
- No admin rights needed for anything else -- the launcher creates an
  isolated environment inside the project folder itself.

---

## 2. Get the files

Either:
- **Download the ZIP** from this repository (green "Code" button ->
  "Download ZIP" on GitHub), then extract it, **or**
- **Clone with git**:
  ```bash
  git clone https://github.com/<your-username>/<your-repo-name>.git
  cd <your-repo-name>
  ```

Either way, you should end up with a folder containing (at least):
```
giwaxs_common.py
giwaxs_2d1d_agent.py
giwaxs_polefigure_agent.py
giwaxs_platform.py
requirements.txt
run_giwaxs_platform.command   (macOS/Linux)
run_giwaxs_platform.bat       (Windows)
```

---

## 3. First run (environment setup)

### macOS/Linux
```bash
cd path/to/the/folder
chmod +x run_giwaxs_platform.command      # once, first time only
./run_giwaxs_platform.command
```
If macOS blocks it as from an "unidentified developer": right-click the
file -> **Open** -> **Open** again to confirm (once).

### Windows
Double-click `run_giwaxs_platform.bat` in File Explorer.

### What happens the first time
```
First run detected -- setting up a local Python environment...
(This only happens once; it will be much faster next time.)
Downloading numpy, matplotlib, pyFAI, fabio, scipy, ...
Successfully installed ...

Starting GIWAXS Processing Platform...
```
This takes a minute or two (downloads ~40-50 MB of packages into a local
`.venv` folder next to the scripts -- nothing is installed system-wide).
Every run after this one skips straight to the wizard.

---

## 4. The setup wizard, step by step

```
======================================================================
GIWAXS Processing Platform -- setup wizard
======================================================================
Enter path to the input TIFF file or directory of TIFFs:
```
Type (or paste) the path to **either** a single `.tif` file **or** a
folder containing several -- a folder processes all of them
automatically (batch mode). **Do not** leave quote marks around a pasted
Windows path (see Troubleshooting below if you copy-pasted from Explorer).

```
Enter destination directory for output files (created automatically if it doesn't exist) [./GIWAXS_output]:
```
Where results get written. Press Enter to accept the default, or type
your own path -- it's created automatically if it doesn't exist yet.

```
--- Calibration ---
Do you have an AgBeh (or other calibrant) image to refine the beam
centre / sample-detector distance? (Recommended for accurate results.) [y/N]:
```
- **Yes**, if you have a separate calibration frame (AgBeh, LaB6, CeO2,
  Si, etc.) from the same experimental session -- you'll then be asked
  for its path and the calibrant name (default `AgBh`). This runs a real
  ring-fitting refinement against that image before processing your data.
- **No**, if you already know the exact calibrated beam centre / distance
  (e.g. from a `.poni` file you already have), or don't have a
  calibration frame at all.

```
--- Geometry ---
Beam centre Y (pixels):
Beam centre X (pixels):
Sample-to-detector distance (metres):
```
If you answered "yes" above, these only need to be **approximate**
(refinement corrects them). If "no", enter the exact calibrated values.

```
Specify wavelength directly (metres)? (No = specify energy in keV instead) [Y/n]:
```
Answer `y` and give wavelength in metres (e.g. `1.5406e-10` for Cu-Kα), or
`n` and give photon energy in keV instead -- whichever you know.

```
Detector rotation rot1/rot2/rot3 (radians) [0.0]:
```
Usually `0.0` unless your setup has a known detector tilt.

```
Use a named pyFAI detector (e.g. 'Pilatus1M')? [y/N]:
```
If your detector matches a common pyFAI-recognized model (Pilatus1M/2M,
Eiger series, etc.), answer `y` and type its name -- pixel size and shape
are filled in automatically. Otherwise answer `n` and enter pixel size +
detector shape (height,width in pixels) manually.

```
Angle of incidence (degrees) [0.1]:
```
Your grazing-incidence angle for this measurement.

```
Use a mask file? [y/N]:
```
If you have a beamstop/detector-gap mask image, answer `y` and give its path.

```
--- Processing steps ---
Run the 2D image + 1D line-cut agent? [Y/n]:
Run the pole figure agent? [Y/n]:
```
Choose which outputs you want. If you say yes to the 2D/1D agent, you'll
then be asked about plot ranges and any extra angular sectors beyond the
two defaults `(-90,-80)` and `(-8,8)`. If yes to pole figures, you'll be
asked for target q value(s), the plot style (`polar`/`cartesian`/`both`),
and whether to compute Herman's orientation factor S.

```
Save this configuration to a file for reuse next time? [Y/n]:
```
**Say yes.** This writes everything you just answered to a JSON file
(e.g. `giwaxs_config.json`). Next time, skip the whole wizard:
```bash
python giwaxs_platform.py --config giwaxs_config.json
```

---

## 5. Reading the output folder

```
GIWAXS_output/
├── images/
│   ├── <name>_2D_GIWAXS.png              # the 2D q-space map
│   └── <name>_sector_<angles>.png        # 2D map with each line-cut sector overlaid
├── linecuts/
│   ├── <name>_lineprofile_<angles>.png   # 1D intensity-vs-Q plot
│   └── <name>_lineprofile_<angles>.txt   # raw Q, intensity data
└── pole_figures/
    ├── <name>_polefigure_polar_q<...>.png    # polar rose-plot style
    ├── <name>_polefigure_chi_q<...>.png      # intensity-vs-chi (log-y) style
    ├── <name>_polefigure_q<...>_chi_profile.txt
    └── herman_orientation_summary.csv         # one row per file x per q, across your whole batch
```

Open `herman_orientation_summary.csv` in Excel/pandas to track orientation
(S) across a whole scan series -- it accumulates one row per frame per
target q every time you run the pole figure agent (including across
separate runs, so re-running with more files just adds more rows).

---

## 6. Batch processing a whole folder

Just point the wizard's first question (or `--input` / the config's
`"input"` field) at a **folder** instead of a single file -- every
`.tif`/`.tiff` inside gets processed the same way, one after another.

---

## 7. Troubleshooting

- **"File not found" right after pasting a path** -- if you copied the
  path via Windows Explorer's "Copy as path", it includes literal
  quote marks. This is now handled automatically (quotes are stripped),
  but if you still see this, retype the path without any `"` characters.
- **"python3: command not found" / "python: command not found"** --
  Python isn't installed or isn't on PATH; see Prerequisites above.
- **pip install fails during first run** -- check your internet
  connection; corporate proxies sometimes block PyPI. If you're on a
  fully offline machine, you'll need to pre-download the wheels
  elsewhere and install them manually into `.venv`.
- **"No .tif/.tiff files found in directory"** -- double-check the path
  points directly at the folder containing the images.
- **Pole figure step: "No integrated data found near q=..."** -- that q
  value is outside what your geometry can reach; check the 1D line-cut
  output first to find real peak positions.
- **Herman's S looks off, or a low-coverage warning appears** -- your
  beam centre may be far off-centre on the detector, causing an
  asymmetric or large missing wedge at that q; see the "Detector-edge
  caveat" section in `README.md`.

---

## 8. Re-running without the wizard (scripted / repeatable)

```bash
source .venv/bin/activate                 # or .venv\Scripts\activate on Windows
python giwaxs_platform.py --config giwaxs_config.json
```
Edit the JSON file directly to change any parameter, or generate a fresh
template with:
```bash
python giwaxs_platform.py --write-example new_config.json
```
