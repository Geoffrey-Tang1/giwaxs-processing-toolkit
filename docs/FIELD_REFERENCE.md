# Field reference (what does this input mean, and what do I put?)

A plain-language explanation of every field in `example_config.json` / the
app sidebar, for when you're not sure what to type.

## Geometry (usually comes from your beamline / calibration, not guessed)

| Field | What it means | If you don't know it |
|---|---|---|
| `poni_file` | Path to an existing pyFAI `.poni` calibration file, which bundles the ENTIRE geometry -- distance, beam centre, rotations, wavelength, AND the detector (including its shape and pixel size). If given, none of the fields below are needed. **This is the recommended way to run the tool if you already have an accurate calibration for this specific sample/session** -- don't reuse a `.poni` (or beam centre/distance) from a different sample's calibration unless you're sure the detector/setup wasn't touched between measurements, since even small changes will shift the whole q-space mapping and give a subtly (or not-so-subtly) wrong result. |
| `calibrant` / `calib_min_intensity` | Not asked in the wizard (silently defaulted to `"AgBh"` and `200.0`) since almost everyone uses AgBeh and the default intensity threshold works for most detectors/exposures. Edit these directly in a saved config JSON if you need a different calibrant or threshold. |
| `beam_center_y` / `beam_center_x` | Pixel row/column where the direct beam hits the detector | Use the AgBeh calibration feature -- it figures this out for you from a ring-fitting fit, you only need an approximate starting guess. Not needed at all if using `poni_file`. |
| `distance` | Sample-to-detector distance, in metres | Same as above -- calibration refines this too |
| `wavelength` | X-ray wavelength, in metres | Most people know **energy** instead (ask your beamline, or check a `.poni` file if you have one) -- use `energy` (keV) instead, they're interchangeable (E = hc/λ) |
| `rot1` / `rot2` / `rot3` | Small detector tilt angles (radians), correcting for the detector not being perfectly perpendicular to the beam | Leave at `0.0` unless a full calibration determined otherwise -- this is the vast majority of setups |
| `detector_name` | A named detector pyFAI already knows (e.g. `Pilatus2M`) | If your detector matches a known model, this alone fills in pixel size and shape for you -- much easier than filling those in by hand |
| `pixel_size` / `detector_shape` | Manual pixel size (metres) and detector dimensions (height, width in pixels) | Only needed if `detector_name` doesn't match anything pyFAI recognizes -- check your detector's spec sheet, or the image's actual pixel dimensions for the shape |
| `incident_angle` | The grazing angle (degrees) the X-ray beam makes with the sample surface | This is a real experimental setting recorded during your scan (often in the filename or a log file) -- it genuinely changes the physics of the calculation, not just a label |
| `incident_angle_from_filename` | If `true`, auto-detects each file's own incident angle from its filename using the `0p095`-style convention (e.g. `sample_0p095_1234.tif` -> 0.095 deg) -- useful for a batch/folder where different frames used different angles. Falls back to `incident_angle` for any file where no such pattern is found. |
| `incident_angle_map` | Path to a JSON/CSV/Excel file assigning a DIFFERENT incident angle to each individual file, for a batch where angles differ but don't follow the `0p095` filename convention -- e.g. `{"sample_0001.tif": 0.095, "sample_0002.tif": 0.1}`. In the wizard, choosing "type a different value for each file" builds this automatically (listing your actual files and letting you type each one's angle, with Enter repeating the previous file's value) -- no need to hand-write the file yourself. |
| `npt` | How many bins to use when re-gridding data | 1000 is a solid default; higher = finer/slower, lower = coarser/faster |

## Processing options

| Field | What it means |
|---|---|
| `qip_plot_range` / `qoop_plot_range` | The visible x/y-axis window (in 1/Å) on the 2D image -- purely a display crop, doesn't change the underlying data |
| `vmin_percentile` | When *not* setting an explicit colour range, this percentile of the darkest pixels gets clipped from the colour scale's minimum, so a few near-zero noisy pixels don't wash out the contrast for everything else |
| `extra_ranges` | Extra angular sectors (beyond the two defaults) to extract 1D line-cut profiles from, as `[start_deg, end_deg]` pairs |
| `pole_figure_q` | The target q value(s) (1/Å), i.e. which diffraction peak(s) you want a pole figure for |
| `pole_figure_q_map` | Path to a JSON or CSV file assigning DIFFERENT target q value(s) to each individual file in a batch (rather than one shared list) -- e.g. `{"sample_0001.tif": [1.673, 0.252], "sample_0002.tif": 1.68}`. Any file not listed falls back to `pole_figure_q`. In the wizard, choosing "type a different value for each file" builds this automatically -- no need to hand-write the file. In the Streamlit app, this is a filename-vs-q editable table instead. |
| `pole_figure_dq` | Half-width of the q window averaged over when extracting that peak's chi profile -- too narrow = noisy, too wide = mixes in neighbouring peaks/background |
| `herman_chi_max` | Maximum tilt angle (from the surface normal) integrated over when computing Herman's orientation factor S; the standard convention uses `90` (the full physically accessible range) |

## Style (purely cosmetic -- doesn't change the science, just the picture)

| Field | What it means |
|---|---|
| `cmap` | Colour scheme for the 2D image. Grouped by type in the app's dropdown: perceptually-uniform (viridis/plasma/inferno/magma/cividis -- recommended for publication figures), high-contrast/warm (turbo/jet/hot), single-hue (gray/bone), diverging (coolwarm/twilight) |
| `vmin_percentile` / `vmax_percentile` | Colour-scale range for the 2D image is set from these percentiles of nonzero pixel intensity (default 1%-99.9%), not the raw min/max -- a few near-zero or saturated/hot pixels would otherwise wash out contrast for everything else. Ignored if `vmin`/`vmax` are given explicitly. |
| `vmin` / `vmax` | Explicit colour-scale min/max (intensity units), if you want manual control instead of the automatic percentile-based one |
| `line_color` | Colour of 1D line-cut / pole-figure curves |
| `sector_line_color` | Colour of the sector-boundary lines overlaid on the 2D image |
| `font_family` / `font_size` | Self-explanatory -- applies to all plot text |
| `dpi` | Output image resolution (dots per inch), default 400. Higher = sharper/larger file, good for publication; lower = smaller file, fine for quick previews |
| `axis_labels` | `"xyz"` (default) for q_xy / q_z axis labels, or `"ip_oop"` for q_ip / q_oop -- purely a label change, identical underlying data either way |

## Plot appearance notes

- All saved figures use a fixed **5x4 inch** layout, at whatever `dpi` you set (default 400).
- The **2D image never has a title** (kept plain for figure/publication use) -- the filename appears in the console log and output filename instead.
- **Line-cut x-axis** is log-scaled, fixed to the range 0.15-2.0 Å⁻¹, with major tick marks placed at clean round values (0.3, 0.6, 0.9, ...) rather than the log-scale default -- ticks look visually non-uniform (an inherent property of a log axis) but the labels themselves are plain round numbers, matching common annealing-series reference figures.
- **Pole-figure x-axis** (χ, in degrees) is linear, with a "χ (°)" label and 20-degree tick spacing by default.
