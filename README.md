# CensorLicensePlate

Segment the licence plate (and traffic sign) region of a vehicle photo with
OpenCV. The detected region stays sharp while the rest of the image is blurred.
The repo also turns a folder of images into a feature dataset for shape
classification.

Bahasa Indonesia: [README_id.md](README_id.md)

## How it works

1. The image is scaled to 1000 px wide with the aspect ratio kept.
2. Grayscale, then an inverted Otsu threshold and a dilated Canny edge map.
3. Contours are sorted by area; the second largest one is the plate in the
   photos of this repo (the largest is the image border).
4. The contour mask crops the region, or keeps it sharp while the rest of the
   image is blurred.

Every image is normalised to 1000 px first so the parameters (rank `-2`, blur
divisor 7) stay consistent. On `data/images/plat/rl2.jpg` the pipeline captures
the whole `KT 6929 IA` plate without distortion; that photo is the reference,
not close-up shots taken at a narrow angle. For more extreme angles the
`geometry` step in `experiments/pipeline_steps.py` shows the angle and aspect
ratio filter that can be added as an extra stage.

## Repository layout

```
censor_license_plate/                 core Python package
  segmentation.py            resize, threshold, contour selection, crop, highlight
  features.py                32x32 resize, 64 element feature vector, vertex count,
                             grayscale and per channel histograms
  plots.py                   histogram figure drawn with OpenCV
  dataset.py                 build the dataset CSV from a folder of images
  __main__.py                CLI
api/segment.py             Vercel function (POST raw image, answer a PNG)
public/index.html          demo page that calls that function
experiments/pipeline_steps.py  experiment log, writes one image per pipeline stage
data/images/               input photos (plat/, rambu/)
data/rambu_dataset.csv     feature dataset produced by the pipeline
```

## Running

Requires Python 3.12 and OpenCV.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Rebuild the dataset from the image folder:

```
.venv/bin/python -m censor_license_plate dataset --images data/images/rambu --csv data/rambu_dataset.csv
```

Crop the plate region of one image, or blur everything around it:

```
.venv/bin/python -m censor_license_plate segment data/images/plat/rl2.jpg --out out/segment.png
.venv/bin/python -m censor_license_plate highlight data/images/plat/rl2.jpg --out out/highlight.png
```

Draw the grayscale and RGB histograms of one image (ported from the histogram
cell of the upstream notebook):

```
.venv/bin/python -m censor_license_plate histogram data/images/plat/rl2.jpg --out out/histogram.png
```

The counts behind that figure are importable as
`censor_license_plate.grayscale_histogram(image)` and `censor_license_plate.color_histograms(image)`.

Render every pipeline stage as an image:

```
.venv/bin/python experiments/pipeline_steps.py data/images/plat/rl2.jpg out/pipeline_steps
```

## Dataset format

`data/rambu_dataset.csv` has these columns:

| column | content |
| --- | --- |
| `nama` | file name of the image in `data/images/rambu/` |
| `fitur` | 64 numbers: active pixel count per column (32) then per row (32) of the binarised 32x32 crop |
| `segi` | vertex count of the contour from `approxPolyDP`, used as the shape label |

## Hosting on Vercel

Production: <https://censor-license-plate.vercel.app>

`api/segment.py` accepts a `POST` with raw image bytes and replies with a PNG of
the highlighted region. `public/index.html` uses that endpoint through a file
upload form.

```
vercel dev                      # local preview
vercel deploy                   # preview deployment
vercel deploy --prod --yes --project censor-license-plate
```

The Vercel CLI does not discover `api/segment.py` on its own, so
`pyproject.toml` declares its entrypoint:

```toml
[tool.vercel]
entrypoint = "api.segment:handler"
```

The CLI also runs `uv lock` with `pyproject.toml` as the project, so the
`[project]` table is required and `[project.dependencies]` must list `numpy` and
`opencv-python-headless`. `requirements.txt` is still what the runtime installs.

This project has Deployment Protection enabled, so the production URL asks for a
Vercel login. Use `vercel curl <url>` or turn the protection off in the
dashboard. `.vercelignore` keeps `.codegraph/`, `.projectmem/`, `.commandcode/`
and `.venv/` out of the upload; files in `data/` are uploaded but not used per
request.

## Notes

- `data/images/plat/` holds licence plate photos, `data/images/rambu/` holds
  traffic signs.
- There is no classifier yet; the repo stops at segmentation and feature
  extraction.
