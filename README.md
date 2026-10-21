# CensorLicensePlate

Segment the license plate (and traffic sign) region of a vehicle photo with OpenCV.
The detected region is then either blurred or shown, depending on the mode.

| mode | detected region | rest of the image |
| --- | --- | --- |
| `censor` (default) | blurred | untouched |
| `highlight` | untouched | blurred |
| `segment` | cropped out on its own | dropped |

The repo also turns a folder of images into a feature dataset for shape classification.

Bahasa Indonesia: [README.id.md](README.id.md)

## Modes in detail

Every mode starts the same way: scale to 1000 px wide, threshold, pick a contour
by area rank, build its filled mask. They differ only in what the mask selects.

| mode | Python | CLI | API | the mask selects |
| --- | --- | --- | --- | --- |
| censor | `censor(image)` | `censor <image> --out out/censor.png` | `POST /api/segment?mode=censor` | blur inside the mask, keep the pixels outside it |
| highlight | `highlight(image)` | `highlight <image> --out out/highlight.png` | `POST /api/segment?mode=highlight` | keep inside the mask sharp, blur the pixels outside it |
| segment | `segment(image)` | `segment <image> --out out/segment.png` | — | crop the mask bounding box, discard the rest |
| histogram | `histogram_figure(image)` | `histogram <image> --out out/histogram.png` | — | no mask: plots intensity counts |
| dataset | `dataset.build_rows(folder)` | `dataset --images DIR --csv FILE` | — | no mask: `segment` plus a feature vector per image |

`censor` and `highlight` are exact inverses. On `data/images/plat/rl2.jpg`, every
pixel inside the plate mask equals the blurred copy in one mode and the original
in the other:

```
censor    inside plate == blurred,  outside plate == original
highlight inside plate == original, outside plate == blurred
```

Blur strength comes from `blur_divisor` (default 7): the box kernel is
`image height / 7`, so on a 1779 px tall photo the kernel is 254 px, wide enough
to destroy the characters. Raise the divisor for a softer blur, lower it for a
stronger one. `segment` applies no blur; it only crops.

`histogram` and `dataset` are analysis helpers, not privacy modes: they do not
modify the image you feed in, they describe it.

## How it works

1. The image is scaled to 1000 px wide with the aspect ratio kept.
2. Grayscale, then an inverted Otsu threshold and a dilated Canny edge map.
3. Contours are sorted by area; the second largest one is the plate in the
   photos of this repo (the largest is the image border).
4. The mask then feeds one of the modes above: blur it (`censor`), keep it sharp
   (`highlight`), or crop it (`segment`).

Every image is normalised to 1000 px first so the parameters (rank `-2`, blur
divisor 7) stay consistent. On `data/images/plat/rl2.jpg` the pipeline captures
the whole `KT 6929 IA` plate without distortion; that photo is the reference,
not close-up shots taken at a narrow angle. For more extreme angles the
`geometry` step in `experiments/pipeline_steps.py` shows the angle and aspect
ratio filter that can be added as an extra stage.

## Repository layout

```tree
censor_license_plate/         # core Python package
  segmentation.py             # resize, threshold, contour selection, censor, highlight, segment
  features.py                 # 32x32 resize, 64 element feature vector, vertex count, grayscale and per channel histograms
  plots.py                    # histogram figure drawn with OpenCV
  dataset.py                  # build the dataset CSV from a folder of images
  __main__.py                 # CLI
api/segment.py                # Vercel function (POST raw image + ?mode=, answer a PNG)
public/index.html             # demo page that calls that function
experiments/pipeline_steps.py # experiment log, writes one image per pipeline stage
data/images/                  # input photos (plat/, rambu/)
data/rambu_dataset.csv        # feature dataset produced by the pipeline
```

## Running

Requires Python 3.12 and OpenCV.

```bash 
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Rebuild the dataset from the image folder:

```bash
.venv/bin/python -m censor_license_plate dataset --images data/images/rambu --csv data/rambu_dataset.csv
```

Blur the plate region so it cannot be read (the default mode):

```bash
.venv/bin/python -m censor_license_plate censor data/images/plat/rl2.jpg --out out/censor.png
```

Crop the plate region, or keep it sharp and blur everything around it:

```bash
.venv/bin/python -m censor_license_plate segment data/images/plat/rl2.jpg --out out/segment.png
.venv/bin/python -m censor_license_plate highlight data/images/plat/rl2.jpg --out out/highlight.png
```

Draw the grayscale and RGB histograms of one image (ported from the histogram
cell of the upstream notebook):

```bash
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

`api/segment.py` accepts a `POST` with raw image bytes and replies with a PNG.
The `mode` query parameter picks the behaviour; omitting it means `censor`:

| request | response |
| --- | --- |
| `POST /api/segment` | plate region blurred |
| `POST /api/segment?mode=censor` | plate region blurred |
| `POST /api/segment?mode=highlight` | plate region sharp, rest blurred |
| `POST /api/segment?mode=bogus` | `422 unknown mode 'bogus', expected one of ['censor', 'highlight']` |

`public/index.html` uses that endpoint through a file upload form with a mode
switch.

```bash
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

- `data/images/plat/` holds license plate photos, `data/images/rambu/` holds
  traffic signs.
- There is no classifier yet; the repo stops at segmentation and feature
  extraction.

## License

MIT, see [LICENSE](LICENSE). © 2026 Fazri Gading.

Source: <https://github.com/fazrigading/censor-license-plate>
