# CensorLicensePlate

Segmentasi region plat nomor (dan rambu) dari foto kendaraan dengan OpenCV.
Region yang terdeteksi lalu diblur atau dipertajam, tergantung mode.

| mode | region terdeteksi | sisa gambar |
| --- | --- | --- |
| `censor` (default) | diblur | dibiarkan |
| `highlight` | dibiarkan | diblur |
| `segment` | di-crop sendiri | dibuang |

Selain itu repo ini bisa mengubah folder gambar menjadi dataset fitur untuk
klasifikasi bentuk.

## Mode

Semua mode memakai langkah awal yang sama: skala ke lebar 1000 px, threshold,
pilih kontur berdasarkan peringkat luas, buat mask-nya. Bedanya hanya apa yang
dipilih mask itu.

| mode | Python | CLI | API | mask memilih |
| --- | --- | --- | --- | --- |
| censor | `censor(image)` | `censor <gambar> --out out/censor.png` | `POST /api/segment?mode=censor` | blur di dalam mask, piksel luar dibiarkan |
| highlight | `highlight(image)` | `highlight <gambar> --out out/highlight.png` | `POST /api/segment?mode=highlight` | dalam mask tetap tajam, luar diblur |
| segment | `segment(image)` | `segment <gambar> --out out/segment.png` | — | crop kotak pembatas mask, sisanya dibuang |
| histogram | `histogram_figure(image)` | `histogram <gambar> --out out/histogram.png` | — | bukan mode mask: gambar histogram |
| dataset | `dataset.build_rows(folder)` | `dataset --images DIR --csv FILE` | — | bukan mode mask: `segment` plus vektor fitur per gambar |

`censor` dan `highlight` berkebalikan persis. Pada `data/images/plat/rl2.jpg`,
setiap piksel di dalam mask plat sama dengan versi blur di satu mode dan sama
dengan gambar asli di mode lain.

Kekuatan blur diatur `blur_divisor` (default 7): kernel kotak = tinggi gambar / 7,
jadi pada foto setinggi 1779 px kernelnya 254 px dan huruf plat tidak terbaca.
Perbesar divisor untuk blur lebih halus, perkecil untuk lebih kuat. `segment`
tidak memakai blur sama sekali.

`histogram` dan `dataset` adalah alat analisis, bukan mode privasi: keduanya
tidak mengubah gambar, hanya menjelaskannya.

## Cara kerja

1. Gambar diskalakan ke lebar 1000 px, aspect ratio dipertahankan.
2. Grayscale, lalu threshold Otsu (INV) dan Canny yang didilatasi.
3. Kontur diurutkan berdasarkan luas; kontur peringkat 2 terbesar adalah plat
   pada foto-foto di repo ini (kontur terbesar adalah tepi gambar).
4. Mask kontur lalu dipakai salah satu mode di atas: diblur (`censor`),
   dipertajam (`highlight`), atau di-crop (`segment`).

Setiap gambar dinormalisasi dulu ke 1000 px agar parameter (`-2`, divisor blur 7)
konsisten. Pada `data/images/plat/rl2.jpg` pipeline ini menangkap seluruh plat
`KT 6929 IA` tanpa distorsi; foto seperti itu yang jadi acuan, bukan foto miring
dari sudut sempit. Untuk sudut yang lebih ekstrem langkah `geometry` di
`experiments/pipeline_steps.py` memperlihatkan filter sudut dan aspect ratio yang
bisa dipakai sebagai tahap tambahan.

## Struktur repo

```
censor_license_plate/          paket Python inti
  segmentation.py            resize, threshold, pemilihan kontur, censor, highlight, segment
  features.py                resize 32x32, vektor fitur 64 elemen, jumlah segi,
                             histogram grayscale dan per kanal
  plots.py                   gambar histogram yang dirender dengan OpenCV
  dataset.py                 bangun CSV dataset dari folder gambar
  __main__.py                CLI
api/segment.py                 fungsi Vercel (POST gambar mentah + ?mode=, balas PNG)
public/index.html              halaman demo yang memanggil fungsi tersebut
experiments/pipeline_steps.py  catatan langkah eksperimen, hasilkan gambar tiap tahap
data/images/                   foto masukan (plat/, rambu/)
data/rambu_dataset.csv         dataset fitur hasil pipeline
```

## Menjalankan

Butuh Python 3.12 dan OpenCV.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Bangun ulang dataset dari folder gambar:

```
.venv/bin/python -m censor_license_plate dataset --images data/images/rambu --csv data/rambu_dataset.csv
```

Blur region plat supaya tidak terbaca (mode default):

```
.venv/bin/python -m censor_license_plate censor data/images/plat/rl2.jpg --out out/censor.png
```

Buat crop region plat dari satu gambar, atau pertajam region itu dan blur sekelilingnya:

```
.venv/bin/python -m censor_license_plate segment data/images/plat/rl2.jpg --out out/segment.png
.venv/bin/python -m censor_license_plate highlight data/images/plat/rl2.jpg --out out/highlight.png
```

Gambar histogram grayscale dan RGB satu gambar (hasil port dari sel histogram di
notebook upstream):

```
.venv/bin/python -m censor_license_plate histogram data/images/plat/rl2.jpg --out out/histogram.png
```

Angka di balik gambar itu bisa diimpor lewat `censor_license_plate.grayscale_histogram(image)`
dan `censor_license_plate.color_histograms(image)`.

Lihat setiap tahap pipeline sebagai gambar:

```
.venv/bin/python experiments/pipeline_steps.py data/images/plat/rl2.jpg out/pipeline_steps
```

## Format dataset

`data/rambu_dataset.csv` berisi kolom:

| kolom | isi |
| --- | --- |
| `nama` | nama berkas gambar di `data/images/rambu/` |
| `fitur` | 64 angka: jumlah piksel aktif per kolom (32) lalu per baris (32) dari crop 32x32 biner |
| `segi` | jumlah titik polygon hasil `approxPolyDP` kontur, dipakai sebagai label bentuk |

## Hosting di Vercel

Produksi: <https://censor-license-plate.vercel.app>

`api/segment.py` menerima `POST` berisi byte gambar mentah dan membalas PNG.
Query `mode` menentukan perilakunya; tanpa query itu artinya `censor`:

| request | balasan |
| --- | --- |
| `POST /api/segment` | region plat diblur |
| `POST /api/segment?mode=censor` | region plat diblur |
| `POST /api/segment?mode=highlight` | region plat tajam, sekitarnya diblur |
| `POST /api/segment?mode=bogus` | `422 unknown mode 'bogus', expected one of ['censor', 'highlight']` |

`public/index.html` memakai endpoint itu lewat form unggah berkas dengan pilihan mode.

```
vercel dev                      # pratinjau lokal
vercel deploy                   # deploy pratinjau
vercel deploy --prod --yes --project censor-license-plate
```

Vercel CLI tidak menemukan `api/segment.py` sendiri, jadi `pyproject.toml`
mendeklarasikan entrypoint-nya:

```toml
[tool.vercel]
entrypoint = "api.segment:handler"
```

CLI juga menjalankan `uv lock` dengan `pyproject.toml` sebagai proyek, jadi
tabel `[project]` wajib ada dan `[project.dependencies]` harus memuat `numpy`
serta `opencv-python-headless`. `requirements.txt` tetap dipakai runtime.

Proyek ini punya Deployment Protection, jadi URL produksi meminta login Vercel.
Gunakan `vercel curl <url>` atau matikan proteksi di dashboard. `.vercelignore`
menyisihkan `.codegraph/`, `.projectmem/`, `.commandcode/`, `.venv/`, dan `out/`
saat unggah; berkas di `data/` ikut terunggah tetapi tidak dipakai saat request.

## Catatan

- `data/images/plat/` adalah foto plat nomor, `data/images/rambu/` adalah rambu.
- Model klasifikasi belum ada; repo ini berhenti di segmentasi dan ekstraksi fitur.
