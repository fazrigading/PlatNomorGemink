# PlatNomorGemink

Segmentasi region plat nomor (dan rambu) dari foto kendaraan dengan OpenCV.
Region yang terdeteksi tetap tajam, sisanya diblur. Selain itu repo ini bisa
mengubah folder gambar menjadi dataset fitur untuk klasifikasi bentuk.

## Cara kerja

1. Gambar diskalakan ke lebar 1000 px, aspect ratio dipertahankan.
2. Grayscale, lalu threshold Otsu (INV) dan Canny yang didilatasi.
3. Kontur diurutkan berdasarkan luas; kontur peringkat 2 terbesar adalah plat
   pada foto-foto di repo ini (kontur terbesar adalah tepi gambar).
4. Mask kontur dipakai untuk crop region, atau untuk menyisakan region tetap
   tajam sementara sisanya diblur.

Setiap gambar dinormalisasi dulu ke 1000 px agar parameter (`-2`, divisor blur 7)
konsisten. Pada `data/images/plat/rl2.jpg` pipeline ini menangkap seluruh plat
`KT 6929 IA` tanpa distorsi; foto seperti itu yang jadi acuan, bukan foto miring
dari sudut sempit. Untuk sudut yang lebih ekstrem langkah `geometry` di
`experiments/pipeline_steps.py` memperlihatkan filter sudut dan aspect ratio yang
bisa dipakai sebagai tahap tambahan.

## Struktur repo

```
platnomor/                 paket Python inti
  segmentation.py            resize, threshold, pemilihan kontur, crop, highlight
  features.py                resize 32x32, vektor fitur 64 elemen, jumlah segi
  dataset.py                 bangun CSV dataset dari folder gambar
  __main__.py                CLI
api/segment.py             fungsi Vercel (POST gambar mentah, balas PNG)
public/index.html          halaman demo yang memanggil fungsi tersebut
experiments/pipeline_steps.py  catatan langkah eksperimen, hasilkan gambar tiap tahap
data/images/               foto masukan (plat/, rambu/)
data/rambu_dataset.csv     dataset fitur hasil pipeline
```

## Menjalankan

Butuh Python 3.12 dan OpenCV.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Bangun ulang dataset dari folder gambar:

```
.venv/bin/python -m platnomor dataset --images data/images/rambu --csv data/rambu_dataset.csv
```

Buat crop region plat dari satu gambar, atau blur sekelilingnya:

```
.venv/bin/python -m platnomor segment data/images/plat/rl2.jpg --out out/segment.png
.venv/bin/python -m platnomor highlight data/images/plat/rl2.jpg --out out/highlight.png
```

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

`api/segment.py` menerima `POST` berisi byte gambar mentah dan membalas PNG
hasil highlight. Halaman `public/index.html` memakai endpoint itu lewat form
unggah berkas.

```
npm i -g vercel
vercel dev      # pratinjau lokal
vercel          # deploy pratinjau
vercel --prod   # deploy produksi
```

Fungsi hanya mengimpor `numpy` dan `opencv-python-headless` dari
`requirements.txt`. Seluruh repo ada di paket deploy karena fungsi mengimpor
`platnomor/`, jadi gambar di `data/` ikut terunggah, tetapi tidak dipakai saat
request.

## Catatan

- `data/images/plat/` adalah foto plat nomor, `data/images/rambu/` adalah rambu.
- Model klasifikasi belum ada; repo ini berhenti di segmentasi dan ekstraksi fitur.
