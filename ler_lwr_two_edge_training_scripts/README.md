# Two-edge LER/LWR metrology-aware training pipeline — run scripts

This is the companion code for the two-edge (correlated left/right edge)
extension of the LER/LWR metrology pipeline: a physically-validated
synthetic-data generator, classical baseline detectors, and a differentiable
metrology-aware training pipeline (soft sub-pixel localization,
differentiable detrending, differentiable Welch PSD, five loss-ablation
variants). It fills in the `[TBD: result]` placeholders in Tables 2-4 of the
manuscript.

Everything here was developed and smoke-tested on a 2-core CPU-only sandbox.
It will run correctly on any machine with the dependencies below; the
**pilot-scale defaults are deliberately small** so you can verify the whole
pipeline runs end-to-end in a few minutes before committing to the full
submission-quality run, which takes much longer (see "Scaling up" below).

## 1. Dependencies

```
python >= 3.9
numpy
scipy
pandas
torch        (CPU-only is fine; CUDA used automatically if available)
opencv-python   (cv2, used by the Canny baseline)
```

Install:

```bash
pip install numpy scipy pandas torch opencv-python
```

No GPU is required. If your PC has a CUDA GPU, PyTorch will use it
automatically for the `train2.py`/`run_experiment{1,2,8}.py` scripts once you
move tensors to `cuda` (see "Using a GPU" below for the one-line change
needed — the scripts default to CPU so they behave identically to how they
were developed and verified here).

## 2. Files

| File | Purpose |
|---|---|
| `generator2.py` | Two-edge correlated K-correlation roughness generator (Eqs. 30-39), Poisson-Gaussian noise, paired-acquisition sampler. Verified numerically against the closed-form variance identity `Var(w) = 2*sigma^2*(1-rho)` (Eq. 34). |
| `generator_base.py` | Shared low-level helpers reused from the single-edge (v1) generator. |
| `classical_two_edge.py` | Threshold, Gaussian-denoise+threshold, and Canny detectors, direction-aware (`falling=True` for the right/trailing edge) so both edges of a two-edge line are handled correctly. |
| `model2.py` | `TinyUNet` backbone, differentiable soft-argmax localization, differentiable linear detrending (closed-form projection matrix), differentiable Welch PSD. |
| `losses2.py` | The five named loss variants (`seg-only`, `seg+pos`, `seg+pos+rms`, `seg+pos+rms+psd`, `full`) and their loss terms (Eqs. 22-29). |
| `train2.py` | Core training/evaluation library imported by all `run_experiment*.py` scripts. Has its own `__main__` smoke test. |
| `run_experiment1.py` | **Table 2**: classical Threshold+sub-pixel baseline, a placeholder row for "Learned denoiser + threshold" (intentionally left blank — see below), and U-Net Seg-only vs. Full. |
| `run_experiment2.py` | **Table 3**: full five-way loss ablation (accuracy + per-variant training cost). |
| `run_experiment5.py` | **Table 4** (classical rows): Threshold × Canny × pixel × sub-pixel factorial comparison on two-edge data. Fast — no training. |
| `run_experiment8.py` | **Table (Section 7.9)**: computational cost — parameter counts, training time/epoch, inference time/image, for every variant plus the classical detectors. |
| `real_data_metrology.py` | Real-SEM-data ingestion: extracts LER/LWR/PSD (and optionally fits K-correlation `xi`/`alpha`) from a directory of real images with ground truth as masks, edge-coordinate JSON, or Labelme polylines. See section 9 below. |

Every script writes `{prefix}_raw.csv` (every trial), `{prefix}_summary.csv`
(the table you paste into the manuscript), and, where relevant,
`{prefix}_cost.csv` (timing/parameter counts). Console output also prints
the summary table directly.

## 3. Quick start (pilot scale — verify everything works, ~10-15 min total)

Run from this directory:

```bash
python3 run_experiment5.py --per-config 5 --rhos 0.0 0.5
python3 run_experiment1.py --epochs 12 --seeds 1 --train-per-config 3 --eval-per-config 4 --classical-per-config 5 --rhos 0.0 0.6
python3 run_experiment2.py --epochs 12 --seeds 1 --train-per-config 3 --eval-per-config 4 --rhos 0.0 0.6
python3 run_experiment8.py --timing-epochs 5 --n-timing-items 32
```

`run_experiment5.py` is fast (under a minute — no training). The other three
each train 2-5 small U-Nets and take a few minutes apiece on a 2-core CPU.
These pilot numbers are **not** the ones to report in the paper — they exist
so you can confirm the pipeline runs cleanly on your machine before
committing to the longer run below. Expect noisier, sometimes counter-
intuitive results at this scale (see "Known pilot-scale finding" below).

## 4. Scaling up to submission quality

The manuscript's own methodology (Section 6) calls for at least 3 seeds, a
fuller parameter grid, and more training epochs. On your own PC:

```bash
python3 run_experiment1.py --epochs 40 --seeds 3 --train-per-config 15 \
    --eval-per-config 15 --classical-per-config 15 --rhos 0.0 0.3 0.6 -0.3

python3 run_experiment2.py --epochs 40 --seeds 3 --train-per-config 15 \
    --eval-per-config 15 --rhos 0.0 0.3 0.6 -0.3

python3 run_experiment5.py --per-config 15 --rhos 0.0 0.3 0.6 -0.3

python3 run_experiment8.py --timing-epochs 10 --n-timing-items 64 --full-epochs 40
```

**Expected runtime.** On this sandbox's 2-core CPU, one `seg-only` training
run (12 epochs, 8 configs × 3 items/config = 24 items) took well under a
minute; the pilot `run_experiment2.py` run above (5 variants × 1 seed, 12
epochs, 16 configs) took about 5-6 minutes total. Each additional seed
roughly multiplies runtime by the number of seeds (it's literally that many
extra training runs), and each additional rho or extra sigma/xi/alpha value
multiplies the config grid (and hence both dataset-build time and epoch
time) by however many more configs it adds. As a rough rule of thumb,
budget **runtime scaling roughly linearly** in
`seeds x epochs x train_per_config x n_configs`. On a modern multi-core
desktop CPU, expect the full-scale command block above (4 rhos, 3 seeds, 40
epochs) to take on the order of a few hours for `run_experiment2.py` alone;
run it overnight, or reduce `--seeds` to 1 first to sanity-check the full
grid before committing to 3 seeds.

If a run is taking too long, reduce `--rhos` to fewer values first (it's the
biggest single multiplier), then `--seeds`, then `--epochs`.

## 5. Using a GPU

**Automatic.** `train2.py` resolves a device at import (`train2.DEVICE`) and
uses CUDA whenever `torch.cuda.is_available()`, otherwise CPU. No edits
needed. Override with the `LER_DEVICE` environment variable:

```bash
LER_DEVICE=cpu    python3 run_experiment2.py ...   # force CPU
LER_DEVICE=cuda   python3 run_experiment2.py ...   # force GPU (errors if unavailable)
LER_DEVICE=cuda:1 python3 run_experiment2.py ...   # pick a specific GPU
```

`_edge_forward` follows whatever device the model is on, and
`model2.py`'s `differentiable_detrend` now defaults its projection-matrix
cache to the device of its input, so the whole forward/backward path moves
together.

**Measured speedup** (RTX A5000, `base=12`, pilot grid — `run_experiment8.py`):

| | CPU (16 threads) | GPU | speedup |
|---|---|---|---|
| training, `full` variant | 6.69 s/epoch | 0.22 s/epoch | ~30x |
| inference | 34.2 ms/image | 1.6 ms/image | ~21x |

This turns the "run it overnight" full-scale block in section 4 into
something on the order of tens of minutes, so on a GPU you can afford
considerably more epochs/seeds/configs — and, importantly, the lambda sweep
that section 6 says Table 3 needs.

**Note on comparing runs across devices.** CPU and GPU convolution kernels
are not bit-identical, so the same seed gives slightly different numbers on
each. A pilot ablation run at identical settings agreed to well within one
standard deviation on every variant (e.g. `full`: `ler_mae` 0.0631 on CPU vs
0.0596 on GPU, against a per-variant std of ~0.045), and the ranking of
variants was unchanged. Don't mix devices *within* one reported table —
pick one and re-run the whole table on it.

## 6. Known pilot-scale finding — read before interpreting your own results

A pilot run of `run_experiment2.py` (8 epochs, 1 seed, 2 train items/config,
16 configs) produced monotonically *worse* edge/LER/LWR MAE as loss terms
were added (`seg-only` best, `full` in the middle, `seg+pos+rms` worst).
This is very likely an artifact of the untuned lambda weights
(`pos=0.05, rms=1.0, psd=0.5, cons=0.5` in `losses2.py`) relative to how few
epochs/items the pilot used — the RMS and PSD terms have much larger
gradients than the segmentation term when the model is still far from
converged, and can dominate/destabilize training before the localization
head has learned anything useful. This is exactly the kind of thing the
manuscript's Section 6 anticipates needing a **lambda sweep** to resolve
(log-spaced sweeps over `lambda_p, lambda_r, lambda_s, lambda_c`) — do not
be surprised if your first full-scale run shows a similar pattern, and treat
tuning those four weights (e.g., a small grid or a short random search
evaluated on a held-out config) as part of getting a reportable Table 3, not
as a bug to chase in the code. Once you have full-scale numbers, if
`full`/`seg+pos+rms+psd` still underperform `seg-only`/`seg+pos` after
tuning, that itself is a legitimate (if less flattering) finding to report
and discuss — do not force a result.

## 7. What is intentionally NOT filled in

`run_experiment1.py`'s Table 2 has a row labeled
`Learned denoiser + threshold [TBD -- not implemented]` — left blank on
purpose. Producing that number honestly requires training a dedicated
denoising network (e.g., a DnCNN- or U-Net-as-denoiser architecture) ahead
of the classical threshold step, which is a separate piece of work from the
metrology-aware *localization* pipeline these scripts implement. Do not
substitute a guessed or borrowed number for this row; either implement and
train that model and report the real result, or state plainly in the
manuscript that this particular comparison is left for future work.

More generally: every number these scripts produce is from a genuine
reimplementation run on synthetic data with known ground truth, using a
compact stand-in U-Net architecture (`TinyUNet`, ~66k parameters at
`base=12`) — not a reproduction of any specific published model's exact
architecture or hyperparameters. Report these results as your own
reimplementation's findings, and keep that distinction explicit in the
manuscript (the LaTeX in the parent directory's v1 draft uses a
`\pending{}`/scope-note convention for exactly this reason — carry the same
discipline into the two-edge tables).

## 8. Real-data metrology (`real_data_metrology.py`)

This is separate from the synthetic-training pipeline above: it does not
train anything, it extracts LER/LWR/PSD numbers directly from real SEM
images that already have some form of edge ground truth. Use it to
characterize a real dataset (e.g. one structured like the "Carinthia-S"-
style layout you described) and to check how it compares to the synthetic
grid the models above were trained on.

**It does not assume you have real data files yet** — you can read the
script and its docstring to see the exact structures it expects before you
have anything to point it at, then run it once you do.

### Supported ground-truth formats (auto-detected per image)

1. **Binary segmentation mask** image next to the SEM image (foreground =
   line), matched by filename — e.g. `image_001.png` + `image_001_mask.png`,
   or matched by the trailing number in the filename if the stems differ.
2. **Pre-extracted edge-coordinate JSON** — e.g. `image_001.tif` +
   `edges_001.json` containing `{"left_edge": [...], "right_edge": [...]}`
   (one value per row).
3. **Labelme polyline JSON** — `{"shapes": [{"label": "left_edge",
   "points": [[x,y], ...]}, {"label": "right_edge", "points": [...]}], ...}`,
   interpolated to one x per row.
4. **CSV/TSV manifest** listing the pairs explicitly, with `image_path` and
   `mask_path` columns (delimiter auto-detected; paths resolved relative to
   the manifest's own directory). Checked *first*, because a manifest-driven
   dataset can use opaque filenames — UUIDs, for instance — that carry
   neither a `mask` token nor digits, which defeats all three name-matching
   patterns above.

Run `discover_dataset()`'s logic (i.e. just run the script) against your
actual unpacked data first — it prints exactly how many pairs it found per
format, and lists any images it could NOT match to ground truth (rather
than silently dropping them). If your real file layout doesn't match any of
the three patterns, adjust the matching logic in `discover_dataset()` (it's
a single function, not spread across the file) rather than renaming your
whole dataset.

### Usage

```bash
python3 real_data_metrology.py --data-dir /path/to/carinthia-s --out-prefix real1

# Sanity-check file matching on a handful of images before running the full set:
python3 real_data_metrology.py --data-dir /path/to/carinthia-s --limit 20 --out-prefix real_smoke

# Also fit K-correlation (xi, alpha) per image and compare against the
# synthetic training grid (sigma in [1.5,3.0], xi in [8,20], alpha in [0.3,0.7]):
python3 real_data_metrology.py --data-dir /path/to/carinthia-s --fit-k-correlation --out-prefix real1
```

Outputs `{prefix}_per_image.csv` (one row per image: sigma_L, sigma_R,
sigma_W, mean linewidth, and xi/alpha if `--fit-k-correlation` was passed)
and `{prefix}_summary.csv` (mean/std/min/max/count across the dataset), and
prints what fraction of the real images fall inside vs. outside the
synthetic training grid — an honest, quantitative domain-gap statement for
the manuscript, not a claim of coverage you haven't actually checked.

**Important caveat about "Carinthia-S" specifically**: I was not able to
verify a public dataset by that exact name in a web search — it did not
turn up under that name, image count, or description on Zenodo, Kaggle,
GitHub, or elsewhere. That doesn't mean it doesn't exist (it may be
internal, differently spelled, or simply not indexed well), but before it
goes into the manuscript as a cited dataset, get a real, checkable source
for it (a paper, a DOI, a dataset landing page) — the same rule this
project has followed for every other citation. This script works on
whatever real data you actually have, regardless of what it's called; the
name just shouldn't appear in the paper unverified.

## 9. Troubleshooting

- **`ModuleNotFoundError: cv2`** — `pip install opencv-python` (not `cv2`,
  and not `opencv-python-headless` unless you specifically want the headless
  build; either works for this code since no GUI functions are used).
- **Training looks like it's hung** — it isn't; `verbose=True` (the default
  in the experiment scripts) prints one line per epoch, so if you don't see
  new lines for a long time on a large grid, check your `--rhos`/`--seeds`/
  `--epochs` product against the runtime guidance above before assuming a
  hang.
- **Results differ slightly between runs at the same settings** — this is
  expected: `torch`'s CPU convolution backward pass is not bit-exact across
  runs/thread counts even with a fixed seed. Differences should be small;
  large swings between runs at the same config usually mean the config
  count × per-config count is too small (too few items to average over).
