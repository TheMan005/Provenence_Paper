# Satellite evidence research baseline

Experimental RGB tamper localization, controlled building edits, and signed processing history for SpaceNet 2. This implementation establishes a reproducible engineering baseline. Synthetic fixtures and short smoke runs are **not research results**. There is no demonstrated cross-city or unseen-generator performance.

## Installation

Use Python 3.11 on Linux. CPU PyTorch is sufficient; no GPU is assumed. Core and tested transitive versions are pinned. All installations below are local to the virtual environment.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

The prepared environment in this workspace already contains the CPU stack. For CUDA, create a separate environment, install the matching torch/torchvision versions from the appropriate official wheel index, and set `training.device: cuda` and `training.amp: true`. Consult the [official installation selector](https://pytorch.org/get-started/locally/) for hardware compatibility. CPU-only lock files should not be used to install CUDA wheels.

## Dataset and current download

The [SpaceNet Las Vegas page](https://spacenet.ai/las-vegas/) identifies the building extraction **training** archive as `SN2_buildings_train_AOI_2_Vegas.tar.gz`. S3 reports 25,640,007,327 bytes (23.88 GiB). It is a gzip-compressed tar archive, not ZIP. The destination is `Building_Dataset/` in this project.

```bash
.venv/bin/sat-evidence --config configs/pilot.yaml status
```

Do not start another transfer while the existing AWS process is active. The downloader's temporary files may be sparse; status uses allocated blocks as an approximate progress indicator, not logical file length. Extraction refuses incomplete/active downloads, verifies the published multipart S3 ETag using 8 MiB chunks, and uses Python's safe tar data filter. It refuses to overwrite extracted files. Keep the archive until checks are complete. A failed extraction may leave a partial extracted tree; inspect and choose a fresh destination before retrying rather than silently overwriting originals.

After the transfer completes, the exact real-data integration command is:

```bash
bash scripts/integrate_real.sh
```

This extracts safely, inventories **20 real tiles**, creates spatial splits, requests **10 controlled edits**, runs **three CPU training steps**, evaluates, renders panels/reports, and validates derivatives. No long training starts automatically. `configs/vegas_smoke.yaml` controls these limits. If a footprint or donor cannot support an edit, the generation summary records the failure and actual count; ten edits are requested, not guaranteed.

To leave an automatic continuation waiting in a separate terminal:

```bash
.venv/bin/python scripts/wait_for_real.py
```

Run only one continuation. It never starts, restarts, deletes, or rewrites a transfer. Once a completed archive exists it invokes the same integration command. A stopped download must be investigated separately; the waiter does not repair it.

Discovery searches recursively for GeoTIFFs in `PS-RGB` or `RGB-PanSharpen`, and matches building GeoJSON by IDs such as `AOI_2_Vegas_img123`. Annotation directories are discovered rather than hardcoded. Missing or ambiguous matches fail. CRS-free GeoJSON uses the configured default `EPSG:4326`; embedded CRS is honored. Missing raster CRS, inconsistent named RGB channels, or footprints wholly outside valid raster coverage fail. Spatial intersection is an alignment sanity check, not manual validation of every footprint. Preview overlays enable that inspection. Empty footprint files are legitimate authentic tiles.

## Engineering smoke test

```bash
.venv/bin/sat-evidence --config configs/smoke.yaml fixtures
.venv/bin/sat-evidence --config configs/smoke.yaml run
.venv/bin/sat-evidence --config configs/smoke.yaml validate
```

Fixture GeoTIFFs are marked `ENGINEERING_FIXTURE` and stored under ignored `outputs/synthetic/`. The first two commands have already run in the prepared workspace; use a new output/data directory to repeat from scratch. Split/normalization/generation manifests refuse overwrite. Never delete an experiment to silently change its assignments. Evaluation reuses frozen thresholds and rejects a changed checkpoint or split binding.

## Entry points and outputs

Paths in YAML are resolved relative to the working directory; run commands from the project root. For a 200–500-tile pilot, edit `inventory.pilot_tiles` in a new copy of `configs/pilot.yaml`, choosing a new `output_root` for each experiment.

```bash
.venv/bin/sat-evidence --config configs/pilot.yaml inventory
.venv/bin/sat-evidence --config configs/pilot.yaml split
.venv/bin/sat-evidence --config configs/pilot.yaml normalize
.venv/bin/sat-evidence --config configs/pilot.yaml previews
.venv/bin/sat-evidence --config configs/pilot.yaml generate
.venv/bin/sat-evidence --config configs/pilot.yaml validate
.venv/bin/sat-evidence --config configs/pilot.yaml train
.venv/bin/sat-evidence --config configs/pilot.yaml evaluate
.venv/bin/sat-evidence --config configs/pilot.yaml report --sample-id AOI_2_Vegas_img123_authentic
```

Use an actual sample ID from `evaluation/detector_results.csv` for the final command. `run` executes the sequence and reports on three test samples. Individual entry points support inspection between stages.

| Artifact | Purpose |
| --- | --- |
| `inventory.csv`, `pilot.csv` | Dimensions, channels, dtype, CRS, transform, bounds, nodata, valid range, footprint density, source hashes |
| `splits.csv`, `split_receipt.json` | Original tile assignments, coverage groups, spatial blocks, SHA-256 integrity binding |
| `normalization.json` | Frozen per-channel normalization fitted on training originals only |
| `previews/` | RGB/footprint alignment overlays |
| `samples/`, `samples.csv` | Lossless image exports, manipulation masks, valid-pixel masks, supervised/evaluation index |
| `ground_truth/generation.jsonl` | Restricted generation truth: source/donor IDs, operation, parameters, seed, history and hashes |
| `detector_inputs.csv` | Image/valid-mask inputs without donor, operation or ground-truth mask |
| `training/` | Best/last checkpoints, losses, parameter count, actual training budget and configuration |
| `evaluation/` | Frozen validation thresholds/calibrator, per-image metrics, robustness results, heatmaps and five-panel figures |
| `reports/` | Deterministic JSON/text assessments reading detector artifacts and provenance, not generation truth |

Ground-truth separation here is an interface boundary, not a security access-control system. Supervised training and evaluation necessarily read labels; inference and report generation do not read donor/operation history. Sample filenames can disclose an operation to a human; rename them to opaque identifiers before a blinded human study.

## Splits, normalization and edits

Splits are created **before** any edits. Tiles form spatial blocks along the wider geographic axis. Contiguous bands are assigned train/validation/test; exclusion corridors at boundaries remove tiles or overlapping coverage components that span partitions. Overlapping interiors are grouped; adjacent tiles from one mosaic can still occupy different blocks with the configured buffer. The validator checks cross-partition geometry distances, identical raster hashes, duplicate IDs, derivative assignments and donor membership. Crops must inherit the source assignment and pass the same derivative validator. No crop generator is included yet. Requested ratios are approximate after geographic exclusions; small or narrow pilots can fail with empty partitions. Increase coverage or adjust block/buffer dimensions explicitly.

For the later city experiment use `inventory.city: auto` and:

```yaml
split:
  block_m: 500
  buffer_m: 30
  ratios: [0.6, 0.2, 0.2]
  city_assignments:
    Vegas: train
    Shanghai: train
    Paris: validation
    Khartoum: test
```

No other city is downloaded by this code. A single inventory must contain those cities when that experiment is actually run. Until then, cross-city claims are unsupported.

RGB band indices are explicit (default `[1,2,3]`) and checked against named raster color interpretation when available. Undefined TIFF color interpretation requires checking the dataset documentation and preview overlays. Native higher-bit-depth pixels are read as float32; they are **not** blindly cast to uint8. The default policy fits channel-wise 1st/99th percentiles using valid pixels from training originals, clips/rescales with those frozen bounds, rounds to uint8, and exports PNG consistently for controls and edits. `normalization.mode: fixed` instead requires three-element `lower` and `upper` arrays, for example `[0,0,0]` and `[2047,2047,2047]` when justified by the sensor data. Nodata is zeroed and excluded from loss, scores and evaluation.

Insertion chooses annotated donor buildings within the recipient split, checks meter-per-pixel scale compatibility, rescales donor support accordingly, and tries valid unoccupied placements. Feathered blending support is included in the saved manipulation mask. Removal uses **classical OpenCV TELEA inpainting**, not diffusion. Footprints select candidate objects; the tamper mask is the operation's edited support including dilation/blending. PNG comparisons enforce unchanged pixels outside that support. Donor/source originals are never written. Generation seeds and actual parameters are recorded; scale/placement artifacts still require manual inspection. A bounded tile cache limits memory during pilots.

## Detection and evaluation

The RGB U-Net uses synchronized flips and nearest-neighbor mask/valid-mask resizing, BCE plus soft Dice loss, seeded loaders, CPU support and optional CUDA AMP. Authentic masks are empty. The optional `training.model: residual` uses separate RGB/high-pass stems fused before the common decoder. It is a testable architecture, not Noiseprint++ or TruFor. `training.base_channels` controls capacity; compare parameter counts and actual optimizer steps in `training/budget.json`, keeping training budget and data fixed. The residual branch has a backward-pass test but no research comparison yet.

CPU training requests strict deterministic algorithms. CUDA uses seeded execution and deterministic cuDNN settings but does not promise bitwise reproducibility because bilinear interpolation backward may be nondeterministic. CUDA has not been tested in the prepared CPU environment. The tested dependency set emits upstream deprecation warnings; they do not indicate failed checks.

The image score is the **mean of the highest configured fraction of valid localization sigmoid values** (default top 1%). It is an uncalibrated ranking score. Image F1 thresholds, a false-positive-constrained operating point and pixel IoU thresholds are selected on validation and reused unchanged on test. The small validation sample cannot establish a population false-positive rate. Calibration fits logistic score calibration on validation only; test Brier score and coverage/error under fixed abstention margins are reported. Calibration and threshold selection currently share validation; reserve a separate calibration cohort for a substantive study.

Metrics include image precision–recall trapezoidal AUC and average precision, precision/recall/F1, confusion matrices, recall/FPR at the validation-selected operating point, per-image manipulated IoU/pixel F1, operation/area breakdowns, and authentic flagged-pixel fractions. Authentic empty masks receive no artificial perfect IoU/F1. Resize, JPEG quality 70 and Gaussian blur are applied deterministically on test with frozen thresholds. Resize transforms masks/validity by nearest neighbor; compression and blur preserve mask geometry. Tiny edits may disappear after downsampling, in which case that condition uses its transformed mask label. Native-resolution localization and grouped confidence intervals remain future research work.

## Provenance and reporting

This is a custom SHA-256/Ed25519 sidecar prototype, **not C2PA-compliant**. It records parent hashes and declared processing operations from ingestion onward; it does not authenticate original satellite capture. Parent hashes are recorded but parent files are not recursively verified.

```bash
.venv/bin/sat-evidence keygen --key credentials/local.pem
# Put the returned public fingerprint into credentials/trusted.json:
# {"trusted_ed25519_sha256": ["PUBLIC_FINGERPRINT"]}
.venv/bin/sat-evidence sign --asset PATH_TO_EXPORT.png --key credentials/local.pem \
  --sidecar PATH_TO_EXPORT.png.provenance.json --operation ingest --operation lossless_export \
  --parent PATH_TO_SOURCE.tif
.venv/bin/sat-evidence verify --asset PATH_TO_EXPORT.png --sidecar PATH_TO_EXPORT.png.provenance.json
```

Keys are created exclusively with mode `0600`; `credentials/`, `*.pem` and `*.key` are ignored by Git. Trust is an explicit local fingerprint allowlist, not implied by any valid signature. Verification distinguishes `missing_credentials`, `invalid_signature`, `asset_hash_mismatch`, `valid_untrusted` and `valid_trusted`. Reports keep this status separate from detector scores, include connected flagged regions in original export pixel coordinates, list checks performed and not performed, and describe unresolved questions. They cannot conclude that a building was removed solely from a flagged region. No LLM API is used.

## Optional external models

`DiffusionInpaintingAdapter` loads only an existing local Diffusers inpainting checkpoint with `local_files_only=True`. Missing weights/dependencies fail clearly. No model is downloaded or trained automatically. Install the `diffusion` extra in a **separate environment** only when experimenting. The adapter is not wired into the baseline generation CLI; integrating it requires recording model/version, prompt and parameters in ground truth and designing an independently held-out generator experiment.

Use the [official TruFor implementation and inference instructions](https://github.com/grip-unina/TruFor/blob/main/TruFor_train_test/README.md) in its own environment, with author-provided weights obtained explicitly. Follow its license and do not replace its model with this U-Net. From the official `TruFor_train_test` directory:

```bash
python test.py -g -1 -in /absolute/path/to/export.png -out /absolute/path/to/official_output \
  -exp trufor_ph3 TEST.MODEL_FILE /absolute/path/to/author_weights.pth.tar
```

Official NPZ output uses `map` for localization. `optional.load_trufor_prediction(path, expected_shape)` imports and validates that map. For a comparison, collect official validation/test maps against the existing split manifest, use the same documented top-pixel aggregation and validation-only `evaluation.select_thresholds`, then `evaluation.summarize` on test. Full automated TruFor comparison is not implemented or run; its dependencies and weights are deliberately absent from the baseline environment.

Datasets, checkpoints, credentials, archives and large generated outputs are ignored by Git. Check SpaceNet's distribution terms before distributing dataset derivatives.
