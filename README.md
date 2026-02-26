# Microarray Workstation

MicroarrayWorkstation is a cross-platform desktop tool (Windows/Linux) for microarray chip image analysis.

## Current MVP capabilities

- Load TIFF chip images (GenePix/InnoScan-compatible scanner outputs)
- Auto-detect spot candidates and infer a grid
- Local peak refinement around predicted grid positions
- Manual grid shift correction in GUI (left/right/up/down)
- Detection parameter tuning in GUI/CLI:
  - spot diameter min/max (px)
  - spot spacing min/max (px)
- Quantify foreground/background grayscale per spot
- Compute chip-level QC metrics (`mean_snr`, `pass_rate_pct`, `qc_status`)
- AI spot confidence scoring:
  - Heuristic quality scoring (default)
  - Optional ONNX model inference with automatic fallback to heuristic
- Generate interpretation calls from YAML templates
- Export raw and interpreted results (`CSV` + `JSON`)
- Submit interpreted results to LIMS via REST API
- Batch analysis mode for directories of chip images
- Batch LIMS submission for `*_summary.json`
- Native `laboratory_management` interface submission:
  - `POST /lab/interface/inbound/<endpoint_code>`
  - supports `none` / `bearer` / `api_key` / `basic` auth
- GUI and CLI workflows

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
microarray-workstation
```

CLI analysis example:

```bash
microarray-cli analyze \
  --image /path/to/chip.tif \
  --template /path/to/template.yaml \
  --output-dir ./outputs \
  --rows 48 \
  --cols 48
```

Batch analysis example:

```bash
microarray-cli analyze-batch \
  --input-dir /path/to/chips \
  --output-dir ./outputs \
  --rows 48 \
  --cols 48
```

Batch LIMS submit example:

```bash
microarray-cli submit-lims-batch \
  --base-url http://lims.local \
  --endpoint /api/results \
  --input-dir ./outputs
```

Laboratory management inbound interface example:

```bash
microarray-cli submit-lab-interface-batch \
  --base-url http://127.0.0.1:8069 \
  --endpoint-code HIS-REST \
  --input-dir ./outputs \
  --auth-type bearer \
  --token <TOKEN>
```

## Architecture

- `src/microarray_workstation/ui`: PySide6 desktop interface
- `src/microarray_workstation/analysis`: image processing, quantification, QC, AI scoring
- `src/microarray_workstation/workflows`: reusable single/batch orchestration
- `src/microarray_workstation/rules`: template-driven interpretation engine
- `src/microarray_workstation/integration`: LIMS API connector
- `src/microarray_workstation/io`: export utilities

## Template format

See `src/microarray_workstation/templates/default_template.yaml`.

## Notes

This repository is optimized for Codex-driven iterative delivery.
