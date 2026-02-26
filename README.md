# Microarray Workstation

MicroarrayWorkstation is a cross-platform desktop tool (Windows/Linux) for microarray chip image analysis.

## Current MVP capabilities

- Load TIFF chip images (GenePix/InnoScan-compatible scanner outputs)
- Auto-detect spot candidates and infer a grid
- Local peak refinement around predicted grid positions
- Manual grid shift correction in GUI (left/right/up/down)
- Quantify foreground/background grayscale per spot
- Compute chip-level QC metrics (`mean_snr`, `pass_rate_pct`, `qc_status`)
- Generate interpretation calls from YAML templates
- Export raw and interpreted results (`CSV` + `JSON`)
- Submit interpreted results to LIMS via REST API
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

## Architecture

- `src/microarray_workstation/ui`: PySide6 desktop interface
- `src/microarray_workstation/analysis`: image processing and quantification pipeline
- `src/microarray_workstation/rules`: template-driven interpretation engine
- `src/microarray_workstation/integration`: LIMS API connector
- `src/microarray_workstation/io`: export utilities

## Template format

See `src/microarray_workstation/templates/default_template.yaml`.

## Notes

This repository is optimized for Codex-driven iterative delivery.
