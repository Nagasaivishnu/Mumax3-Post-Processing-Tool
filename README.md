# MuMax3 Post-Processing Tool

A modular desktop application for analysing MuMax3 simulation output.  
Supports **Hysteresis** and **FMR (Ferromagnetic Resonance)** analysis from `table.txt` files.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

---

## Workflow

### 1 · Load files
- Click **Add Files** (or `Ctrl+O`) to select one or more `table.txt` files.
- Double-click the **Label** cell to rename a dataset — labels appear in plot legends and CSV column headers.

### 2 · Hysteresis tab
| Step | Action |
|------|--------|
| Select columns | Choose X and Y axis columns from the dropdowns (auto-populated from the file headers). |
| Plot | Click **Plot** — all loaded datasets are overlaid. |
| Export | Click **Export CSV** to save merged data (outer-joined on X, NaN for missing points). |

Options: grid, log X/Y scale, legend toggle, inward ticks, line width, marker style.

### 3 · FMR tab
| Step | Action |
|------|--------|
| Set parameters | Enter **dt** (saving interval in seconds), **Bias field direction**, **MW field direction**. |
| Run | Click **Run FMR Processing** — FFT is computed in a background thread. |
| Heatmap | Switch to the **Heatmap** sub-tab to inspect |χ(f, B)|. |
| Slice | Switch to the **Slice Viewer** sub-tab — choose fixed frequency or fixed field, then click **Plot Slice**. |
| Export | Both sub-tabs have dedicated **Export CSV** buttons. |

---

## Project Structure

```
mumax_postproc/
├── main.py                   ← entry point
├── requirements.txt
├── gui/
│   ├── main_window.py        ← top-level window + menu
│   ├── file_manager.py       ← dataset list (add / label / remove)
│   ├── hysteresis_tab.py     ← hysteresis plotting + export
│   ├── fmr_tab.py            ← FMR heatmap + slice viewer + export
│   └── plot_canvas.py        ← reusable Matplotlib canvas widget
├── processing/
│   ├── data_loader.py        ← read table.txt → DataFrame
│   ├── hysteresis.py         ← data helpers for hysteresis
│   └── fmr.py                ← FFT / susceptibility (EDIT THIS for physics)
└── export/
    └── csv_export.py         ← all CSV-writing helpers
```

### Where to customise the physics
Open **`processing/fmr.py`** — that is the only file you need to touch to change:
- How `m(t)` is pre-processed before FFT
- Which FFT variant or windowing is applied
- How the susceptibility is normalised

---

## Edge Cases Handled
- Unequal file lengths → incomplete field runs are dropped with a warning.
- Missing columns → error dialog, no crash.
- Duplicate time-points → deduplicated automatically.
- Unequal dataset lengths in hysteresis export → outer join, NaN fill.
- GUI stays responsive during FMR processing (background QThread).
