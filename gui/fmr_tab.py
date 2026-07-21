"""
FMR Tab
========
FMR analysis: parameter inputs → heatmap generation → slice viewer.
The heavy lifting is in processing/fmr.py; this file is UI only.
"""

from __future__ import annotations
import logging

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QCheckBox, QLabel, QGroupBox,
    QFileDialog, QMessageBox, QSplitter, QDoubleSpinBox,
    QTabWidget, QRadioButton, QButtonGroup, QProgressBar,
    QScrollArea, QLineEdit,
)

from gui.plot_canvas import PlotCanvas
from gui.plot_style import style_axis
from processing.fmr import calc_susceptibility, get_absorption_curve, get_mfft_at_field
from export.csv_export import export_dataframe, build_heatmap_dataframe, build_slice_dataframe
import pandas as pd

logger = logging.getLogger(__name__)

COLORMAPS = ["jet", "inferno", "plasma", "viridis", "hot", "gray", "RdBu_r"]


# ---------------------------------------------------------------------------
# Background worker so the GUI stays responsive during FFT processing
# ---------------------------------------------------------------------------

class FMRWorker(QThread):
    finished = pyqtSignal(list)    # list of (label, fields, f, mFFTs)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)     # 0-100

    def __init__(self, entries, dt, bias_dir, mw_dir, interpolate, parent=None):
        super().__init__(parent)
        self.entries     = entries
        self.dt          = dt
        self.bias_dir    = bias_dir
        self.mw_dir      = mw_dir
        self.interpolate = interpolate

    def run(self):
        results = []
        n = len(self.entries)
        for i, entry in enumerate(self.entries):
            try:
                fields, f, mFFTs = calc_susceptibility(
                    entry.df, self.dt,
                    interpolate=self.interpolate,
                    BiasFieldDir=self.bias_dir,
                    MWFieldDir=self.mw_dir,
                )
                results.append((entry.label, fields, f, mFFTs))
            except Exception as exc:
                self.error.emit(f"Error processing '{entry.label}':\n{exc}")
            self.progress.emit(int((i + 1) / n * 100))
        self.finished.emit(results)


# ---------------------------------------------------------------------------
# Main FMR Tab
# ---------------------------------------------------------------------------

class FMRTab(QWidget):
    """
    Layout
    ------
    Left panel : parameters + run + progress
    Right panel: QTabWidget with "Heatmap" and "Slice Viewer" sub-tabs
    """

    def __init__(self, file_manager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fm      = file_manager
        self._results: list = []    # list of (label, fields, f, mFFTs)
        self._worker  = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(splitter)

        # ── LEFT: parameter panel ─────────────────────────────────────
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_widget.setMaximumWidth(270)
        splitter.addWidget(ctrl_widget)

        # Simulation parameters
        param_grp  = QGroupBox("FMR Parameters")
        param_form = QFormLayout(param_grp)

        self._dt_edit = QLineEdit("2e-11")
        self._dt_edit.setToolTip(
            "Saving interval dt [seconds]  –  scientific notation OK, e.g. 2e-11"
        )
        self._dt_edit.setPlaceholderText("e.g. 5e-12")
        param_form.addRow("dt (s):", self._dt_edit)

        self._bias_combo = QComboBox()
        self._bias_combo.addItems(["x", "y", "z"])
        self._bias_combo.setCurrentText("x")
        param_form.addRow("Bias field dir:", self._bias_combo)

        self._mw_combo = QComboBox()
        self._mw_combo.addItems(["x", "y", "z"])
        self._mw_combo.setCurrentText("y")
        param_form.addRow("MW field dir:", self._mw_combo)

        self._interp_chk = QCheckBox("Interpolate m(t)")
        self._interp_chk.setToolTip("Cubic spline re-sampling for uniform timestep")
        param_form.addRow("", self._interp_chk)

        ctrl_layout.addWidget(param_grp)

        # Heatmap display options
        disp_grp  = QGroupBox("Heatmap Options")
        disp_form = QFormLayout(disp_grp)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(COLORMAPS)
        disp_form.addRow("Colormap:", self._cmap_combo)

        self._log_chk = QCheckBox("Log colour scale")
        self._log_chk.setChecked(True)
        disp_form.addRow("", self._log_chk)

        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setRange(1e-10, 1.0)
        self._vmin_spin.setDecimals(10)
        self._vmin_spin.setValue(1e-3)
        self._vmin_spin.setToolTip("Relative vmin as fraction of max")
        disp_form.addRow("vmin (frac.):", self._vmin_spin)

        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(1e-10, 1.0)
        self._vmax_spin.setDecimals(10)
        self._vmax_spin.setValue(1e-1)
        self._vmax_spin.setToolTip("Relative vmax as fraction of max")
        disp_form.addRow("vmax (frac.):", self._vmax_spin)

        self._freq_max_spin = QDoubleSpinBox()
        self._freq_max_spin.setRange(0.1, 1000.0)
        self._freq_max_spin.setValue(10.0)
        self._freq_max_spin.setSuffix(" GHz")
        disp_form.addRow("Max freq:", self._freq_max_spin)

        ctrl_layout.addWidget(disp_grp)

        # Progress + run
        ctrl_layout.addStretch()
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        ctrl_layout.addWidget(self._progress)

        self._run_btn = QPushButton("Run FMR Processing")
        self._run_btn.clicked.connect(self._do_run)
        ctrl_layout.addWidget(self._run_btn)

        self._refresh_btn = QPushButton("Refresh Heatmaps")
        self._refresh_btn.clicked.connect(self._refresh_heatmaps)
        ctrl_layout.addWidget(self._refresh_btn)

        # ── RIGHT: analysis tabs ──────────────────────────────────────
        right_tabs = QTabWidget()
        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(1, 1)

        # -- Heatmap sub-tab --
        self._heatmap_tab = _HeatmapSubTab(self)
        right_tabs.addTab(self._heatmap_tab, "Heatmap")

        # -- Fixed-frequency slice tab --
        self._freq_slice_tab = _FreqSliceTab(self)
        right_tabs.addTab(self._freq_slice_tab, "Slice: Fixed Frequency")

        # -- Fixed-field slice tab --
        self._field_slice_tab = _FieldSliceTab(self)
        right_tabs.addTab(self._field_slice_tab, "Slice: Fixed Field")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _do_run(self) -> None:
        entries = self._fm.get_loaded_entries()
        if not entries:
            QMessageBox.information(self, "No Data", "Add files in the File Manager first.")
            return

        # Parse dt from the text field – accepts scientific notation like 2e-11
        try:
            dt_val = float(self._dt_edit.text().strip())
            if dt_val <= 0:
                raise ValueError("dt must be positive")
        except ValueError:
            QMessageBox.warning(
                self, "Invalid dt",
                f"Cannot parse dt value '{self._dt_edit.text()}'.\n"
                "Enter a positive number, e.g.  5e-12  or  0.000000000005"
            )
            return

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        self._worker = FMRWorker(
            entries,
            dt          = dt_val,
            bias_dir    = self._bias_combo.currentText(),
            mw_dir      = self._mw_combo.currentText(),
            interpolate = self._interp_chk.isChecked(),
        )
        self._worker.progress.connect(self._progress.setValue)
        self._worker.error.connect(lambda msg: QMessageBox.warning(self, "Processing Error", msg))
        self._worker.finished.connect(self._on_results_ready)
        self._worker.start()

    def _on_results_ready(self, results: list) -> None:
        self._results = results
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)

        if not results:
            QMessageBox.warning(self, "No Results", "Processing produced no valid results.")
            return

        self._refresh_heatmaps()
        self._freq_slice_tab.load_results(results)
        self._field_slice_tab.load_results(results)

    def _refresh_heatmaps(self) -> None:
        if not self._results:
            return
        self._heatmap_tab.plot_heatmaps(
            self._results,
            cmap        = self._cmap_combo.currentText(),
            log_scale   = self._log_chk.isChecked(),
            vmin_frac   = self._vmin_spin.value(),
            vmax_frac   = self._vmax_spin.value(),
            freq_max_ghz= self._freq_max_spin.value(),
        )

    # ── public accessor for slice tab ─────────────────────────────────
    @property
    def results(self):
        return self._results


# ---------------------------------------------------------------------------
# Heatmap sub-tab
# ---------------------------------------------------------------------------

class _HeatmapSubTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area so many datasets don't crush the view
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll)

        self._container = QWidget()
        self._c_layout  = QVBoxLayout(self._container)
        self._scroll.setWidget(self._container)

        # Export button
        self._export_btn = QPushButton("Export All Heatmap Data (CSV)…")
        self._export_btn.clicked.connect(self._do_export)
        layout.addWidget(self._export_btn)

        self._canvases: list[PlotCanvas] = []
        self._last_results = []

    def plot_heatmaps(self, results, cmap, log_scale, vmin_frac, vmax_frac, freq_max_ghz):
        self._last_results = results
        import matplotlib.colors as mcolors

        # clear previous
        for c in self._canvases:
            self._c_layout.removeWidget(c)
            c.deleteLater()
        self._canvases.clear()

        for label, fields, f, mFFTs in results:
            canvas = PlotCanvas(self._container, n_rows=1, n_cols=1, figsize=(8, 4))
            ax = canvas.single_ax
            self._c_layout.addWidget(canvas)
            self._canvases.append(canvas)

            # Frequency mask
            f_ghz   = f * 1e-9
            f_mask  = (f_ghz >= 2) & (f_ghz <= freq_max_ghz)
            f_plot  = f_ghz[f_mask]
            Z       = np.abs(mFFTs[f_mask, :])

            norm = None
            if log_scale:
                vmax = vmax_frac
                vmin = vmin_frac #vmax * vmin_frac if vmax > 0 else 1e-30
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

            extent = [fields[0], fields[-1], f_plot[0], f_plot[-1]]
            im = ax.imshow(
                Z, aspect="auto", origin="lower", extent=extent,
                cmap=cmap, norm=norm,
            )
            ax.set_xlabel("Magnetic Field (T)", fontsize=11)
            ax.set_ylabel("Frequency (GHz)", fontsize=11)
            ax.set_title(label, fontsize=12)
            # Origin style: box + inward ticks (no minor locators on an image)
            style_axis(ax, minor=False)
            cbar = canvas.fig.colorbar(im, ax=ax, label="|χ| (arb. units)")
            cbar.ax.tick_params(direction="in")
            canvas.draw()

    def _do_export(self):
        if not self._last_results:
            QMessageBox.information(self, "Nothing to export",
                                    "Run FMR processing first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Heatmap CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            frames = []
            for label, fields, f, mFFTs in self._last_results:
                df = build_heatmap_dataframe(fields, f, np.abs(mFFTs), label)
                frames.append(df)
            combined = frames[0] if len(frames) == 1 else \
                pd.merge(frames[0], frames[1], on=["Field_T", "Frequency_Hz"], how="outer")
            for extra in frames[2:]:
                combined = pd.merge(combined, extra, on=["Field_T", "Frequency_Hz"], how="outer")
            export_dataframe(combined, path)
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# ---------------------------------------------------------------------------
# Shared base for slice tabs — keeps common plot/export logic in one place
# ---------------------------------------------------------------------------

class _BaseSliceTab(QWidget):
    """
    Common scaffold for fixed-frequency and fixed-field slice tabs.
    Subclasses implement `_compute_slice(results, val)` and provide
    `_x_label`, `_y_label_template`, and `_val_unit` attributes.
    """

    _val_unit: str = ""           # displayed next to the input, e.g. "GHz" or "T"
    _val_default: float = 5.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list = []
        self._last_export_data = None   # (x_array, [(y, label), …], x_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── controls ──────────────────────────────────────────────────
        ctrl_grp  = QGroupBox("Slice Parameters")
        ctrl_form = QFormLayout(ctrl_grp)

        val_row = QHBoxLayout()
        self._val_edit = QLineEdit(str(self._val_default))
        self._val_edit.setPlaceholderText(f"value in {self._val_unit}")
        val_row.addWidget(self._val_edit)
        val_row.addWidget(QLabel(self._val_unit))
        ctrl_form.addRow("Value:", val_row)

        self._log_chk = QCheckBox("Log Y scale")
        ctrl_form.addRow("", self._log_chk)

        layout.addWidget(ctrl_grp)

        btn_row = QHBoxLayout()
        self._plot_btn   = QPushButton("Plot Slice")
        self._export_btn = QPushButton("Export CSV…")
        self._plot_btn.clicked.connect(self._do_plot)
        self._export_btn.clicked.connect(self._do_export)
        btn_row.addWidget(self._plot_btn)
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

        self._canvas = PlotCanvas(self, n_rows=1, n_cols=1, figsize=(8, 5))
        layout.addWidget(self._canvas)

    # ── public ────────────────────────────────────────────────────────

    def load_results(self, results: list) -> None:
        self._results = results

    # ── slots ─────────────────────────────────────────────────────────

    def _do_plot(self) -> None:
        if not self._results:
            QMessageBox.information(self, "No Data", "Run FMR processing first.")
            return

        try:
            val = float(self._val_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid Value",
                                f"Enter a number in {self._val_unit}.")
            return

        ax = self._canvas.single_ax
        self._canvas.clear_axes()

        import matplotlib.cm as cm
        cmap   = cm.get_cmap("tab10")
        y_sets = []
        x_plot = None

        for i, (label, fields, f, mFFTs) in enumerate(self._results):
            x_plot, y, x_label, y_label = self._compute_slice(
                fields, f, mFFTs, val
            )
            ax.plot(x_plot, y, label=label, color=cmap(i % 10))
            y_sets.append((y, label))

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.legend(frameon=False, fontsize=10)
        if self._log_chk.isChecked():
            ax.set_yscale("log")
        style_axis(ax)   # publication ("Origin") styling
        self._canvas.draw()

        self._last_export_data = (x_plot, y_sets, x_label)

    def _do_export(self) -> None:
        if self._last_export_data is None:
            QMessageBox.information(self, "Nothing to export", "Plot a slice first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Slice CSV", "", "CSV (*.csv)")
        if not path:
            return
        x, y_sets, x_label = self._last_export_data
        try:
            df = build_slice_dataframe(x, y_sets, x_label)
            export_dataframe(df, path)
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ── override in subclass ──────────────────────────────────────────

    def _compute_slice(self, fields, f, mFFTs, val):
        """Return (x_plot, y, x_label, y_label)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fixed-frequency slice  →  Absorption vs. Magnetic Field
# ---------------------------------------------------------------------------

class _FreqSliceTab(_BaseSliceTab):
    """
    User enters a frequency in GHz.
    Plots |mFFT| at that frequency as a function of applied field.
    """
    _val_unit    = "GHz"
    _val_default = 5.0

    def _compute_slice(self, fields, f, mFFTs, val):
        Fmeas = val * 1e9                          # GHz → Hz
        x, y  = get_absorption_curve(fields, mFFTs, f, Fmeas=Fmeas)
        x_label = "Magnetic Field (T)"
        y_label = f"Absorption at {val:.4g} GHz  (arb. units)"
        return x, y, x_label, y_label


# ---------------------------------------------------------------------------
# Fixed-field slice  →  |FFT(m)| vs. Frequency
# ---------------------------------------------------------------------------

class _FieldSliceTab(_BaseSliceTab):
    """
    User enters a field value in T.
    Plots |FFT(m(t))| at that field as a function of frequency.
    """
    _val_unit    = "T"
    _val_default = 0.05

    def _compute_slice(self, fields, f, mFFTs, val):
        x, y    = get_mfft_at_field(fields, mFFTs, f, Bstat=val)
        x_plot  = x * 1e-9                         # Hz → GHz
        x_label = "Frequency (GHz)"
        y_label = f"|FFT(m)| at {val:.4g} T  (arb. units)"
        return x_plot, y, x_label, y_label
