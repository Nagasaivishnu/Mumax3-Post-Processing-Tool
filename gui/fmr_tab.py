"""
FMR Tab
========
FMR analysis: parameter inputs → heatmap generation → slice viewer.
The heavy lifting is in processing/fmr.py; this file is UI only.
"""

from __future__ import annotations
import logging

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
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
# Shared heatmap renderer  (used by the on-screen view AND the PPT export)
# ---------------------------------------------------------------------------

def render_heatmap(
    fig,
    ax,
    label: str,
    fields: np.ndarray,
    f: np.ndarray,
    mFFTs: np.ndarray,
    cmap: str,
    log_scale: bool,
    auto_range: bool,
    vmin: float,
    vmax: float,
    freq_max_ghz: float,
    freq_min_ghz: float = 2.0,
) -> None:
    """
    Draw one |χ(f, B)| heatmap onto (fig, ax) in publication ("Origin") style.

    Colour limits
    -------------
    auto_range=True   → limits taken from the data (log: min-positive…max).
    auto_range=False  → vmin/vmax are ABSOLUTE |χ| values.
    """
    import matplotlib.colors as mcolors

    f_ghz  = f * 1e-9
    f_mask = (f_ghz >= freq_min_ghz) & (f_ghz <= freq_max_ghz)
    f_plot = f_ghz[f_mask]
    Z      = np.abs(mFFTs[f_mask, :])

    Zmax = float(Z.max()) if Z.size else 1.0

    # ── resolve colour limits ──────────────────────────────────────────
    if auto_range:
        if log_scale:
            pos  = Z[Z > 0]
            vmin = float(pos.min()) if pos.size else Zmax * 1e-3
            vmax = Zmax
        else:
            vmin = vmax = None      # let imshow autoscale
    else:
        # absolute values as entered by the user
        if log_scale and (vmin is None or vmin <= 0):
            vmin = (vmax if vmax else Zmax) * 1e-3

    norm = None
    imshow_kw = {}
    if log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        imshow_kw = {"vmin": vmin, "vmax": vmax}

    extent = [fields[0], fields[-1], f_plot[0], f_plot[-1]]
    im = ax.imshow(
        Z, aspect="auto", origin="lower", extent=extent,
        cmap=cmap, norm=norm, **imshow_kw,
    )
    ax.set_xlabel("Magnetic Field (T)", fontsize=11)
    ax.set_ylabel("Frequency (GHz)", fontsize=11)
    ax.set_title(label, fontsize=12)
    style_axis(ax, minor=False)          # Origin box + inward ticks
    cbar = fig.colorbar(im, ax=ax, label="|χ| (arb. units)")
    cbar.ax.tick_params(direction="in")


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

        self._auto_range_chk = QCheckBox("Auto range")
        self._auto_range_chk.setChecked(True)
        self._auto_range_chk.setToolTip(
            "Take the colour limits from the data.\n"
            "Uncheck to set vmin/vmax manually (as a fraction of the max)."
        )
        self._auto_range_chk.toggled.connect(self._on_auto_range_toggled)
        disp_form.addRow("", self._auto_range_chk)

        self._vmin_edit = QLineEdit("1e-3")
        self._vmin_edit.setToolTip("Absolute vmin for |χ| (scientific notation OK, e.g. 1e-3)")
        disp_form.addRow("vmin:", self._vmin_edit)

        self._vmax_edit = QLineEdit("1e-1")
        self._vmax_edit.setToolTip("Absolute vmax for |χ| (scientific notation OK, e.g. 1e-1)")
        disp_form.addRow("vmax:", self._vmax_edit)

        # start disabled — Auto range is on by default
        self._vmin_edit.setEnabled(False)
        self._vmax_edit.setEnabled(False)

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

    def _on_auto_range_toggled(self, checked: bool) -> None:
        self._vmin_edit.setEnabled(not checked)
        self._vmax_edit.setEnabled(not checked)

    def _refresh_heatmaps(self) -> None:
        if not self._results:
            return
        auto = self._auto_range_chk.isChecked()
        vmin = vmax = None
        if not auto:
            try:
                vmin = float(self._vmin_edit.text())
                vmax = float(self._vmax_edit.text())
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid Range",
                    "vmin and vmax must be numbers (e.g. 1e-3). "
                    "Falling back to Auto range."
                )
                auto = True
        self._heatmap_tab.plot_heatmaps(
            self._results,
            cmap        = self._cmap_combo.currentText(),
            log_scale   = self._log_chk.isChecked(),
            auto_range  = auto,
            vmin        = vmin,
            vmax        = vmax,
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
        self._settings = QSettings("MuMax3Tool", "FMRHeatmap")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area so many datasets don't crush the view
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll)

        self._container = QWidget()
        self._c_layout  = QVBoxLayout(self._container)
        self._scroll.setWidget(self._container)

        # ── PowerPoint export row ─────────────────────────────────────
        ppt_row = QHBoxLayout()
        ppt_row.addWidget(QLabel("PPT:"))
        self._ppt_path_edit = QLineEdit(self._settings.value("ppt_path", ""))
        self._ppt_path_edit.setPlaceholderText("presentation.pptx")
        self._ppt_path_edit.setToolTip(
            "Target .pptx file. A new slide with the heatmap image(s) —\n"
            "resized and restyled in Origin style — is appended at the end."
        )
        ppt_row.addWidget(self._ppt_path_edit)
        ppt_browse = QPushButton("…")
        ppt_browse.setMaximumWidth(30)
        ppt_browse.clicked.connect(self._browse_ppt)
        ppt_row.addWidget(ppt_browse)
        self._ppt_btn = QPushButton("Save Heatmaps to PPT")
        self._ppt_btn.clicked.connect(self._do_save_ppt)
        ppt_row.addWidget(self._ppt_btn)
        layout.addLayout(ppt_row)

        # Export button
        self._export_btn = QPushButton("Export All Heatmap Data (CSV)…")
        self._export_btn.clicked.connect(self._do_export)
        layout.addWidget(self._export_btn)

        self._canvases: list[PlotCanvas] = []
        self._last_results = []
        self._last_params: dict = {}

    def plot_heatmaps(self, results, cmap, log_scale, auto_range,
                      vmin, vmax, freq_max_ghz):
        self._last_results = results
        self._last_params = dict(
            cmap=cmap, log_scale=log_scale, auto_range=auto_range,
            vmin=vmin, vmax=vmax, freq_max_ghz=freq_max_ghz,
        )

        # clear previous
        for c in self._canvases:
            self._c_layout.removeWidget(c)
            c.deleteLater()
        self._canvases.clear()

        for label, fields, f, mFFTs in results:
            canvas = PlotCanvas(self._container, n_rows=1, n_cols=1, figsize=(8, 4))
            self._c_layout.addWidget(canvas)
            self._canvases.append(canvas)
            render_heatmap(
                canvas.fig, canvas.single_ax, label, fields, f, mFFTs,
                cmap=cmap, log_scale=log_scale, auto_range=auto_range,
                vmin=vmin, vmax=vmax, freq_max_ghz=freq_max_ghz,
            )
            canvas.draw()

    # ── PowerPoint export ─────────────────────────────────────────────

    def _browse_ppt(self):
        start = self._ppt_path_edit.text().strip()
        path, _ = QFileDialog.getSaveFileName(
            self, "Select PowerPoint file", start, "PowerPoint (*.pptx)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if path:
            self._ppt_path_edit.setText(path)

    def _do_save_ppt(self):
        if not self._last_results:
            QMessageBox.information(self, "Nothing to export",
                                    "Run FMR processing first.")
            return
        path_txt = self._ppt_path_edit.text().strip()
        if not path_txt:
            QMessageBox.warning(self, "No PPT File",
                                "Enter or browse the target .pptx file first.")
            return
        from pathlib import Path
        ppt_path = Path(path_txt)
        if ppt_path.suffix.lower() != ".pptx":
            ppt_path = ppt_path.with_suffix(".pptx")
        self._settings.setValue("ppt_path", str(ppt_path))

        import tempfile
        from matplotlib.figure import Figure

        p = self._last_params
        tmpdir = Path(tempfile.mkdtemp(prefix="mumax_fmr_ppt_"))
        images = []
        for i, (label, fields, f, mFFTs) in enumerate(self._last_results):
            # Fresh figure at a fixed publication size, Origin-styled.
            fig = Figure(figsize=(6.5, 4.0), dpi=300, tight_layout=True)
            ax  = fig.add_subplot(111)
            render_heatmap(
                fig, ax, label, fields, f, mFFTs,
                cmap=p["cmap"], log_scale=p["log_scale"],
                auto_range=p["auto_range"], vmin=p["vmin"],
                vmax=p["vmax"], freq_max_ghz=p["freq_max_ghz"],
            )
            png = tmpdir / f"heatmap_{i}.png"
            fig.savefig(png)
            images.append(str(png))

        title = f"FMR heatmaps — {len(images)} dataset" \
                f"{'s' if len(images) != 1 else ''}"
        try:
            from export.ppt_export import append_images_slide
            append_images_slide(ppt_path, images, title)
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "python-pptx is not installed.\n\nInstall it with:\n"
                "    pip install python-pptx"
            )
            return
        except PermissionError:
            QMessageBox.critical(
                self, "PPT Export Error",
                f"Cannot write to:\n{ppt_path}\n\n"
                "The file is probably open in PowerPoint — close it and try again."
            )
            return
        except Exception as e:
            QMessageBox.critical(self, "PPT Export Error", str(e))
            return
        QMessageBox.information(
            self, "Saved to PPT",
            f"Appended 1 slide with {len(images)} heatmap"
            f"{'s' if len(images) != 1 else ''} to:\n{ppt_path}"
        )

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
