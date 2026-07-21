"""
Spin Wave Mode Profile Tab
===========================
Spatial spin-wave mode profiling from MuMax3 time-domain OVF snapshots.

Layout
------
Left panel  – all controls (loading, processing, peaks, visualisation)
Right panel – FMR spectrum plot (peaks labelled after Find Peaks)

Popup windows (ModeProfilePopup) open for each selected mode when the
user clicks "Show Selected Modes".

This tab is fully self-contained.  It does not modify any other tab.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QCheckBox, QLabel, QGroupBox,
    QFileDialog, QMessageBox, QSplitter, QLineEdit,
    QProgressBar, QSpinBox, QDialog, QScrollArea,
)

from gui.plot_canvas import PlotCanvas
from gui.plot_style import style_axis
from processing.mode_profile import (
    load_dataset, compute_fft, find_fmr_peaks, get_spatial_profile,
    fft_cache_path, save_fft_result, load_fft_result,
    COMPONENT_MAP, AVG_AXIS_MAP, AVG_TO_VIEW,
)

logger = logging.getLogger(__name__)

COLORMAPS = ["inferno", "viridis", "plasma", "magma", "jet", "gray"]

# colours for mode peak lines on the FMR spectrum plot
PEAK_COLORS = ["tab:red", "tab:green", "tab:blue", "tab:orange",
               "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class _LoadWorker(QThread):
    """Loads OVF/NPY dataset in a background thread."""
    progress = pyqtSignal(int, int)   # done, total
    status   = pyqtSignal(str)
    finished = pyqtSignal(object)     # np.ndarray | None
    error    = pyqtSignal(str)

    def __init__(self, sim_dir: str, parent=None) -> None:
        super().__init__(parent)
        self.sim_dir = sim_dir

    def run(self) -> None:
        try:
            data = load_dataset(
                self.sim_dir,
                progress_cb=lambda d, t: self.progress.emit(d, t),
                status_cb=lambda m: self.status.emit(m),
            )
            self.finished.emit(data)
        except Exception as exc:
            self.error.emit(str(exc))


class _FFTWorker(QThread):
    """
    Runs the FFT in a background thread, with file-based caching.

    If *use_cache* is True and a cache file matching the FFT parameters
    exists in *sim_dir*, it is loaded instead of recomputing.
    Otherwise the FFT is computed and the result saved to that cache file.
    """
    finished = pyqtSignal(object, str)   # result dict, source description
    error    = pyqtSignal(str)

    def __init__(self, m_raw, component, dt, t_start, t_end,
                 sim_dir: str, use_cache: bool = True, parent=None):
        super().__init__(parent)
        self.m_raw     = m_raw
        self.component = component
        self.dt        = dt
        self.t_start   = t_start
        self.t_end     = t_end
        self.sim_dir   = sim_dir
        self.use_cache = use_cache

    def run(self) -> None:
        try:
            cache = fft_cache_path(
                self.sim_dir, self.component,
                self.dt, self.t_start, self.t_end,
            )

            # ── fast path: load cached FFT ────────────────────────────
            if self.use_cache and cache.exists():
                result = load_fft_result(cache)
                self.finished.emit(result, f"cache: {cache.name}")
                return

            # ── compute + save ────────────────────────────────────────
            if self.m_raw is None:
                raise ValueError(
                    "No dataset loaded and no matching FFT cache file found.\n"
                    "Load the OVF dataset first."
                )
            result = compute_fft(
                self.m_raw, self.component,
                self.dt, self.t_start, self.t_end,
            )
            try:
                save_fft_result(
                    self.sim_dir, self.component,
                    self.dt, self.t_start, self.t_end, result,
                )
                source = f"computed, saved → {cache.name}"
            except OSError as exc:
                logger.warning("Could not save FFT cache: %s", exc)
                source = "computed (cache save failed)"
            self.finished.emit(result, source)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

class SpinWaveModeProfileTab(QWidget):

    def __init__(self, file_manager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fm = file_manager
        self._settings = QSettings("MuMax3Tool", "ModeProfileTab")

        # ── runtime state ─────────────────────────────────────────────
        self._raw_data: np.ndarray | None = None   # (n_time, nz, ny, nx, 3)
        self._fft_result: dict | None     = None   # from compute_fft
        self._peaks: list[dict]           = []
        self._load_worker: _LoadWorker | None = None
        self._fft_worker:  _FFTWorker  | None = None

        self._build_ui()
        self._fm.datasets_changed.connect(self._refresh_dataset_combo)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(splitter)

        # ── LEFT: scroll-able control panel ───────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(270)
        scroll.setMaximumWidth(290)

        ctrl_widget = QWidget()
        ctrl = QVBoxLayout(ctrl_widget)
        ctrl.setContentsMargins(6, 6, 6, 6)
        ctrl.setSpacing(6)
        scroll.setWidget(ctrl_widget)
        splitter.addWidget(scroll)

        # 1 ─ Dataset selection ────────────────────────────────────────
        ds_grp  = QGroupBox("Dataset")
        ds_form = QFormLayout(ds_grp)
        self._ds_combo = QComboBox()
        self._ds_combo.setToolTip("Select which loaded dataset to analyse")
        ds_form.addRow("Source:", self._ds_combo)
        ctrl.addWidget(ds_grp)

        # 2 ─ OVF loading ──────────────────────────────────────────────
        load_grp    = QGroupBox("Data Loading")
        load_layout = QVBoxLayout(load_grp)

        self._load_btn = QPushButton("Load OVF Dataset")
        self._load_btn.setToolTip(
            "Load OVF/NPY raw data.\n"
            "Skipped automatically when an FFT cache file matching the\n"
            "current Processing parameters already exists (the FFT is\n"
            "loaded from it directly instead)."
        )
        self._load_btn.clicked.connect(self._do_load)
        load_layout.addWidget(self._load_btn)

        self._force_load_chk = QCheckBox("Load raw data even if FFT cache exists")
        self._force_load_chk.setToolTip(
            "Override the FFT-cache check: always load the OVF/NPY raw\n"
            "data. Needed e.g. before 'Recompute FFT (ignore cache)' or\n"
            "when trying different FFT parameters."
        )
        load_layout.addWidget(self._force_load_chk)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        load_layout.addWidget(self._progress)

        self._status_lbl = QLabel("No data loaded.")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color: gray; font-size: 11px;")
        load_layout.addWidget(self._status_lbl)

        ctrl.addWidget(load_grp)

        # 3 ─ Processing parameters ────────────────────────────────────
        proc_grp  = QGroupBox("Processing")
        proc_form = QFormLayout(proc_grp)

        self._comp_combo = QComboBox()
        self._comp_combo.addItems(["Mx", "My", "Mz"])
        self._comp_combo.setCurrentText("My")
        proc_form.addRow("Component:", self._comp_combo)

        self._dt_edit = QLineEdit("5e-12")
        self._dt_edit.setPlaceholderText("e.g. 5e-12")
        self._dt_edit.setToolTip("Saving interval [s] – scientific notation OK")
        proc_form.addRow("dt (s):", self._dt_edit)

        self._tstart_edit = QLineEdit("3e-9")
        self._tstart_edit.setToolTip("Window start time [s]")
        proc_form.addRow("t start (s):", self._tstart_edit)

        self._tend_edit = QLineEdit("25e-9")
        self._tend_edit.setToolTip("Window end time [s]")
        proc_form.addRow("t end (s):", self._tend_edit)

        self._fft_btn = QPushButton("Compute FFT")
        self._fft_btn.setToolTip(
            "Uses a saved FFT cache file if one matches the current\n"
            "parameters (component, dt, time window); otherwise computes\n"
            "and saves a new cache file in the simulation directory."
        )
        self._fft_btn.clicked.connect(lambda: self._do_fft(use_cache=True))
        proc_form.addRow("", self._fft_btn)

        self._fft_force_btn = QPushButton("Recompute FFT (ignore cache)")
        self._fft_force_btn.setToolTip(
            "Recompute the FFT from the loaded dataset even if a cache\n"
            "file exists, and overwrite the cache file."
        )
        self._fft_force_btn.clicked.connect(lambda: self._do_fft(use_cache=False))
        proc_form.addRow("", self._fft_force_btn)

        ctrl.addWidget(proc_grp)

        # 4 ─ Peak detection ───────────────────────────────────────────
        peak_grp  = QGroupBox("Peak Detection")
        peak_form = QFormLayout(peak_grp)

        self._npeaks_spin = QSpinBox()
        self._npeaks_spin.setRange(1, 20)
        self._npeaks_spin.setValue(5)
        peak_form.addRow("No. of peaks:", self._npeaks_spin)

        self._fmin_edit = QLineEdit("1.0")
        self._fmin_edit.setToolTip("Lower frequency limit for peak search [GHz]")
        peak_form.addRow("f min (GHz):", self._fmin_edit)

        self._fmax_edit = QLineEdit("20.0")
        self._fmax_edit.setToolTip("Upper frequency limit for peak search [GHz]")
        peak_form.addRow("f max (GHz):", self._fmax_edit)

        self._peaks_btn = QPushButton("Find Peaks")
        self._peaks_btn.clicked.connect(self._do_find_peaks)
        peak_form.addRow("", self._peaks_btn)

        ctrl.addWidget(peak_grp)

        # 5 ─ Visualisation ────────────────────────────────────────────
        vis_grp  = QGroupBox("Mode Visualisation")
        vis_form = QFormLayout(vis_grp)

        self._modes_edit = QLineEdit("1,2,3")
        self._modes_edit.setToolTip("Comma-separated mode numbers to display, e.g. 1,4")
        vis_form.addRow("Modes:", self._modes_edit)

        self._avg_combo = QComboBox()
        self._avg_combo.addItems(["Z", "Y", "X", "None"])
        self._avg_combo.setToolTip(
            "Average the 3-D power along this axis before displaying.\n"
            "Z → XY plane,  Y → XZ plane,  X → YZ plane."
        )
        self._avg_combo.currentTextChanged.connect(self._on_avg_changed)
        vis_form.addRow("Average along:", self._avg_combo)

        self._plane_combo = QComboBox()
        self._plane_combo.addItems(["XY", "XZ", "YZ"])
        self._plane_combo.setToolTip("Display plane (auto-suggested after averaging)")
        vis_form.addRow("View plane:", self._plane_combo)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(COLORMAPS)
        vis_form.addRow("Colormap:", self._cmap_combo)

        self._scale_combo = QComboBox()
        self._scale_combo.addItems(["Log10", "Linear"])
        vis_form.addRow("Scale:", self._scale_combo)

        self._auto_range_chk = QCheckBox("Auto range")
        self._auto_range_chk.setChecked(True)
        self._auto_range_chk.toggled.connect(self._on_auto_range_toggled)
        vis_form.addRow("", self._auto_range_chk)

        self._vmin_edit = QLineEdit("-10")
        self._vmax_edit = QLineEdit("0")
        self._vmin_edit.setEnabled(False)
        self._vmax_edit.setEnabled(False)
        vis_form.addRow("Vmin:", self._vmin_edit)
        vis_form.addRow("Vmax:", self._vmax_edit)

        self._show_btn = QPushButton("Show Selected Modes")
        self._show_btn.clicked.connect(self._do_show_modes)
        vis_form.addRow("", self._show_btn)

        ctrl.addWidget(vis_grp)

        # 6 ─ PowerPoint export ────────────────────────────────────────
        ppt_grp    = QGroupBox("Export to PowerPoint")
        ppt_layout = QVBoxLayout(ppt_grp)

        path_row = QHBoxLayout()
        self._ppt_path_edit = QLineEdit(self._settings.value("ppt_path", ""))
        self._ppt_path_edit.setPlaceholderText("presentation.pptx")
        self._ppt_path_edit.setToolTip(
            "Target .pptx file. Created if it does not exist;\n"
            "otherwise a new slide is appended at the end."
        )
        path_row.addWidget(self._ppt_path_edit)

        ppt_browse_btn = QPushButton("…")
        ppt_browse_btn.setMaximumWidth(30)
        ppt_browse_btn.setToolTip("Browse for the target .pptx file")
        ppt_browse_btn.clicked.connect(self._browse_ppt)
        path_row.addWidget(ppt_browse_btn)
        ppt_layout.addLayout(path_row)

        self._ppt_save_btn = QPushButton("Save FFT + Modes to PPT")
        self._ppt_save_btn.setToolTip(
            "Appends one slide at the end of the selected .pptx containing\n"
            "the current FMR spectrum and the spatial profiles of the\n"
            "modes listed in the Modes field."
        )
        self._ppt_save_btn.clicked.connect(self._do_save_ppt)
        ppt_layout.addWidget(self._ppt_save_btn)

        ctrl.addWidget(ppt_grp)
        ctrl.addStretch()

        # ── RIGHT: FMR spectrum plot ───────────────────────────────────
        self._canvas = PlotCanvas(self, n_rows=1, n_cols=1, figsize=(8, 5))
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    # Dataset combo management
    # ------------------------------------------------------------------

    def _refresh_dataset_combo(self) -> None:
        prev = self._ds_combo.currentText()
        self._ds_combo.blockSignals(True)
        self._ds_combo.clear()
        for entry in self._fm.entries:
            self._ds_combo.addItem(entry.label, userData=entry)
        # restore selection if still present
        idx = self._ds_combo.findText(prev)
        if idx >= 0:
            self._ds_combo.setCurrentIndex(idx)
        self._ds_combo.blockSignals(False)

    def _current_entry(self):
        return self._ds_combo.currentData()

    def _current_sim_dir(self) -> Path | None:
        entry = self._current_entry()
        if entry is None:
            return None
        return Path(entry.path).parent

    # ------------------------------------------------------------------
    # Loading slot
    # ------------------------------------------------------------------

    def _current_cache_path(self, sim_dir: Path) -> Path | None:
        """FFT cache path for the current Processing parameters, or None if
        the parameter fields cannot be parsed."""
        try:
            dt      = float(self._dt_edit.text())
            t_start = float(self._tstart_edit.text())
            t_end   = float(self._tend_edit.text())
        except ValueError:
            return None
        component = COMPONENT_MAP[self._comp_combo.currentText()]
        return fft_cache_path(sim_dir, component, dt, t_start, t_end)

    def _do_load(self) -> None:
        sim_dir = self._current_sim_dir()
        if sim_dir is None:
            QMessageBox.information(self, "No Dataset",
                                    "Add table.txt files in the File Manager first.")
            return

        # ── skip raw-data loading when a matching FFT cache exists ────
        if not self._force_load_chk.isChecked():
            cache = self._current_cache_path(sim_dir)
            if cache is not None and cache.exists():
                self._status_lbl.setText(
                    f"FFT cache found ({cache.name}) — raw data load skipped. "
                    "Loading FFT from cache …"
                )
                self._do_fft(use_cache=True)
                return

        self._raw_data  = None
        self._fft_result = None
        self._peaks     = []
        self._load_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_lbl.setText("Starting …")

        self._load_worker = _LoadWorker(str(sim_dir), self)
        self._load_worker.progress.connect(self._on_load_progress)
        self._load_worker.status.connect(self._status_lbl.setText)
        self._load_worker.finished.connect(self._on_load_done)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _on_load_progress(self, done: int, total: int) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_load_done(self, data: np.ndarray) -> None:
        self._raw_data = data
        self._load_btn.setEnabled(True)
        self._progress.setVisible(False)
        shape_str = " × ".join(map(str, data.shape))
        self._status_lbl.setText(f"Loaded  [{shape_str}]  (t, z, y, x, comp)")

    def _on_load_error(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_lbl.setText("Load failed.")
        QMessageBox.critical(self, "Load Error", msg)

    # ------------------------------------------------------------------
    # FFT slot
    # ------------------------------------------------------------------

    def _do_fft(self, use_cache: bool = True) -> None:
        sim_dir = self._current_sim_dir()
        if sim_dir is None:
            QMessageBox.information(self, "No Dataset",
                                    "Add table.txt files in the File Manager first.")
            return

        try:
            dt      = float(self._dt_edit.text())
            t_start = float(self._tstart_edit.text())
            t_end   = float(self._tend_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "dt, t start, and t end must be numbers.")
            return

        component  = COMPONENT_MAP[self._comp_combo.currentText()]
        cache      = fft_cache_path(sim_dir, component, dt, t_start, t_end)
        has_cache  = cache.exists()

        # Raw data is only required when we actually have to compute
        if self._raw_data is None and not (use_cache and has_cache):
            QMessageBox.information(
                self, "No Data",
                "Load a dataset first.\n"
                "(No matching FFT cache file was found for these parameters.)"
            )
            return

        self._fft_btn.setEnabled(False)
        self._fft_force_btn.setEnabled(False)
        self._status_lbl.setText(
            "Loading FFT from cache …" if (use_cache and has_cache)
            else "Computing FFT …"
        )

        self._fft_worker = _FFTWorker(
            self._raw_data, component, dt, t_start, t_end,
            sim_dir=str(sim_dir), use_cache=use_cache, parent=self,
        )
        self._fft_worker.finished.connect(self._on_fft_done)
        self._fft_worker.error.connect(self._on_fft_error)
        self._fft_worker.start()

    def _on_fft_done(self, result: dict, source: str) -> None:
        self._fft_result = result
        self._peaks = []            # old peaks refer to the previous FFT
        self._fft_btn.setEnabled(True)
        self._fft_force_btn.setEnabled(True)
        n_freq = len(result["f"])
        self._status_lbl.setText(
            f"FFT done ({source})  –  {n_freq} frequency points  "
            f"({result['f'][-1]/1e9:.1f} GHz max)"
        )
        self._plot_spectrum()

    def _on_fft_error(self, msg: str) -> None:
        self._fft_btn.setEnabled(True)
        self._fft_force_btn.setEnabled(True)
        self._status_lbl.setText("FFT failed.")
        QMessageBox.critical(self, "FFT Error", msg)

    # ------------------------------------------------------------------
    # Peak detection slot
    # ------------------------------------------------------------------

    def _do_find_peaks(self) -> None:
        if self._fft_result is None:
            QMessageBox.information(self, "No FFT", "Compute FFT first.")
            return

        try:
            n_peaks   = self._npeaks_spin.value()
            f_min_ghz = float(self._fmin_edit.text())
            f_max_ghz = float(self._fmax_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "Frequency limits must be numbers.")
            return

        try:
            self._peaks = find_fmr_peaks(
                self._fft_result["f"],
                self._fft_result["P_int"],
                n_peaks, f_min_ghz, f_max_ghz,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Peak Detection", str(exc))
            return

        self._plot_spectrum(with_peaks=True)

        # Show summary in the status label
        summary = "  ".join(
            f"M{p['mode']}:{p['f_peak']/1e9:.3f} GHz" for p in self._peaks
        )
        self._status_lbl.setText(f"Peaks: {summary}")

    # ------------------------------------------------------------------
    # FMR spectrum plotting
    # ------------------------------------------------------------------

    def _plot_spectrum(self, with_peaks: bool = False) -> None:
        if self._fft_result is None:
            return

        f     = self._fft_result["f"]
        P_int = self._fft_result["P_int"]
        ax    = self._canvas.single_ax
        self._canvas.clear_axes()

        try:
            f_min_ghz = float(self._fmin_edit.text())
            f_max_ghz = float(self._fmax_edit.text())
        except ValueError:
            f_min_ghz, f_max_ghz = 0.0, f[-1] * 1e-9

        # Frequency window mask for display
        f_ghz = f * 1e-9
        mask  = (f_ghz >= f_min_ghz) & (f_ghz <= f_max_ghz)

        ax.plot(f_ghz[mask], P_int[mask], color="black", linewidth=1.5)
        ax.set_xlabel("Frequency (GHz)", fontsize=12)
        ax.set_ylabel("Integrated Power (arb. units)", fontsize=12)
        style_axis(ax)   # publication ("Origin") styling

        # Overlay peak markers
        if with_peaks and self._peaks:
            for pk in self._peaks:
                color     = PEAK_COLORS[(pk["mode"] - 1) % len(PEAK_COLORS)]
                f_ghz_pk  = pk["f_peak"] * 1e-9
                ax.axvline(f_ghz_pk, color=color, linestyle="--",
                           linewidth=1.5, label=f"Mode {pk['mode']}")
                ax.text(
                    f_ghz_pk + (f_max_ghz - f_min_ghz) * 0.005,
                    ax.get_ylim()[1] * 0.85,
                    f"Mode {pk['mode']}\n{f_ghz_pk:.3f} GHz",
                    color=color, fontsize=9, va="top", rotation=90,
                )
            ax.legend(frameon=False, fontsize=9)

        self._canvas.draw()

    # ------------------------------------------------------------------
    # Show mode profiles
    # ------------------------------------------------------------------

    def _do_show_modes(self) -> None:
        if self._fft_result is None:
            QMessageBox.information(self, "No FFT", "Compute FFT first.")
            return
        if not self._peaks:
            QMessageBox.information(self, "No Peaks",
                                    "Run Find Peaks first.")
            return

        # Parse requested mode numbers
        try:
            requested = [
                int(x.strip())
                for x in self._modes_edit.text().split(",")
                if x.strip()
            ]
        except ValueError:
            QMessageBox.warning(self, "Invalid Input",
                                "Modes must be comma-separated integers, e.g. 1,2,4")
            return

        peak_map = {p["mode"]: p for p in self._peaks}
        avg_axis = self._avg_combo.currentText()
        cmap     = self._cmap_combo.currentText()
        scale    = self._scale_combo.currentText()
        auto     = self._auto_range_chk.isChecked()

        try:
            vmin_val = float(self._vmin_edit.text()) if not auto else None
            vmax_val = float(self._vmax_edit.text()) if not auto else None
        except ValueError:
            vmin_val = vmax_val = None

        for mode_num in requested:
            if mode_num not in peak_map:
                QMessageBox.warning(
                    self, "Mode not found",
                    f"Mode {mode_num} was not detected.\n"
                    f"Available modes: {sorted(peak_map)}"
                )
                continue

            pk   = peak_map[mode_num]
            sp2d = get_spatial_profile(
                self._fft_result["P"], pk["pk_idx"], avg_axis
            )

            popup = ModeProfilePopup(
                sp2d       = sp2d,
                mode_num   = pk["mode"],
                f_peak_hz  = pk["f_peak"],
                avg_axis   = avg_axis,
                cmap       = cmap,
                scale      = scale,
                vmin       = vmin_val,
                vmax       = vmax_val,
                parent     = self,
            )
            popup.show()

    # ------------------------------------------------------------------
    # PowerPoint export
    # ------------------------------------------------------------------

    def _browse_ppt(self) -> None:
        start = self._ppt_path_edit.text().strip()
        path, _ = QFileDialog.getSaveFileName(
            self, "Select PowerPoint file", start,
            "PowerPoint (*.pptx)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if path:
            self._ppt_path_edit.setText(path)

    def _do_save_ppt(self) -> None:
        if self._fft_result is None:
            QMessageBox.information(self, "No FFT",
                                    "Compute FFT first — nothing to save.")
            return

        path_txt = self._ppt_path_edit.text().strip()
        if not path_txt:
            QMessageBox.warning(self, "No PPT File",
                                "Enter or browse the target .pptx file first.")
            return
        ppt_path = Path(path_txt)
        if ppt_path.suffix.lower() != ".pptx":
            ppt_path = ppt_path.with_suffix(".pptx")
        self._settings.setValue("ppt_path", str(ppt_path))

        import tempfile
        from matplotlib.figure import Figure

        tmpdir = Path(tempfile.mkdtemp(prefix="mumax_ppt_"))
        images: list[str] = []

        # 1 ─ FFT spectrum (current right-panel plot, incl. peak markers)
        spec_png = tmpdir / "fmr_spectrum.png"
        self._canvas.fig.savefig(spec_png, dpi=150, bbox_inches="tight")
        images.append(str(spec_png))

        # 2 ─ spatial mode profiles (modes listed in the Modes field)
        n_modes = 0
        if self._peaks:
            try:
                requested = [
                    int(x.strip())
                    for x in self._modes_edit.text().split(",")
                    if x.strip()
                ]
            except ValueError:
                QMessageBox.warning(self, "Invalid Input",
                                    "Modes must be comma-separated integers, e.g. 1,2,4")
                return

            peak_map = {p["mode"]: p for p in self._peaks}
            avg_axis = self._avg_combo.currentText()
            cmap     = self._cmap_combo.currentText()
            scale    = self._scale_combo.currentText()
            auto     = self._auto_range_chk.isChecked()
            try:
                vmin = float(self._vmin_edit.text()) if not auto else None
                vmax = float(self._vmax_edit.text()) if not auto else None
            except ValueError:
                vmin = vmax = None

            for mode_num in requested:
                pk = peak_map.get(mode_num)
                if pk is None:
                    logger.warning("PPT export: mode %d not detected, skipped", mode_num)
                    continue
                sp2d = get_spatial_profile(
                    self._fft_result["P"], pk["pk_idx"], avg_axis
                )
                fig = Figure(figsize=(5, 4), tight_layout=True)
                ax  = fig.add_subplot(111)
                draw_mode_profile(fig, ax, sp2d, pk["mode"], pk["f_peak"],
                                  avg_axis, cmap, scale, vmin, vmax)
                png = tmpdir / f"mode_{mode_num}.png"
                fig.savefig(png, dpi=150)
                images.append(str(png))
                n_modes += 1

        entry = self._current_entry()
        label = entry.label if entry is not None else "dataset"
        title = (
            f"{label} — FMR spectrum"
            + (f" + {n_modes} mode profile{'s' if n_modes != 1 else ''}"
               if n_modes else "")
            + f"  ({self._comp_combo.currentText()}, dt={self._dt_edit.text()} s, "
              f"window {self._tstart_edit.text()}–{self._tend_edit.text()} s)"
        )

        try:
            from export.ppt_export import append_images_slide
            append_images_slide(ppt_path, images, title)
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "python-pptx is not installed.\n\n"
                "Install it with:\n    pip install python-pptx"
            )
            return
        except PermissionError:
            QMessageBox.critical(
                self, "PPT Export Error",
                f"Cannot write to:\n{ppt_path}\n\n"
                "The file is probably open in PowerPoint — close it and "
                "try again."
            )
            return
        except Exception as exc:
            QMessageBox.critical(self, "PPT Export Error", str(exc))
            return

        QMessageBox.information(
            self, "Saved to PPT",
            f"Appended 1 slide with {len(images)} image"
            f"{'s' if len(images) != 1 else ''} to:\n{ppt_path}"
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _on_avg_changed(self, avg: str) -> None:
        """Auto-suggest the view plane when averaging direction changes."""
        suggested = AVG_TO_VIEW.get(avg, "XY")
        self._plane_combo.setCurrentText(suggested)

    def _on_auto_range_toggled(self, checked: bool) -> None:
        self._vmin_edit.setEnabled(not checked)
        self._vmax_edit.setEnabled(not checked)


# ---------------------------------------------------------------------------
# Shared mode-profile drawing (used by the popup AND the PPT export)
# ---------------------------------------------------------------------------

def draw_mode_profile(
    fig,
    ax,
    sp2d: np.ndarray,
    mode_num: int,
    f_peak_hz: float,
    avg_axis: str,
    cmap:  str = "inferno",
    scale: str = "Log10",
    vmin:  float | None = None,
    vmax:  float | None = None,
) -> None:
    """Render one 2-D spatial mode profile onto (fig, ax)."""
    view_label = AVG_TO_VIEW.get(avg_axis, "XY")

    if scale == "Log10":
        eps = sp2d[sp2d > 0].min() * 1e-3 if (sp2d > 0).any() else 1e-30
        Z   = np.log10(np.maximum(sp2d, eps))
    else:
        Z = sp2d

    if vmin is None or vmax is None:
        vmin_plot = float(Z.min())
        vmax_plot = float(Z.max())
    else:
        vmin_plot, vmax_plot = float(vmin), float(vmax)

    im = ax.imshow(
        Z, origin="lower", cmap=cmap, aspect="equal",
        vmin=vmin_plot, vmax=vmax_plot,
    )
    ax.set_title(
        f"Mode {mode_num}  –  {f_peak_hz/1e9:.4f} GHz\n"
        f"Avg: {avg_axis}  →  {view_label} plane",
        fontsize=11,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    x_lbl, y_lbl = list(view_label)
    ax.set_xlabel(x_lbl, fontsize=10)
    ax.set_ylabel(y_lbl, fontsize=10)

    scale_lbl = r"$\log_{10}$(Power)" if scale == "Log10" else "Power"
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(scale_lbl, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Spatial mode profile popup window
# ---------------------------------------------------------------------------

class ModeProfilePopup(QDialog):
    """
    Non-modal popup showing the 2-D spatial power map for one mode.
    The user can adjust colormap, scale, and colour limits interactively.
    """

    def __init__(
        self,
        sp2d:      np.ndarray,
        mode_num:  int,
        f_peak_hz: float,
        avg_axis:  str,
        cmap:      str   = "inferno",
        scale:     str   = "Log10",
        vmin:      float | None = None,
        vmax:      float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sp2d      = sp2d
        self._mode_num  = mode_num
        self._f_peak_hz = f_peak_hz
        self._avg_axis  = avg_axis

        freq_ghz = f_peak_hz * 1e-9
        self.setWindowTitle(f"Mode {mode_num}  –  {freq_ghz:.4f} GHz")
        self.setWindowFlag(Qt.WindowType.Window, True)  # stays open alongside main
        self.resize(550, 520)

        self._build_ui(cmap, scale, vmin, vmax)
        self._draw(cmap, scale, vmin, vmax)

    def _build_ui(self, cmap, scale, vmin, vmax) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── plot canvas ───────────────────────────────────────────────
        self._canvas = PlotCanvas(self, figsize=(5, 4))
        layout.addWidget(self._canvas)

        # ── control row ───────────────────────────────────────────────
        ctrl = QHBoxLayout()
        layout.addLayout(ctrl)

        ctrl.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(COLORMAPS)
        self._cmap_combo.setCurrentText(cmap)
        ctrl.addWidget(self._cmap_combo)

        ctrl.addWidget(QLabel("Scale:"))
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(["Log10", "Linear"])
        self._scale_combo.setCurrentText(scale)
        ctrl.addWidget(self._scale_combo)

        self._auto_chk = QCheckBox("Auto range")
        self._auto_chk.setChecked(vmin is None)
        self._auto_chk.toggled.connect(self._on_auto_toggled)
        ctrl.addWidget(self._auto_chk)

        ctrl.addWidget(QLabel("Vmin:"))
        self._vmin_edit = QLineEdit("" if vmin is None else str(vmin))
        self._vmin_edit.setMaximumWidth(60)
        self._vmin_edit.setEnabled(vmin is not None)
        ctrl.addWidget(self._vmin_edit)

        ctrl.addWidget(QLabel("Vmax:"))
        self._vmax_edit = QLineEdit("" if vmax is None else str(vmax))
        self._vmax_edit.setMaximumWidth(60)
        self._vmax_edit.setEnabled(vmax is not None)
        ctrl.addWidget(self._vmax_edit)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._on_refresh)
        ctrl.addWidget(refresh_btn)

    def _draw(self, cmap, scale, vmin, vmax) -> None:
        ax = self._canvas.single_ax
        self._canvas.clear_axes()
        draw_mode_profile(
            self._canvas.fig, ax, self._sp2d,
            self._mode_num, self._f_peak_hz, self._avg_axis,
            cmap, scale, vmin, vmax,
        )
        self._canvas.draw()

    def _on_refresh(self) -> None:
        cmap  = self._cmap_combo.currentText()
        scale = self._scale_combo.currentText()
        auto  = self._auto_chk.isChecked()
        try:
            vmin = None if auto else float(self._vmin_edit.text())
            vmax = None if auto else float(self._vmax_edit.text())
        except ValueError:
            vmin = vmax = None
        self._draw(cmap, scale, vmin, vmax)

    def _on_auto_toggled(self, checked: bool) -> None:
        self._vmin_edit.setEnabled(not checked)
        self._vmax_edit.setEnabled(not checked)
