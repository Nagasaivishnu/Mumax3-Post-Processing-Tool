"""
Hysteresis Tab
===============
Plotting and export for MuMax3 hysteresis table.txt data.
"""

from __future__ import annotations
import logging

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QCheckBox, QLabel, QGroupBox,
    QFileDialog, QMessageBox, QSplitter, QDoubleSpinBox,
    QSpinBox, QProgressBar,
)

from gui.plot_canvas import PlotCanvas
from gui.plot_style import style_axis
from processing.hysteresis import extract_xy, merge_datasets
from processing.ovf_video import make_video, video_cache_path, find_ovf_frames
from export.csv_export import export_dataframe

logger = logging.getLogger(__name__)

# Matplotlib line styles and markers available in the UI
LINE_STYLES = ["-", "--", "-.", ":", "None"]
MARKERS     = ["None", "o", "s", "^", "v", "D", "x", "+"]
COLORMAPS   = ["tab10", "Dark2", "Set1", "Set2"]


class _VideoWorker(QThread):
    """Builds an OVF → MP4 movie in a background thread."""
    progress = pyqtSignal(int, int)   # done, total
    status   = pyqtSignal(str)
    finished = pyqtSignal(str)        # video path
    error    = pyqtSignal(str)

    def __init__(self, sim_dir: str, fps: int, parent=None) -> None:
        super().__init__(parent)
        self.sim_dir = sim_dir
        self.fps     = fps

    def run(self) -> None:
        try:
            path = make_video(
                self.sim_dir, fps=self.fps,
                progress_cb=lambda d, t: self.progress.emit(d, t),
                status_cb=lambda m: self.status.emit(m),
            )
            self.finished.emit(str(path))
        except Exception as exc:
            self.error.emit(str(exc))


class HysteresisTab(QWidget):
    """
    Layout
    ------
    ┌──────────────────────────────────────────────────────┐
    │  Controls (left)  │  Plot canvas (right)             │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, file_manager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fm = file_manager
        self._build_ui()
        # Refresh dropdowns whenever file list changes
        self._fm.datasets_changed.connect(self._refresh_columns)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(splitter)

        # ── LEFT: controls ────────────────────────────────────────────
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_widget.setMaximumWidth(260)
        splitter.addWidget(ctrl_widget)

        # Axis selection
        axis_grp = QGroupBox("Axis Selection")
        axis_form = QFormLayout(axis_grp)
        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        axis_form.addRow("X axis:", self._x_combo)
        axis_form.addRow("Y axis:", self._y_combo)
        ctrl_layout.addWidget(axis_grp)

        # Plot appearance
        app_grp = QGroupBox("Appearance")
        app_form = QFormLayout(app_grp)

        self._lw_spin = QDoubleSpinBox()
        self._lw_spin.setRange(0.5, 5.0)
        self._lw_spin.setSingleStep(0.5)
        self._lw_spin.setValue(1.5)
        app_form.addRow("Line width:", self._lw_spin)

        self._ms_spin = QSpinBox()
        self._ms_spin.setRange(0, 12)
        self._ms_spin.setValue(0)
        app_form.addRow("Marker size:", self._ms_spin)

        self._marker_combo = QComboBox()
        self._marker_combo.addItems(MARKERS)
        app_form.addRow("Marker:", self._marker_combo)

        ctrl_layout.addWidget(app_grp)

        # Toggles
        tog_grp = QGroupBox("Options")
        tog_layout = QVBoxLayout(tog_grp)
        self._chk_grid    = QCheckBox("Show grid")
        self._chk_logx    = QCheckBox("Log X scale")
        self._chk_logy    = QCheckBox("Log Y scale")
        self._chk_legend  = QCheckBox("Show legend")
        self._chk_legend.setChecked(True)
        self._chk_ticks   = QCheckBox("Inward ticks")
        self._chk_ticks.setChecked(True)
        for chk in (self._chk_grid, self._chk_logx, self._chk_logy,
                    self._chk_legend, self._chk_ticks):
            tog_layout.addWidget(chk)
        ctrl_layout.addWidget(tog_grp)

        # Axis labels (manual override)
        lbl_grp = QGroupBox("Axis Labels")
        lbl_form = QFormLayout(lbl_grp)
        from PyQt6.QtWidgets import QLineEdit
        self._xlabel_edit = QLineEdit()
        self._ylabel_edit = QLineEdit()
        self._xlabel_edit.setPlaceholderText("auto")
        self._ylabel_edit.setPlaceholderText("auto")
        lbl_form.addRow("X label:", self._xlabel_edit)
        lbl_form.addRow("Y label:", self._ylabel_edit)
        ctrl_layout.addWidget(lbl_grp)

        # X-axis range (manual override)
        xr_grp  = QGroupBox("X-axis Range")
        xr_form = QFormLayout(xr_grp)
        self._xauto_chk = QCheckBox("Auto range")
        self._xauto_chk.setChecked(True)
        self._xauto_chk.toggled.connect(self._on_xauto_toggled)
        xr_form.addRow("", self._xauto_chk)
        self._xmin_edit = QLineEdit()
        self._xmax_edit = QLineEdit()
        self._xmin_edit.setPlaceholderText("min")
        self._xmax_edit.setPlaceholderText("max")
        self._xmin_edit.setToolTip("Lower X limit (scientific notation OK, e.g. -0.05)")
        self._xmax_edit.setToolTip("Upper X limit (scientific notation OK, e.g. 0.05)")
        self._xmin_edit.setEnabled(False)
        self._xmax_edit.setEnabled(False)
        xr_form.addRow("X min:", self._xmin_edit)
        xr_form.addRow("X max:", self._xmax_edit)
        ctrl_layout.addWidget(xr_grp)

        # ── OVF movie ─────────────────────────────────────────────────
        vid_grp    = QGroupBox("OVF Movie")
        vid_layout = QVBoxLayout(vid_grp)

        vid_form = QFormLayout()
        self._vid_ds_combo = QComboBox()
        self._vid_ds_combo.setToolTip(
            "Dataset whose directory holds the numbered OVF frames\n"
            "(0000.ovf, 0001.ovf, …)."
        )
        vid_form.addRow("Source:", self._vid_ds_combo)

        self._vid_fps_spin = QSpinBox()
        self._vid_fps_spin.setRange(1, 60)
        self._vid_fps_spin.setValue(10)
        self._vid_fps_spin.setSuffix(" fps")
        vid_form.addRow("Frame rate:", self._vid_fps_spin)
        vid_layout.addLayout(vid_form)

        self._vid_rebuild_chk = QCheckBox("Rebuild (ignore cached video)")
        self._vid_rebuild_chk.setToolTip(
            "Regenerate the movie even if ovf_movie.mp4 already exists."
        )
        vid_layout.addWidget(self._vid_rebuild_chk)

        self._vid_btn = QPushButton("Create / View Video")
        self._vid_btn.setToolTip(
            "Convert the numbered OVF frames to an MP4 (via mumax3-convert)\n"
            "and open it. The video is cached, so next time it opens directly."
        )
        self._vid_btn.clicked.connect(self._do_video)
        vid_layout.addWidget(self._vid_btn)

        self._vid_progress = QProgressBar()
        self._vid_progress.setVisible(False)
        vid_layout.addWidget(self._vid_progress)

        self._vid_status = QLabel("")
        self._vid_status.setWordWrap(True)
        self._vid_status.setStyleSheet("color: gray; font-size: 11px;")
        vid_layout.addWidget(self._vid_status)

        ctrl_layout.addWidget(vid_grp)

        self._vid_worker = None

        ctrl_layout.addStretch()

        # Action buttons
        self._plot_btn   = QPushButton("Plot")
        self._export_btn = QPushButton("Export CSV")
        self._plot_btn.clicked.connect(self._do_plot)
        self._export_btn.clicked.connect(self._do_export)
        ctrl_layout.addWidget(self._plot_btn)
        ctrl_layout.addWidget(self._export_btn)

        # ── RIGHT: canvas ─────────────────────────────────────────────
        self._canvas = PlotCanvas(self, n_rows=1, n_cols=1, figsize=(8, 6))
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(1, 1)

        # Internal state
        self._last_merge_df = None   # used for export

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _refresh_columns(self) -> None:
        """Repopulate x/y dropdowns from the first loaded file's columns."""
        entries = self._fm.entries
        self._refresh_video_datasets()
        if not entries:
            self._x_combo.clear()
            self._y_combo.clear()
            return
        try:
            cols = list(entries[0].df.columns)
        except Exception:
            return

        self._x_combo.blockSignals(True)
        self._y_combo.blockSignals(True)
        prev_x = self._x_combo.currentText()
        prev_y = self._y_combo.currentText()
        self._x_combo.clear()
        self._y_combo.clear()
        self._x_combo.addItems(cols)
        self._y_combo.addItems(cols)
        # restore previous selection if still available
        if prev_x in cols:
            self._x_combo.setCurrentText(prev_x)
        if prev_y in cols:
            self._y_combo.setCurrentText(prev_y)
        self._x_combo.blockSignals(False)
        self._y_combo.blockSignals(False)

    def _do_plot(self) -> None:
        entries = self._fm.get_loaded_entries()
        if not entries:
            QMessageBox.information(self, "No Data", "Please add files first.")
            return

        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()
        if not x_col or not y_col:
            QMessageBox.warning(self, "No Columns", "Select X and Y columns.")
            return

        ax = self._canvas.single_ax
        self._canvas.clear_axes()

        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab10")
        lw     = self._lw_spin.value()
        ms     = self._ms_spin.value() or None
        marker = self._marker_combo.currentText()
        if marker == "None":
            marker = None

        merge_inputs = []

        for i, entry in enumerate(entries):
            df = entry.df
            if x_col not in df.columns or y_col not in df.columns:
                logger.warning("'%s' missing column '%s' or '%s'", entry.label, x_col, y_col)
                continue
            x, y = extract_xy(df, x_col, y_col)
            color = cmap(i % 10)
            ax.plot(
                x, y,
                label=entry.label,
                linewidth=lw,
                marker=marker,
                markersize=ms,
                color=color,
            )
            merge_inputs.append((x, y, entry.label))

        # Store for export
        self._last_merge_df = merge_datasets(merge_inputs, x_col) if merge_inputs else None

        # Apply options
        x_label = self._xlabel_edit.text() or x_col
        y_label = self._ylabel_edit.text() or y_col
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        if self._chk_grid.isChecked():
            ax.grid(True, alpha=0.4)

        if self._chk_logx.isChecked():
            ax.set_xscale("log")
        if self._chk_logy.isChecked():
            ax.set_yscale("log")

        if self._chk_legend.isChecked():
            ax.legend(frameon=False, fontsize=10)

        # Publication ("Origin") styling: full box, inward major+minor ticks.
        # The "Inward ticks" checkbox toggles the all-sides tick marks.
        style_axis(ax)
        if not self._chk_ticks.isChecked():
            ax.tick_params(which="both", top=False, right=False)

        # Manual X-axis range
        if not self._xauto_chk.isChecked():
            self._apply_xlim(ax)

        self._canvas.draw()

    def _on_xauto_toggled(self, checked: bool) -> None:
        self._xmin_edit.setEnabled(not checked)
        self._xmax_edit.setEnabled(not checked)

    def _apply_xlim(self, ax) -> None:
        """Set the X limits from the min/max fields; blanks keep the auto edge."""
        lo_txt = self._xmin_edit.text().strip()
        hi_txt = self._xmax_edit.text().strip()
        cur_lo, cur_hi = ax.get_xlim()
        try:
            lo = float(lo_txt) if lo_txt else cur_lo
            hi = float(hi_txt) if hi_txt else cur_hi
        except ValueError:
            QMessageBox.warning(self, "Invalid X Range",
                                "X min and X max must be numbers (e.g. -0.05).")
            return
        if lo == hi:
            return
        ax.set_xlim(lo, hi)

    # ------------------------------------------------------------------
    # OVF movie
    # ------------------------------------------------------------------

    def _refresh_video_datasets(self) -> None:
        prev = self._vid_ds_combo.currentText()
        self._vid_ds_combo.blockSignals(True)
        self._vid_ds_combo.clear()
        for entry in self._fm.entries:
            self._vid_ds_combo.addItem(entry.label, userData=entry)
        idx = self._vid_ds_combo.findText(prev)
        if idx >= 0:
            self._vid_ds_combo.setCurrentIndex(idx)
        self._vid_ds_combo.blockSignals(False)

    def _video_sim_dir(self) -> Path | None:
        entry = self._vid_ds_combo.currentData()
        if entry is None:
            return None
        return Path(entry.path).parent

    def _open_video(self, path: str) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _do_video(self) -> None:
        sim_dir = self._video_sim_dir()
        if sim_dir is None:
            QMessageBox.information(self, "No Dataset",
                                    "Add table.txt files first, then pick a source.")
            return

        cache = video_cache_path(sim_dir)

        # ── cached video → just open it ──────────────────────────────
        if cache.exists() and not self._vid_rebuild_chk.isChecked():
            self._vid_status.setText(f"Opening cached {cache.name} …")
            self._open_video(str(cache))
            return

        # ── otherwise build it (needs OVF frames) ────────────────────
        if not find_ovf_frames(sim_dir):
            QMessageBox.information(
                self, "No OVF Frames",
                "No numbered OVF frames (0000.ovf, 0001.ovf, …) were found in:\n"
                f"{sim_dir}"
            )
            return

        self._vid_btn.setEnabled(False)
        self._vid_progress.setVisible(True)
        self._vid_progress.setValue(0)
        self._vid_status.setText("Starting …")

        self._vid_worker = _VideoWorker(str(sim_dir), self._vid_fps_spin.value(), self)
        self._vid_worker.progress.connect(self._on_video_progress)
        self._vid_worker.status.connect(self._vid_status.setText)
        self._vid_worker.finished.connect(self._on_video_done)
        self._vid_worker.error.connect(self._on_video_error)
        self._vid_worker.start()

    def _on_video_progress(self, done: int, total: int) -> None:
        self._vid_progress.setMaximum(total)
        self._vid_progress.setValue(done)

    def _on_video_done(self, path: str) -> None:
        self._vid_btn.setEnabled(True)
        self._vid_progress.setVisible(False)
        self._vid_status.setText(f"Saved {Path(path).name} — opening …")
        self._open_video(path)

    def _on_video_error(self, msg: str) -> None:
        self._vid_btn.setEnabled(True)
        self._vid_progress.setVisible(False)
        self._vid_status.setText("Video failed.")
        QMessageBox.critical(self, "Video Error", msg)

    def _do_export(self) -> None:
        if self._last_merge_df is None or self._last_merge_df.empty:
            QMessageBox.information(self, "Nothing to export",
                                    "Plot something first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            export_dataframe(self._last_merge_df, path)
            QMessageBox.information(self, "Exported",
                                    f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
