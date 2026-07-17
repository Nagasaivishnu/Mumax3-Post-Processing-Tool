"""
Hysteresis Tab
===============
Plotting and export for MuMax3 hysteresis table.txt data.
"""

from __future__ import annotations
import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QCheckBox, QLabel, QGroupBox,
    QFileDialog, QMessageBox, QSplitter, QDoubleSpinBox,
    QSpinBox,
)

from gui.plot_canvas import PlotCanvas
from processing.hysteresis import extract_xy, merge_datasets
from export.csv_export import export_dataframe

logger = logging.getLogger(__name__)

# Matplotlib line styles and markers available in the UI
LINE_STYLES = ["-", "--", "-.", ":", "None"]
MARKERS     = ["None", "o", "s", "^", "v", "D", "x", "+"]
COLORMAPS   = ["tab10", "Dark2", "Set1", "Set2"]


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

        if self._chk_ticks.isChecked():
            ax.tick_params(direction="in", which="both",
                           top=True, right=True, length=5)

        self._canvas.draw()

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
