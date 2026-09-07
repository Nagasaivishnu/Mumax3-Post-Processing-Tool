"""
Hysteresis Tab
===============
Plotting and export for MuMax3 hysteresis table.txt data.
"""

from __future__ import annotations
import logging

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QSettings
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QCheckBox, QLabel, QGroupBox,
    QFileDialog, QMessageBox, QSplitter, QDoubleSpinBox,
    QSpinBox, QProgressBar, QButtonGroup,
    QDialog, QGridLayout, QScrollArea, QDialogButtonBox,
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
    """
    Builds (or reuses cached) OVF → MP4 movies for one or more directories.
    """
    progress = pyqtSignal(int, int)   # frame progress for the current directory
    status   = pyqtSignal(str)
    finished = pyqtSignal(list, list)  # produced video paths, error messages
    error    = pyqtSignal(str)         # fatal (nothing produced)

    def __init__(self, sim_dirs: list[str], fps: int, rebuild: bool, parent=None) -> None:
        super().__init__(parent)
        self.sim_dirs = sim_dirs
        self.fps      = fps
        self.rebuild  = rebuild

    def run(self) -> None:
        produced, errors = [], []
        n = len(self.sim_dirs)
        for i, d in enumerate(self.sim_dirs):
            tag = f"[{i + 1}/{n}] {Path(d).name}"
            try:
                cache = video_cache_path(d)
                if cache.exists() and not self.rebuild:
                    self.status.emit(f"{tag}: using cached {cache.name}")
                    produced.append(str(cache))
                    continue
                if not find_ovf_frames(d):
                    errors.append(f"{Path(d).name}: no numbered OVF frames")
                    continue
                self.status.emit(f"{tag}: building …")
                path = make_video(
                    d, fps=self.fps,
                    progress_cb=lambda a, b: self.progress.emit(a, b),
                    status_cb=lambda m, t=tag: self.status.emit(f"{t}: {m}"),
                )
                produced.append(str(path))
            except Exception as exc:
                errors.append(f"{Path(d).name}: {exc}")

        if not produced and errors:
            self.error.emit("\n".join(errors))
            return
        self.finished.emit(produced, errors)


class PlotSelectionDialog(QDialog):
    """
    Per-file plot selection.

    Each loaded file gets an *include* checkbox plus Mx/My/Mz checkboxes.
    Checking a component plots  m<comp> vs B_ext<comp>  for that file (so a
    hard axis and an easy axis from different files can share one graph).
    Leaving all component boxes unchecked falls back to the global X/Y axes.
    """

    COMPS = ("x", "y", "z")

    def __init__(self, entries, state: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot Selection")
        self.resize(470, 380)
        self._checks: dict = {}

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Choose which files (and components) to plot.\n"
            "Mx/My/Mz plot m<comp> vs B_ext<comp>; leave them unchecked to "
            "use the global X/Y axes."
        ))

        grid = QGridLayout()
        for j, h in enumerate(["Label", "Plot", "Mx", "My", "Mz"]):
            grid.addWidget(QLabel(f"<b>{h}</b>"), 0, j)

        for i, e in enumerate(entries, start=1):
            grid.addWidget(QLabel(e.label), i, 0)
            st = state.get(e.path, {})
            row = {}
            inc = QCheckBox()
            inc.setChecked(st.get("include", True))
            grid.addWidget(inc, i, 1, Qt.AlignmentFlag.AlignCenter)
            row["include"] = inc
            for j, c in enumerate(self.COMPS, start=2):
                cb = QCheckBox()
                cb.setChecked(bool(st.get(c)))
                grid.addWidget(cb, i, j, Qt.AlignmentFlag.AlignCenter)
                row[c] = cb
            self._checks[e.path] = row

        cont = QWidget()
        cont.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(cont)
        lay.addWidget(scroll)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def result_state(self) -> dict:
        out = {}
        for path, row in self._checks.items():
            out[path] = {
                "include": row["include"].isChecked(),
                **{c: row[c].isChecked() for c in self.COMPS},
            }
        return out


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
        self._settings = QSettings("MuMax3Tool", "HysteresisTab")
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

        # ── LEFT: scroll-able controls ────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(270)
        scroll.setMaximumWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(ctrl_widget)
        splitter.addWidget(scroll)

        # Axis selection
        axis_grp = QGroupBox("Axis Selection")
        axis_form = QFormLayout(axis_grp)

        # Quick component select: x/y/z (mutually exclusive) → sets both axes
        comp_row = QHBoxLayout()
        self._axis_group = QButtonGroup(self)
        self._axis_group.setExclusive(True)
        self._axis_checks = {}
        for d in ("x", "y", "z"):
            cb = QCheckBox(d)
            cb.setToolTip(f"Plot m{d} vs B_ext{d}")
            comp_row.addWidget(cb)
            self._axis_group.addButton(cb)
            self._axis_checks[d] = cb
            cb.toggled.connect(
                lambda checked, dd=d: self._apply_axis_component(dd) if checked else None
            )
        axis_form.addRow("Component:", comp_row)

        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        axis_form.addRow("X axis:", self._x_combo)
        axis_form.addRow("Y axis:", self._y_combo)

        self._plot_sel_btn = QPushButton("Plot Selection…")
        self._plot_sel_btn.setToolTip(
            "Pick per-file which datasets and components (Mx/My/Mz) to plot —\n"
            "e.g. overlay a hard axis and an easy axis in one graph."
        )
        self._plot_sel_btn.clicked.connect(self._open_plot_selection)
        axis_form.addRow("", self._plot_sel_btn)
        ctrl_layout.addWidget(axis_grp)

        # per-file plot selection state: {path: {'include':bool,'x','y','z':bool}}
        self._plot_sel: dict = {}

        # Plot appearance
        app_grp = QGroupBox("Appearance")
        app_form = QFormLayout(app_grp)

        self._lw_spin = QDoubleSpinBox()
        self._lw_spin.setRange(0.5, 5.0)
        self._lw_spin.setSingleStep(0.5)
        self._lw_spin.setValue(2.2)
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

        # Title + axis labels (manual override)
        lbl_grp = QGroupBox("Title & Axis Labels")
        lbl_form = QFormLayout(lbl_grp)
        from PyQt6.QtWidgets import QLineEdit
        self._title_edit  = QLineEdit()
        self._xlabel_edit = QLineEdit()
        self._ylabel_edit = QLineEdit()
        self._title_edit.setPlaceholderText("plot title (optional)")
        self._xlabel_edit.setPlaceholderText("auto")
        self._ylabel_edit.setPlaceholderText("auto")
        lbl_form.addRow("Title:", self._title_edit)
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

        # Plot size (inches) + DPI — used for the canvas and the saved image
        size_grp  = QGroupBox("Plot Size")
        size_form = QFormLayout(size_grp)
        self._w_spin = QDoubleSpinBox()
        self._w_spin.setRange(2.0, 30.0)
        self._w_spin.setSingleStep(0.5)
        self._w_spin.setValue(7.0)
        self._w_spin.setSuffix(" in")
        self._h_spin = QDoubleSpinBox()
        self._h_spin.setRange(2.0, 30.0)
        self._h_spin.setSingleStep(0.5)
        self._h_spin.setValue(5.0)
        self._h_spin.setSuffix(" in")
        self._dpi_spin = QSpinBox()
        self._dpi_spin.setRange(50, 1200)
        self._dpi_spin.setValue(300)
        self._dpi_spin.setToolTip("Resolution used when saving (PPT / toolbar save)")
        size_form.addRow("Width:", self._w_spin)
        size_form.addRow("Height:", self._h_spin)
        size_form.addRow("Save DPI:", self._dpi_spin)
        ctrl_layout.addWidget(size_grp)

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

        self._vid_all_chk = QCheckBox("All loaded directories")
        self._vid_all_chk.setToolTip(
            "Build/cache a movie in every loaded dataset's directory."
        )
        self._vid_all_chk.toggled.connect(
            lambda checked: self._vid_ds_combo.setEnabled(not checked)
        )
        vid_form.addRow("", self._vid_all_chk)

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

        # PowerPoint export
        from PyQt6.QtWidgets import QLineEdit
        ppt_row = QHBoxLayout()
        ppt_row.addWidget(QLabel("PPT:"))
        self._ppt_path_edit = QLineEdit(self._settings.value("ppt_path", ""))
        self._ppt_path_edit.setPlaceholderText("presentation.pptx")
        self._ppt_path_edit.setToolTip(
            "Target .pptx file. The current plot is appended as a new slide."
        )
        ppt_row.addWidget(self._ppt_path_edit)
        ppt_browse = QPushButton("…")
        ppt_browse.setMaximumWidth(30)
        ppt_browse.clicked.connect(self._browse_ppt)
        ppt_row.addWidget(ppt_browse)
        ctrl_layout.addLayout(ppt_row)

        self._ppt_btn = QPushButton("Save Plot to PPT")
        self._ppt_btn.clicked.connect(self._do_save_ppt)
        ctrl_layout.addWidget(self._ppt_btn)

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

    @staticmethod
    def _find_col(cols: list[str], token: str) -> str | None:
        """First column whose first whitespace-token equals *token* (e.g. 'mx')."""
        for c in cols:
            if c.split() and c.split()[0] == token:
                return c
        return None

    def _apply_axis_component(self, d: str) -> None:
        """x/y/z quick-select → X axis = B_ext<d>, Y axis = m<d>."""
        entries = self._fm.entries
        if not entries:
            return
        try:
            cols = list(entries[0].df.columns)
        except Exception:
            return

        x_col = self._find_col(cols, f"B_ext{d}")
        y_col = self._find_col(cols, f"m{d}")

        missing = []
        if x_col:
            self._x_combo.setCurrentText(x_col)
        else:
            missing.append(f"B_ext{d}")
        if y_col:
            self._y_combo.setCurrentText(y_col)
        else:
            missing.append(f"m{d}")

        if missing:
            QMessageBox.warning(
                self, "Columns not found",
                "These columns are not in the dataset:\n  "
                + ", ".join(missing)
            )

    def _open_plot_selection(self) -> None:
        entries = self._fm.entries
        if not entries:
            QMessageBox.information(self, "No Data", "Add files first.")
            return
        dlg = PlotSelectionDialog(entries, self._plot_sel, self)
        if dlg.exec():
            self._plot_sel = dlg.result_state()

    def _build_plot_plan(self, entries):
        """
        Return (plan, comp_used) where plan is a list of
        (df, x_col, y_col, label) curves to draw.
        """
        gx = self._x_combo.currentText()
        gy = self._y_combo.currentText()
        sel = self._plot_sel

        included = [e for e in entries if sel.get(e.path, {}).get("include")]
        use_sel  = bool(included)
        targets  = included if use_sel else entries

        plan, comp_used = [], False
        for e in targets:
            cols = list(e.df.columns)
            st = sel.get(e.path, {}) if use_sel else {}
            comps = [c for c in ("x", "y", "z") if st.get(c)]
            if comps:
                comp_used = True
                for c in comps:
                    xc = self._find_col(cols, f"B_ext{c}")
                    yc = self._find_col(cols, f"m{c}")
                    if xc and yc:
                        plan.append((e.df, xc, yc, f"{e.label} · m{c}"))
                    else:
                        logger.warning("'%s' missing B_ext%s or m%s", e.label, c, c)
            else:
                if gx in cols and gy in cols:
                    plan.append((e.df, gx, gy, e.label))
                else:
                    logger.warning("'%s' missing '%s' or '%s'", e.label, gx, gy)
        return plan, comp_used

    def _do_plot(self) -> None:
        entries = self._fm.get_loaded_entries()
        if not entries:
            QMessageBox.information(self, "No Data", "Please add files first.")
            return

        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()

        plan, comp_used = self._build_plot_plan(entries)
        if not plan:
            QMessageBox.warning(
                self, "Nothing to plot",
                "No curves selected. Check X/Y columns or your Plot Selection."
            )
            return

        # Store for export (generic x column name when components are mixed)
        merge_inputs = [(extract_xy(df, xc, yc)[0], extract_xy(df, xc, yc)[1], label)
                        for (df, xc, yc, label) in plan]
        merge_x_label = "Field (T)" if comp_used else x_col
        self._last_merge_df = merge_datasets(merge_inputs, merge_x_label) \
            if merge_inputs else None

        self._canvas.fig.set_size_inches(self._w_spin.value(), self._h_spin.value())
        ax = self._canvas.single_ax
        self._canvas.clear_axes()
        self._render_curves(ax, plan, comp_used, x_col, y_col, warn=True)
        self._canvas.draw()

    def _render_curves(self, ax, plan, comp_used, x_col, y_col, warn=False):
        """
        Draw the hysteresis curves + Origin styling onto *ax*.

        Shared by the on-screen canvas and the PPT export so both look
        identical (publication / "Origin" style).
        """
        import matplotlib.cm as cm
        cmap   = cm.get_cmap("tab10")
        lw     = self._lw_spin.value()
        ms     = self._ms_spin.value() or None
        marker = self._marker_combo.currentText()
        if marker == "None":
            marker = None

        for i, (df, xc, yc, label) in enumerate(plan):
            x, y = extract_xy(df, xc, yc)
            ax.plot(x, y, label=label, linewidth=lw, marker=marker,
                    markersize=ms, color=cmap(i % 10))

        x_label = self._xlabel_edit.text() or ("Field (T)" if comp_used else x_col)
        y_label = self._ylabel_edit.text() or ("m" if comp_used else y_col)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

        title = self._title_edit.text().strip()
        if title:
            ax.set_title(title, fontsize=13)

        if self._chk_grid.isChecked():
            ax.grid(True, alpha=0.4)
        if self._chk_logx.isChecked():
            ax.set_xscale("log")
        if self._chk_logy.isChecked():
            ax.set_yscale("log")
        if self._chk_legend.isChecked():
            ax.legend(frameon=False, fontsize=10)

        # Publication ("Origin") styling: full box, inward major+minor ticks.
        style_axis(ax)
        if not self._chk_ticks.isChecked():
            ax.tick_params(which="both", top=False, right=False)

        # Manual X-axis range
        if not self._xauto_chk.isChecked():
            self._apply_xlim(ax, warn=warn)

    def _on_xauto_toggled(self, checked: bool) -> None:
        self._xmin_edit.setEnabled(not checked)
        self._xmax_edit.setEnabled(not checked)

    def _apply_xlim(self, ax, warn: bool = True) -> None:
        """Set the X limits from the min/max fields; blanks keep the auto edge."""
        lo_txt = self._xmin_edit.text().strip()
        hi_txt = self._xmax_edit.text().strip()
        cur_lo, cur_hi = ax.get_xlim()
        try:
            lo = float(lo_txt) if lo_txt else cur_lo
            hi = float(hi_txt) if hi_txt else cur_hi
        except ValueError:
            if warn:
                QMessageBox.warning(self, "Invalid X Range",
                                    "X min and X max must be numbers (e.g. -0.05).")
            return
        if lo == hi:
            return
        ax.set_xlim(lo, hi)

    # ------------------------------------------------------------------
    # PowerPoint export
    # ------------------------------------------------------------------

    def _browse_ppt(self) -> None:
        start = self._ppt_path_edit.text().strip()
        path, _ = QFileDialog.getSaveFileName(
            self, "Select PowerPoint file", start, "PowerPoint (*.pptx)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if path:
            self._ppt_path_edit.setText(path)

    def _do_save_ppt(self) -> None:
        if self._last_merge_df is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Plot something first.")
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

        # Re-render into a fresh publication ("Origin") figure so the export
        # is clean and independent of the on-screen canvas state.
        entries = self._fm.get_loaded_entries()
        plan, comp_used = self._build_plot_plan(entries)
        if not plan:
            QMessageBox.warning(self, "Nothing to save", "Plot something first.")
            return

        fig = Figure(figsize=(self._w_spin.value(), self._h_spin.value()),
                     dpi=self._dpi_spin.value(), tight_layout=True)
        ax  = fig.add_subplot(111)
        self._render_curves(ax, plan, comp_used,
                            self._x_combo.currentText(),
                            self._y_combo.currentText(), warn=False)

        tmpdir = Path(tempfile.mkdtemp(prefix="mumax_hyst_ppt_"))
        png = tmpdir / "hysteresis.png"
        fig.savefig(png)

        title = self._title_edit.text().strip() or "Hysteresis"
        try:
            from export.ppt_export import append_images_slide
            append_images_slide(ppt_path, [str(png)], title)
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

        QMessageBox.information(self, "Saved to PPT",
                                f"Appended 1 slide to:\n{ppt_path}")

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

    def _all_video_dirs(self) -> list[Path]:
        """Unique parent directories of all loaded datasets."""
        seen, out = set(), []
        for e in self._fm.entries:
            d = Path(e.path).parent.resolve()
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def _open_video(self, path: str) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _do_video(self) -> None:
        if self._vid_all_chk.isChecked():
            dirs = self._all_video_dirs()
            if not dirs:
                QMessageBox.information(self, "No Datasets", "Add table.txt files first.")
                return
        else:
            sim_dir = self._video_sim_dir()
            if sim_dir is None:
                QMessageBox.information(self, "No Dataset",
                                        "Add table.txt files first, then pick a source.")
                return
            dirs = [sim_dir]

        self._vid_btn.setEnabled(False)
        self._vid_progress.setVisible(True)
        self._vid_progress.setValue(0)
        self._vid_status.setText("Starting …")

        self._vid_worker = _VideoWorker(
            [str(d) for d in dirs],
            self._vid_fps_spin.value(),
            self._vid_rebuild_chk.isChecked(),
            self,
        )
        self._vid_worker.progress.connect(self._on_video_progress)
        self._vid_worker.status.connect(self._vid_status.setText)
        self._vid_worker.finished.connect(self._on_video_done)
        self._vid_worker.error.connect(self._on_video_error)
        self._vid_worker.start()

    def _on_video_progress(self, done: int, total: int) -> None:
        self._vid_progress.setMaximum(total)
        self._vid_progress.setValue(done)

    def _on_video_done(self, paths: list, errors: list) -> None:
        self._vid_btn.setEnabled(True)
        self._vid_progress.setVisible(False)
        for pth in paths:
            self._open_video(pth)
        n = len(paths)
        self._vid_status.setText(
            f"{n} video{'s' if n != 1 else ''} ready"
            + (f", {len(errors)} skipped" if errors else "")
        )
        if errors:
            QMessageBox.warning(
                self, "Some videos skipped",
                "Built " + f"{n} video(s). The following were skipped:\n\n"
                + "\n".join(errors)
            )

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
