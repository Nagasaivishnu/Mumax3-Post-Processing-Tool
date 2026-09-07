"""
Plotter Tab
===========
Collect plots from other tabs into a *bundle*, edit common display settings
for all of them at once, preview them (Origin style), save/reopen the bundle
as a file, and export to PowerPoint (up to 9 plots per slide).
"""

from __future__ import annotations
import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QCheckBox, QLineEdit, QListWidget, QSplitter,
    QScrollArea, QDoubleSpinBox, QFileDialog, QMessageBox, QInputDialog,
)

from gui.plot_canvas import PlotCanvas
from gui.plot_style import style_axis
from processing.plot_bundle import (
    new_bundle, make_plot_item, save_bundle, load_bundle,
    read_registry, bundles_dir, BUNDLE_EXT,
)

logger = logging.getLogger(__name__)

PREVIEW_COLS = 3


# ---------------------------------------------------------------------------
# Shared rendering (Origin style) — used by preview and PPT export
# ---------------------------------------------------------------------------

def render_plot_item(fig, ax, item: dict, common: dict) -> None:
    ax.plot(item["x"], item["y"], color="black", linewidth=2.2)
    for pk in item.get("peaks", []):
        ax.axvline(pk["f"], color="tab:red", linestyle="--", linewidth=1.4)

    # Axis titles: common override wins, else the plot's own label
    font = common.get("font") or None
    fs   = float(common.get("fontsize", 12.0))
    fkw = {"fontfamily": font} if font else {}
    x_label = common.get("xlabel") or item.get("x_label", "")
    y_label = common.get("ylabel") or item.get("y_label", "")
    if x_label:
        ax.set_xlabel(x_label, fontsize=fs, **fkw)
    if y_label:
        ax.set_ylabel(y_label, fontsize=fs, **fkw)
    if item.get("title"):
        ax.set_title(item["title"], fontsize=fs + 1, **fkw)

    if common.get("logx"):
        ax.set_xscale("log")
    if common.get("logy"):
        ax.set_yscale("log")

    xmin, xmax = common.get("xmin"), common.get("xmax")
    if xmin is not None or xmax is not None:
        lo, hi = ax.get_xlim()
        ax.set_xlim(xmin if xmin is not None else lo, xmax if xmax is not None else hi)
    ymin, ymax = common.get("ymin"), common.get("ymax")
    if ymin is not None or ymax is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else lo, ymax if ymax is not None else hi)

    style_axis(ax)

    # Tick label size + font (after styling)
    ax.tick_params(which="major", labelsize=max(6.0, fs - 1))
    if font:
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontfamily(font)


class PlotterTab(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = QSettings("MuMax3Tool", "PlotterTab")
        self._bundle = new_bundle()
        self._path: Path | None = None
        self._preview_canvas: PlotCanvas | None = None
        self._build_ui()
        self._refresh_registry()
        self._refresh_plot_list()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(splitter)

        # left controls (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        # No max width — the splitter can drag the panel wider.
        # Horizontal scrollbar appears when the content needs more width.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ctrl_w = QWidget()
        ctrl = QVBoxLayout(ctrl_w)
        scroll.setWidget(ctrl_w)
        splitter.addWidget(scroll)

        # ── Bundle group ──────────────────────────────────────────────
        b_grp = QGroupBox("Bundle")
        b_lay = QVBoxLayout(b_grp)

        reg_row = QHBoxLayout()
        self._reg_combo = QComboBox()
        self._reg_combo.setToolTip("Existing bundles found in the repo")
        reg_row.addWidget(self._reg_combo, 1)
        self._load_reg_btn = QPushButton("Open")
        self._load_reg_btn.clicked.connect(self._open_from_registry)
        reg_row.addWidget(self._load_reg_btn)
        b_lay.addLayout(reg_row)

        row2 = QHBoxLayout()
        for text, slot in [("New", self._new_bundle),
                           ("Open File…", self._open_file),
                           ("Save", self._save),
                           ("Save As…", self._save_as)]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            row2.addWidget(btn)
        b_lay.addLayout(row2)

        self._desc_edit = QLineEdit(self._bundle["description"])
        self._desc_edit.setPlaceholderText("bundle description")
        b_lay.addWidget(self._desc_edit)

        self._bundle_lbl = QLabel("Unsaved bundle")
        self._bundle_lbl.setStyleSheet("color: gray; font-size: 11px;")
        b_lay.addWidget(self._bundle_lbl)
        ctrl.addWidget(b_grp)

        # ── Plots group ───────────────────────────────────────────────
        p_grp = QGroupBox("Plots in Bundle")
        p_lay = QVBoxLayout(p_grp)
        self._plot_list = QListWidget()
        self._plot_list.currentRowChanged.connect(self._on_plot_selected)
        p_lay.addWidget(self._plot_list)

        ptitle_row = QHBoxLayout()
        ptitle_row.addWidget(QLabel("Title:"))
        self._ptitle_edit = QLineEdit()
        self._ptitle_edit.setPlaceholderText("(select a plot)")
        self._ptitle_edit.editingFinished.connect(self._apply_plot_title)
        ptitle_row.addWidget(self._ptitle_edit, 1)
        p_lay.addLayout(ptitle_row)

        rm_btn = QPushButton("Remove Selected Plot")
        rm_btn.clicked.connect(self._remove_plot)
        p_lay.addWidget(rm_btn)
        ctrl.addWidget(p_grp)

        # ── Common settings ───────────────────────────────────────────
        c_grp = QGroupBox("Common Settings (all plots)")
        c_form = QFormLayout(c_grp)
        self._logx_chk = QCheckBox("Log X")
        self._logy_chk = QCheckBox("Log Y")
        c_form.addRow(self._logx_chk, self._logy_chk)
        self._xmin_edit = QLineEdit(); self._xmin_edit.setPlaceholderText("auto")
        self._xmax_edit = QLineEdit(); self._xmax_edit.setPlaceholderText("auto")
        self._ymin_edit = QLineEdit(); self._ymin_edit.setPlaceholderText("auto")
        self._ymax_edit = QLineEdit(); self._ymax_edit.setPlaceholderText("auto")
        c_form.addRow("X min:", self._xmin_edit)
        c_form.addRow("X max:", self._xmax_edit)
        c_form.addRow("Y min:", self._ymin_edit)
        c_form.addRow("Y max:", self._ymax_edit)
        self._aspect_spin = QDoubleSpinBox()
        self._aspect_spin.setRange(0.3, 4.0)
        self._aspect_spin.setSingleStep(0.1)
        self._aspect_spin.setValue(1.4)
        self._aspect_spin.setToolTip("Width / height of each plot picture")
        c_form.addRow("Aspect (W/H):", self._aspect_spin)

        self._xlabel_edit = QLineEdit()
        self._xlabel_edit.setPlaceholderText("auto (per plot)")
        self._ylabel_edit = QLineEdit()
        self._ylabel_edit.setPlaceholderText("auto (per plot)")
        c_form.addRow("X title:", self._xlabel_edit)
        c_form.addRow("Y title:", self._ylabel_edit)

        self._font_combo = QComboBox()
        self._font_combo.addItems([
            "Default", "Arial", "Times New Roman", "Calibri", "Cambria",
            "DejaVu Sans", "Courier New",
        ])
        self._font_combo.setToolTip("Font family for titles, axis labels, and ticks")
        c_form.addRow("Font:", self._font_combo)

        self._fontsize_spin = QDoubleSpinBox()
        self._fontsize_spin.setRange(6.0, 40.0)
        self._fontsize_spin.setSingleStep(1.0)
        self._fontsize_spin.setValue(12.0)
        self._fontsize_spin.setToolTip(
            "Base font size (axis labels; title is +1, tick labels −1)")
        c_form.addRow("Font size:", self._fontsize_spin)

        apply_btn = QPushButton("Apply / Refresh Preview")
        apply_btn.clicked.connect(self._apply_common)
        c_form.addRow("", apply_btn)
        ctrl.addWidget(c_grp)

        # ── Export ────────────────────────────────────────────────────
        e_grp = QGroupBox("Export to PowerPoint (9 / slide)")
        e_lay = QVBoxLayout(e_grp)
        ppt_row = QHBoxLayout()
        self._ppt_edit = QLineEdit(self._settings.value("ppt_path", ""))
        self._ppt_edit.setPlaceholderText("presentation.pptx")
        ppt_row.addWidget(self._ppt_edit, 1)
        ppt_browse = QPushButton("…")
        ppt_browse.setMaximumWidth(30)
        ppt_browse.clicked.connect(self._browse_ppt)
        ppt_row.addWidget(ppt_browse)
        e_lay.addLayout(ppt_row)
        ppt_btn = QPushButton("Save Bundle to PPT")
        ppt_btn.clicked.connect(self._save_ppt)
        e_lay.addWidget(ppt_btn)
        ctrl.addWidget(e_grp)

        ctrl.addStretch()

        # right: preview
        self._preview_area = QScrollArea()
        self._preview_area.setWidgetResizable(True)
        splitter.addWidget(self._preview_area)
        splitter.setStretchFactor(1, 1)
        self._render_preview()

    # ------------------------------------------------------------------
    # Public: called by other tabs
    # ------------------------------------------------------------------
    def add_plot(self, item: dict) -> None:
        """Append a plot item (from another tab) to the current bundle."""
        self._bundle["plots"].append(item)
        self._refresh_plot_list()
        self._render_preview()

    # ------------------------------------------------------------------
    # Registry / bundle IO
    # ------------------------------------------------------------------
    def _refresh_registry(self) -> None:
        self._reg_combo.clear()
        for e in read_registry():
            desc = f" — {e['description']}" if e.get("description") else ""
            self._reg_combo.addItem(f"{e['name']} ({e.get('n_plots', 0)}){desc}",
                                    userData=e.get("path"))

    def _open_from_registry(self) -> None:
        path = self._reg_combo.currentData()
        if not path:
            QMessageBox.information(self, "No Bundle", "No existing bundle selected.")
            return
        self._load_path(path)

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Bundle", str(bundles_dir()),
            f"MuMax bundle (*{BUNDLE_EXT})")
        if path:
            self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            self._bundle = load_bundle(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return
        self._path = Path(path)
        self._desc_edit.setText(self._bundle.get("description", ""))
        self._load_common_into_ui()
        self._refresh_plot_list()
        self._render_preview()
        self._bundle_lbl.setText(f"Loaded: {self._path.name}")

    def _new_bundle(self) -> None:
        self._bundle = new_bundle()
        self._path = None
        self._desc_edit.clear()
        self._load_common_into_ui()
        self._refresh_plot_list()
        self._render_preview()
        self._bundle_lbl.setText("Unsaved bundle")

    def _save(self) -> None:
        if self._path is None:
            self._save_as()
            return
        self._commit_ui_to_bundle()
        try:
            save_bundle(self._bundle, self._path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return
        self._bundle_lbl.setText(f"Saved: {self._path.name}")
        self._refresh_registry()

    def _save_as(self) -> None:
        if not self._bundle["plots"]:
            QMessageBox.information(self, "Empty", "Add plots before saving.")
            return
        start = str(bundles_dir() / "bundle")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Bundle", start, f"MuMax bundle (*{BUNDLE_EXT})")
        if not path:
            return
        self._commit_ui_to_bundle()
        try:
            self._path = save_bundle(self._bundle, path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return
        self._bundle_lbl.setText(f"Saved: {self._path.name}")
        self._refresh_registry()

    # ------------------------------------------------------------------
    # Plot list / titles
    # ------------------------------------------------------------------
    def _refresh_plot_list(self) -> None:
        self._plot_list.blockSignals(True)
        self._plot_list.clear()
        for i, pl in enumerate(self._bundle["plots"]):
            self._plot_list.addItem(pl.get("title") or f"(plot {i + 1})")
        self._plot_list.blockSignals(False)

    def _on_plot_selected(self, row: int) -> None:
        if 0 <= row < len(self._bundle["plots"]):
            self._ptitle_edit.setText(self._bundle["plots"][row].get("title", ""))
        else:
            self._ptitle_edit.clear()

    def _apply_plot_title(self) -> None:
        row = self._plot_list.currentRow()
        if 0 <= row < len(self._bundle["plots"]):
            self._bundle["plots"][row]["title"] = self._ptitle_edit.text().strip()
            self._refresh_plot_list()
            self._plot_list.setCurrentRow(row)
            self._render_preview()

    def _remove_plot(self) -> None:
        row = self._plot_list.currentRow()
        if 0 <= row < len(self._bundle["plots"]):
            self._bundle["plots"].pop(row)
            self._refresh_plot_list()
            self._render_preview()

    # ------------------------------------------------------------------
    # Common settings
    # ------------------------------------------------------------------
    @staticmethod
    def _parse(edit) -> float | None:
        t = edit.text().strip()
        try:
            return float(t) if t else None
        except ValueError:
            return None

    def _commit_ui_to_bundle(self) -> None:
        self._bundle["description"] = self._desc_edit.text().strip()
        font = self._font_combo.currentText()
        self._bundle["common"] = {
            "logx": self._logx_chk.isChecked(),
            "logy": self._logy_chk.isChecked(),
            "xmin": self._parse(self._xmin_edit),
            "xmax": self._parse(self._xmax_edit),
            "ymin": self._parse(self._ymin_edit),
            "ymax": self._parse(self._ymax_edit),
            "aspect": self._aspect_spin.value(),
            "xlabel": self._xlabel_edit.text().strip(),
            "ylabel": self._ylabel_edit.text().strip(),
            "font": "" if font == "Default" else font,
            "fontsize": self._fontsize_spin.value(),
        }

    def _load_common_into_ui(self) -> None:
        c = self._bundle.get("common", {})
        self._logx_chk.setChecked(bool(c.get("logx")))
        self._logy_chk.setChecked(bool(c.get("logy")))
        for edit, key in [(self._xmin_edit, "xmin"), (self._xmax_edit, "xmax"),
                          (self._ymin_edit, "ymin"), (self._ymax_edit, "ymax")]:
            v = c.get(key)
            edit.setText("" if v is None else str(v))
        self._aspect_spin.setValue(float(c.get("aspect", 1.4)))
        self._xlabel_edit.setText(c.get("xlabel", ""))
        self._ylabel_edit.setText(c.get("ylabel", ""))
        font = c.get("font", "")
        idx = self._font_combo.findText(font) if font else 0
        self._font_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._fontsize_spin.setValue(float(c.get("fontsize", 12.0)))

    def _apply_common(self) -> None:
        self._commit_ui_to_bundle()
        self._render_preview()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _render_preview(self) -> None:
        plots = self._bundle["plots"]
        common = self._bundle.get("common", {})
        n = len(plots)
        if n == 0:
            lbl = QLabel("No plots yet.\nSend a plot from another tab, or open a bundle.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: gray;")
            self._preview_area.setWidget(lbl)
            return

        ncols = min(PREVIEW_COLS, n)
        nrows = (n + ncols - 1) // ncols
        aspect = float(common.get("aspect", 1.4))
        cell_h = 3.2
        canvas = PlotCanvas(None, n_rows=nrows, n_cols=ncols,
                            figsize=(aspect * cell_h * ncols, cell_h * nrows))
        axes = canvas.axes
        for idx in range(nrows * ncols):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            if idx < n:
                render_plot_item(canvas.fig, ax, plots[idx], common)
            else:
                ax.axis("off")
        canvas.draw()
        self._preview_area.setWidget(canvas)
        self._preview_canvas = canvas

    # ------------------------------------------------------------------
    # PPT export (9 per slide)
    # ------------------------------------------------------------------
    def _browse_ppt(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Select PowerPoint file", self._ppt_edit.text().strip(),
            "PowerPoint (*.pptx)", options=QFileDialog.Option.DontConfirmOverwrite)
        if path:
            self._ppt_edit.setText(path)

    def _save_ppt(self) -> None:
        plots = self._bundle["plots"]
        if not plots:
            QMessageBox.information(self, "Empty", "No plots to export.")
            return
        path_txt = self._ppt_edit.text().strip()
        if not path_txt:
            QMessageBox.warning(self, "No PPT File", "Choose a target .pptx first.")
            return
        ppt_path = Path(path_txt)
        if ppt_path.suffix.lower() != ".pptx":
            ppt_path = ppt_path.with_suffix(".pptx")
        self._settings.setValue("ppt_path", str(ppt_path))
        self._commit_ui_to_bundle()

        import tempfile
        from matplotlib.figure import Figure
        common = self._bundle["common"]
        aspect = float(common.get("aspect", 1.4))
        h = 4.0
        tmpdir = Path(tempfile.mkdtemp(prefix="mumax_bundle_ppt_"))

        # render every plot to a PNG
        pngs = []
        for i, pl in enumerate(plots):
            fig = Figure(figsize=(aspect * h, h), dpi=200, tight_layout=True)
            ax = fig.add_subplot(111)
            render_plot_item(fig, ax, pl, common)
            p = tmpdir / f"plot_{i}.png"
            fig.savefig(p)
            pngs.append(str(p))

        # 9 per slide
        try:
            from export.ppt_export import append_images_slide
            desc = self._bundle.get("description") or "Plot bundle"
            for s in range(0, len(pngs), 9):
                chunk = pngs[s:s + 9]
                title = desc if len(pngs) <= 9 else f"{desc}  ({s + 1}–{s + len(chunk)})"
                append_images_slide(ppt_path, chunk, title)
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency",
                                 "python-pptx is not installed.\n\n"
                                 "Install it with:\n    pip install python-pptx")
            return
        except PermissionError:
            QMessageBox.critical(self, "PPT Export Error",
                                 f"Cannot write to:\n{ppt_path}\n\n"
                                 "The file is probably open in PowerPoint — close it and retry.")
            return
        except Exception as e:
            QMessageBox.critical(self, "PPT Export Error", str(e))
            return

        n_slides = (len(pngs) + 8) // 9
        QMessageBox.information(
            self, "Saved to PPT",
            f"Exported {len(pngs)} plot(s) across {n_slides} slide(s) to:\n{ppt_path}")
