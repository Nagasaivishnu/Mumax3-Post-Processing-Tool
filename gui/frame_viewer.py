"""
OVF Frame Viewer
================
A compact, dockable panel that previews a simulation's OVF frames as images
(scrub + play), and opens the currently-shown frame's OVF file in MuView.

Frames are the PNGs produced by ``mumax3-convert`` (cached next to the OVFs).
Because each on-screen frame maps to an exact OVF file, the MuView button
always opens the right file.
"""

from __future__ import annotations
import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel,
    QSlider, QSpinBox, QProgressBar, QCheckBox, QFileDialog, QMessageBox,
    QSizePolicy,
)

from processing.ovf_video import ensure_frames, find_ovf_frames, open_in_muview, find_muview

logger = logging.getLogger(__name__)


class _FrameLoadWorker(QThread):
    """Converts/collects OVF frames in the background."""
    progress = pyqtSignal(int, int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, sim_dir: str, rebuild: bool, parent=None) -> None:
        super().__init__(parent)
        self.sim_dir = sim_dir
        self.rebuild = rebuild

    def run(self) -> None:
        try:
            pairs = ensure_frames(
                self.sim_dir, rebuild=self.rebuild,
                progress_cb=lambda d, t: self.progress.emit(d, t),
                status_cb=lambda m: self.status.emit(m),
            )
            self.finished.emit(pairs)
        except Exception as exc:
            self.error.emit(str(exc))


class FrameViewerWidget(QWidget):
    """Self-contained frame viewer, driven by the shared FileManager."""

    def __init__(self, file_manager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fm = file_manager
        self._settings = QSettings("MuMax3Tool", "FrameViewer")
        self._frames: list[dict] = []      # [{'index','ovf','png'}]
        self._cur = 0
        self._pixmap: QPixmap | None = None
        self._worker = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

        self._build_ui()
        self._fm.datasets_changed.connect(self._refresh_datasets)
        self._refresh_datasets()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # dataset + load
        top = QHBoxLayout()
        self._ds_combo = QComboBox()
        self._ds_combo.setToolTip("Dataset whose OVF frames to preview")
        top.addWidget(self._ds_combo, 1)
        self._rebuild_chk = QCheckBox("Rebuild")
        self._rebuild_chk.setToolTip("Re-convert PNG frames even if cached")
        top.addWidget(self._rebuild_chk)
        self._load_btn = QPushButton("Load")
        self._load_btn.clicked.connect(self._do_load)
        top.addWidget(self._load_btn)
        lay.addLayout(top)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        # image
        self._image = QLabel("No frames loaded.")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumSize(180, 140)
        self._image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._image.setStyleSheet("background:#111; color:#888;")
        lay.addWidget(self._image, 1)

        # scrubber
        scrub = QHBoxLayout()
        self._play_btn = QPushButton("▶")
        self._play_btn.setMaximumWidth(34)
        self._play_btn.setToolTip("Play / pause")
        self._play_btn.clicked.connect(self._toggle_play)
        scrub.addWidget(self._play_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider)
        scrub.addWidget(self._slider, 1)

        self._frame_lbl = QLabel("—")
        self._frame_lbl.setMinimumWidth(90)
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scrub.addWidget(self._frame_lbl)
        lay.addLayout(scrub)

        # controls
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("fps:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(10)
        self._fps_spin.valueChanged.connect(self._apply_fps)
        ctl.addWidget(self._fps_spin)
        ctl.addStretch()
        self._muview_btn = QPushButton("Open in MuView")
        self._muview_btn.setToolTip("Open the current frame's OVF in MuView")
        self._muview_btn.clicked.connect(self._open_muview)
        ctl.addWidget(self._muview_btn)
        lay.addLayout(ctl)

        # muview path
        mv = QHBoxLayout()
        mv.addWidget(QLabel("MuView:"))
        from PyQt6.QtWidgets import QLineEdit
        self._muview_edit = QLineEdit(self._settings.value("muview_path", ""))
        self._muview_edit.setPlaceholderText("auto-detect on PATH")
        mv.addWidget(self._muview_edit, 1)
        mv_browse = QPushButton("…")
        mv_browse.setMaximumWidth(30)
        mv_browse.clicked.connect(self._browse_muview)
        mv.addWidget(mv_browse)
        lay.addLayout(mv)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        lay.addWidget(self._status)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def _refresh_datasets(self) -> None:
        prev = self._ds_combo.currentText()
        self._ds_combo.blockSignals(True)
        self._ds_combo.clear()
        for e in self._fm.entries:
            self._ds_combo.addItem(e.label, userData=e)
        idx = self._ds_combo.findText(prev)
        if idx >= 0:
            self._ds_combo.setCurrentIndex(idx)
        self._ds_combo.blockSignals(False)

    def _sim_dir(self) -> Path | None:
        e = self._ds_combo.currentData()
        return Path(e.path).parent if e is not None else None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _do_load(self) -> None:
        sim_dir = self._sim_dir()
        if sim_dir is None:
            QMessageBox.information(self, "No Dataset", "Add table.txt files first.")
            return
        if not find_ovf_frames(sim_dir):
            QMessageBox.information(
                self, "No OVF Frames",
                "No numbered OVF frames (0000.ovf, 0001.ovf, …) found in:\n"
                f"{sim_dir}"
            )
            return
        self._stop_play()
        self._load_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Preparing frames …")

        self._worker = _FrameLoadWorker(str(sim_dir), self._rebuild_chk.isChecked(), self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._status.setText)
        self._worker.finished.connect(self._on_loaded)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_loaded(self, pairs: list) -> None:
        self._load_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._frames = pairs
        self._cur = 0
        self._slider.blockSignals(True)
        self._slider.setEnabled(True)
        self._slider.setRange(0, len(pairs) - 1)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._show_frame(0)
        self._status.setText(f"{len(pairs)} frames loaded.")

    def _on_error(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status.setText("Load failed.")
        QMessageBox.critical(self, "Frame Load Error", msg)

    # ------------------------------------------------------------------
    # Display / playback
    # ------------------------------------------------------------------
    def _show_frame(self, i: int) -> None:
        if not self._frames:
            return
        i = max(0, min(i, len(self._frames) - 1))
        self._cur = i
        fr = self._frames[i]
        self._pixmap = QPixmap(str(fr["png"]))
        self._rescale()
        self._frame_lbl.setText(f"{i + 1}/{len(self._frames)}  ({fr['ovf'].name})")

    def _rescale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self._image.setText("(frame image unavailable)")
            return
        self._image.setPixmap(self._pixmap.scaled(
            self._image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _on_slider(self, v: int) -> None:
        self._show_frame(v)

    def _next_frame(self) -> None:
        if not self._frames:
            return
        nxt = (self._cur + 1) % len(self._frames)
        self._slider.setValue(nxt)   # triggers _show_frame

    def _toggle_play(self) -> None:
        if not self._frames:
            return
        if self._timer.isActive():
            self._stop_play()
        else:
            self._apply_fps()
            self._timer.start()
            self._play_btn.setText("⏸")

    def _stop_play(self) -> None:
        self._timer.stop()
        self._play_btn.setText("▶")

    def _apply_fps(self) -> None:
        self._timer.setInterval(int(1000 / max(1, self._fps_spin.value())))

    # ------------------------------------------------------------------
    # MuView
    # ------------------------------------------------------------------
    def _browse_muview(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Locate MuView executable")
        if path:
            self._muview_edit.setText(path)
            self._settings.setValue("muview_path", path)

    def _open_muview(self) -> None:
        if not self._frames:
            QMessageBox.information(self, "No Frame", "Load frames first.")
            return
        exe = self._muview_edit.text().strip() or find_muview()
        if not exe:
            QMessageBox.warning(
                self, "MuView Not Found",
                "Set the MuView executable path below, or add MuView to your PATH."
            )
            return
        self._settings.setValue("muview_path", self._muview_edit.text().strip())
        ovf = self._frames[self._cur]["ovf"]
        try:
            open_in_muview(ovf, muview_exe=exe)
        except Exception as exc:
            QMessageBox.critical(self, "MuView Error", str(exc))
