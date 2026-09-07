"""
Main Window
============
Top-level QMainWindow. Assembles the File Manager panel (left) and the
analysis tab widget (right) into a horizontal splitter.
"""

from __future__ import annotations
import logging

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QTabWidget, QWidget,
    QHBoxLayout, QStatusBar, QApplication, QDockWidget,
)
from PyQt6.QtGui import QKeySequence, QAction

from gui.file_manager import FileManagerWidget
from gui.hysteresis_tab import HysteresisTab
from gui.fmr_tab import FMRTab
from gui.mode_profile_tab import SpinWaveModeProfileTab
from gui.plotter_tab import PlotterTab
from gui.frame_viewer import FrameViewerWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MuMax3 Post-Processing Tool")
        self.resize(1300, 750)

        self._settings = QSettings("MuMax3Tool", "MainWindow")
        self._build_ui()
        self._build_menu()
        self._restore_geometry()

        # connect file-manager status messages to the status bar
        self._fm.datasets_changed.connect(self._on_datasets_changed)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── File manager ──────────────────────────────────────────────
        self._fm = FileManagerWidget()
        splitter.addWidget(self._fm)

        # ── Analysis tabs ─────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._hyst_tab    = HysteresisTab(self._fm)
        self._fmr_tab     = FMRTab(self._fm)
        self._mode_tab    = SpinWaveModeProfileTab(self._fm)
        self._plotter_tab = PlotterTab()
        self._tabs.addTab(self._hyst_tab,  "Hysteresis")
        self._tabs.addTab(self._fmr_tab,   "FMR")
        self._tabs.addTab(self._mode_tab,  "Spin Wave Mode Profile")
        self._tabs.addTab(self._plotter_tab, "Plotter")
        splitter.addWidget(self._tabs)

        # let the mode-profile tab send plots to the Plotter tab
        self._mode_tab.set_plotter(self._plotter_tab)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1020])

        # ── OVF frame viewer (dockable, corner) ───────────────────────
        self._frame_viewer = FrameViewerWidget(self._fm)
        self._frame_dock = QDockWidget("OVF Frame Viewer", self)
        self._frame_dock.setObjectName("FrameViewerDock")
        self._frame_dock.setWidget(self._frame_viewer)
        self._frame_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._frame_dock)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready – add table.txt files to begin.")

    def _build_menu(self) -> None:
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("File")

        add_act = QAction("Add Files…", self)
        add_act.setShortcut(QKeySequence("Ctrl+O"))
        add_act.triggered.connect(self._fm._add_files)
        file_menu.addAction(add_act)

        file_menu.addSeparator()

        quit_act = QAction("Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(QApplication.quit)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = menu.addMenu("View")
        for i, name in enumerate(["Hysteresis", "FMR", "Spin Wave Mode Profile"]):
            act = QAction(name, self)
            act.setShortcut(QKeySequence(f"Ctrl+{i+1}"))
            act.triggered.connect(lambda _, idx=i: self._tabs.setCurrentIndex(idx))
            view_menu.addAction(act)
        view_menu.addSeparator()
        view_menu.addAction(self._frame_dock.toggleViewAction())

        # Help menu
        help_menu = menu.addMenu("Help")
        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_datasets_changed(self) -> None:
        n = len(self._fm.entries)
        self.statusBar().showMessage(
            f"{n} dataset{'s' if n != 1 else ''} loaded."
        )

    def _show_about(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self, "About",
            "<b>MuMax3 Post-Processing Tool</b><br>"
            "Hysteresis and FMR analysis for MuMax3 table.txt files.<br><br>"
            "Axes → Hysteresis tab: select X/Y columns, plot, export.<br>"
            "FMR tab: set dt, bias/MW directions, run, explore heatmaps.",
        )

    # ------------------------------------------------------------------
    # Window geometry persistence
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        geom = self._settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)

    def closeEvent(self, event) -> None:
        self._settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)
