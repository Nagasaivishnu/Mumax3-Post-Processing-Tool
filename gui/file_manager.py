"""
File Manager
=============
A panel that lets the user add, label, and remove MuMax3 table.txt files.
Emits `datasets_changed` whenever the list is modified so the analysis
tabs can refresh automatically.
"""

from __future__ import annotations
import logging
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt, QSettings
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QHeaderView, QLabel,
    QAbstractItemView,
)

from processing.data_loader import load_table, LoadError

logger = logging.getLogger(__name__)


class DataEntry:
    """Holds one dataset: its path, user label, and (lazily) the DataFrame."""

    def __init__(self, path: str, label: str) -> None:
        self.path  = path
        self.label = label
        self._df   = None

    @property
    def df(self):
        if self._df is None:
            self._df = load_table(self.path)
        return self._df

    def reload(self) -> None:
        self._df = None

    def __repr__(self) -> str:  # pragma: no cover
        return f"DataEntry(label={self.label!r}, path={self.path!r})"


class FileManagerWidget(QWidget):
    """
    Left-panel file manager.

    Signals
    -------
    datasets_changed : emitted after any add / remove / label change
    """

    datasets_changed = pyqtSignal()

    # column indices
    COL_LABEL = 0
    COL_PATH  = 1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[DataEntry] = []
        self._settings = QSettings("MuMax3Tool", "FileManager")
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[DataEntry]:
        """Return a copy of the current dataset list."""
        return list(self._entries)

    def get_loaded_entries(self) -> list[DataEntry]:
        """
        Return entries whose DataFrames can be loaded.
        Shows a warning dialog for any that fail.
        """
        good = []
        for entry in self._entries:
            try:
                _ = entry.df  # triggers load if not cached
                good.append(entry)
            except LoadError as e:
                QMessageBox.warning(self, "Load Error", str(e))
        return good

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel("<b>Datasets</b>"))

        # ── table ──────────────────────────────────────────────────────
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Label", "File"])
        self._table.horizontalHeader().setSectionResizeMode(
            self.COL_PATH, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            self.COL_LABEL, QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
        )
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table)

        # ── buttons ────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        for text, slot in [
            ("Add Files", self._add_files),
            ("Remove",    self._remove_selected),
            ("Clear All", self._clear_all),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _add_files(self) -> None:
        last_dir = self._settings.value("last_dir", "")
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select table.txt files",
            last_dir,
            "MuMax3 table files (*.txt);;All files (*)",
        )
        if not paths:
            return

        self._settings.setValue("last_dir", str(Path(paths[0]).parent))

        existing_paths = {e.path for e in self._entries}
        added = 0
        for path in paths:
            if path in existing_paths:
                continue
            label = self._auto_label(path)
            label = self._ensure_unique_label(label)
            entry = DataEntry(path, label)
            self._entries.append(entry)
            self._add_table_row(entry)
            added += 1

        if added:
            self.datasets_changed.emit()

    def _remove_selected(self) -> None:
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self._entries.pop(row)
            self._table.removeRow(row)
        if rows:
            self.datasets_changed.emit()

    def _clear_all(self) -> None:
        if not self._entries:
            return
        reply = QMessageBox.question(
            self, "Clear All", "Remove all datasets?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._entries.clear()
            self._table.setRowCount(0)
            self.datasets_changed.emit()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != self.COL_LABEL:
            return
        row   = item.row()
        new_label = item.text().strip()
        if not new_label:
            # revert to previous label
            self._table.blockSignals(True)
            item.setText(self._entries[row].label)
            self._table.blockSignals(False)
            return
        # ensure uniqueness
        unique = self._ensure_unique_label(new_label, exclude_row=row)
        self._entries[row].label = unique
        if unique != new_label:
            self._table.blockSignals(True)
            item.setText(unique)
            self._table.blockSignals(False)
        self.datasets_changed.emit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_table_row(self, entry: DataEntry) -> None:
        self._table.blockSignals(True)
        row = self._table.rowCount()
        self._table.insertRow(row)

        label_item = QTableWidgetItem(entry.label)
        path_item  = QTableWidgetItem(entry.path)
        path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        self._table.setItem(row, self.COL_LABEL, label_item)
        self._table.setItem(row, self.COL_PATH,  path_item)
        self._table.blockSignals(False)

    def _auto_label(self, path: str) -> str:
        """Generate a label from the parent folder name."""
        return Path(path).parent.stem or Path(path).stem

    def _ensure_unique_label(self, label: str, exclude_row: int = -1) -> str:
        existing = {
            e.label for i, e in enumerate(self._entries) if i != exclude_row
        }
        if label not in existing:
            return label
        counter = 2
        while f"{label}_{counter}" in existing:
            counter += 1
        return f"{label}_{counter}"
