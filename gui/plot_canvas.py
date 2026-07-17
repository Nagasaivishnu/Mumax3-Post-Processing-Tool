"""
Plot Canvas
============
Thin wrapper around a Matplotlib figure embedded in PyQt6.
Used by both the Hysteresis and FMR tabs.
"""

from __future__ import annotations
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavToolbar,
)
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class PlotCanvas(QWidget):
    """
    A self-contained widget that holds a Matplotlib figure + navigation toolbar.

    Usage
    -----
    canvas = PlotCanvas(parent, n_rows=1, n_cols=1)
    ax = canvas.axes[0][0]  # or canvas.single_ax for the common 1×1 case
    ax.plot(x, y)
    canvas.draw()
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        n_rows: int = 1,
        n_cols: int = 1,
        figsize: tuple[float, float] = (7, 5),
    ) -> None:
        super().__init__(parent)

        self._fig, self._axes_grid = Figure(figsize=figsize, tight_layout=True), None
        self._axes_grid = self._fig.subplots(n_rows, n_cols, squeeze=False)

        self._canvas  = FigureCanvas(self._fig)
        self._toolbar = NavToolbar(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def fig(self) -> Figure:
        return self._fig

    @property
    def axes(self):
        """2-D array of Axes, shape (n_rows, n_cols)."""
        return self._axes_grid

    @property
    def single_ax(self):
        """Shortcut for the common 1×1 layout."""
        return self._axes_grid[0][0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def draw(self) -> None:
        self._canvas.draw_idle()

    def clear_axes(self) -> None:
        for row in self._axes_grid:
            for ax in row:
                ax.cla()

    def replace_layout(self, n_rows: int, n_cols: int) -> None:
        """Re-create the subplot grid (e.g. when number of datasets changes)."""
        self._fig.clf()
        self._axes_grid = self._fig.subplots(n_rows, n_cols, squeeze=False)
        self.draw()
