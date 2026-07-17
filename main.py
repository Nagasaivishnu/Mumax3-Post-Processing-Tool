"""
MuMax3 Post-Processing Tool
============================
Entry point.  Run with:

    python main.py

Requires:  PyQt6, matplotlib, pandas, numpy, scipy
"""

import sys
import logging

# Set up logging before importing Qt
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s : %(message)s",
)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow


def main() -> None:
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("MuMax3 Post-Processing Tool")
    app.setOrganizationName("MuMax3Tool")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
