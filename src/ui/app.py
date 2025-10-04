"""Application bootstrap for the Lockless GUI."""
from __future__ import annotations

import sys
from typing import Optional

from PyQt5.QtWidgets import QApplication

from .main_window import LocklessMainWindow


def run_gui(config_path: Optional[str] = None) -> int:
    """Start the PyQt application and return the exit code."""

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True

    window = LocklessMainWindow(config_path=config_path)
    window.show()

    if created:
        return app.exec()
    return 0
