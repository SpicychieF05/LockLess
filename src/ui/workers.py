"""Threading helpers for the Lockless UI."""
from __future__ import annotations

import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal


class OperationWorker(QObject):
    """Execute a callable in a background thread and stream updates."""

    started = pyqtSignal()
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        task: Callable[[], Any],
        description: str = "",
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._task = task
        self._description = description or "Operation"

    def run(self) -> None:
        """Invoke the task and emit results."""

        self.started.emit()
        try:
            result = self._task()
            payload: Dict[str, Any]
            if is_dataclass(result) and not isinstance(result, type):
                payload = asdict(result)
            elif isinstance(result, dict):
                payload = result  # type: ignore[assignment]
            else:
                payload = {"result": result}
            self.finished.emit(payload)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            traceback.print_exc()
            self.failed.emit(f"{self._description} failed: {exc}")


class WorkerRunner:
    """Utility wrapper to connect an OperationWorker to a QThread."""

    def __init__(self, worker: OperationWorker) -> None:
        self.worker = worker
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)

    def start(self) -> None:
        self.thread.start()

    def dispose(self) -> None:
        self.thread.quit()
        self.thread.wait()
