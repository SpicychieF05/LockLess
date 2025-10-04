"""Main application window for the Lockless biometric desktop UI."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .services import (
    ServiceResponse,
    authenticate_user_service,
    delete_user_service,
    enroll_user_service,
    get_logo_path,
    get_system_summary,
    list_users_service,
    test_camera_service,
)
from .workers import OperationWorker, WorkerRunner


class LocklessMainWindow(QMainWindow):
    """Primary window hosting all UI flows."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        super().__init__()
        self._config_path = config_path
        self._runners: List[WorkerRunner] = []

        self.setWindowTitle("Lockless Biometric Suite")
        self.resize(1024, 720)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Build individual tabs
        self._build_dashboard_tab()
        self._build_enrollment_tab()
        self._build_authentication_tab()
        self._build_templates_tab()

        # Initial data population
        self._populate_system_summary()
        self._refresh_users()

    # region UI construction -------------------------------------------------
    def _build_dashboard_tab(self) -> None:
        dashboard = QWidget()
        layout = QVBoxLayout(dashboard)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        pixmap = QPixmap(str(get_logo_path()))
        if not pixmap.isNull():
            scaled = pixmap.scaledToWidth(
                320, Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(scaled)
        else:
            logo_label.setText("Lockless")
            font = QFont()
            font.setPointSize(28)
            font.setBold(True)
            logo_label.setFont(font)

        layout.addWidget(logo_label)

        intro = QLabel(
            "Privacy-first biometric authentication\n"
            "Run enrollment, authentication, and diagnostics from one place."
        )
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.system_info_box = QGroupBox("System summary")
        self.system_info_form = QFormLayout(self.system_info_box)
        layout.addWidget(self.system_info_box)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self.dashboard_camera_button = QPushButton("Test camera")
        self.dashboard_camera_button.clicked.connect(
            self._on_dashboard_camera_test)
        button_row.addWidget(self.dashboard_camera_button)

        self.dashboard_refresh_button = QPushButton("Refresh user list")
        self.dashboard_refresh_button.clicked.connect(self._refresh_users)
        button_row.addWidget(self.dashboard_refresh_button)

        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.tabs.addTab(dashboard, "Dashboard")

    def _build_enrollment_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        form_box = QGroupBox("Enroll new user")
        form_layout = QFormLayout(form_box)

        self.enroll_user_input = QLineEdit()
        self.enroll_user_input.setPlaceholderText("Username or identifier")
        form_layout.addRow("User ID", self.enroll_user_input)

        self.enroll_password_input = QLineEdit()
        self.enroll_password_input.setEchoMode(QLineEdit.Password)
        self.enroll_password_input.setPlaceholderText("Master password")
        form_layout.addRow("Password", self.enroll_password_input)

        self.enroll_config_input = QLineEdit()
        self.enroll_config_input.setPlaceholderText(
            "Optional custom config path")
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(
            partial(self._browse_config, self.enroll_config_input))

        config_row = QHBoxLayout()
        config_row.addWidget(self.enroll_config_input)
        config_row.addWidget(browse_button)
        form_layout.addRow("Config", config_row)

        layout.addWidget(form_box)

        self.enroll_output = QPlainTextEdit()
        self.enroll_output.setReadOnly(True)
        self.enroll_output.setPlaceholderText(
            "Results and guidance will appear here once enrollment starts."
        )
        layout.addWidget(self.enroll_output, stretch=1)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.enroll_button = QPushButton("Start enrollment")
        self.enroll_button.clicked.connect(self._start_enrollment)
        action_row.addWidget(self.enroll_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.tabs.addTab(page, "Enrollment")

    def _build_authentication_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        form_box = QGroupBox("Authenticate user")
        form_layout = QFormLayout(form_box)

        self.auth_user_input = QLineEdit()
        self.auth_user_input.setPlaceholderText("Enrolled user ID")
        form_layout.addRow("User ID", self.auth_user_input)

        self.auth_password_input = QLineEdit()
        self.auth_password_input.setEchoMode(QLineEdit.Password)
        self.auth_password_input.setPlaceholderText("Master password")
        form_layout.addRow("Password", self.auth_password_input)

        self.auth_config_input = QLineEdit()
        self.auth_config_input.setPlaceholderText(
            "Optional custom config path")
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(
            partial(self._browse_config, self.auth_config_input))

        config_row = QHBoxLayout()
        config_row.addWidget(self.auth_config_input)
        config_row.addWidget(browse_button)
        form_layout.addRow("Config", config_row)

        layout.addWidget(form_box)

        self.auth_output = QPlainTextEdit()
        self.auth_output.setReadOnly(True)
        self.auth_output.setPlaceholderText(
            "Authentication and liveness results will appear here."
        )
        layout.addWidget(self.auth_output, stretch=1)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.auth_button = QPushButton("Start authentication")
        self.auth_button.clicked.connect(self._start_authentication)
        action_row.addWidget(self.auth_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.tabs.addTab(page, "Authentication")

    def _build_templates_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        description = QLabel(
            "Manage enrolled templates. "
            "Select a user to remove their biometric template."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.user_list = QListWidget()
        layout.addWidget(self.user_list, stretch=1)

        button_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._refresh_users)
        self.delete_button = QPushButton("Delete selected user")
        self.delete_button.clicked.connect(self._delete_selected_user)

        button_row.addWidget(self.refresh_button)
        button_row.addWidget(self.delete_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.templates_output = QPlainTextEdit()
        self.templates_output.setReadOnly(True)
        self.templates_output.setPlaceholderText(
            "Template management logs will appear here."
        )
        layout.addWidget(self.templates_output, stretch=1)

        self.tabs.addTab(page, "Templates")

    # endregion -------------------------------------------------------------

    # region helpers ------------------------------------------------------
    def _browse_config(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select configuration",
            "",
            "YAML (*.yaml *.yml)",
        )
        if path:
            target.setText(path)

    def _populate_system_summary(self) -> None:
        summary = get_system_summary(self._config_path)
        for i in reversed(range(self.system_info_form.count())):
            self.system_info_form.removeRow(i)

        for key, value in summary.items():
            label = QLabel(str(value))
            self.system_info_form.addRow(key.replace("_", " ").title(), label)

    def _run_async(
        self,
        description: str,
        task: Callable[[], ServiceResponse],
        callback: Callable[[ServiceResponse], None],
        on_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        worker = OperationWorker(lambda: task(), description)
        runner = WorkerRunner(worker)
        self._runners.append(runner)

        def _cleanup(payload: Dict[str, Any]) -> None:
            runner.dispose()
            if runner in self._runners:
                self._runners.remove(runner)
            response = ServiceResponse(**payload)
            callback(response)
            if on_complete:
                on_complete()

        def _handle_failure(message: str) -> None:
            runner.dispose()
            if runner in self._runners:
                self._runners.remove(runner)
            QMessageBox.critical(self, "Operation failed", message)
            if on_complete:
                on_complete()

        def _announce_start() -> None:
            self.status_bar.showMessage(f"{description} in progress…")

        worker.started.connect(_announce_start)
        worker.finished.connect(_cleanup)
        worker.failed.connect(_handle_failure)
        runner.start()

    # endregion -------------------------------------------------------------

    # region operations ---------------------------------------------------
    def _start_enrollment(self) -> None:
        user = self.enroll_user_input.text().strip()
        password = self.enroll_password_input.text()
        config = self.enroll_config_input.text().strip() or None

        if not user or not password:
            QMessageBox.warning(
                self,
                "Missing information",
                "User ID and password are required.",
            )
            return

        self.enroll_button.setEnabled(False)
        self.enroll_output.clear()

        def task() -> ServiceResponse:
            return enroll_user_service(user, password, config)

        def on_complete() -> None:
            self.enroll_button.setEnabled(True)
            self.status_bar.clearMessage()
            self._refresh_users()

        self._run_async("Enrollment", task,
                        self._handle_enrollment_result, on_complete)

    def _handle_enrollment_result(self, response: ServiceResponse) -> None:
        if response.success:
            self.enroll_output.appendPlainText(response.message)
            payload = response.payload or {}
            result = payload.get("result", {})
            samples = result.get("samples_collected")
            avg_quality = result.get("average_quality")
            if isinstance(samples, int) and isinstance(
                avg_quality, (int, float)
            ):
                self.enroll_output.appendPlainText(
                    f"Samples: {samples} | Average quality: {avg_quality:.3f}"
                )
        else:
            self.enroll_output.appendPlainText(response.message)

    def _start_authentication(self) -> None:
        user = self.auth_user_input.text().strip()
        password = self.auth_password_input.text()
        config = self.auth_config_input.text().strip() or None

        if not user or not password:
            QMessageBox.warning(
                self,
                "Missing information",
                "User ID and password are required.",
            )
            return

        self.auth_button.setEnabled(False)
        self.auth_output.clear()

        def task() -> ServiceResponse:
            return authenticate_user_service(user, password, config)

        def on_complete() -> None:
            self.auth_button.setEnabled(True)
            self.status_bar.clearMessage()

        self._run_async("Authentication", task,
                        self._handle_authentication_result, on_complete)

    def _handle_authentication_result(self, response: ServiceResponse) -> None:
        if response.success:
            self.auth_output.appendPlainText(response.message)
            payload = response.payload or {}
            data = payload.get("response", {})
            confidence = data.get("confidence")
            quality = data.get("quality_score")
            processing = data.get("processing_time")
            if isinstance(confidence, (int, float)):
                self.auth_output.appendPlainText(
                    f"Confidence: {confidence:.3f}")
            if isinstance(quality, (int, float)):
                self.auth_output.appendPlainText(
                    f"Quality score: {quality:.3f}")
            if isinstance(processing, (int, float)):
                self.auth_output.appendPlainText(
                    f"Processing time: {processing:.2f}s")
        else:
            self.auth_output.appendPlainText(response.message)

    def _refresh_users(self) -> None:
        def task() -> ServiceResponse:
            return list_users_service()

        def on_complete() -> None:
            self.status_bar.clearMessage()

        def callback(response: ServiceResponse) -> None:
            self.user_list.clear()
            if response.success and response.payload:
                for user_id in response.payload.get("users", []):
                    QListWidgetItem(str(user_id), self.user_list)
                self.templates_output.appendPlainText(response.message)
            else:
                self.templates_output.appendPlainText(response.message)

        self._run_async("Refreshing users", task, callback, on_complete)

    def _delete_selected_user(self) -> None:
        current_item = self.user_list.currentItem()
        if not current_item:
            QMessageBox.information(
                self, "No selection", "Select a user to delete.")
            return

        user_id = current_item.text()
        confirmation = QMessageBox.question(
            self,
            "Confirm deletion",
            f"Delete biometric template for '{user_id}'?",
        )
        if confirmation != QMessageBox.Yes:
            return

        def task() -> ServiceResponse:
            return delete_user_service(user_id)

        def callback(response: ServiceResponse) -> None:
            self.templates_output.appendPlainText(response.message)
            if response.success:
                self._refresh_users()

        self._run_async("Deleting user", task, callback)

    def _on_dashboard_camera_test(self) -> None:
        button = self.dashboard_camera_button
        button.setEnabled(False)

        def task() -> ServiceResponse:
            summary = get_system_summary(self._config_path)
            camera_id = summary.get("camera_id", 0)
            return test_camera_service(camera_id)

        def callback(response: ServiceResponse) -> None:
            QMessageBox.information(
                self,
                "Camera test",
                response.message,
            )

        def on_complete() -> None:
            button.setEnabled(True)
            self.status_bar.clearMessage()

        self._run_async("Camera test", task, callback, on_complete)

    # endregion -------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        for runner in list(self._runners):
            runner.dispose()
        self._runners.clear()
        super().closeEvent(event)


def run_standalone() -> int:
    """Allow running the UI by executing this module directly."""

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication([])
        created = True

    window = LocklessMainWindow()
    window.show()

    if created:
        return app.exec()
    return 0
