from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from microarray_workstation.analysis.image_loader import load_image, normalize_to_uint8
from microarray_workstation.analysis.pipeline import run_analysis, to_dataframe
from microarray_workstation.io.exporters import export_dataframe_csv, export_json
from microarray_workstation.rules.interpreter import interpret, load_template, summarize_calls


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Microarray Workstation")
        self.resize(1400, 900)

        self.current_image_path: str | None = None
        self.current_gray: np.ndarray | None = None
        self.latest_df = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        left = QVBoxLayout()
        right = QVBoxLayout()

        control_row = QHBoxLayout()
        self.rows_input = QSpinBox()
        self.rows_input.setRange(1, 1000)
        self.rows_input.setValue(8)
        self.cols_input = QSpinBox()
        self.cols_input.setRange(1, 1000)
        self.cols_input.setValue(12)
        self.template_input = QLineEdit()
        self.template_input.setPlaceholderText("Template YAML path (optional)")

        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.on_analyze)

        control_row.addWidget(QLabel("Rows"))
        control_row.addWidget(self.rows_input)
        control_row.addWidget(QLabel("Cols"))
        control_row.addWidget(self.cols_input)
        control_row.addWidget(QLabel("Template"))
        control_row.addWidget(self.template_input)
        control_row.addWidget(analyze_btn)

        self.image_label = QLabel("Open a microarray image to start")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 500)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)

        left.addLayout(control_row)
        left.addWidget(self.image_label, stretch=1)
        left.addWidget(self.log)

        self.table = QTableWidget()
        right.addWidget(self.table)

        layout.addLayout(left, stretch=3)
        layout.addLayout(right, stretch=2)

        menu = self.menuBar().addMenu("File")
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.on_open_image)
        menu.addAction(open_action)

        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self.on_export)
        menu.addAction(export_action)

    def log_info(self, message: str) -> None:
        self.log.append(message)

    def on_open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Microarray Image",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if not path:
            return

        self.current_image_path = path
        self.current_gray = load_image(path)
        self._render_image(self.current_gray)
        self.log_info(f"Loaded image: {path}")

    def _render_image(self, gray: np.ndarray, points: list[tuple[float, float]] | None = None) -> None:
        img = normalize_to_uint8(gray)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if points:
            for x, y in points:
                cv2.circle(bgr, (int(x), int(y)), 4, (0, 255, 255), 1)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self.current_gray is not None:
            self._render_image(self.current_gray)

    def on_analyze(self) -> None:
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        rows = int(self.rows_input.value())
        cols = int(self.cols_input.value())
        template_path = self.template_input.text().strip()

        try:
            result = run_analysis(self.current_image_path, rows=rows, cols=cols)
            df = to_dataframe(result)

            if template_path:
                template = load_template(template_path)
            else:
                template = {
                    "interpretation": {
                        "snr_threshold": 2.0,
                        "net_median_threshold": 200.0,
                    }
                }
            interpreted = interpret(df, template)
            self.latest_df = interpreted

            points = [(float(v["x"]), float(v["y"])) for _, v in interpreted.head(500).iterrows()]
            if self.current_gray is not None:
                self._render_image(self.current_gray, points=points)

            self._fill_table(interpreted)
            summary = summarize_calls(interpreted)
            self.log_info(
                f"Analysis complete: total={summary['total']} positive={summary['positive']} "
                f"negative={summary['negative']} review={summary['review']}"
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Analysis Failed", str(exc))
            self.log_info(f"Analysis failed: {exc}")

    def _fill_table(self, df) -> None:
        view_cols = ["row", "col", "net_median", "snr", "flag", "call", "target"]
        self.table.clear()
        self.table.setColumnCount(len(view_cols))
        self.table.setHorizontalHeaderLabels(view_cols)
        self.table.setRowCount(len(df))

        for r in range(len(df)):
            for c, col in enumerate(view_cols):
                self.table.setItem(r, c, QTableWidgetItem(str(df.iloc[r][col])))

        self.table.resizeColumnsToContents()

    def on_export(self) -> None:
        if self.latest_df is None:
            QMessageBox.warning(self, "No Results", "Run analysis before export.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not output_dir:
            return

        stem = Path(self.current_image_path).stem if self.current_image_path else "microarray"
        csv_path = Path(output_dir) / f"{stem}_interpreted.csv"
        json_path = Path(output_dir) / f"{stem}_summary.json"

        export_dataframe_csv(self.latest_df, csv_path)
        summary = summarize_calls(self.latest_df)
        export_json(summary, json_path)
        self.log_info(f"Exported: {csv_path}")
        self.log_info(f"Exported: {json_path}")
