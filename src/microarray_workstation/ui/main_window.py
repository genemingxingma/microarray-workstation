from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
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
from microarray_workstation.analysis.ai_classifier import classify_spot_quality
from microarray_workstation.integration.lims_client import LIMSClient
from microarray_workstation.integration.lab_interface_client import LaboratoryManagementInterfaceClient
from microarray_workstation.io.exporters import export_dataframe_csv, export_json
from microarray_workstation.rules.interpreter import interpret, load_template, summarize_calls
from microarray_workstation.workflows.analysis_workflow import (
    IMAGE_SUFFIXES,
    analyze_one_image,
    build_lab_interface_jobs_from_summaries,
    list_summary_payloads,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Microarray Workstation")
        self.resize(1500, 920)

        self.current_image_path: str | None = None
        self.current_gray: np.ndarray | None = None
        self.latest_df = None
        self.last_summary = None
        self.grid_shift_x = 0.0
        self.grid_shift_y = 0.0

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

        adjust_row = QHBoxLayout()
        self.shift_step_input = QSpinBox()
        self.shift_step_input.setRange(1, 50)
        self.shift_step_input.setValue(2)

        left_btn = QPushButton("Shift Left")
        right_btn = QPushButton("Shift Right")
        up_btn = QPushButton("Shift Up")
        down_btn = QPushButton("Shift Down")
        reset_btn = QPushButton("Reset Shift")

        left_btn.clicked.connect(lambda: self.on_shift_grid(-self.shift_step_input.value(), 0))
        right_btn.clicked.connect(lambda: self.on_shift_grid(self.shift_step_input.value(), 0))
        up_btn.clicked.connect(lambda: self.on_shift_grid(0, -self.shift_step_input.value()))
        down_btn.clicked.connect(lambda: self.on_shift_grid(0, self.shift_step_input.value()))
        reset_btn.clicked.connect(self.on_reset_shift)

        adjust_row.addWidget(QLabel("Shift(px)"))
        adjust_row.addWidget(self.shift_step_input)
        adjust_row.addWidget(left_btn)
        adjust_row.addWidget(right_btn)
        adjust_row.addWidget(up_btn)
        adjust_row.addWidget(down_btn)
        adjust_row.addWidget(reset_btn)

        batch_input_row = QHBoxLayout()
        self.batch_input_dir = QLineEdit()
        self.batch_input_dir.setPlaceholderText("Batch Input Dir")
        browse_batch_in_btn = QPushButton("Browse In")
        browse_batch_in_btn.clicked.connect(self.on_browse_batch_input)
        batch_input_row.addWidget(QLabel("Batch In"))
        batch_input_row.addWidget(self.batch_input_dir)
        batch_input_row.addWidget(browse_batch_in_btn)

        batch_output_row = QHBoxLayout()
        self.batch_output_dir = QLineEdit()
        self.batch_output_dir.setPlaceholderText("Batch Output Dir")
        browse_batch_out_btn = QPushButton("Browse Out")
        browse_batch_out_btn.clicked.connect(self.on_browse_batch_output)
        run_batch_btn = QPushButton("Run Batch")
        run_batch_btn.clicked.connect(self.on_run_batch)
        batch_output_row.addWidget(QLabel("Batch Out"))
        batch_output_row.addWidget(self.batch_output_dir)
        batch_output_row.addWidget(browse_batch_out_btn)
        batch_output_row.addWidget(run_batch_btn)

        lims_row = QHBoxLayout()
        self.lims_base_url_input = QLineEdit()
        self.lims_base_url_input.setPlaceholderText("LIMS Base URL")
        self.lims_endpoint_input = QLineEdit()
        self.lims_endpoint_input.setPlaceholderText("LIMS Endpoint (/api/results) or Endpoint Code (HIS-REST)")
        self.lims_token_input = QLineEdit()
        self.lims_token_input.setPlaceholderText("Token (optional)")
        submit_batch_btn = QPushButton("Submit Batch To LIMS")
        submit_batch_btn.clicked.connect(self.on_submit_batch_lims)
        lims_row.addWidget(self.lims_base_url_input)
        lims_row.addWidget(self.lims_endpoint_input)
        lims_row.addWidget(self.lims_token_input)
        lims_row.addWidget(submit_batch_btn)

        self.batch_progress = QProgressBar()
        self.batch_progress.setMinimum(0)
        self.batch_progress.setMaximum(100)
        self.batch_progress.setValue(0)

        self.image_label = QLabel("Open a microarray image to start")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(620, 480)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(190)

        left.addLayout(control_row)
        left.addLayout(adjust_row)
        left.addLayout(batch_input_row)
        left.addLayout(batch_output_row)
        left.addLayout(lims_row)
        left.addWidget(self.batch_progress)
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
        self.grid_shift_x = 0.0
        self.grid_shift_y = 0.0
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

    def _resolve_template(self) -> dict:
        template_path = self.template_input.text().strip()
        if template_path:
            return load_template(template_path)
        return {
            "interpretation": {
                "snr_threshold": 2.0,
                "net_median_threshold": 200.0,
            }
        }

    def _run_analysis_and_render(self) -> None:
        if not self.current_image_path:
            return

        rows = int(self.rows_input.value())
        cols = int(self.cols_input.value())
        result = run_analysis(
            self.current_image_path,
            rows=rows,
            cols=cols,
            grid_shift=(self.grid_shift_x, self.grid_shift_y),
        )
        df = to_dataframe(result)
        if self.current_gray is None:
            self.current_gray = load_image(self.current_image_path)
        ai_df, _ = classify_spot_quality(self.current_gray, df, model_path=None)
        template = self._resolve_template()
        interpreted = interpret(ai_df, template)
        self.latest_df = interpreted

        points = [(float(v["x"]), float(v["y"])) for _, v in interpreted.head(500).iterrows()]
        if self.current_gray is not None:
            self._render_image(self.current_gray, points=points)

        self._fill_table(interpreted)
        self.last_summary = summarize_calls(interpreted)
        qc = result.metadata.get("qc", {})
        self.log_info(
            f"Analysis complete: total={self.last_summary['total']} positive={self.last_summary['positive']} "
            f"negative={self.last_summary['negative']} review={self.last_summary['review']} "
            f"qc={qc.get('qc_status', 'NA')} mean_snr={qc.get('mean_snr', 0)}"
        )

    def on_analyze(self) -> None:
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        try:
            self._run_analysis_and_render()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Analysis Failed", str(exc))
            self.log_info(f"Analysis failed: {exc}")

    def on_shift_grid(self, dx: float, dy: float) -> None:
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        self.grid_shift_x += float(dx)
        self.grid_shift_y += float(dy)
        self.log_info(f"Grid shift updated: dx={self.grid_shift_x:.1f}, dy={self.grid_shift_y:.1f}")
        try:
            self._run_analysis_and_render()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Shift Re-analysis Failed", str(exc))
            self.log_info(f"Shift re-analysis failed: {exc}")

    def on_reset_shift(self) -> None:
        self.grid_shift_x = 0.0
        self.grid_shift_y = 0.0
        self.log_info("Grid shift reset to dx=0.0, dy=0.0")
        if self.current_image_path:
            try:
                self._run_analysis_and_render()
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Reset Re-analysis Failed", str(exc))
                self.log_info(f"Reset re-analysis failed: {exc}")

    def on_browse_batch_input(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Batch Input Directory")
        if directory:
            self.batch_input_dir.setText(directory)

    def on_browse_batch_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Batch Output Directory")
        if directory:
            self.batch_output_dir.setText(directory)

    def on_run_batch(self) -> None:
        input_dir = self.batch_input_dir.text().strip()
        output_dir = self.batch_output_dir.text().strip()
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "Missing Path", "Please set batch input/output directory.")
            return

        image_files = [p for p in sorted(Path(input_dir).iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
        if not image_files:
            QMessageBox.warning(self, "No Images", f"No supported image files in {input_dir}")
            return

        rows = int(self.rows_input.value())
        cols = int(self.cols_input.value())
        template_path = self.template_input.text().strip() or None

        self.batch_progress.setMaximum(len(image_files))
        self.batch_progress.setValue(0)

        batch_rows = []
        for idx, image in enumerate(image_files, start=1):
            out = analyze_one_image(
                image_path=str(image),
                rows=rows,
                cols=cols,
                template_path=template_path,
                output_dir=output_dir,
                channel=None,
                ai_model=None,
            )
            batch_rows.append(
                {
                    "image": out["image"],
                    "qc_status": out["qc_status"],
                    "positive": out["summary"]["positive"],
                    "negative": out["summary"]["negative"],
                    "review": out["summary"]["review"],
                    "ai_mode": out["ai_summary"]["mode"],
                    "ai_mean_score": out["ai_summary"]["mean_ai_score"],
                    "summary_json": out["summary_json"],
                }
            )
            self.batch_progress.setValue(idx)
            self.log_info(f"Batch processed {idx}/{len(image_files)}: {image.name}")
            QApplication.processEvents()

        out_dir = Path(output_dir)
        batch_df = pd.DataFrame(batch_rows)
        summary_csv = out_dir / "batch_summary.csv"
        summary_json = out_dir / "batch_summary.json"
        export_dataframe_csv(batch_df, summary_csv)
        export_json({"total_images": len(batch_rows), "records": batch_rows}, summary_json)
        self.log_info(f"Batch complete: {len(batch_rows)} files")
        self.log_info(f"Batch summary: {summary_csv}")

    def on_submit_batch_lims(self) -> None:
        output_dir = self.batch_output_dir.text().strip()
        base_url = self.lims_base_url_input.text().strip()
        endpoint = self.lims_endpoint_input.text().strip()
        token = self.lims_token_input.text().strip() or None

        if not output_dir or not base_url or not endpoint:
            QMessageBox.warning(self, "Missing LIMS Info", "Set batch output dir, base URL and endpoint first.")
            return

        try:
            if endpoint.startswith("/"):
                payloads = list_summary_payloads(output_dir)
                client = LIMSClient(base_url=base_url, token=token)
                result = client.submit_batch_results(endpoint=endpoint, payloads=payloads)
                out_path = Path(output_dir) / "lims_submit_summary.json"
            else:
                jobs = build_lab_interface_jobs_from_summaries(output_dir)
                client = LaboratoryManagementInterfaceClient(
                    base_url=base_url,
                    auth_type="bearer" if token else "none",
                    token=token,
                )
                result = client.submit_batch_result_auto(endpoint_code=endpoint, jobs=jobs)
                out_path = Path(output_dir) / "lab_interface_submit_summary.json"
            export_json(result, out_path)
            self.log_info(
                f"LIMS submit complete: total={result['total']} success={result['success']} failed={result['failed']}"
            )
            self.log_info(f"LIMS summary: {out_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "LIMS Batch Submit Failed", str(exc))
            self.log_info(f"LIMS batch submit failed: {exc}")

    def _fill_table(self, df) -> None:
        view_cols = ["row", "col", "net_median", "snr", "ai_score", "ai_label", "flag", "call", "target"]
        self.table.clear()
        self.table.setColumnCount(len(view_cols))
        self.table.setHorizontalHeaderLabels(view_cols)
        self.table.setRowCount(len(df))

        for r in range(len(df)):
            for c, col in enumerate(view_cols):
                if col in df.columns:
                    self.table.setItem(r, c, QTableWidgetItem(str(df.iloc[r][col])))
                else:
                    self.table.setItem(r, c, QTableWidgetItem(""))

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
        summary = self.last_summary if self.last_summary is not None else summarize_calls(self.latest_df)
        payload = {
            "summary": summary,
            "grid_shift": {"dx": self.grid_shift_x, "dy": self.grid_shift_y},
        }
        export_json(payload, json_path)
        self.log_info(f"Exported: {csv_path}")
        self.log_info(f"Exported: {json_path}")
