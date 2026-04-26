"""
WIPWindow — experimental Photoshop-style layout tab.

Phases:
  Phase 1  Structural skeleton: three fixed zones + three workspace columns.
  Phase 2  Center canvas with image navigation wired up.
  Phase 3+ Right panel, tool palette, and status bar wired up incrementally.

Rules (consistent with other views):
  V1  All file/folder dialogs live here, not in controllers.
  V2  State is never accessed directly; all data reads go through model query APIs.
  V3  No disk I/O here; controllers handle all file operations.
  V4  Colors are (r, g, b) tuples until the last Qt draw boundary.

Color scheme: no explicit stylesheet colors — Qt platform palette only.
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
    QPushButton, QFileDialog,
)

from views.annomate.image_label import ImageLabel


class WIPWindow(QWidget):
    """Experimental Photoshop-style layout tab.

    Receives the dataset model and IO controller so later phases can wire up
    the canvas, navigator, and class panel without touching AppWindow.

    Args:
        dataset_model: DatasetTableModel instance.
        io_controller: IOController instance.
        parent: Optional Qt parent widget.
    """

    def __init__(self, dataset_model, io_controller, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.io_controller = io_controller
        self._current_row: int = -1
        self._init_ui()

        self.dataset_model.modelReset.connect(self._on_model_reset)
        self.canvas.polygonFinished.connect(self._on_polygon_finished)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._top_bar = self._build_top_bar()
        root.addWidget(self._top_bar)

        self._workspace = self._build_workspace()
        root.addWidget(self._workspace, stretch=1)

        self._status_bar = self._build_status_bar()
        root.addWidget(self._status_bar)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(54)

        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Strip A — menu labels (24 px)
        strip_a = QWidget()
        strip_a.setFixedHeight(24)
        a_layout = QHBoxLayout(strip_a)
        a_layout.setContentsMargins(4, 0, 4, 0)
        a_layout.setSpacing(0)
        for label in ("File", "Edit", "Image", "Layer", "Type", "Select", "View", "Window", "Help"):
            lbl = QLabel(label)
            lbl.setContentsMargins(8, 0, 8, 0)
            a_layout.addWidget(lbl)
        a_layout.addStretch()
        layout.addWidget(strip_a)

        # Strip B — navigation controls (30 px)
        strip_b = QWidget()
        strip_b.setFixedHeight(30)
        b_layout = QHBoxLayout(strip_b)
        b_layout.setContentsMargins(8, 2, 8, 2)
        b_layout.setSpacing(6)

        btn_open = QPushButton("Open Folder")
        btn_open.clicked.connect(self._open_folder)
        b_layout.addWidget(btn_open)

        b_layout.addSpacing(12)

        self._btn_prev = QPushButton("< Prev")
        self._btn_prev.clicked.connect(self._prev_image)
        b_layout.addWidget(self._btn_prev)

        self._btn_next = QPushButton("Next >")
        self._btn_next.clicked.connect(self._next_image)
        b_layout.addWidget(self._btn_next)

        self._lbl_image_name = QLabel("")
        b_layout.addWidget(self._lbl_image_name)

        b_layout.addStretch()
        layout.addWidget(strip_b)

        return bar

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        h_layout = QHBoxLayout(workspace)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # Left tool palette — 52 px fixed, placeholder until Phase 5
        self._tool_palette = QWidget()
        self._tool_palette.setFixedWidth(52)
        h_layout.addWidget(self._tool_palette)

        # Center artboard — fluid, real ImageLabel
        self.canvas = ImageLabel(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        h_layout.addWidget(self.canvas, stretch=1)

        # Right panel — 280 px fixed, placeholder until Phase 3
        self._right_panel = QWidget()
        self._right_panel.setFixedWidth(280)
        h_layout.addWidget(self._right_panel)

        return workspace

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(24)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(16)

        self._lbl_status_left = QLabel("Zoom: 100%  |  — × — px  |  Tool: None")
        layout.addWidget(self._lbl_status_left)

        layout.addStretch()

        self._lbl_status_right = QLabel("Ready")
        layout.addWidget(self._lbl_status_right)

        return bar

    # ------------------------------------------------------------------ #
    # Navigation slots
    # ------------------------------------------------------------------ #

    def _open_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open Image Folder", os.getcwd())
        if not directory:
            return
        self.io_controller.load_folder(directory)

    def _on_model_reset(self) -> None:
        if self.dataset_model.rowCount() > 0:
            self._load_row(0)
        else:
            self._current_row = -1
            self._lbl_image_name.setText("")

    def _load_row(self, row: int) -> None:
        bgr = self.io_controller.load_image_for_display(row)
        if bgr is None:
            return
        self.canvas.set_image(bgr)
        self._current_row = row
        self._refresh_overlays()
        total = self.dataset_model.rowCount()
        filename = self.dataset_model.get_image_filename(row)
        self._lbl_image_name.setText(f"{row + 1} / {total}  —  {filename}")

    def _prev_image(self) -> None:
        if self._current_row > 0:
            self._load_row(self._current_row - 1)

    def _next_image(self) -> None:
        if self._current_row < self.dataset_model.rowCount() - 1:
            self._load_row(self._current_row + 1)

    # ------------------------------------------------------------------ #
    # Annotation
    # ------------------------------------------------------------------ #

    def _on_polygon_finished(self, pts: list) -> None:
        if self._current_row < 0 or not pts:
            return
        class_names = self.dataset_model.get_class_names()
        active_class = class_names[0] if class_names else "defect"
        self.dataset_model.add_annotation(self._current_row, active_class, pts)
        self._refresh_overlays()

    def _refresh_overlays(self) -> None:
        """Rebuild canvas overlays from the current row's annotations. V4: QColor constructed here."""
        annos = self.dataset_model.get_annotations(self._current_row)
        overlays = [
            (a["polygon"], QColor(*self.dataset_model.get_class_color(a["category_name"])))
            for a in annos
        ]
        self.canvas.set_overlays(overlays)
