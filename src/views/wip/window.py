"""
WIPWindow — experimental Photoshop-style layout tab.

Rules (consistent with other views):
  V1  All file/folder dialogs live here, not in controllers.
  V2  State is never accessed directly; all data reads go through model query APIs.
  V3  No disk I/O here; controllers handle all file operations.
  V4  Colors are (r, g, b) tuples until the last Qt draw boundary.

Color scheme: no explicit stylesheet colors — Qt platform palette only.
"""

import os

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QFileDialog, QSplitter, QToolButton,
)

from views.annomate.image_label import ImageLabel
from views.wip.right_panel import RightPanel
from views.wip.tool_palette import ToolPalette
from views.wip.top_bar import TopBar
from views.wip.status_bar import WIPStatusBar


class _ZoomToolbar(QFrame):
    """Floating vertical toolbar with zoom-in, zoom-out, and reset-view buttons."""

    _MARGIN = 10
    _BTN_SIZE = 30

    def __init__(self, canvas, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)

        font = QFont()
        font.setPointSize(14)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        for text, tip, slot in (
            ("+",  "Zoom In",    canvas.zoom_in),
            ("−",  "Zoom Out",   canvas.zoom_out),
            ("⊙",  "Reset View", canvas.reset_view),
        ):
            btn = QToolButton()
            btn.setText(text)
            btn.setToolTip(tip)
            btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
            btn.setFont(font)
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        self.adjustSize()

    def reposition(self, canvas_size) -> None:
        x = self._MARGIN
        y = canvas_size.height() - self.height() - self._MARGIN
        self.move(x, y)


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
        self._active_class: str = ""
        self._active_tool: str = ""
        self._init_ui()

        # Dataset changes
        self.dataset_model.modelReset.connect(self._on_model_reset)

        # Canvas → navigation / annotation
        self.canvas.polygonFinished.connect(self._on_polygon_finished)
        self.canvas.toolCanceled.connect(self._on_tool_canceled)

        # Canvas → status bar (live feedback)
        self.canvas.zoom_changed.connect(self.status_bar.set_zoom)
        self.canvas.image_loaded.connect(self.status_bar.set_dimensions)

        # Top bar
        self.top_bar.open_folder_requested.connect(self._open_folder)

        # Right panel
        self.right_panel.image_selected.connect(self._load_row)
        self.right_panel.class_selected.connect(self._set_active_class)
        self.right_panel.prev_requested.connect(self._prev_image)
        self.right_panel.next_requested.connect(self._next_image)

        # Tool palette
        self.tool_palette.tool_selected.connect(self._on_tool_selected)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.top_bar = TopBar(self)
        root.addWidget(self.top_bar)

        root.addWidget(self._build_workspace(), stretch=1)

        self.status_bar = WIPStatusBar(self)
        root.addWidget(self.status_bar)

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        h_layout = QHBoxLayout(workspace)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        self.tool_palette = ToolPalette(self)
        h_layout.addWidget(self.tool_palette)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setChildrenCollapsible(False)

        self.canvas = ImageLabel(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.canvas)

        self._zoom_toolbar = _ZoomToolbar(self.canvas, self.canvas)
        self._zoom_toolbar.raise_()
        self.canvas.installEventFilter(self)

        self.right_panel = RightPanel(self.dataset_model, self)
        self.right_panel.setMinimumWidth(160)
        splitter.addWidget(self.right_panel)

        splitter.setSizes([700, 280])
        h_layout.addWidget(splitter, stretch=1)

        return workspace

    # ------------------------------------------------------------------ #
    # Zoom toolbar positioning
    # ------------------------------------------------------------------ #

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._zoom_toolbar.reposition(self.canvas.size())

    def eventFilter(self, obj, event) -> bool:
        if obj is self.canvas and event.type() == QEvent.Resize:
            self._zoom_toolbar.reposition(event.size())
        return super().eventFilter(obj, event)

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

    def _load_row(self, row: int) -> None:
        bgr = self.io_controller.load_image_for_display(row)
        if bgr is None:
            return
        self.canvas.set_image(bgr)
        self.status_bar.set_zoom(1.0)   # set_image resets zoom without emitting zoom_changed
        self._current_row = row
        self._refresh_overlays()
        total = self.dataset_model.rowCount()
        self.right_panel.set_counter(row, total)
        self.right_panel.select_row(row)
        self.right_panel.set_current_row(row)

    def _prev_image(self) -> None:
        if self._current_row > 0:
            self._load_row(self._current_row - 1)

    def _next_image(self) -> None:
        if self._current_row < self.dataset_model.rowCount() - 1:
            self._load_row(self._current_row + 1)

    # ------------------------------------------------------------------ #
    # Tool slots
    # ------------------------------------------------------------------ #

    def _on_tool_selected(self, tool_name: str) -> None:
        self._active_tool = tool_name
        self.canvas.set_tool("polygon" if tool_name == "polygon" else None)
        self.status_bar.set_tool(tool_name)
        self.top_bar.set_active_tool(tool_name)

    def _on_tool_canceled(self) -> None:
        self.tool_palette.deselect_all()
        self._active_tool = ""
        self.status_bar.set_tool("")
        self.top_bar.set_active_tool("")

    # ------------------------------------------------------------------ #
    # Annotation slots
    # ------------------------------------------------------------------ #

    def _set_active_class(self, name: str) -> None:
        self._active_class = name

    def _on_polygon_finished(self, pts: list) -> None:
        if self._current_row < 0 or not pts:
            return
        class_names = self.dataset_model.get_class_names()
        if not class_names:
            return
        target = self._active_class if self._active_class in class_names else class_names[0]
        self.dataset_model.add_annotation(self._current_row, target, pts)
        self._refresh_overlays()

    def _refresh_overlays(self) -> None:
        """Rebuild canvas overlays from the current row's annotations. V4: QColor here."""
        annos = self.dataset_model.get_annotations(self._current_row)
        overlays = [
            (a["polygon"], QColor(*self.dataset_model.get_class_color(a["category_name"])))
            for a in annos
        ]
        self.canvas.set_overlays(overlays)
