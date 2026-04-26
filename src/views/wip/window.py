"""
WIPWindow — experimental Photoshop-style layout tab.

Phases:
  Phase 1  Structural skeleton: three fixed zones + three workspace columns.
  Phase 2+ Canvas, panels, tool palette, and status bar wired up incrementally.

Rules (consistent with other views):
  V1  All file/folder dialogs live here, not in controllers.
  V2  State is never accessed directly; all data reads go through model query APIs.
  V3  No disk I/O here; controllers handle all file operations.
  V4  Colors are (r, g, b) tuples until the last Qt draw boundary.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
)


def _placeholder(text: str, bg: str, fg: str = "#888888") -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"background-color: {bg}; color: {fg}; font-size: 11px;"
    )
    return lbl


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
        self._init_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Zone 1: Top Bar ─────────────────────────────────────────────
        self._top_bar = self._build_top_bar()
        root.addWidget(self._top_bar)

        # ── Zone 2: Main Workspace ──────────────────────────────────────
        self._workspace = self._build_workspace()
        root.addWidget(self._workspace, stretch=1)

        # ── Zone 3: Bottom Status Bar ───────────────────────────────────
        self._status_bar = self._build_status_bar()
        root.addWidget(self._status_bar)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(54)
        bar.setStyleSheet("background-color: #2b2b2b;")

        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Strip A — menu labels (24 px)
        strip_a = QWidget()
        strip_a.setFixedHeight(24)
        strip_a.setStyleSheet("background-color: #323232; border-bottom: 1px solid #1a1a1a;")
        a_layout = QHBoxLayout(strip_a)
        a_layout.setContentsMargins(4, 0, 4, 0)
        a_layout.setSpacing(0)
        for label in ("File", "Edit", "Image", "Layer", "Type", "Select", "View", "Window", "Help"):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #cccccc; font-size: 12px; padding: 0 8px;")
            a_layout.addWidget(lbl)
        a_layout.addStretch()
        layout.addWidget(strip_a)

        # Strip B — tool options placeholder (30 px)
        strip_b = QWidget()
        strip_b.setFixedHeight(30)
        strip_b.setStyleSheet("background-color: #2b2b2b;")
        b_layout = QHBoxLayout(strip_b)
        b_layout.setContentsMargins(8, 0, 8, 0)
        b_layout.addWidget(_placeholder("Tool Options Bar", "#2b2b2b", "#555555"))
        layout.addWidget(strip_b)

        return bar

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        h_layout = QHBoxLayout(workspace)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # Left tool palette (52 px fixed)
        self._tool_palette = QWidget()
        self._tool_palette.setFixedWidth(52)
        self._tool_palette.setStyleSheet("background-color: #3c3c3c; border-right: 1px solid #1a1a1a;")
        tp_layout = QVBoxLayout(self._tool_palette)
        tp_layout.setContentsMargins(0, 0, 0, 0)
        tp_layout.addWidget(_placeholder("Tools", "#3c3c3c", "#666666"))
        h_layout.addWidget(self._tool_palette)

        # Center artboard container (fluid)
        self._canvas_container = QWidget()
        self._canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas_container.setStyleSheet("background-color: #1e1e1e;")
        cc_layout = QVBoxLayout(self._canvas_container)
        cc_layout.setContentsMargins(0, 0, 0, 0)
        cc_layout.addWidget(_placeholder("Canvas / Artboard Area", "#1e1e1e", "#444444"))
        h_layout.addWidget(self._canvas_container, stretch=1)

        # Right panel (280 px fixed)
        self._right_panel = QWidget()
        self._right_panel.setFixedWidth(280)
        self._right_panel.setStyleSheet("background-color: #252526; border-left: 1px solid #1a1a1a;")
        rp_layout = QVBoxLayout(self._right_panel)
        rp_layout.setContentsMargins(0, 0, 0, 0)
        rp_layout.addWidget(_placeholder("Right Panels", "#252526", "#555555"))
        h_layout.addWidget(self._right_panel)

        return workspace

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(24)
        bar.setStyleSheet("background-color: #1e1e1e; border-top: 1px solid #1a1a1a;")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(16)

        lbl_left = QLabel("Zoom: 100%  |  — × — px  |  Tool: None")
        lbl_left.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(lbl_left)

        layout.addStretch()

        lbl_right = QLabel("Ready")
        lbl_right.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(lbl_right)

        return bar
