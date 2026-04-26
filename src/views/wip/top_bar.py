"""
TopBar — minimal top bar for the WIP tab.

Open Folder button on the left, tool-context hint on the right, HLine
separator at the bottom to visually separate it from the canvas.

V1: dialogs stay in WIPWindow; TopBar only emits open_folder_requested.
No custom background/border colors — system palette only.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QSizePolicy,
)


class TopBar(QWidget):
    """Single content row + bottom separator.

    Signals:
        open_folder_requested (): emitted when the user clicks Open Folder.
    """

    open_folder_requested = Signal()

    _TOOL_PAGE: dict = {"polygon": 1}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(38)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Content row
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(8)

        btn_open = QPushButton("Open Folder")
        btn_open.clicked.connect(self.open_folder_requested)
        h.addWidget(btn_open)

        h.addSpacing(8)

        self._tool_stack = QStackedWidget()
        self._tool_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._tool_stack.addWidget(QLabel(""))                              # 0: default

        poly_hint = QLabel(
            "Click to place points · Double-click or Enter to close · Esc to cancel"
        )
        poly_hint.setStyleSheet("color: black;")
        self._tool_stack.addWidget(poly_hint)                               # 1: polygon

        self._tool_stack.setCurrentIndex(0)
        h.addWidget(self._tool_stack, stretch=1)

        root.addWidget(row)

        # Bottom separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

    def set_active_tool(self, name: str) -> None:
        """Switch the tool-hint page for *name*."""
        self._tool_stack.setCurrentIndex(self._TOOL_PAGE.get(name, 0))
