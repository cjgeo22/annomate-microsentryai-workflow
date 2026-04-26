"""
TopBar — two-strip top bar for the WIP tab.

Strip A: flat menu buttons (File, Edit, …). File opens a real menu with
         "Open Folder"; all others show a "Coming soon" stub.
Strip B: navigation controls on the left + a QStackedWidget on the right
         that shows tool-specific hints/options for the active tool.

No custom colors — system palette only.
V1: dialogs live in WIPWindow, not here. TopBar emits signals; the window
    calls QFileDialog.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QMenu, QSizePolicy,
)


class TopBar(QWidget):
    """Fixed-height top bar with menu strip and navigation/tool-options strip.

    Signals:
        open_folder_requested (): User chose "Open Folder" (File menu or button).
        prev_requested ():        "< Prev" button clicked.
        next_requested ():        "Next >" button clicked.
    """

    open_folder_requested = Signal()
    prev_requested        = Signal()
    next_requested        = Signal()

    # Maps tool name → QStackedWidget page index
    _TOOL_PAGE: dict = {"polygon": 1}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(54)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_strip_a())

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        root.addWidget(self._build_strip_b())

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _build_strip_a(self) -> QWidget:
        strip = QWidget()
        strip.setFixedHeight(24)
        h = QHBoxLayout(strip)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(0)

        for label in ("File", "Edit", "Image", "Layer", "Type", "Select", "View", "Window", "Help"):
            btn = QPushButton(label)
            btn.setFlat(True)
            btn.setContentsMargins(8, 0, 8, 0)
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            if label == "File":
                btn.clicked.connect(self._show_file_menu)
            else:
                btn.clicked.connect(self._show_stub_menu)
            h.addWidget(btn)

        h.addStretch()
        return strip

    def _build_strip_b(self) -> QWidget:
        strip = QWidget()
        strip.setFixedHeight(29)
        h = QHBoxLayout(strip)
        h.setContentsMargins(8, 2, 8, 2)
        h.setSpacing(6)

        btn_open = QPushButton("Open Folder")
        btn_open.clicked.connect(self.open_folder_requested)
        h.addWidget(btn_open)

        h.addSpacing(12)

        self._btn_prev = QPushButton("< Prev")
        self._btn_prev.clicked.connect(self.prev_requested)
        h.addWidget(self._btn_prev)

        self._btn_next = QPushButton("Next >")
        self._btn_next.clicked.connect(self.next_requested)
        h.addWidget(self._btn_next)

        self._lbl_image = QLabel("")
        h.addWidget(self._lbl_image)

        h.addSpacing(16)

        # Tool-context stack — swaps content based on active tool
        self._tool_stack = QStackedWidget()

        self._tool_stack.addWidget(QLabel(""))                              # 0: default
        poly = QLabel("Click to place points · Double-click or Enter to close · Esc to cancel")
        poly.setEnabled(False)
        self._tool_stack.addWidget(poly)                                    # 1: polygon

        self._tool_stack.setCurrentIndex(0)
        h.addWidget(self._tool_stack, stretch=1)

        return strip

    # ------------------------------------------------------------------ #
    # Menu handlers
    # ------------------------------------------------------------------ #

    def _show_file_menu(self) -> None:
        btn = self.sender()
        menu = QMenu(self)
        menu.addAction("Open Folder…", self.open_folder_requested)
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _show_stub_menu(self) -> None:
        btn = self.sender()
        menu = QMenu(self)
        menu.addAction("Coming soon")
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_active_tool(self, name: str) -> None:
        """Switch the Strip B tool-hints page for *name*."""
        self._tool_stack.setCurrentIndex(self._TOOL_PAGE.get(name, 0))

    def set_image_label(self, text: str) -> None:
        """Update the filename / counter label in Strip B."""
        self._lbl_image.setText(text)
