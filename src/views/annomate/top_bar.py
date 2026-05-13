"""
TopBar — collapsible horizontal context bar above the main canvas.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QToolButton,
)


class TopBar(QWidget):
    """A collapsible top bar for tool options and review actions.

    Currently acts as a blank template container.

    Signals:
        toggled (bool): Emitted when the bar expands (True) or collapses (False).
    """

    toggled = Signal(bool)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Main Bar ---
        self._bar = QFrame()
        self._bar.setFrameShape(QFrame.StyledPanel)
        self._bar.setStyleSheet(
            "QFrame { border-bottom: 1px solid palette(mid); background: palette(window); }"
        )
        self._bar.setMinimumHeight(44)

        self._layout = QHBoxLayout(self._bar)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(8)

        # Blank content area for future widgets
        self._content = QWidget()
        self._layout.addWidget(self._content, stretch=1)

        # --- Collapse Button (Inside the bar) ---
        self._btn_collapse = QToolButton()
        self._btn_collapse.setText("▴")
        self._btn_collapse.setToolTip("Collapse Top Bar")
        self._btn_collapse.setFixedSize(32, 24)
        self._btn_collapse.setStyleSheet(
            "QToolButton { "
            "background: palette(button); "
            "border: 1px solid palette(mid); "
            "border-radius: 3px; "
            "font-size: 14px; "
            "}"
            "QToolButton:hover { background: palette(light); }"
        )
        self._btn_collapse.clicked.connect(lambda: self.set_expanded(False))
        self._layout.addWidget(self._btn_collapse)

        root.addWidget(self._bar)

        # --- Expand Handle (Visible only when collapsed) ---
        self._handle_container = QWidget()
        hc_layout = QHBoxLayout(self._handle_container)
        hc_layout.setContentsMargins(0, 0, 16, 0)  # Offset slightly from the right edge
        hc_layout.setSpacing(0)
        hc_layout.setAlignment(Qt.AlignRight)

        self._btn_expand = QToolButton()
        self._btn_expand.setText("▾")
        self._btn_expand.setToolTip("Expand Top Bar")
        self._btn_expand.setFixedSize(40, 18)
        self._btn_expand.setStyleSheet(
            "QToolButton { "
            "background: palette(button); "
            "border: 1px solid palette(mid); "
            "border-top: none; "
            "border-bottom-left-radius: 4px; "
            "border-bottom-right-radius: 4px; "
            "font-size: 12px; "
            "}"
            "QToolButton:hover { background: palette(light); }"
        )
        self._btn_expand.clicked.connect(lambda: self.set_expanded(True))

        hc_layout.addWidget(self._btn_expand)
        self._handle_container.setVisible(False)

        root.addWidget(self._handle_container)

        self._expanded = True

    def set_expanded(self, expanded: bool) -> None:
        """Expand or collapse the top bar programmatically."""
        if self._expanded == expanded:
            return
        self._expanded = expanded
        self._bar.setVisible(expanded)
        self._handle_container.setVisible(not expanded)
        self.toggled.emit(expanded)

    def is_expanded(self) -> bool:
        return self._expanded
