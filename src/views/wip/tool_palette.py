"""
ToolPalette — left tool column for the WIP tab.

Single-column panel with checkable QToolButtons aligned to the top.
Only the polygon tool is active; additional tools can be appended to _TOOLS
in later phases without touching window.py.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QToolButton, QButtonGroup, QSizePolicy,
)

_TOOLS = [
    ("⬠", "polygon", "Polygon"),
]


class ToolPalette(QFrame):
    """Single-column tool panel.

    Uses QFrame.StyledPanel for a visual border consistent with the rest of
    the workspace. Buttons use the system palette — no custom colors.

    Signals:
        tool_selected (str): Emitted with the tool name when a button is pressed.
    """

    tool_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setFixedWidth(46)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._btn_tool: dict = {}
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignTop)

        font = QFont()
        font.setPointSize(13)

        for icon, tool_name, tooltip in _TOOLS:
            btn = QToolButton()
            btn.setText(icon)
            btn.setToolTip(tooltip)
            btn.setFixedSize(32, 30)
            btn.setCheckable(True)
            btn.setFont(font)
            self._btn_tool[btn] = tool_name
            self._btn_group.addButton(btn)
            layout.addWidget(btn)

        self._btn_group.buttonClicked.connect(self._on_btn_clicked)

    def _on_btn_clicked(self, btn: QToolButton) -> None:
        self.tool_selected.emit(self._btn_tool.get(btn, ""))

    def deselect_all(self) -> None:
        """Uncheck all tool buttons (called when the canvas cancels a tool)."""
        self._btn_group.setExclusive(False)
        for btn in self._btn_tool:
            btn.setChecked(False)
        self._btn_group.setExclusive(True)
