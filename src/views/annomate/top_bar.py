"""
TopBar — collapsible horizontal context bar above the main canvas.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QToolButton,
    QStackedWidget,
    QLabel,
    QSlider,
    QButtonGroup,
    QSizePolicy,
)


class TopBar(QWidget):
    """A collapsible top bar for tool options and review actions.
    
    Signals:
        toggled (bool): Emitted when the bar expands (True) or collapses (False).
        thickness_changed (float): Emitted when the brush thickness slider moves.
        draw_mode_changed (str): Emitted when the user changes the drawing shape.
    """

    toggled = Signal(bool)
    thickness_changed = Signal(float)
    draw_mode_changed = Signal(str)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        
        # Prevent the bar from expanding vertically to take up canvas space
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        
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
        
        # --- Stacked Content Area ---
        self._stack = QStackedWidget()
        
        # Page 0: Default (Empty)
        self._page_empty = QWidget()
        self._stack.addWidget(self._page_empty)
        
        # Page 1: Polygon Tool Settings
        self._build_polygon_page()
        
        # Page 2: Edit Existing Polygon (Thickness only)
        self._build_edit_page()
        
        # Page 3: SAM Tool Settings (Blank for now)
        self._page_sam = QWidget()
        self._stack.addWidget(self._page_sam)
        
        self._layout.addWidget(self._stack, stretch=1)
        
        # --- Collapse Button ---
        self._btn_collapse = QToolButton()
        self._btn_collapse.setText("▴")
        self._btn_collapse.setToolTip("Collapse Top Bar")
        self._btn_collapse.setFixedSize(32, 24)
        self._btn_collapse.setStyleSheet(
            "QToolButton { background: palette(button); border: 1px solid palette(mid); border-radius: 3px; font-size: 14px; }"
            "QToolButton:hover { background: palette(light); }"
        )
        self._btn_collapse.clicked.connect(lambda: self.set_expanded(False))
        self._layout.addWidget(self._btn_collapse)
        
        root.addWidget(self._bar)
        
        # --- Expand Handle (Visible only when collapsed) ---
        self._handle_container = QWidget()
        hc_layout = QHBoxLayout(self._handle_container)
        hc_layout.setContentsMargins(0, 0, 16, 0)
        hc_layout.setSpacing(0)
        hc_layout.setAlignment(Qt.AlignRight)
        
        self._btn_expand = QToolButton()
        self._btn_expand.setText("▾")
        self._btn_expand.setToolTip("Expand Top Bar")
        self._btn_expand.setFixedSize(40, 18)
        self._btn_expand.setStyleSheet(
            "QToolButton { background: palette(button); border: 1px solid palette(mid); border-top: none; border-bottom-left-radius: 4px; border-bottom-right-radius: 4px; font-size: 12px; }"
            "QToolButton:hover { background: palette(light); }"
        )
        self._btn_expand.clicked.connect(lambda: self.set_expanded(True))
        
        hc_layout.addWidget(self._btn_expand)
        self._handle_container.setVisible(False)
        
        root.addWidget(self._handle_container)
        
        self._expanded = True
        self.set_expanded(False)

    def _build_polygon_page(self) -> None:
        self._page_polygon = QWidget()
        layout = QHBoxLayout(self._page_polygon)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # 1. Draw Mode Buttons (No Label)
        mode_container = QWidget()
        mode_layout = QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(2)
        
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self._mode_map = {}
        
        font = QFont()
        font.setPointSize(14)

        def add_mode(icon: str, tooltip: str, mode_id: str, checked: bool = False):
            btn = QToolButton()
            btn.setText(icon)
            btn.setToolTip(tooltip)
            btn.setFont(font)
            btn.setFixedSize(32, 28)
            btn.setCheckable(True)
            btn.setChecked(checked)
            self.mode_group.addButton(btn)
            self._mode_map[btn] = mode_id
            mode_layout.addWidget(btn)

        add_mode("⬠", "Point-based Polygon", "point", checked=True)
        add_mode("◯", "Circle Shape", "circle")
        add_mode("▭", "Rectangle Shape", "rectangle")
        add_mode("✎", "Freebrush", "brush")
        
        self.mode_group.buttonClicked.connect(self._on_mode_clicked)
        layout.addWidget(mode_container)
        
        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setFrameShadow(QFrame.Sunken)
        layout.addWidget(div)
        
        # 2. Thickness Slider (No Label)
        self.slider_thickness = QSlider(Qt.Horizontal)
        self.slider_thickness.setRange(1, 40)
        self.slider_thickness.setValue(8)  # 8 × 0.25 = 2.00 px
        self.slider_thickness.setFixedWidth(150)
        self.slider_thickness.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_thickness.setTickInterval(4)
        self.slider_thickness.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider_thickness)
        
        self.lbl_thickness = QLabel("2.00 px")
        self.lbl_thickness.setFixedWidth(55)
        layout.addWidget(self.lbl_thickness)
        
        layout.addStretch()
        self._stack.addWidget(self._page_polygon)

    def _build_edit_page(self) -> None:
        self._page_edit = QWidget()
        layout = QHBoxLayout(self._page_edit)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # Edit Thickness Slider (No Label)
        self.slider_thickness_edit = QSlider(Qt.Horizontal)
        self.slider_thickness_edit.setRange(1, 40)
        self.slider_thickness_edit.setValue(8)
        self.slider_thickness_edit.setFixedWidth(150)
        self.slider_thickness_edit.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_thickness_edit.setTickInterval(4)
        self.slider_thickness_edit.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider_thickness_edit)
        
        self.lbl_thickness_edit = QLabel("2.00 px")
        self.lbl_thickness_edit.setFixedWidth(55)
        layout.addWidget(self.lbl_thickness_edit)
        
        layout.addStretch()
        self._stack.addWidget(self._page_edit)

    def set_thickness(self, thickness: float) -> None:
        """Update all thickness sliders to match the given value."""
        val = int(thickness * 4)
        text = f"{thickness:.2f} px"
        
        # Update Polygon page slider
        self.slider_thickness.blockSignals(True)
        self.slider_thickness.setValue(val)
        self.lbl_thickness.setText(text)
        self.slider_thickness.blockSignals(False)
        
        # Update Edit page slider
        self.slider_thickness_edit.blockSignals(True)
        self.slider_thickness_edit.setValue(val)
        self.lbl_thickness_edit.setText(text)
        self.slider_thickness_edit.blockSignals(False)

    def set_context(self, tool_name: str) -> None:
        """Switch the top bar to show settings for the active tool."""
        if tool_name == "polygon":
            self._stack.setCurrentWidget(self._page_polygon)
        elif tool_name == "sam_bbox":
            self._stack.setCurrentWidget(self._page_sam)
        elif tool_name == "edit_polygon":
            self._stack.setCurrentWidget(self._page_edit)
        else:
            self._stack.setCurrentWidget(self._page_empty)

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

    def _on_mode_clicked(self, btn) -> None:
        self.draw_mode_changed.emit(self._mode_map[btn])

    def _on_slider_changed(self, value: int) -> None:
        thickness = value * 0.25
        self.set_thickness(thickness)  # Sync both sliders and labels visually
        self.thickness_changed.emit(thickness)