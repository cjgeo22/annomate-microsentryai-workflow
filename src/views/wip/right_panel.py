"""
Right panel for the WIP tab.

Phase 3: _CollapsibleSection scaffold + DataNavigatorSection.
Phase 4: ClassesSection added inside a second collapsible.

Rules: no explicit theme colors on container widgets. Dot indicators use
fixed semantic colors (orange = in review, green = reviewed) — these are
data-state indicators, not theme colors.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QSizePolicy,
)


_COLOR_REVIEWED  = "#4caf50"
_COLOR_IN_REVIEW = "#ff9800"


def _dot(color: str) -> QLabel:
    lbl = QLabel()
    lbl.setFixedSize(10, 10)
    lbl.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
    return lbl


# ======================================================================= #
# Collapsible section scaffold
# ======================================================================= #

class _CollapsibleSection(QWidget):
    """Toggle-header + collapsible body. No custom colors — uses system palette.

    Args:
        title: Label shown in the header button.
        parent: Optional Qt parent.
    """

    def __init__(self, title: str, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._toggle_btn = QPushButton()
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(True)
        self._toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle_btn.setStyleSheet("text-align: left; padding: 4px 6px;")
        self._toggle_btn.clicked.connect(self._on_toggle)
        self._refresh_label()
        root.addWidget(self._toggle_btn)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 4, 4, 4)
        self._body_layout.setSpacing(4)
        root.addWidget(self._body)

    def body_layout(self) -> QVBoxLayout:
        """Returns the layout inside the collapsible body."""
        return self._body_layout

    def _refresh_label(self) -> None:
        arrow = "▾" if self._expanded else "▸"   # ▾ / ▸
        self._toggle_btn.setText(f"{arrow}  {self._title}")

    def _on_toggle(self, checked: bool) -> None:
        self._expanded = checked
        self._body.setVisible(checked)
        self._refresh_label()


# ======================================================================= #
# Dataset Navigator content
# ======================================================================= #

class DataNavigatorSection(QWidget):
    """Legend + bounded image list. Lives inside a _CollapsibleSection body.

    Args:
        dataset_model: DatasetTableModel instance.
        parent: Optional Qt parent widget.

    Signals:
        image_selected (int): Emitted when the user clicks a row.
    """

    image_selected = Signal(int)

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._dot_labels: list = []
        self._init_ui()
        self.dataset_model.modelReset.connect(self._rebuild_list)
        self.dataset_model.dataChanged.connect(self._on_data_changed)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(self._build_legend())

        self.list_widget = QListWidget()
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Bounded height — list scrolls internally; doesn't push other sections away
        self.list_widget.setMinimumHeight(80)
        self.list_widget.setMaximumHeight(320)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.list_widget)

    def _build_legend(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for color, text in ((_COLOR_IN_REVIEW, "In Review"), (_COLOR_REVIEWED, "Reviewed")):
            layout.addWidget(_dot(color))
            layout.addWidget(QLabel(text))
            layout.addSpacing(6)
        layout.addStretch()
        return widget

    def _rebuild_list(self) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self._dot_labels.clear()

        for row in range(self.dataset_model.rowCount()):
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 24))
            self.list_widget.addItem(item)

            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(4, 0, 4, 0)
            h.setSpacing(6)

            dot = _dot(_COLOR_REVIEWED if self.dataset_model.is_reviewed(row) else _COLOR_IN_REVIEW)
            self._dot_labels.append(dot)
            h.addWidget(dot)
            h.addWidget(QLabel(Path(self.dataset_model.get_image_filename(row)).stem))
            h.addStretch()

            self.list_widget.setItemWidget(item, container)

        self.list_widget.blockSignals(False)

    def _on_data_changed(self, top_left, bottom_right, roles=None) -> None:
        for row in range(top_left.row(), bottom_right.row() + 1):
            if row < len(self._dot_labels):
                color = _COLOR_REVIEWED if self.dataset_model.is_reviewed(row) else _COLOR_IN_REVIEW
                self._dot_labels[row].setStyleSheet(
                    f"background-color: {color}; border-radius: 5px;"
                )

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.image_selected.emit(row)

    def select_row(self, row: int) -> None:
        """Silently highlight *row* without emitting image_selected."""
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(row)
        self.list_widget.blockSignals(False)


# ======================================================================= #
# Right panel shell
# ======================================================================= #

class RightPanel(QWidget):
    """Scrollable right panel containing collapsible sections.

    Width is controlled by the parent QSplitter, not fixed here.
    Phase 4 will insert a ClassesSection collapsible above the navigator.

    Args:
        dataset_model: DatasetTableModel instance.
        parent: Optional Qt parent widget.

    Signals:
        image_selected (int): Forwarded from DataNavigatorSection.
    """

    image_selected = Signal(int)

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        nav_sec = _CollapsibleSection("Dataset Navigator")
        self.navigator = DataNavigatorSection(dataset_model)
        self.navigator.image_selected.connect(self.image_selected)
        nav_sec.body_layout().addWidget(self.navigator)
        cl.addWidget(nav_sec)

        cl.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def select_row(self, row: int) -> None:
        """Silently highlight *row* in the navigator list."""
        self.navigator.select_row(row)
