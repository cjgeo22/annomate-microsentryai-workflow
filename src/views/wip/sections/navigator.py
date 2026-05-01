from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QToolButton,
)

from ._shared import _dot, _COLOR_REVIEWED, _COLOR_IN_REVIEW


class DataNavigatorSection(QWidget):
    """Legend + bounded image list. Lives inside a _CollapsibleSection body.

    Signals:
        image_selected (int): Emitted when the user clicks a row.
        prev_requested (): Emitted when Prev button is clicked.
        next_requested (): Emitted when Next button is clicked.
    """

    image_selected = Signal(int)
    prev_requested = Signal()
    next_requested = Signal()

    def __init__(self, dataset_model, inference_model=None, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self._dot_labels: list = []
        self._ai_labels: list = []
        self._init_ui()
        self.dataset_model.modelReset.connect(self._rebuild_list)
        self.dataset_model.dataChanged.connect(self._on_data_changed)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        nav_row = QWidget()
        nav_h = QHBoxLayout(nav_row)
        nav_h.setContentsMargins(0, 0, 0, 0)
        nav_h.setSpacing(4)

        self._btn_prev = QToolButton()
        self._btn_prev.setText("‹ Prev")
        self._btn_prev.setToolTip("Previous image")
        self._btn_prev.clicked.connect(self.prev_requested)
        self._btn_prev.setVisible(False)
        nav_h.addWidget(self._btn_prev)

        self._btn_next = QToolButton()
        self._btn_next.setText("Next ›")
        self._btn_next.setToolTip("Next image")
        self._btn_next.clicked.connect(self.next_requested)
        self._btn_next.setVisible(False)
        nav_h.addWidget(self._btn_next)

        self._lbl_counter = QLabel("No images loaded")
        self._lbl_counter.setStyleSheet("color: black;")
        nav_h.addWidget(self._lbl_counter)
        nav_h.addStretch()

        layout.addWidget(nav_row)

        self._legend = self._build_legend()
        layout.addWidget(self._legend)

        self.list_widget = QListWidget()
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setMinimumHeight(80)
        self.list_widget.setMaximumHeight(320)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.list_widget)

        self._legend.setVisible(False)
        self.list_widget.setVisible(False)

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
        self._ai_labels.clear()

        total = self.dataset_model.rowCount()
        has_images = total > 0
        self._btn_prev.setVisible(has_images)
        self._btn_next.setVisible(has_images)
        self._legend.setVisible(has_images)
        self.list_widget.setVisible(has_images)

        if not has_images:
            self._lbl_counter.setText("No images loaded")
            self.list_widget.blockSignals(False)
            return

        self._lbl_counter.setText(f"{total} image{'s' if total != 1 else ''} loaded")

        for row in range(total):
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

            processed = (
                self.inference_model is not None
                and self.inference_model.is_processed(self.dataset_model.get_image_path(row))
            )
            ai_dot = _dot("#2196f3")
            ai_dot.setToolTip("Processed by AI")
            ai_dot.setVisible(processed)
            self._ai_labels.append(ai_dot)
            h.addWidget(ai_dot)

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

    def set_counter(self, current: int, total: int) -> None:
        """Update the position counter label (e.g. '5 / 24')."""
        if total > 0:
            self._lbl_counter.setText(f"{current + 1} / {total}")

    def set_row_processed(self, row: int, processed: bool) -> None:
        """Show or hide the AI indicator dot for a specific row."""
        if row < len(self._ai_labels):
            self._ai_labels[row].setVisible(processed)

    def refresh_all_processed(self) -> None:
        """Update all AI indicators from inference_model."""
        if self.inference_model is None:
            return
        for row in range(self.dataset_model.rowCount()):
            path = self.dataset_model.get_image_path(row)
            self.set_row_processed(row, self.inference_model.is_processed(path))
