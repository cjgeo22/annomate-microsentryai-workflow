from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QToolButton,
)

_DOT_W   = 22
_COUNT_W = 52
_TRASH_W = 28


class _AnnotationRow(QWidget):
    """One annotation row: color dot | class name | vertex count | trash.

    Clicking anywhere on the row (except the trash button) emits row_clicked.
    """

    row_clicked      = Signal(int)   # annotation index
    delete_requested = Signal(int)   # annotation index

    def __init__(self, idx: int, name: str, verts: int,
                 r: int, g: int, b: int,
                 parent: QWidget = None) -> None:
        super().__init__(parent)
        self._idx = idx
        self._original_palette = QPalette(self.palette())
        self.setCursor(Qt.PointingHandCursor)

        h = QHBoxLayout(self)
        h.setContentsMargins(6, 3, 6, 3)
        h.setSpacing(8)

        dot = QLabel()
        dot.setFixedSize(_DOT_W, _DOT_W)
        dot.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border-radius: {_DOT_W // 2}px;"
        )
        h.addWidget(dot)

        h.addWidget(QLabel(name), stretch=1)

        lbl_v = QLabel(str(verts))
        lbl_v.setFixedWidth(_COUNT_W)
        lbl_v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(lbl_v)

        trash = QToolButton()
        trash.setText("🗑")
        trash.setFixedSize(_TRASH_W, _TRASH_W)
        trash.setToolTip("Delete annotation")
        trash.clicked.connect(lambda: self.delete_requested.emit(self._idx))
        h.addWidget(trash)

    def set_selected(self, selected: bool) -> None:
        if selected:
            p = QPalette(self.palette())
            p.setColor(QPalette.Window, self.palette().color(QPalette.Highlight))
            p.setColor(QPalette.WindowText, self.palette().color(QPalette.HighlightedText))
            self.setAutoFillBackground(True)
            self.setPalette(p)
        else:
            self.setAutoFillBackground(False)
            self.setPalette(self._original_palette)

    def mousePressEvent(self, event) -> None:
        self.row_clicked.emit(self._idx)
        super().mousePressEvent(event)


class AnnotationsSection(QWidget):
    """Flat list of annotations for the currently displayed image.

    Columns: color dot | class name | vertex count | trash.
    Clicking a row emits annotation_selected(idx) for canvas highlighting.
    Rebuilt whenever the current row changes or dataChanged fires.

    Signals:
        annotation_selected (int): annotation index within the current image.
    """

    annotation_selected = Signal(int)

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._current_row: int = -1
        self._selected_idx: int = -1
        self._row_widgets: list = []
        self._init_ui()
        self.dataset_model.dataChanged.connect(self._rebuild)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header_h = QHBoxLayout(header)
        header_h.setContentsMargins(6, 2, 6, 2)
        header_h.setSpacing(8)

        lbl_color = QLabel("Color")
        lbl_color.setFixedWidth(_DOT_W)
        lbl_color.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_color)

        lbl_class = QLabel("Class")
        lbl_class.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_class, stretch=1)

        lbl_verts = QLabel("Vertices")
        lbl_verts.setFixedWidth(_COUNT_W)
        lbl_verts.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_verts.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_verts)

        header_h.addSpacing(_TRASH_W + header_h.spacing())

        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(1)
        layout.addLayout(self._rows_layout)

        self._empty_lbl = QLabel("No annotations")
        self._empty_lbl.setStyleSheet("color: gray; font-size: 11px;")
        self._empty_lbl.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._empty_lbl)

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self._selected_idx = -1
        self._rebuild()

    def _rebuild(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._row_widgets.clear()

        annos = self.dataset_model.get_annotations(self._current_row) if self._current_row >= 0 else []
        self._empty_lbl.setVisible(not annos)

        # Clamp selection in case annotations were deleted
        if self._selected_idx >= len(annos):
            self._selected_idx = -1

        for idx, anno in enumerate(annos):
            name = anno["category_name"]
            verts = len(anno["polygon"])
            r, g, b = self.dataset_model.get_class_color(name)

            row_w = _AnnotationRow(idx, name, verts, r, g, b, self)
            row_w.row_clicked.connect(self._on_row_clicked)
            row_w.delete_requested.connect(self._on_delete_requested)
            if idx == self._selected_idx:
                row_w.set_selected(True)
            self._row_widgets.append(row_w)
            self._rows_layout.addWidget(row_w)

    def _on_row_clicked(self, idx: int) -> None:
        if self._selected_idx >= 0 and self._selected_idx < len(self._row_widgets):
            self._row_widgets[self._selected_idx].set_selected(False)
        self._selected_idx = idx
        if idx < len(self._row_widgets):
            self._row_widgets[idx].set_selected(True)
        self.annotation_selected.emit(idx)

    def _on_delete_requested(self, idx: int) -> None:
        if self._selected_idx == idx:
            self._selected_idx = -1
        self.dataset_model.delete_annotation(self._current_row, idx)
