from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
)


class AnnotationsSection(QWidget):
    """Flat list of annotations for the currently displayed image.

    Columns: color dot | class name | vertex count.
    Rebuilt whenever the current row changes or dataChanged fires.
    """

    _COUNT_W = 52

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._current_row: int = -1
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
        lbl_color.setFixedWidth(20)
        lbl_color.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_color)

        lbl_class = QLabel("Class")
        lbl_class.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_class, stretch=1)

        lbl_verts = QLabel("Vertices")
        lbl_verts.setFixedWidth(self._COUNT_W)
        lbl_verts.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_verts.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_verts)

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
        self._rebuild()

    def _rebuild(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        annos = self.dataset_model.get_annotations(self._current_row) if self._current_row >= 0 else []
        self._empty_lbl.setVisible(not annos)

        for anno in annos:
            name = anno["category_name"]
            verts = len(anno["polygon"])
            r, g, b = self.dataset_model.get_class_color(name)

            row_w = QWidget()
            h = QHBoxLayout(row_w)
            h.setContentsMargins(6, 3, 6, 3)
            h.setSpacing(8)

            dot = QLabel()
            dot.setFixedSize(14, 14)
            dot.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border-radius: 7px;"
            )
            h.addWidget(dot)

            h.addWidget(QLabel(name), stretch=1)

            lbl_v = QLabel(str(verts))
            lbl_v.setFixedWidth(self._COUNT_W)
            lbl_v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            h.addWidget(lbl_v)

            self._rows_layout.addWidget(row_w)
