from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QToolButton, QInputDialog, QMessageBox, QColorDialog,
)


class _ClassRow(QWidget):
    """One selectable row: color swatch | class name | per-image count | total count | trash.

    Clicking the swatch opens QColorDialog to change the class color.
    The trash button deletes the class immediately.
    Selection background uses QPalette.Highlight (OS-adaptive).

    Signals:
        row_clicked (str): class name when the row body is pressed.
        delete_requested (str): emitted when the trash button is clicked.
        color_changed (str, int, int, int): name + new (r, g, b) after swatch dialog.
    """

    row_clicked      = Signal(str)
    delete_requested = Signal(str)
    color_changed    = Signal(str, int, int, int)

    _COUNT_W = 40
    _SWATCH_W = 33

    def __init__(self, name: str, r: int, g: int, b: int,
                 per_image_count: int, total_count: int,
                 parent: QWidget = None) -> None:
        super().__init__(parent)
        self._name = name
        self._color = (r, g, b)
        self._original_palette = QPalette(self.palette())
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self._swatch = QToolButton()
        self._swatch.setFixedSize(self._SWATCH_W, self._SWATCH_W)
        self._swatch.setToolTip("Change color")
        self._swatch.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid gray; border-radius: 3px;"
        )
        self._swatch.clicked.connect(self._on_swatch_clicked)
        layout.addWidget(self._swatch)

        layout.addWidget(QLabel(name), stretch=1)

        self._image_lbl = QLabel(str(per_image_count))
        self._image_lbl.setFixedWidth(self._COUNT_W)
        self._image_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._image_lbl)

        self._total_lbl = QLabel(str(total_count))
        self._total_lbl.setFixedWidth(self._COUNT_W)
        self._total_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._total_lbl)

        trash = QToolButton()
        trash.setText("🗑")
        trash.setFixedSize(28, 28)
        trash.setToolTip(f'Delete "{name}"')
        trash.clicked.connect(lambda: self.delete_requested.emit(self._name))
        layout.addWidget(trash)

    def update_counts(self, per_image: int, total: int) -> None:
        self._image_lbl.setText(str(per_image))
        self._total_lbl.setText(str(total))

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
        self.row_clicked.emit(self._name)
        super().mousePressEvent(event)

    def _on_swatch_clicked(self) -> None:
        color = QColorDialog.getColor(QColor(*self._color), self, "Change Class Color")
        if not color.isValid():
            return
        r, g, b = color.red(), color.green(), color.blue()
        self._color = (r, g, b)
        self._swatch.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid gray; border-radius: 3px;"
        )
        self.color_changed.emit(self._name, r, g, b)


class ClassesSection(QWidget):
    """Directly-laid-out class rows with add/delete and per-class annotation counts.

    Signals:
        class_selected (str): Emitted when the user clicks a class row.
    """

    class_selected = Signal(str)

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._class_names: list = []
        self._rows: dict = {}
        self._selected_name: str = ""
        self._current_row: int = -1
        self._init_ui()
        self.dataset_model.modelReset.connect(self._rebuild_classes)
        self.dataset_model.dataChanged.connect(self._on_data_changed)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header_h = QHBoxLayout(header)
        header_h.setContentsMargins(6, 2, 6, 2)
        header_h.setSpacing(6)

        lbl_color = QLabel("Color")
        lbl_color.setFixedWidth(_ClassRow._SWATCH_W)
        lbl_color.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_color)

        lbl_class = QLabel("Class")
        lbl_class.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_class, stretch=0)

        lbl_image = QLabel("Image")
        lbl_image.setFixedWidth(_ClassRow._COUNT_W)
        lbl_image.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_image.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_image)

        lbl_total = QLabel("Total")
        lbl_total.setFixedWidth(_ClassRow._COUNT_W)
        lbl_total.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_total.setStyleSheet("font-size: 12px; font-weight: bold;")
        header_h.addWidget(lbl_total)

        header_h.addSpacing(28 + header_h.spacing())

        layout.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(1)
        layout.addLayout(self._rows_layout)

        btn_add = QPushButton("+ Add Class")
        btn_add.clicked.connect(self._add_class)
        layout.addWidget(btn_add)

    def set_current_row(self, row: int) -> None:
        self._current_row = row
        self._update_counts()

    def _rebuild_classes(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        self._class_names = self.dataset_model.get_class_names()

        for name in self._class_names:
            r, g, b = self.dataset_model.get_class_color(name)
            per_image = self._count_image_annotations(name, self._current_row)
            total = self._count_annotations(name)
            row = _ClassRow(name, r, g, b, per_image, total, self)
            row.row_clicked.connect(self._on_row_clicked)
            row.delete_requested.connect(self._on_delete_requested)
            row.color_changed.connect(self._on_color_changed)
            if name == self._selected_name:
                row.set_selected(True)
            self._rows[name] = row
            self._rows_layout.addWidget(row)

    def _update_counts(self) -> None:
        for name, row in self._rows.items():
            per_image = self._count_image_annotations(name, self._current_row)
            total = self._count_annotations(name)
            row.update_counts(per_image, total)

    def _count_image_annotations(self, class_name: str, image_row: int) -> int:
        if image_row < 0:
            return 0
        return sum(
            1 for a in self.dataset_model.get_annotations(image_row)
            if a["category_name"] == class_name
        )

    def _count_annotations(self, class_name: str) -> int:
        total = 0
        for row in range(self.dataset_model.rowCount()):
            for anno in self.dataset_model.get_annotations(row):
                if anno["category_name"] == class_name:
                    total += 1
        return total

    def _on_data_changed(self, top_left, bottom_right, roles=None) -> None:
        if self.dataset_model.get_class_names() != self._class_names:
            self._rebuild_classes()
        else:
            self._update_counts()

    def _on_row_clicked(self, name: str) -> None:
        if self._selected_name in self._rows:
            self._rows[self._selected_name].set_selected(False)
        self._selected_name = name
        if name in self._rows:
            self._rows[name].set_selected(True)
        self.class_selected.emit(name)

    def _on_color_changed(self, name: str, r: int, g: int, b: int) -> None:
        self.dataset_model.set_class_color(name, (r, g, b))

    def _on_delete_requested(self, name: str) -> None:
        self.dataset_model.delete_class(name)
        if self._selected_name == name:
            self._selected_name = ""
        self._rebuild_classes()

    def _add_class(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self.dataset_model.get_class_names():
            QMessageBox.warning(self, "Add Class", f'"{name}" already exists.')
            return
        color = QColorDialog.getColor(QColor(Qt.white), self, "Choose Class Color")
        if not color.isValid():
            return
        self.dataset_model.add_class(name, (color.red(), color.green(), color.blue()))
        self._rebuild_classes()
        self._on_row_clicked(name)
