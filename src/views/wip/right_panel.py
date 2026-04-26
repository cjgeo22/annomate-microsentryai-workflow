"""
Right panel for the WIP tab.

Phase 3: _CollapsibleSection scaffold + DataNavigatorSection.
Phase 4: ClassesSection — direct row widgets, no inner list box.

Rules: no explicit theme colors on container widgets. Dot indicators use
fixed semantic colors (orange / green) — data-state indicators, not theme.
Class row selection uses QPalette.Highlight so it adapts to the OS theme.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QSizePolicy, QToolButton,
    QMenu, QInputDialog, QMessageBox, QColorDialog,
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
    """Bold toggle-header + separator + collapsible body.

    Styled after example_UI.py: bold header button, HLine separator,
    body with comfortable insets. No custom background colors.
    """

    def __init__(self, title: str, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self._title = title
        self._expanded = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._toggle_btn = QPushButton(f"▾  {title}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(True)
        self._toggle_btn.setStyleSheet(
            "text-align: left; font-weight: bold; padding: 5px 10px;"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)
        root.addWidget(self._toggle_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 6, 8, 8)
        self._body_layout.setSpacing(4)
        root.addWidget(self._body)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def _on_toggle(self, checked: bool) -> None:
        self._expanded = checked
        self._body.setVisible(checked)
        arrow = "▾" if checked else "▸"
        self._toggle_btn.setText(f"{arrow}  {self._title}")


# ======================================================================= #
# Dataset Navigator content
# ======================================================================= #

class DataNavigatorSection(QWidget):
    """Legend + bounded image list. Lives inside a _CollapsibleSection body.

    Signals:
        image_selected (int): Emitted when the user clicks a row.
    """

    image_selected = Signal(int)
    prev_requested = Signal()
    next_requested = Signal()

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

        # Nav row: Prev · Next · counter
        nav_row = QWidget()
        nav_h = QHBoxLayout(nav_row)
        nav_h.setContentsMargins(0, 0, 0, 0)
        nav_h.setSpacing(4)

        btn_prev = QToolButton()
        btn_prev.setText("‹")
        btn_prev.setToolTip("Previous image")
        btn_prev.clicked.connect(self.prev_requested)
        nav_h.addWidget(btn_prev)

        btn_next = QToolButton()
        btn_next.setText("›")
        btn_next.setToolTip("Next image")
        btn_next.clicked.connect(self.next_requested)
        nav_h.addWidget(btn_next)

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

        # Start hidden — shown once images are loaded
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

        total = self.dataset_model.rowCount()
        has_images = total > 0
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


# ======================================================================= #
# Annotation Classes — individual selectable rows
# ======================================================================= #

class _ClassRow(QWidget):
    """One selectable row: colored dot + class name + annotation count.

    Selection is highlighted using QPalette.Highlight (OS theme color).
    Right-click emits delete_requested so the parent can handle deletion.

    Signals:
        row_clicked (str): class name when the row is pressed.
        delete_requested (str): class name when "Delete" is chosen from context menu.
    """

    row_clicked      = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, name: str, r: int, g: int, b: int, count: int,
                 parent: QWidget = None) -> None:
        super().__init__(parent)
        self._name = name
        self._original_palette = QPalette(self.palette())
        self.setCursor(Qt.PointingHandCursor)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # Colored dot — V4: (r,g,b) → inline QSS at draw boundary
        dot = QLabel("●")
        dot.setStyleSheet(f"color: rgb({r},{g},{b}); font-size: 14px;")
        dot.setFixedWidth(18)
        layout.addWidget(dot)

        layout.addWidget(QLabel(name), stretch=1)

        self._count_lbl = QLabel(str(count))
        self._count_lbl.setStyleSheet("font-size: 10px;")
        layout.addWidget(self._count_lbl)

    def update_count(self, count: int) -> None:
        self._count_lbl.setText(str(count))

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

    def _on_context_menu(self, pos) -> None:
        menu = QMenu(self)
        action = menu.addAction(f'Delete "{self._name}"')
        if menu.exec(self.mapToGlobal(pos)) == action:
            self.delete_requested.emit(self._name)


class ClassesSection(QWidget):
    """Directly-laid-out class rows with add/delete and per-class annotation counts.

    No inner scroll box — rows are direct children of the body layout so the
    collapsible section header controls visibility. The outer RightPanel
    QScrollArea handles overflow when many classes are present.

    Signals:
        class_selected (str): Emitted when the user clicks a class row.
    """

    class_selected = Signal(str)

    def __init__(self, dataset_model, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.dataset_model = dataset_model
        self._class_names: list = []
        self._rows: dict = {}        # name → _ClassRow
        self._selected_name: str = ""
        self._init_ui()
        self.dataset_model.dataChanged.connect(self._on_data_changed)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(1)
        layout.addLayout(self._rows_layout)

        btn_add = QPushButton("+ Add Class")
        btn_add.clicked.connect(self._add_class)
        layout.addWidget(btn_add)

    def _rebuild_classes(self) -> None:
        # Remove existing row widgets
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        self._class_names = self.dataset_model.get_class_names()

        for name in self._class_names:
            r, g, b = self.dataset_model.get_class_color(name)
            count = self._count_annotations(name)
            row = _ClassRow(name, r, g, b, count, self)
            row.row_clicked.connect(self._on_row_clicked)
            row.delete_requested.connect(self._on_delete_requested)
            if name == self._selected_name:
                row.set_selected(True)
            self._rows[name] = row
            self._rows_layout.addWidget(row)

    def _update_counts(self) -> None:
        for name, row in self._rows.items():
            row.update_count(self._count_annotations(name))

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


# ======================================================================= #
# Right panel shell
# ======================================================================= #

class RightPanel(QWidget):
    """Scrollable right panel with collapsible Annotation Classes and Dataset Navigator.

    Width is controlled by the parent QSplitter.

    Signals:
        image_selected (int): Forwarded from DataNavigatorSection.
        class_selected (str): Forwarded from ClassesSection.
    """

    image_selected = Signal(int)
    class_selected = Signal(str)
    prev_requested = Signal()
    next_requested = Signal()

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

        classes_sec = _CollapsibleSection("Annotation Classes")
        self.classes = ClassesSection(dataset_model)
        self.classes.class_selected.connect(self.class_selected)
        classes_sec.body_layout().addWidget(self.classes)
        cl.addWidget(classes_sec)

        nav_sec = _CollapsibleSection("Dataset Navigator")
        self.navigator = DataNavigatorSection(dataset_model)
        self.navigator.image_selected.connect(self.image_selected)
        self.navigator.prev_requested.connect(self.prev_requested)
        self.navigator.next_requested.connect(self.next_requested)
        nav_sec.body_layout().addWidget(self.navigator)
        cl.addWidget(nav_sec)

        cl.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def select_row(self, row: int) -> None:
        """Silently highlight *row* in the navigator list."""
        self.navigator.select_row(row)

    def set_counter(self, current: int, total: int) -> None:
        """Update the image position counter in the navigator."""
        self.navigator.set_counter(current, total)
