from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea

from views.wip.sections import (
    _CollapsibleSection,
    DataNavigatorSection,
    ClassesSection,
    AnnotationsSection,
)


class RightPanel(QWidget):
    """Scrollable right panel with collapsible sections for the WIP tab.

    Signals:
        image_selected (int): Forwarded from DataNavigatorSection.
        class_selected (str): Forwarded from ClassesSection.
        prev_requested (): Forwarded from DataNavigatorSection.
        next_requested (): Forwarded from DataNavigatorSection.
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

        nav_sec = _CollapsibleSection("Dataset Navigator")
        self.navigator = DataNavigatorSection(dataset_model)
        self.navigator.image_selected.connect(self.image_selected)
        self.navigator.prev_requested.connect(self.prev_requested)
        self.navigator.next_requested.connect(self.next_requested)
        nav_sec.body_layout().addWidget(self.navigator)
        cl.addWidget(nav_sec)
        
        classes_sec = _CollapsibleSection("Annotation Classes")
        self.classes = ClassesSection(dataset_model)
        self.classes.class_selected.connect(self.class_selected)
        classes_sec.body_layout().addWidget(self.classes)
        cl.addWidget(classes_sec)

        annos_sec = _CollapsibleSection("Current Image Annotations")
        self.annotations = AnnotationsSection(dataset_model)
        annos_sec.body_layout().addWidget(self.annotations)
        cl.addWidget(annos_sec)


        cl.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    def select_row(self, row: int) -> None:
        """Silently highlight *row* in the navigator list."""
        self.navigator.select_row(row)

    def set_counter(self, current: int, total: int) -> None:
        """Update the image position counter in the navigator."""
        self.navigator.set_counter(current, total)

    def set_current_row(self, row: int) -> None:
        """Update per-image counts and the annotations list for the new image."""
        self.classes.set_current_row(row)
        self.annotations.set_current_row(row)
