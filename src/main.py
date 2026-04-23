import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QInputDialog,
    QFileDialog, QMessageBox,
)
from PySide6.QtGui import QAction, QKeySequence

from core.utils.logger import setup_logging
setup_logging()

from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from core.states.validation_state import ValidationState
from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from models.validation_model import ValidationModel

from controllers.io_controller import IOController
from controllers.inference_controller import InferenceController
from controllers.validation_controller import ValidationController

from views.annomate.window import ImageAnnotator
from views.microsentry.window import MicroSentryWindow
from views.validation.window import ValidationWindow

from core.project_io import ProjectIO
from core.autosave import AutosaveManager

_APP_TITLE = "AnnoMate & MicroSentryAI"


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_APP_TITLE)
        self.resize(1400, 900)

        # Domain state
        self.dataset_state    = DatasetState()
        self.inference_state  = InferenceState()
        self.validation_state = ValidationState()

        # Models
        self.dataset_model    = DatasetTableModel(self.dataset_state)
        self.inference_model  = InferenceModel(self.inference_state)
        self.validation_model = ValidationModel(self.validation_state)

        # Controllers
        self.io_controller         = IOController(self.dataset_model)
        self.inference_controller  = InferenceController(
            self.dataset_model, self.inference_model
        )
        self.validation_controller = ValidationController(self.validation_model)

        # Views
        self.annomate_view    = ImageAnnotator(self.dataset_model, self.io_controller)
        self.sentry_view      = MicroSentryWindow(
            self.dataset_model,
            self.inference_model,
            self.inference_controller,
            self.io_controller,
        )
        self.validation_view  = ValidationWindow(
            self.validation_model, self.validation_controller
        )

        # Cross-tab row sync via each view's public API
        self.annomate_view.row_selection_changed.connect(self.sentry_view.select_row)
        self.sentry_view.row_selection_changed.connect(self.annomate_view.select_row)

        # Polygon transfer: MicroSentry → AnnoMate
        self.sentry_view.polygonsSent.connect(self._handle_polygon_transfer)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.annomate_view,   "AnnoMate")
        self.tabs.addTab(self.sentry_view,     "MicroSentry AI")
        self.tabs.addTab(self.validation_view, "Validation")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)

        # Project state
        self._project_dir: Optional[str] = None
        self._project_name: str = ""
        self._is_dirty: bool = False
        self._created_at: Optional[str] = None

        # Project I/O helpers
        self.project_io = ProjectIO()
        self.autosave_manager = AutosaveManager(interval_minutes=5, parent=self)
        self.autosave_manager.save_requested.connect(self._do_autosave)

        # Dirty-state tracking
        self.dataset_model.dataChanged.connect(self._mark_dirty)
        self.dataset_model.modelReset.connect(self._mark_dirty)

        self._build_menu()

    # ================================================================== #
    # Menu bar
    # ================================================================== #

    def _build_menu(self) -> None:
        """Build the File menu with project and export/import actions."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        act_new = QAction("New Project", self)
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QAction("Open Project…", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        act_save = QAction("Save Project", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save Project As…", self)
        act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_export_coco = QAction("Export COCO JSON…", self)
        act_export_coco.triggered.connect(self._export_coco)
        file_menu.addAction(act_export_coco)

        act_import_coco = QAction("Import COCO JSON…", self)
        act_import_coco.triggered.connect(self._import_coco)
        file_menu.addAction(act_import_coco)

        file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

    # ================================================================== #
    # Project slots
    # ================================================================== #

    def _new_project(self) -> None:
        """Prompt to save if dirty, then clear all state and reset project tracking."""
        if self._is_dirty and not self._confirm_discard():
            return

        self.dataset_state.clear()
        self.inference_state.clear()
        self.validation_state.clear()
        self.dataset_model.beginResetModel()
        self.dataset_model.endResetModel()

        self._project_dir = None
        self._project_name = ""
        self._created_at = None
        self.autosave_manager.stop()
        self._clear_dirty()

    def _open_project(self) -> None:
        """Open a file dialog to select a .annoproj file, then load the project."""
        if self._is_dirty and not self._confirm_discard():
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", os.getcwd(), "AnnoMate Project (*.annoproj)"
        )
        if not path:
            return

        try:
            project_data = self.project_io.load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Project", f"Could not read project file:\n{exc}")
            return

        image_dir = project_data.get("dataset", {}).get("image_dir", "")
        if image_dir and os.path.isdir(image_dir):
            try:
                self.io_controller.load_folder(image_dir)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Open Project",
                    f"Image directory not accessible:\n{image_dir}\n\n{exc}\n\n"
                    "Annotations will be loaded but images may not display."
                )
        elif image_dir:
            QMessageBox.warning(
                self, "Open Project",
                f"Image directory not found:\n{image_dir}\n\n"
                "Annotations will be loaded but images may not display."
            )

        try:
            self.project_io.apply_project_to_states(
                project_data,
                self.dataset_state,
                self.validation_state,
                self.inference_state,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Open Project", f"Could not apply project data:\n{exc}")
            return

        # Trigger full view refresh after state is populated
        self.dataset_model.beginResetModel()
        self.dataset_model.endResetModel()
        self.annomate_view.refresh_class_combo()

        self._project_dir = str(Path(path).parent)
        self._project_name = project_data.get("project_name", Path(path).stem)
        self._created_at = project_data.get("created_at")
        self.autosave_manager.set_project_dir(self._project_dir)
        self._clear_dirty()

    def _save_project(self) -> None:
        """Save to the current project directory, or prompt for one if not set."""
        if self._project_dir is None:
            self._save_project_as()
            return

        self._write_project(self._project_dir, self._project_name)

    def _save_project_as(self) -> None:
        """Prompt for a project name and parent directory, then save."""
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Choose Project Folder", os.getcwd()
        )
        if not parent_dir:
            return

        default_name = (
            Path(self.dataset_state.image_dir).name
            if self.dataset_state.image_dir
            else "project"
        )
        name, ok = QInputDialog.getText(
            self, "Project Name", "Enter project name:", text=default_name
        )
        if not ok or not name.strip():
            return

        name = name.strip()
        project_dir = os.path.join(parent_dir, name)

        self._project_dir = project_dir
        self._project_name = name
        self._write_project(project_dir, name)
        self.autosave_manager.set_project_dir(project_dir)

    def _write_project(self, project_dir: str, project_name: str) -> None:
        """Perform the actual disk write and update dirty/title state."""
        try:
            annoproj_path = self.project_io.save_project(
                project_dir=project_dir,
                project_name=project_name,
                dataset_state=self.dataset_state,
                validation_state=self.validation_state,
                inference_state=self.inference_state,
                created_at=self._created_at,
                save_score_maps=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Project", f"Could not save project:\n{exc}")
            return

        if self._created_at is None:
            self._created_at = datetime.now(timezone.utc).isoformat()

        self._clear_dirty()
        self.statusBar().showMessage(f"Saved: {annoproj_path}", 4000)

    def _do_autosave(self, project_dir: str) -> None:
        """Write an autosave snapshot when the timer fires."""
        if not self._is_dirty:
            return

        autosave_dir = os.path.join(project_dir, "autosave")
        try:
            self.project_io.save_project(
                project_dir=autosave_dir,
                project_name=f"{self._project_name}.autosave",
                dataset_state=self.dataset_state,
                validation_state=self.validation_state,
                inference_state=self.inference_state,
                created_at=self._created_at,
                save_score_maps=False,
            )
            self.statusBar().showMessage("Autosaved", 3000)
        except Exception as exc:
            self.statusBar().showMessage(f"Autosave failed: {exc}", 5000)

    # ================================================================== #
    # COCO export / import
    # ================================================================== #

    def _export_coco(self) -> None:
        """Open a save-file dialog and export annotations as a COCO JSON file."""
        default_name = (
            f"{Path(self.dataset_state.image_dir).name}_coco.json"
            if self.dataset_state.image_dir
            else "annotations.coco.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Export COCO JSON", default_name, "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.project_io.export_coco(path, self.dataset_state)
            QMessageBox.information(self, "Export COCO", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export COCO", f"Export failed:\n{exc}")

    def _import_coco(self) -> None:
        """Open a file dialog and import a COCO JSON file into the dataset."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import COCO JSON", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.io_controller.import_data_json(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import COCO", f"Import failed:\n{exc}")
            return
        self.annomate_view.refresh_class_combo()

    # ================================================================== #
    # Dirty state & title
    # ================================================================== #

    def _mark_dirty(self) -> None:
        self._is_dirty = True
        self._update_title()

    def _clear_dirty(self) -> None:
        self._is_dirty = False
        self._update_title()

    def _update_title(self) -> None:
        if self._project_name:
            dirty_marker = "*" if self._is_dirty else ""
            self.setWindowTitle(f"{self._project_name}{dirty_marker} — {_APP_TITLE}")
        else:
            self.setWindowTitle(_APP_TITLE)

    def _confirm_discard(self) -> bool:
        """Ask whether to save or discard unsaved changes. Returns True to proceed."""
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Save before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Save:
            self._save_project()
            return True
        elif reply == QMessageBox.Discard:
            return True
        return False  # Cancel

    def closeEvent(self, event) -> None:
        """Prompt to save unsaved changes before closing."""
        if self._is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self._save_project()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
                return
        self.autosave_manager.stop()
        super().closeEvent(event)

    # ================================================================== #
    # Polygon transfer
    # ================================================================== #

    def _handle_polygon_transfer(self, polygons: list, default_class: str):
        """Show class-selection dialog then forward polygons to AnnoMate."""
        class_names = self.dataset_model.get_class_names()
        if not class_names:
            class_names = [default_class]

        chosen, ok = QInputDialog.getItem(
            self, "Choose Class", "Assign polygons to class:",
            class_names,
            class_names.index(default_class) if default_class in class_names else 0,
            False,
        )
        if ok and chosen:
            self.annomate_view.receive_polygons(polygons, chosen)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = AppWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
