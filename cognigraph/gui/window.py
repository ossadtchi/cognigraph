from typing import List
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.Qt import QSizePolicy
from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QMainWindow,
    QMdiArea,
    QAction,
    QMdiSubWindow,
    QProgressDialog,
    QFileDialog,
    QMessageBox,
)
from ..nodes.pipeline import Pipeline
from .async_pipeline_update import AsyncPipelineInitializer, AsyncUpdater
from .controls import Controls
from .. import PIPELINES_DIR
from .. import nodes

import logging
import json


class _HookedSubWindow(QMdiSubWindow):
    """Subwindow with intercepted X button behaviour"""

    def closeEvent(self, close_event):
        """
        Before closing subwindow delete output node
        from pipeline and tree widget

        """
        node = self.widget().pipeline_node
        tw = self.parent().parent().parent()._controls.tree_widget
        item = tw.fetch_item_by_node(node)
        tw.remove_item(item)
        QMdiSubWindow.closeEvent(self, close_event)


class GUIWindow(QMainWindow):
    def __init__(self, app, pipeline=Pipeline()):
        super().__init__()
        self._app = app
        self._logger = logging.getLogger(self.__class__.__name__)
        self._is_initialized = False
        self.init_basic(pipeline)
        self.init_ui()
        self.init_controls()

    def init_basic(self, pipeline):
        self._pipeline = pipeline  # type: Pipeline
        self._updater = AsyncUpdater(self._app, pipeline)
        self._pipeline._signal_sender.long_operation_started.connect(
            self._show_progress_dialog
        )
        self._pipeline._signal_sender.long_operation_finished.connect(
            self._hide_progress_dialog
        )
        self._pipeline._signal_sender.request_message.connect(
            self._show_message
        )
        self._pipeline._signal_sender.node_widget_added.connect(
            self._on_node_widget_added
        )
        self._controls = Controls(pipeline=self._pipeline)
        self._controls.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding
        )

        self._controls.tree_widget.node_removed.connect(self._on_node_removed)
        if hasattr(self, "central_widget"):
            for w in self.central_widget.subWindowList():
                self.central_widget.removeSubWindow(w)

    def init_controls(self):
        self.controls_dock.setWidget(self._controls)
        self.run_toggle_action.triggered.disconnect()
        self.run_toggle_action.triggered.connect(self._updater.toggle)
        self._updater._sender.run_toggled.connect(self._on_run_button_toggled)
        self._updater._sender.errored.connect(self._show_message)
        self.is_initialized = False

    def init_ui(self):
        self.central_widget = QMdiArea()
        self.setCentralWidget(self.central_widget)

        # -------- controls widget -------- #
        self.controls_dock = QDockWidget("Processing pipeline setup", self)
        self.controls_dock.setObjectName("Controls")
        self.controls_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        self.controls_dock.visibilityChanged.connect(
            self._update_pipeline_tree_widget_action_text
        )

        self.addDockWidget(Qt.LeftDockWidgetArea, self.controls_dock)

        # self._controls.setMinimumWidth(800)
        # --------------------------------- #

        file_menu = self.menuBar().addMenu("&File")  # file menu
        load_pipeline_action = self._createAction(
            "&Load pipeline", self._load_pipeline
        )
        save_pipeline_action = self._createAction(
            "&Save pipeline", self._save_pipeline
        )
        file_menu.addAction(load_pipeline_action)
        file_menu.addAction(save_pipeline_action)

        # -------- view menu & toolbar -------- #
        tile_windows_action = self._createAction(
            "&Tile windows", self.central_widget.tileSubWindows
        )
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(tile_windows_action)
        view_toolbar = self.addToolBar("View")
        view_toolbar.addAction(tile_windows_action)
        # ------------------------------------- #

        edit_menu = self.menuBar().addMenu("&Edit")
        self._toggle_pipeline_tree_widget_action = self._createAction(
            "&Hide pipeline settings", self._toggle_pipeline_tree_widget
        )
        edit_menu.addAction(self._toggle_pipeline_tree_widget_action)
        edit_toolbar = self.addToolBar("Edit")
        edit_toolbar.setObjectName("edit_toolbar")
        edit_toolbar.addAction(self._toggle_pipeline_tree_widget_action)

        # -------- run menu & toolbar -------- #
        self.run_toggle_action = self._createAction(
            "&Start", self._on_run_button_toggled
        )
        run_menu = self.menuBar().addMenu("&Run")
        self.initialize_pipeline = self._createAction(
            "&Initialize pipeline", self.initialize
        )
        run_menu.addAction(self.run_toggle_action)
        run_menu.addAction(self.initialize_pipeline)
        run_toolbar = self.addToolBar("Run")
        run_toolbar.setObjectName("run_toolbar")
        run_toolbar.addAction(self.run_toggle_action)
        run_toolbar.addAction(self.initialize_pipeline)
        # ------------------------------------ #

    def _toggle_pipeline_tree_widget(self):
        if self.controls_dock.isHidden():
            self.controls_dock.show()
        else:
            self.controls_dock.hide()

    def _update_pipeline_tree_widget_action_text(self, is_visible):
        if is_visible:
            self._toggle_pipeline_tree_widget_action.setText(
                "&Hide pipelne settings"
            )
        else:
            self._toggle_pipeline_tree_widget_action.setText(
                "&Show pipelne settings"
            )

    def _load_pipeline(self):
        file_dialog = QFileDialog(
            caption="Select pipeline file", directory=PIPELINES_DIR
        )
        ext_filter = "JSON file (*.json);; All files (*.*)"
        pipeline_path = file_dialog.getOpenFileName(filter=ext_filter)[0]
        if pipeline_path:
            self._logger.info(
                "Loading pipeline configuration from %s" % pipeline_path
            )
            if not self._updater.is_paused:
                self.run_toggle_action.trigger()
            with open(pipeline_path, "r") as db:
                try:
                    params_dict = json.load(db)
                except json.decoder.JSONDecodeError as e:
                    self._show_message(
                        "Bad pipeline configuration file", detailed_text=str(e)
                    )

                pipeline = self.assemble_pipeline(params_dict, "Pipeline")
                self.init_basic(pipeline)
                self.init_controls()
                # self.resize(self.sizeHint())
        else:
            return

    def _save_pipeline(self):
        self._logger.info("Saving pipeline")
        file_dialog = QFileDialog(
            caption="Select pipeline file", directory=PIPELINES_DIR
        )
        ext_filter = "JSON file (*.json);; All files (*.*)"
        pipeline_path = file_dialog.getSaveFileName(filter=ext_filter)[0]
        if pipeline_path:
            self._logger.info(
                "Saving pipeline configuration to %s" % pipeline_path
            )
            try:
                self._pipeline.save_pipeline(pipeline_path)
            except Exception as exc:
                self._show_message(
                    "Cant`t save pipeline configuration to %s" % pipeline_path,
                    detailed_text=str(exc),
                )
                self._logger.exception(exc)

    def assemble_pipeline(self, d, class_name):
        node_class = getattr(nodes, class_name)
        node = node_class(**d["init_args"])
        for child_class_name in d["children"]:
            child = self.assemble_pipeline(
                d["children"][child_class_name], child_class_name
            )
            node.add_child(child)
        return node

    def initialize(self):
        is_paused = self._updater.is_paused
        if not is_paused:
            self._updater.stop()
        self._logger.info("Initializing all nodes")
        async_initer = AsyncPipelineInitializer(
            pipeline=self._pipeline, parent=self
        )
        async_initer.no_blocking_execution()
        for node in self._pipeline.all_nodes:
            if hasattr(node, "widget"):
                if not node.widget.parent():  # widget not added to QMdiArea
                    self._add_subwindow(node.widget, repr(node))
        self.central_widget.tileSubWindows()
        self.run_toggle_action.setDisabled(False)
        if not is_paused:
            self._updater.start()

    def _finish_initialization(self):
        self.progress_dialog.hide()
        self.progress_dialog.deleteLater()
        for node in self._pipeline.all_nodes:
            if hasattr(node, "widget"):
                self._add_subwindow(node.widget, repr(node))
        self.central_widget.tileSubWindows()

    def _add_subwindow(self, widget, title):
        sw = _HookedSubWindow(self.central_widget)
        sw.setWidget(widget)
        sw.setWindowTitle(title)
        widget.show()

    def _show_progress_dialog(self, text):
        # -------- setup progress dialog -------- #
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setLabelText(text)
        self.progress_dialog.setCancelButtonText(None)
        self.progress_dialog.setRange(0, 0)
        self.progress_dialog.show()

    def _hide_progress_dialog(self):
        self.progress_dialog.hide()
        self.progress_dialog.deleteLater()

    def _on_subwindow_close(self, close_event):
        pass

    def _on_node_widget_added(self, widget, widget_name):
        self._add_subwindow(widget, widget_name)
        self.central_widget.tileSubWindows()

    def _on_node_removed(self, tree_item):
        if hasattr(tree_item.node, "widget"):
            try:
                self.central_widget.removeSubWindow(
                    tree_item.node.widget.parent()
                )
            except AttributeError:
                pass
            except Exception as exc:
                self._show_message(
                    "Can`t remove widget for %s" % tree_item.node,
                    detailed_text=str(exc),
                )
                self._logger.exception(exc)

    def _show_message(self, text, detailed_text=None, level="error"):
        if level == "error":
            icon = QMessageBox.Critical
        elif level == "warning":
            icon = QMessageBox.Warning
        elif level == "info":
            icon = QMessageBox.Information
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setText(text)
        msg.setDetailedText(detailed_text)
        msg.show()

    def _createAction(
        self,
        text,
        slot=None,
        shortcut=None,
        icon=None,
        tip=None,
        checkable=False,
    ):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
        return action

    def moveEvent(self, event):
        return super(GUIWindow, self).moveEvent(event)

    def _on_run_button_toggled(self, is_paused=True):
        if is_paused:
            self.run_toggle_action.setText("Start")
        else:
            self.run_toggle_action.setText("Pause")

    @property
    def is_initialized(self):
        return self._is_initialized

    @is_initialized.setter
    def is_initialized(self, value):
        if value:
            self.run_toggle_action.setDisabled(False)
        else:
            self.run_toggle_action.setDisabled(True)
        self._is_initialized = value

    @property
    def _node_widgets(self) -> List[QWidget]:
        node_widgets = list()
        for node in self._pipeline.all_nodes:
            try:
                node_widgets.append(node.widget)
            except AttributeError:
                pass
        return node_widgets
