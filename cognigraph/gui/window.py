from typing import List
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.Qt import QSizePolicy
from PyQt5.QtWidgets import (QDockWidget, QWidget, QMainWindow, QDesktopWidget,
                             QMdiArea, QAction, QMdiSubWindow)
from ..pipeline import Pipeline
from .async_pipeline_update import AsyncPipelineInitializer
from .controls import Controls

import logging
logger = logging.getLogger(name=__name__)


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
    def __init__(self, pipeline=Pipeline()):
        super().__init__()
        self._pipeline = pipeline  # type: Pipeline
        self._controls = Controls(pipeline=self._pipeline)
        self._controls.setSizePolicy(QSizePolicy.Preferred,
                                     QSizePolicy.Expanding)

        # Resize screen
        self.resize(QSize(QDesktopWidget().availableGeometry().width() * 0.9,
                          QDesktopWidget().availableGeometry().height() * 0.9))
        self._controls.tree_widget.node_added.connect(self._on_node_added)
        self._controls.tree_widget.node_removed.connect(self._on_node_removed)

    def init_ui(self):
        self.central_widget = QMdiArea()
        self.setCentralWidget(self.central_widget)

        # -------- controls widget -------- #
        controls_dock = QDockWidget('Processing pipeline setup', self)
        controls_dock.setObjectName('Controls')
        controls_dock.setAllowedAreas(Qt.LeftDockWidgetArea |
                                      Qt.RightDockWidgetArea)

        controls_dock.setWidget(self._controls)
        self.addDockWidget(Qt.LeftDockWidgetArea, controls_dock)

        # self._controls.setMinimumWidth(800)
        # --------------------------------- #

        self.menuBar().addMenu('&File')  # file menu

        # -------- view menu & toolbar -------- #
        tile_windows_action = self.createAction(
            '&Tile windows', self.central_widget.tileSubWindows)
        view_menu = self.menuBar().addMenu('&View')
        view_menu.addAction(tile_windows_action)
        view_toolbar = self.addToolBar('View')
        view_toolbar.addAction(tile_windows_action)
        # ------------------------------------- #

        # -------- run menu & toolbar -------- #
        self.run_toggle_action = self.createAction(
            '&Start', self._on_run_button_toggled)
        run_menu = self.menuBar().addMenu('&Run')
        run_menu.addAction(self.run_toggle_action)
        run_toolbar = self.addToolBar('Run')
        run_toolbar.setObjectName('run_toolbar')
        run_toolbar.addAction(self.run_toggle_action)
        # ------------------------------------ #

    def initialize(self):
        logger.debug('Initializing all nodes')
        async_initer = AsyncPipelineInitializer(pipeline=self._pipeline,
                                                parent=self)
        async_initer.no_blocking_execution()
        for node in self._pipeline.all_nodes:
            if hasattr(node, 'widget'):
                self._add_subwindow(node.widget, repr(node))
        self.central_widget.tileSubWindows()

    def _add_subwindow(self, widget, title):
        sw = _HookedSubWindow(self.central_widget)
        sw.setWidget(widget)
        sw.setWindowTitle(title)
        widget.show()

    def _on_subwindow_close(self, close_event):
        pass

    def _on_node_added(self, node, _):
        if hasattr(node, 'widget'):
            self._add_subwindow(node.widget, repr(node))
            self.central_widget.tileSubWindows()

    def _on_node_removed(self, tree_item):
        if hasattr(tree_item.node, 'widget'):
            self.central_widget.removeSubWindow(tree_item.node.widget.parent())

    def createAction(self, text, slot=None, shortcut=None, icon=None,
                     tip=None, checkable=False):
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

    def _on_run_button_toggled(self):
        if self.run_toggle_action.text() == "Pause":
            self.run_toggle_action.setText("Start")
        else:
            self.run_toggle_action.setText("Pause")

    @property
    def _node_widgets(self) -> List[QWidget]:
        node_widgets = list()
        for node in self._pipeline.all_nodes:
            try:
                node_widgets.append(node.widget)
            except AttributeError:
                pass
        return node_widgets
