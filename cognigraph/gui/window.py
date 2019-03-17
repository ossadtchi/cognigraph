from typing import List
from PyQt5.QtGui import QIcon
# from PyQt5 import Qt
from PyQt5.QtCore import Qt, QSize
from PyQt5.Qt import QSizePolicy
from PyQt5.QtWidgets import (QDockWidget, QWidget, QMainWindow, QDesktopWidget,
                             QMdiArea, QAction)
from ..pipeline import Pipeline
from .async_pipeline_update import AsyncPipelineInitializer
from .controls import Controls
# from .screen_recorder import ScreenRecorder

import logging
logger = logging.getLogger(name=__name__)


class GUIWindow(QMainWindow):
    def __init__(self, pipeline=Pipeline()):
        super().__init__()
        self._pipeline = pipeline  # type: Pipeline
        self._controls = Controls(pipeline=self._pipeline)
        self._controls_widget = self._controls.widget
        self._controls_widget.setSizePolicy(QSizePolicy.Preferred,
                                            QSizePolicy.Expanding)

        # Start button

        # Resize screen
        self.resize(QSize(
            QDesktopWidget().availableGeometry().width() * 0.9,
            QDesktopWidget().availableGeometry().height() * 0.9))

    def init_ui(self):
        self.central_widget = QMdiArea()
        self.setCentralWidget(self.central_widget)

        # -------- controls widget -------- #
        self._controls.initialize()

        controls_dock = QDockWidget('Controls', self)
        controls_dock.setObjectName('Controls')
        controls_dock.setAllowedAreas(Qt.LeftDockWidgetArea |
                                      Qt.RightDockWidgetArea)

        controls_dock.setWidget(self._controls_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, controls_dock)

        self._controls_widget.setMinimumWidth(800)
        # --------------------------------- #

        file_menu = self.menuBar().addMenu('&File')

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
        for node_widget in self._node_widgets:
            if node_widget:
                node_widget.setMinimumWidth(600)
                self.central_widget.addSubWindow(node_widget)
                node_widget.show()
            else:
                raise ValueError('Node widget is not defined')
        self.central_widget.tileSubWindows()

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
