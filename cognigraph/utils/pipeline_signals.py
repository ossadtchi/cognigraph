from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget


class Communicate(QObject):
    open_dialog = pyqtSignal()
    # -------- used in Node class -------- #
    long_operation_started = pyqtSignal(str)
    long_operation_finished = pyqtSignal()
    # ------------------------------------ #
    init_widget_sig = pyqtSignal()
    draw_sig = pyqtSignal('PyQt_PyObject')
    screenshot_sig = pyqtSignal()
    open_fwd_dialog = pyqtSignal()
    reinit_widget = pyqtSignal(QWidget)
    disabled_changed = pyqtSignal(bool)
    request_message = pyqtSignal(str, str, str)
    node_widget_added = pyqtSignal(QWidget, str)
    initialized = pyqtSignal()
    fps_updated = pyqtSignal(float)
