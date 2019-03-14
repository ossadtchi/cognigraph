from PyQt5.QtCore import QThread, QObject, Qt, pyqtSignal, QEventLoop
from PyQt5.QtWidgets import QProgressDialog, QApplication, QMessageBox
import logging
import time


class Communicate(QObject):
    """Pyqt signals sender"""
    sync_signal = pyqtSignal()


class AsyncUpdater(QThread):
    _stop_flag = False

    def __init__(self, app, pipeline):
        super(AsyncUpdater, self).__init__()
        self.sender = Communicate()
        self.sender.sync_signal.connect(
            self.process_events_on_main_thread,
            type=Qt.BlockingQueuedConnection)
        self.is_paused = True
        self.app = app
        self.logger = logging.getLogger(type(self).__name__)
        self.pipeline = pipeline

    def process_events_on_main_thread(self):
        self.app.processEvents()

    def run(self):
        self._stop_flag = False
        self.logger.info('Start pipeline')

        is_first_iter = True
        while True:
            # start = time.time()
            self.pipeline.update_all_nodes()
            # end = time.time()
            if is_first_iter:
                # without this hack widgets are not updated unless
                # you click on them
                time.sleep(0.05)
                is_first_iter = False

            self.sender.sync_signal.emit()
            if self._stop_flag is True:
                QApplication.processEvents()
                break

    def stop(self):
        self.logger.info('Stop pipeline')
        self._stop_flag = True

    def toggle(self):
        if self.is_paused:
            self.is_paused = False
            self.start()
        else:
            self.is_paused = True
            self.stop()
            self.wait(1000)
