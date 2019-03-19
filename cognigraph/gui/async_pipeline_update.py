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


class ThreadToBeWaitedFor(QThread):
    """
    Thread running a long process which must be waited for.
    Disables its parent and shows a progress dialog while keeping both
    responsive.
    In case of exception shows message box with exception details.

    Abstract class.
    Define _run and augment __init__ method in children.

    To run the thread call no_blocking_execution method.

    """
    exception_ocurred = pyqtSignal(Exception)
    progress_updated = pyqtSignal(int)

    def __init__(self, parent=None,
                 progress_text='Operation is running... Please be patient',
                 error_text='Operation failed', is_show_progress=False):
        super().__init__(parent)
        self.is_successful = True
        self.is_show_progress = is_show_progress
        self.progress_text = progress_text
        self.error_text = error_text

        self.exception_ocurred.connect(self._on_run_exception)

        self.logger = logging.getLogger(type(self).__name__)

    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.is_successful = False
            self.logger.exception(str(exc))
            self.exception_ocurred.emit(exc)

    def _run(self):
        raise NotImplementedError()

    def no_blocking_execution(self):
        """Returns True if computation went through without exceptions"""
        q = QEventLoop()
        # -------- setup progress dialog -------- #
        progress_dialog = QProgressDialog(self.parent())
        progress_dialog.setLabelText(self.progress_text)
        progress_dialog.setCancelButtonText(None)
        if self.is_show_progress:
            progress_dialog.setRange(0, 100)
        else:
            progress_dialog.setRange(0, 0)
        self.progress_updated.connect(progress_dialog.setValue)
        progress_dialog.show()
        # --------------------------------------- #
        if self.parent():
            self.parent().setDisabled(True)
        progress_dialog.setDisabled(False)
        self.finished.connect(q.quit)
        # self.finished.connect(self._on_finished)
        self.start()
        q.exec(QEventLoop.ExcludeUserInputEvents)
        progress_dialog.hide()
        if self.parent():
            self.parent().setDisabled(False)
        return self.is_successful

    # def _on_finished(self):
    #     print('Finished')

    def _on_run_exception(self, exception):
        msg = QMessageBox(self.parent())
        msg.setText(self.error_text)
        msg.setDetailedText(str(exception))
        msg.setIcon(QMessageBox.Warning)
        msg.show()


class AsyncPipelineInitializer(ThreadToBeWaitedFor):
    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.is_show_progress = False
        self.progress_text = ('Initializing the data processing pipeline...'
                              ' Please be patient')
        self.error_text = 'Initialization failed. See log for details.'
        self.pipeline = pipeline

    def _run(self):
        self.pipeline.initialize_all_nodes()
