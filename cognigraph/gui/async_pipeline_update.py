from PyQt5.QtCore import QThread, QObject, Qt, pyqtSignal, QEventLoop, pyqtSlot
from PyQt5.QtWidgets import QProgressDialog, QApplication, QMessageBox
import logging
import time
import traceback


class _Communicate(QObject):
    """Pyqt signals sender"""

    sync_signal = pyqtSignal()
    run_toggled = pyqtSignal(bool)
    errored = pyqtSignal(str, str, str)


class AsyncUpdater(QThread):
    _stop_flag = False

    def __init__(self, app, pipeline):
        QThread.__init__(self)
        self._sender = _Communicate()
        self._sender.sync_signal.connect(
            self.process_events_on_main_thread,
            type=Qt.BlockingQueuedConnection,
        )
        self.is_paused = True
        self.app = app
        self._logger = logging.getLogger(type(self).__name__)
        self._pipeline = pipeline

    def process_events_on_main_thread(self):
        self.app.processEvents()

    def run(self):
        self._stop_flag = False

        is_first_iter = True
        while True:
            # start = time.time()
            try:
                self._pipeline.update()
            except Exception as exc:
                self.stop()
                self._logger.exception(exc)
                self._sender.errored.emit(
                    "Can't update pipeline!",
                    str(exc),
                    "error",
                )
            # end = time.time()
            if is_first_iter:
                # without this hack widgets are not updated unless
                # you click on them
                time.sleep(0.05)
                is_first_iter = False

            self._sender.sync_signal.emit()
            if self._stop_flag is True:
                QApplication.processEvents()
                break

    def start(self):
        if self.is_paused:
            self.is_paused = False
            self._stop_flag = False
            self._logger.info("Start pipeline")
            QThread.start(self)
            self._sender.run_toggled.emit(self.is_paused)

    def stop(self):
        if not self.is_paused:
            self.is_paused = True
            self._logger.info("Stop pipeline")
            self._stop_flag = True
            self._sender.run_toggled.emit(self.is_paused)

    def toggle(self):
        if self.is_paused:
            self.start()
        else:
            self.stop()


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

    exception_ocurred = pyqtSignal(Exception, str)
    progress_updated = pyqtSignal(int)

    def __init__(
        self,
        parent=None,
        progress_text="Operation is running... Please be patient",
        error_text="Operation failed",
        is_show_progress=False,
    ):
        QThread.__init__(self, parent)
        self.is_successful = True
        self.is_show_progress = is_show_progress
        self.progress_text = progress_text
        self.error_text = error_text

        self.exception_ocurred.connect(self._on_run_exception)

        self._logger = logging.getLogger(type(self).__name__)

    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.is_successful = False
            self._logger.exception(exc)
            tb = traceback.format_exc()
            self.exception_ocurred.emit(exc, tb)

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

    def _on_run_exception(self, exception, tb):
        msg = QMessageBox(self.parent())
        msg.setText(self.error_text)
        msg.setDetailedText(tb)
        msg.setIcon(QMessageBox.Warning)
        msg.show()
        self._logger.exception(exception)


class AsyncPipelineInitializer(ThreadToBeWaitedFor):
    def __init__(self, pipeline, parent=None):
        ThreadToBeWaitedFor.__init__(self, parent)
        self.is_show_progress = False
        self.progress_text = (
            "Initializing the data processing pipeline... Please be patient"
        )
        self.error_text = "Initialization failed. See log for details."
        self._pipeline = pipeline

    def _run(self):
        self._pipeline.chain_initialize()


class AsyncPipelineResetter(ThreadToBeWaitedFor):
    """Reset pipeline nodes asyncronously"""

    def __init__(self, node, key, old_value, new_value, parent=None):
        ThreadToBeWaitedFor.__init__(self, parent)
        self._node = node
        self._key = key
        self._old_value = old_value
        self._new_value = new_value

    def _run(self):
        with self._node.not_triggering_reset():
            is_output_hist_invalid = self._node._on_critical_attr_change(
                self._key, self._old_value, self._new_value
            )
        for child in self._node._children:
            child.on_upstream_change(is_output_hist_invalid)


class NodeInitWorker(QObject):
    finished = pyqtSignal()
    errored = pyqtSignal(str)
    succeeded = pyqtSignal()

    def __init__(self, node):
        QObject.__init__(self)
        self.node = node
        self._logger = logging.getLogger(type(self).__name__)

    @pyqtSlot()
    def run(self):
        try:
            self.node.initialize()
            self.succeeded.emit()
        except Exception as exc:
            self.errored.emit(str(exc))
            self._logger.exception(exc)
        finally:
            self.finished.emit()
