"""Main launching script"""

import argparse
import sys
import time
import os.path as op
import logging
from PyQt5 import QtCore, QtWidgets
import mne
import numpy as np

from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs
from cognigraph.gui.window import GUIWindow

np.warnings.filterwarnings('ignore')  # noqa

# ----------------------------- setup logging  ----------------------------- #
logfile = None
format = '%(asctime)s:%(name)-17s:%(levelname)s:%(message)s'
logging.basicConfig(level=logging.DEBUG, filename=logfile, format=format)
logger = logging.getLogger(__name__)
mne.set_log_level('INFO')
mne.set_log_file(fname=logfile, output_format=format)
# -------------------------------------------------------------------------- #

# ----------------------------- setup argparse ----------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=argparse.FileType('r'),
                    help='data path')
parser.add_argument('-f', '--forward', type=argparse.FileType('r'),
                    help='forward model path')
args = parser.parse_args()
# -------------------------------------------------------------------------- #

sys.path.append('../vendor/nfb')  # For nfb submodule

SURF_DIR = op.join(mne.datasets.sample.data_path(), 'subjects')
SUBJECT = 'sample'
DATA_DIR = '/home/dmalt/Code/python/cogni_submodules/tests/data'
FWD_MODEL_NAME = 'dmalt_custom_mr-fwd.fif'


def assemble_pipeline(file_path, inverse_method='mne'):
    pipeline = Pipeline()
    source = sources.FileSource(file_path=file_path)
    # source = sources.FileSource()
    source.loop_the_file = True
    source.MAX_SAMPLES_IN_CHUNK = 10000
    pipeline.source = source

    # ----------------------------- processors ----------------------------- #
    preprocessing = processors.Preprocessing(collect_for_x_seconds=120)
    pipeline.add_processor(preprocessing)

    linear_filter = processors.LinearFilter(lower_cutoff=8.0,
                                            upper_cutoff=12.0)
    pipeline.add_processor(linear_filter)

    if inverse_method == 'mne':
        inverse_model = processors.InverseModel(method='MNE', snr=1.0,
                                                forward_model_path=fwd_path)
        pipeline.add_processor(inverse_model)
        envelope_extractor = processors.EnvelopeExtractor(0.99)
        pipeline.add_processor(envelope_extractor)
    elif inverse_method == 'beamformer':
        inverse_model = processors.Beamformer(
            forward_model_path=fwd_path, is_adaptive=True,
            output_type='activation', forgetting_factor_per_second=0.95)
        pipeline.add_processor(inverse_model)
        envelope_extractor = processors.EnvelopeExtractor(0.99)
        pipeline.add_processor(envelope_extractor)
    elif inverse_method == 'mce':
        inverse_model = processors.MCE(forward_model_path=fwd_path, snr=1.0)
        pipeline.add_processor(inverse_model)
        envelope_extractor = processors.EnvelopeExtractor(0.995)
        pipeline.add_processor(envelope_extractor)
    # ---------------------------------------------------------------------- #

    # ------------------------------ outputs ------------------------------ #
    global_mode = outputs.BrainViewer.LIMITS_MODES.GLOBAL

    brain_viewer = outputs.BrainViewer(
        limits_mode=global_mode, buffer_length=6,
        surfaces_dir=op.join(SURF_DIR, SUBJECT))
    pipeline.add_output(brain_viewer, input_node=envelope_extractor)

    roi_average = processors.AtlasViewer(SUBJECT, SURF_DIR)
    roi_average.input_node = inverse_model
    pipeline.add_processor(roi_average)

    aec = processors.AmplitudeEnvelopeCorrelations()
    aec.input_node = roi_average
    pipeline.add_processor(aec)

    # pipeline.add_output(outputs.LSLStreamOutput())
    signal_viewer = outputs.SignalViewer()
    signal_viewer_src = outputs.SignalViewer()
    pipeline.add_output(signal_viewer, input_node=linear_filter)
    pipeline.add_output(signal_viewer_src, input_node=roi_average)
    # --------------------------------------------------------------------- #
    return pipeline


class Communicate(QtCore.QObject):
    """Pyqt signals sender"""
    sync_signal = QtCore.pyqtSignal()


class AsyncUpdater(QtCore.QThread):
    _stop_flag = False

    def __init__(self):
        super(AsyncUpdater, self).__init__()
        self.sender = Communicate()
        self.sender.sync_signal.connect(
            self.process_events_on_main_thread,
            type=QtCore.Qt.BlockingQueuedConnection)
        self.is_paused = True

    def process_events_on_main_thread(self):
        app.processEvents()

    def run(self):
        self._stop_flag = False
        logger.info('Start pipeline')

        is_first_iter = True
        while True:
            start = time.time()
            pipeline.update_all_nodes()
            end = time.time()
            if is_first_iter:
                # without this hack widgets are not updated unless
                # you click on them
                time.sleep(0.05)
                is_first_iter = False

            self.sender.sync_signal.emit()
            if self._stop_flag is True:
                QtWidgets.QApplication.processEvents()
                break

    def stop(self):
        logger.info('Stop pipeline')
        self._stop_flag = True

    def toggle(self):
        if self.is_paused:
            self.is_paused = False
            self.start()
        else:
            self.is_paused = True
            self.stop()
            self.wait(1000)


def on_main_window_close():
    thread.stop()
    thread.wait(100)
    app.processEvents()
    thread.quit()
    try:
        logger.info('Deleting main window ...')
        window.deleteLater()
    except RuntimeError:
        logger.info('Window has already been deleted')
    # del pipeline
    # thread.deleteLater()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    if not args.data:
        try:
            file_tuple = QtWidgets.QFileDialog.getOpenFileName(
                caption="Select Data",
                filter="Brainvision (*.eeg *.vhdr *.vmrk);;"
                       "MNE-python (*.fif);;"
                       "European Data Format (*.edf)")
            file_path = file_tuple[0]
        except:
            logger.error("DATA FILE IS MANDATORY!")
    else:
        file_path = args.data.name

    if not file_path:
        raise Exception("DATA PATH IS MANDATORY!")

    if not args.forward:
        try:
            fwd_tuple = QtWidgets.QFileDialog.getOpenFileName(
                caption="Select forward model",
                filter= "MNE-python forward (*-fwd.fif)")
            fwd_path = fwd_tuple[0]
        except:
            logger.error("FORWARD SOLUTION IS MANDATORY!")
    else:
        fwd_path = args.forward.name

    if not fwd_path:
        raise Exception("FORWARD SOLUTION IS MANDATORY!")
        logger.info('Exiting ...')

    logger.debug('Assembling pipeline')
    pipeline = assemble_pipeline(file_path, inverse_method='mne')
    logger.debug('Finished assembling pipeline')
    # Create window
    window = GUIWindow(pipeline=pipeline)
    window.init_ui()
    window.initialize()  # initializes all pipeline nodes
    window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

    thread = AsyncUpdater()
    window.run_button.clicked.connect(thread.toggle)
    # window.destroyed.connect(on_main_window_close)

    # Show window and exit on close
    window.show()
    app.aboutToQuit.connect(on_main_window_close)
    sys.exit(app.exec_())
