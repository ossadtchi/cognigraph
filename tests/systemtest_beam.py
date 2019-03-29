"""Integration test to check mce performance"""
import time
import sys
import os.path as op

import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'  # noqa
from PyQt5 import QtCore, QtWidgets
# import pyqtgraph

# pyqtgraph.setConfigOption('useOpenGL', True)  # noqa

# from cognigraph.helpers.brainvision import read_fif_data
from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs
# from cognigraph import TIME_AXIS
from cognigraph.gui.window import GUIWindow
import logging
import mne
mne.set_log_level('WARNING')

app = QtWidgets.QApplication(sys.argv)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(name)-17s:%(levelname)s:%(message)s')

logger = logging.getLogger(__name__)
pipeline = Pipeline()

cur_dir = '/home/dmalt/Code/python/cogni_submodules'
test_data_path = cur_dir + '/tests/data/'
print(test_data_path)
sim_data_fname = 'raw_sim_nobads.fif'
# sim_data_fname = 'DF_2018-03-02_11-34-38.edf'
# sim_data_fname = 'Koleno.fif'
fwd_fname = 'dmalt_custom_mr-fwd.fif'
# fwd_fname = 'sample_1005-eeg-ico-4-fwd.fif'
# fwd_fname = 'sample_1005-eeg-oct-6-fwd.fif'
# fwd_fname = 'DF_2018-03-02_11-34-38-fwd.fif'

surf_dir = '/home/dmalt/mne_data/MNE-sample-data/subjects/sample/surf'

fwd_path = op.join(test_data_path, fwd_fname)
sim_data_path = op.join(test_data_path, sim_data_fname)

source = sources.FileSource(file_path=sim_data_path)
# source = sources.LSLStreamSource(stream_name='Mitsar')
source.MAX_SAMPLES_IN_CHUNK = 100
source.loop_the_file = True
pipeline.source = source

# Processors
preprocessing = processors.Preprocessing(collect_for_x_seconds=30)
pipeline.add_processor(preprocessing)

linear_filter = processors.LinearFilter(lower_cutoff=8.0, upper_cutoff=12.0)
pipeline.add_processor(linear_filter)

beamformer = processors.Beamformer(forward_model_path=fwd_path,
                                   is_adaptive=True, output_type='activation',
                                   forgetting_factor_per_second=0.95)
# inverse_model = processors.InverseModel(
#         method='MNE', snr=1.0,
#         forward_model_path=fwd_path)
pipeline.add_processor(beamformer)
# pipeline.add_processor(inverse_model)


envelope_extractor = processors.EnvelopeExtractor(0.995)
pipeline.add_processor(envelope_extractor)

# Outputs
global_mode = outputs.BrainViewer.LIMITS_MODES.GLOBAL
three_dee_brain = outputs.BrainViewer(
        limits_mode=global_mode, buffer_length=10, surfaces_dir=surf_dir)
pipeline.add_output(three_dee_brain)
# pipeline.add_output(outputs.LSLStreamOutput())
# pipeline.initialize_all_nodes()
# file_output = outputs.FileOutput()
# torch_output = outputs.TorchOutput()

signal_viewer = outputs.SignalViewer()
pipeline.add_output(signal_viewer, parent=linear_filter)
# pipeline.add_output(file_output, parent=beamformer)
# pipeline.add_output(torch_output, parent=source)

window = GUIWindow(pipeline=pipeline)
window.init_ui()
window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
window.show()


base_controls = window._controls._base_controls
source_controls = base_controls.source_controls
processors_controls = base_controls.processors_controls
outputs_controls = base_controls.outputs_controls

source_controls.source_type_combo.setValue(
        source_controls.SOURCE_TYPE_PLACEHOLDER)


linear_filter_controls = processors_controls.children()[0]

envelope_controls = processors_controls.children()[2]
# envelope_controls.disabled.setValue(True)


three_dee_brain_controls = outputs_controls.children()[0]
three_dee_brain_controls.limits_mode_combo.setValue('Global')
three_dee_brain_controls.threshold_slider.setValue(45)
# three_dee_brain_controls.limits_mode_combo.setValue('Local')

window.initialize()

# start_s, stop_s = 80, 100
# with source.not_triggering_reset():
#     source.data, _ = read_fif_data(sim_data_path, time_axis=TIME_AXIS, start_s=start_s, stop_s=stop_s)

# def run():
#     logging.debug('Start iteration')
#     pipeline.update_all_nodes()
#     logging.debug('End iteration')
    # pass
    # print(pipeline.source._samples_already_read / 500)

class Communicate(QtCore.QObject):
    sync_signal = QtCore.pyqtSignal()


class AsyncUpdater(QtCore.QThread):
    _stop_flag = False

    def __init__(self):
        super(AsyncUpdater, self).__init__()
        self.sender = Communicate()
        self.sender.sync_signal.connect(
            self.process_events_on_main_thread,
            type=QtCore.Qt.BlockingQueuedConnection)
        # self.setAutoDelete(False)

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
                time.sleep(0.05)
                is_first_iter = False

            self.sender.sync_signal.emit()
            # Force sleep to update at 10Hz
            # if end - start < 0.05:
            #     time.sleep(0.05 - (end - start))
            # QtWidgets.QApplication.processEvents()
            if self._stop_flag is True:
                QtWidgets.QApplication.processEvents()
                break

    def stop(self):
        logger.info('Stop pipeline')
        self._stop_flag = True


# pool = QtCore.QThreadPool.globalInstance()
pool = AsyncUpdater()
# updater = AsyncUpdater()
is_paused = True


def toggle_updater():
    global pool
    global updater
    global is_paused

    if is_paused:
        is_paused = False
        # pool.start(updater)
        pool.start()
    else:
        is_paused = True
        pool.stop()
        # pool.waitForDone()


def on_main_window_close():
    global pipeline
    global window
    global app
    logger.info('Exiting ...')
    pool.stop()
    pool.wait(100)
    app.processEvents()
    pool.quit()
    try:
        logger.info('Deleting main window ...')
        window.deleteLater()
    except RuntimeError:
        logger.info('Window has already been deleted')
    # del pipeline
    # pool.deleteLater()


window.run_button.clicked.connect(toggle_updater)
# window.destroyed.connect(on_main_window_close)

# Show window and exit on close
window.show()
# updater.stop()
# pool.waitForDone()
app.aboutToQuit.connect(on_main_window_close)
sys.exit(app.exec_())
