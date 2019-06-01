"""Launch main cognigraph gui window"""

import argparse
import sys

# import os.path as op
import logging
import mne
import numpy as np
from PyQt5 import QtWidgets
from cognigraph.nodes.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs
from cognigraph.gui.window import GUIWindow


np.warnings.filterwarnings("ignore")  # noqa

# ----------------------------- setup argparse ----------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data", type=argparse.FileType("r"), help="data path"
)
# parser.add_argument(
#     "-f", "--forward", type=argparse.FileType("r"), help="forward model path"
# )
parser.add_argument(
    "-l", "--logfile", type=argparse.FileType("w"), default=None
)
args = parser.parse_args()
# -------------------------------------------------------------------------- #

# ----------------------------- setup logging  ----------------------------- #
if args.logfile:
    logfile = args.logfile.name
else:
    logfile = None
format = "%(asctime)s:%(name)-17s:%(levelname)s:%(message)s"
logging.basicConfig(level=logging.INFO, filename=logfile, format=format)
logger = logging.getLogger(__name__)
mne.set_log_level("ERROR")
mne.set_log_file(fname=logfile, output_format=format)
# -------------------------------------------------------------------------- #

sys.path.append("../vendor/nfb")  # For nfb submodule


def assemble_pipeline(
    file_path=None,
    fwd_path=None,
    subject=None,
    subjects_dir=None,
    inverse_method="mne",
):
    pipeline = Pipeline()
    source = sources.FileSource(file_path=file_path)
    source.loop_the_file = True
    source.MAX_SAMPLES_IN_CHUNK = 10000
    pipeline.add_child(source)

    # ----------------------------- processors ----------------------------- #
    preprocessing = processors.Preprocessing(collect_for_x_seconds=120)
    source.add_child(preprocessing)

    linear_filter = processors.LinearFilter(
        lower_cutoff=8.0, upper_cutoff=12.0
    )
    preprocessing.add_child(linear_filter)

    if inverse_method == "mne":
        inverse_model = processors.MNE(
            method="MNE", snr=1.0, forward_model_path=fwd_path
        )
        # inverse_model = processors.MneGcs(snr=1.0, seed=1000,
        #                                   forward_model_path=fwd_path)
        linear_filter.add_child(inverse_model)
        envelope_extractor = processors.EnvelopeExtractor(0.99)
        inverse_model.add_child(envelope_extractor)
    elif inverse_method == "beamformer":
        inverse_model = processors.Beamformer(
            fwd_path=fwd_path,
            is_adaptive=True,
            output_type="activation",
            forgetting_factor_per_second=0.95,
        )
        linear_filter.add_child(inverse_model)
        envelope_extractor = processors.EnvelopeExtractor(0.99)
        inverse_model.add_child(envelope_extractor)
    elif inverse_method == "mce":
        inverse_model = processors.MCE(forward_model_path=fwd_path, snr=1.0)
        linear_filter.add_child(inverse_model)
        envelope_extractor = processors.EnvelopeExtractor(0.995)
        inverse_model.add_child(envelope_extractor)
    # ---------------------------------------------------------------------- #

    # ------------------------------ outputs ------------------------------ #
    global_mode = outputs.BrainViewer.LIMITS_MODES.GLOBAL

    brain_viewer = outputs.BrainViewer(
        limits_mode=global_mode, buffer_length=6
    )
    envelope_extractor.add_child(brain_viewer)

    # roi_average = processors.AtlasViewer(SUBJECT, subjects_dir)
    # roi_average.parent = inverse_model
    # pipeline.add_processor(roi_average)

    # aec = processors.AmplitudeEnvelopeCorrelations(
    #     method=None,
    #     seed=1000
    #     # method='temporal_orthogonalization',
    #     # method='geometric_correction',
    #     # seed=0
    # )
    # pipeline.add_processor(aec)
    # aec.parent = inverse_model
    # # coh = processors.Coherence(
    # #     method='coh', seed=0)
    # aec_env = processors.EnvelopeExtractor(0.995)
    # pipeline.add_processor(aec_env)

    # seed_viewer = outputs.BrainViewer(
    #     limits_mode=global_mode, buffer_length=6,
    #     surfaces_dir=op.join(subjects_dir, SUBJECT))

    # pipeline.add_output(seed_viewer, parent=aec_env)

    # pipeline.add_output(outputs.LSLStreamOutput())
    # signal_viewer = outputs.SignalViewer()
    # signal_viewer_src = outputs.SignalViewer()
    # pipeline.add_output(signal_viewer, parent=linear_filter)
    # pipeline.add_output(signal_viewer_src, parent=roi_average)
    # con_viewer = outputs.ConnectivityViewer(
    #     surfaces_dir=op.join(subjects_dir, SUBJECT))
    # pipeline.add_output(con_viewer, parent=aec)
    # --------------------------------------------------------------------- #
    return pipeline


def main():
    def on_main_window_close():
        window._updater.stop()
        window._updater.wait(100)
        app.processEvents()
        window._updater.quit()
        try:
            logger.info("Deleting main window ...")
            window.deleteLater()
        except RuntimeError:
            logger.info("Window has already been deleted")

    app = QtWidgets.QApplication(sys.argv)

    logger.debug("Assembling pipeline")
    pipeline = assemble_pipeline(None, None, inverse_method="beamformer")
    # pipeline = load_pipeline("/home/dmalt/my_pipeline.json")
    logger.debug("Finished assembling pipeline")
    # Create window
    window = GUIWindow(app, pipeline=pipeline)
    # window.init_ui()
    window.show()

    if not args.data:
        try:
            file_tuple = QtWidgets.QFileDialog.getOpenFileName(
                caption="Select Data",
                filter="Brainvision (*.eeg *.vhdr *.vmrk);;"
                "MNE-python (*.fif);;"
                "European Data Format (*.edf)",
            )
            file_path = file_tuple[0]
        except Exception:
            logger.error("DATA FILE IS MANDATORY!")
    else:
        file_path = args.data.name

    # if not file_path:
    #     raise Exception("DATA PATH IS MANDATORY!")

    # if not args.forward:
    #     dialog = FwdSetupDialog()
    #     dialog.exec()
    #     fwd_path = dialog.fwd_path
    #     subject = dialog.subject
    #     subjects_dir = dialog.subjects_dir
    # else:
    #     fwd_path = args.forward.name

    # if not fwd_path:
    #     raise Exception("FORWARD SOLUTION IS MANDATORY!")
    #     logger.info("Exiting ...")
    # pipeline.save_pipeline("/home/dmalt/test.pipeline")
    pipeline._children[0].file_path = file_path
    pipeline._children[0].loop_the_file = True
    # pipeline.all_nodes[4].fwd_path = fwd_path
    # pipeline.all_nodes[4].surfaces_dir = op.join(subjects_dir, subject)
    # pipeline.subjects_dir = subjects_dir
    # pipeline.subject = subject

    # sys.exit(app.exec_())
    # QTimer.singleShot(0, window.initialize)  # initializes all pipeline nodes
    # pipeline.save_pipeline("/home/dmalt/test.pipeline")

    # pipeline.updater = AsyncUpdater(app, pipeline)

    # Show window and exit on close
    app.aboutToQuit.connect(on_main_window_close)
    # sys.exit(app.exec_())
    app.exec_()
