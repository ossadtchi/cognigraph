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
logging.basicConfig(level=logging.DEBUG, filename=logfile, format=format)
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
    source.MAX_SAMPLES_IN_CHUNK = 1000
    pipeline.add_child(source)

    # ----------------------------- processors ----------------------------- #
    preprocessing = processors.Preprocessing(collect_for_x_seconds=10)
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

    pipeline._children[0].loop_the_file = True
    app.aboutToQuit.connect(on_main_window_close)
    app.exec_()
