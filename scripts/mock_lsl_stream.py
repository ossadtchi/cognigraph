import time
import sys
from multiprocessing import Process

import pylsl as lsl
import numpy as np
import mne
from mne.datasets import sample


from cognigraph.utils.lsl import create_lsl_outlet
from cognigraph.utils.channels import read_channel_types
# from cognigraph import MISC_CHANNEL_TYPE


class MockLSLStream(Process):
    # TODO: implement so that we do not have to run this file as a script

    def __init__(self, meg_cnt, eeg_cnt, other_cnt):
        self.meg_cnt = meg_cnt
        self.eeg_cnt = eeg_cnt
        self.other_cnt = other_cnt


frequency = 100
name = "cognigraph-mock-stream"
stream_type = "EEG"
channel_format = lsl.cf_float32

channel_labels_1005 = mne.channels.read_montage("standard_1005").ch_names

# Get neuromag channels from a random raw file
data_path = sample.data_path()
# raw_fname = data_path + "/MEG/sample/sample_audvis_filt-0-40_raw.fif"
raw_fname = '/home/dmalt/Data/cognigraph/evoked_mne.fif'
info = mne.io.read_info(raw_fname)
channel_labels_neuromag = info["ch_names"]
channel_types_neuromag = read_channel_types(info)

meg_cnt = eeg_cnt = other_cnt = 8
# channel_labels = (
#     [
#         name
#         for name in channel_labels_1005
#         if any(char.isdigit() for char in name) or name.endswith("z")
#     ][:eeg_cnt]
#     + channel_labels_neuromag[:meg_cnt]
#     + ["Other {}".format(i + 1) for i in range(other_cnt)]
# )
channel_labels = [c for c in info['ch_names']]
channel_types = ["eeg" for c in channel_labels]
# channel_types = (
#     ["eeg"] * eeg_cnt
#     + channel_types_neuromag[:meg_cnt]
#     + [MISC_CHANNEL_TYPE] * other_cnt
# )
channel_count = len(channel_labels)


outlet = create_lsl_outlet(
    name=name,
    type=stream_type,
    frequency=frequency,
    channel_format=channel_format,
    channel_labels=channel_labels,
    channel_types=channel_types,
)

while True:
    try:
        mysample = np.random.random((channel_count, 1))
        outlet.push_sample(mysample)
        time.sleep(1 / frequency)
    except Exception:
        del outlet
        sys.exit()
