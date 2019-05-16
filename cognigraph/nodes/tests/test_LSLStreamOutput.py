"""Tests for LSLStreamOutput node"""
import pytest
from cognigraph.nodes.outputs import LSLStreamOutput
from cognigraph.nodes.sources import FileSource

from cognigraph.nodes.tests.prepare_tests_data import info, data_path  # noqa
import numpy as np


@pytest.fixture  # noqa
def lsl_streamer(info, data_path):  # noqa
    lsl_streamer = LSLStreamOutput()
    lsl_streamer.mne_info = info
    N_SEN = len(info['ch_names'])
    lsl_streamer.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    lsl_streamer.parent = parent
    return lsl_streamer


def test_change_api_attributes(lsl_streamer):
    """
    Change stream_name and check if initialize is called which resets _outlet.

    """
    arbitrary_name = 'somename'
    arbitrary_value = 200

    lsl_streamer.parent.initialize()
    lsl_streamer.initialize()
    lsl_streamer._outlet = arbitrary_value

    lsl_streamer.stream_name = arbitrary_name
    assert lsl_streamer._outlet is not None


def test_input_hist_invalidation_resets_statistics(lsl_streamer):
    """Check that upstream history change doesn't break the node"""
    lsl_streamer.parent.initialize()
    lsl_streamer.initialize()

    lsl_streamer.parent.source_name = 'new_name'  # triggers reset for source
