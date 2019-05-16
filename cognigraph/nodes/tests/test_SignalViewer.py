"""Tests for SignalViewer node"""
import pytest
from cognigraph.nodes.outputs import SignalViewer
from cognigraph.nodes.sources import FileSource

from cognigraph.nodes.tests.prepare_tests_data import info, data_path  # noqa
import numpy as np


@pytest.fixture  # noqa
def signal_viewer(info, data_path):  # noqa
    signal_viewer = SignalViewer()
    signal_viewer.mne_info = info
    N_SEN = len(info['ch_names'])
    signal_viewer.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    signal_viewer.parent = parent
    return signal_viewer


def test_change_api_attributes(signal_viewer):
    """Check that appropriate method is defined"""
    signal_viewer._on_critical_attr_change(None, None, None)


# Doesn't work in headless mode
# commenting out signal_viewer.initialize() doesn't help either
# def test_input_hist_invalidation_resets_statistics(signal_viewer):
#     """Check that upstream history change doesn't break the node"""
#     signal_viewer.parent.initialize()
#     signal_viewer.initialize()
#     signal_viewer.parent.source_name = 'new_name' # triggers reset for source
