"""Tests for ICARejection node"""
import pytest
from cognigraph.nodes.processors import ICARejection
from cognigraph.nodes.sources import FileSource

from cognigraph.nodes.tests.prepare_tests_data import info, data_path  # noqa
import numpy as np


@pytest.fixture  # noqa
def ica_rejector(info, data_path):  # noqa
    ica_rejector = ICARejection()
    ica_rejector.mne_info = info
    N_SEN = len(info['ch_names'])
    ica_rejector.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    ica_rejector.parent = parent
    return ica_rejector


def test_change_api_attributes(ica_rejector):
    """Change collect_for_x_seconds and check if _samples_collected is reset"""
    arbitrary_value = 200

    ica_rejector.collect_for_x_seconds = 10
    ica_rejector.initialize()
    ica_rejector._samples_collected = arbitrary_value
    ica_rejector.collect_for_x_seconds = 20
    ica_rejector.reset()
    assert ica_rejector._samples_collected == 0


def test_input_hist_invalidation_resets_statistics(ica_rejector):
    arbitrary_value = 200

    ica_rejector.parent.initialize()
    ica_rejector.initialize()

    ica_rejector._samples_collected = arbitrary_value
    ica_rejector.on_input_history_invalidation()

    assert ica_rejector._samples_collected == 0
