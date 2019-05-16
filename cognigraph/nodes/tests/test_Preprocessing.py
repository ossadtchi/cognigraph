"""Tests for Preprocessing node"""
import pytest
from cognigraph.nodes.processors import Preprocessing
from cognigraph.nodes.sources import FileSource

from cognigraph.nodes.tests.prepare_tests_data import info, data_path  # noqa
import numpy as np


@pytest.fixture  # noqa
def preprocessor(info, data_path):  # noqa
    preprocessor = Preprocessing()
    preprocessor.mne_info = info
    N_SEN = len(info['ch_names'])
    preprocessor.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    preprocessor.parent = parent
    return preprocessor


def test_change_api_attributes(preprocessor):
    """Change collect_for_x_seconds and check if _samples_collected is reset"""
    arbitrary_value = 200

    preprocessor.collect_for_x_seconds = 10
    preprocessor.initialize()
    preprocessor._samples_collected = arbitrary_value
    preprocessor.collect_for_x_seconds = 20
    assert preprocessor._samples_collected == 0

    preprocessor._samples_collected = arbitrary_value
    preprocessor.dsamp_freq = 8
    assert preprocessor._samples_collected == arbitrary_value


def test_input_hist_invalidation_resets_statistics(preprocessor):
    arbitrary_value = 200

    preprocessor.parent.initialize()
    preprocessor.initialize()

    preprocessor._samples_collected = arbitrary_value
    preprocessor.parent.source_name = 'new_name'  # triggers reset for source

    assert preprocessor._samples_collected == 0
