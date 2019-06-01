"""Tests for LinearFilter node"""
import pytest
from cognigraph.nodes.processors import LinearFilter
from cognigraph.nodes.sources import FileSource

from cognigraph.nodes.tests.prepare_tests_data import info, data_path  # noqa
import numpy as np


@pytest.fixture  # noqa
def lin_filter(info, data_path):  # noqa
    lin_filter = LinearFilter()
    lin_filter.mne_info = info
    N_SEN = len(info['ch_names'])
    lin_filter.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    lin_filter.parent = parent
    return lin_filter


def test_change_api_attributes(lin_filter):
    """Change upper_cutoff and check if _linear_filter is reset"""
    arbitrary_value_1 = 50
    arbitrary_value_2 = 60

    # filter_init = lin_filter._linear_filter

    lin_filter.upper_cutoff = arbitrary_value_1
    lin_filter.initialize()
    lin_filter._linear_filter = None
    lin_filter.upper_cutoff = arbitrary_value_2
    lin_filter.update()

    assert lin_filter._linear_filter is not None


def test_input_hist_invalidation_resets_filter_delays(lin_filter):
    arbitrary_value = 200

    lin_filter.parent.initialize()
    lin_filter.initialize()

    zi_ini = lin_filter._linear_filter.zi
    lin_filter._linear_filter.zi = arbitrary_value

    lin_filter.on_input_history_invalidation()

    assert np.array_equal(lin_filter._linear_filter.zi, zi_ini)
