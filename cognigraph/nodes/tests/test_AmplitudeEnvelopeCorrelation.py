"""Tests for AmplitudeEnvelopeCorrelations node"""
import pytest
from cognigraph.nodes.processors import AmplitudeEnvelopeCorrelations
from cognigraph.nodes.sources import FileSource

from cognigraph.nodes.tests.prepare_tests_data import info, data_path  # noqa
import numpy as np


@pytest.fixture  # noqa
def amp_correlator(info, data_path):  # noqa
    amp_correlator = AmplitudeEnvelopeCorrelations()
    amp_correlator.mne_info = info
    N_SEN = len(info['ch_names'])
    amp_correlator.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    amp_correlator.parent = parent
    return amp_correlator


def test_change_api_attributes(amp_correlator):
    """Change factor and check if _envelope_extractor is reset"""
    arbitrary_value_1 = 0.5
    arbitrary_value_2 = 0.6

    # filter_init = amp_correlator._linear_filter

    amp_correlator.factor = arbitrary_value_1
    amp_correlator.initialize()
    amp_correlator._envelope_extractor = None
    amp_correlator.factor = arbitrary_value_2
    amp_correlator.update()

    assert amp_correlator._envelope_extractor is not None


def test_input_hist_invalidation_resets_filter_delays(amp_correlator):
    arbitrary_value = 200

    amp_correlator.parent.initialize()
    amp_correlator.initialize()

    zi_ini = amp_correlator._envelope_extractor.zi
    amp_correlator._envelope_extractor.zi = arbitrary_value

    amp_correlator._on_input_history_invalidation()

    assert np.array_equal(amp_correlator._envelope_extractor.zi, zi_ini)
