"""Tests for EnvelopeExtractor node"""
import pytest
from cognigraph.nodes.processors import EnvelopeExtractor
from cognigraph.nodes.sources import FileSource

from cognigraph.nodes.tests.prepare_tests_data import info, data_path  # noqa
import numpy as np


@pytest.fixture  # noqa
def env_extractor(info, data_path):  # noqa
    env_extractor = EnvelopeExtractor()
    env_extractor.mne_info = info
    N_SEN = len(info['ch_names'])
    env_extractor.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    env_extractor.parent = parent
    return env_extractor


def test_change_api_attributes(env_extractor):
    """Change factor and check if _envelope_extractor is reset"""
    arbitrary_value_1 = 0.5
    arbitrary_value_2 = 0.6

    # filter_init = env_extractor._linear_filter

    env_extractor.factor = arbitrary_value_1
    env_extractor.initialize()
    env_extractor._envelope_extractor = None
    env_extractor.factor = arbitrary_value_2

    assert env_extractor._envelope_extractor is not None


def test_input_hist_invalidation_resets_filter_delays(env_extractor):
    arbitrary_value = 200

    env_extractor.parent.initialize()
    env_extractor.initialize()

    zi_ini = env_extractor._envelope_extractor.zi
    env_extractor._envelope_extractor.zi = arbitrary_value

    env_extractor.parent.source_name = 'new_name'  # triggers reset for source

    assert np.array_equal(env_extractor._envelope_extractor.zi, zi_ini)
