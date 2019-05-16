"""Tests for Coherence node"""
import pytest
from cognigraph.nodes.processors import Coherence
from cognigraph.nodes.sources import FileSource

from cognigraph.nodes.tests.prepare_tests_data import info, data_path  # noqa
import numpy as np


@pytest.fixture  # noqa
def coh_computer(info, data_path):  # noqa
    coh_computer = Coherence()
    coh_computer.mne_info = info
    N_SEN = len(info['ch_names'])
    coh_computer.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    coh_computer.parent = parent
    return coh_computer


def test_change_api_attributes(coh_computer):
    """Check if _on_critical_attr_change is defined"""
    coh_computer._on_critical_attr_change(None, None, None)
