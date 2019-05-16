"""Tests for MneGcs node"""
import numpy as np

import pytest
from cognigraph.nodes.processors import MneGcs
from cognigraph.nodes.sources import FileSource
from cognigraph.nodes.tests.prepare_tests_data import (info,  # noqa
                                                       fwd_model_path,
                                                       data_path)


@pytest.fixture
def mne_gcs(info, fwd_model_path, data_path):  # noqa
    snr = 1
    mne_gcs = MneGcs(snr=snr, forward_model_path=fwd_model_path, seed=0)
    mne_gcs.mne_info = info
    N_SEN = len(info['ch_names'])
    mne_gcs.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    mne_gcs.parent = parent
    return mne_gcs


def test_initialize(mne_gcs):
    mne_gcs.initialize()


def test_change_api_attributes(mne_gcs):
    mne_gcs.initialize()
    l2_old = mne_gcs._lambda2
    snr_old = mne_gcs.snr

    arbitrary_value = 1
    mne_gcs.snr += arbitrary_value

    assert l2_old != mne_gcs._lambda2
    assert mne_gcs._lambda2 == 1 / (snr_old + arbitrary_value) ** 2


def test_input_hist_invalidation_defined(mne_gcs):
    """
    Change source attribute which triggers on_upstream_change and see if
    mne_gcs fails

    """
    mne_gcs.parent.initialize()
    mne_gcs.initialize()

    mne_gcs.parent.source_name = 'new_name'  # triggers reset for source


def test_update(mne_gcs):
    mne_gcs._initialize()
    mne_gcs._update()


def test_check_value(mne_gcs):
    with pytest.raises(ValueError):
        mne_gcs.snr = -1
