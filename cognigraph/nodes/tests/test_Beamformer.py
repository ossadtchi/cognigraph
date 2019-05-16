import pytest
from cognigraph.nodes.processors import Beamformer
from cognigraph.nodes.sources import FileSource
from cognigraph.nodes.tests.prepare_tests_data import (info,  # noqa
                                                       fwd_model_path,
                                                       data_path)
import numpy as np


@pytest.fixture(scope='function')  # noqa
def beamformer(info, fwd_model_path, data_path):  # noqa
    is_adaptive = True
    beamformer = Beamformer(forward_model_path=fwd_model_path,
                            is_adaptive=is_adaptive)
    beamformer.mne_info = info
    N_SEN = len(info['ch_names'])
    beamformer.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    beamformer.parent = parent
    return beamformer


@pytest.fixture  # noqa
def beamformer_default(info):  # noqa
    beamformer_default = Beamformer()
    parent = FileSource()
    parent.mne_info = info
    parent.output = np.random.rand(info['nchan'], 1)
    beamformer_default.parent = parent
    return beamformer_default


def test_defaults(beamformer_default):
    assert beamformer_default.mne_forward_model_file_path is None
    assert beamformer_default.mne_info is None


def test_initialize(beamformer):
    beamformer.initialize()
    assert hasattr(beamformer, '_filters')
    assert beamformer.mne_info is not None


def test_reg_change(beamformer):
    """
    Change regulariation parameter and see if filters changed but
    covariance matrix didn't reset to default

    """
    beamformer.initialize()
    # -------- modify covariance so it's not equal to inital -------- #
    nchans = beamformer._mne_info['nchan']
    ntimes = 100
    beamformer._update_covariance_matrix(np.random.rand(nchans, ntimes))
    # --------------------------------------------------------------- #
    data_cov_old = beamformer._data_cov.data
    filters_old = beamformer._filters.copy()

    beamformer.reg = 5

    assert not np.array_equal(filters_old, beamformer._filters)
    assert np.array_equal(beamformer._data_cov.data, data_cov_old)


def test_adaptiveness_change(beamformer):
    """
    Change is_adaptive and see if reinitialization happens

    """
    beamformer.is_adaptive = True
    beamformer.initialize()

    data_cov_init = beamformer._data_cov.data

    # -------- modify covariance so it's not equal to inital -------- #
    nchans = beamformer._mne_info['nchan']
    ntimes = 100
    beamformer._update_covariance_matrix(np.random.rand(nchans, ntimes))
    # --------------------------------------------------------------- #

    filters = beamformer._filters.copy()
    beamformer.is_adaptive = False
    assert not np.array_equal(filters, beamformer._filters)
    assert np.array_equal(beamformer._data_cov.data, data_cov_init)


def test_input_hist_inval_triggers_reinit_for_adaptive_beamformer(beamformer):
    beamformer.parent.initialize()
    beamformer.initialize()

    data_cov_init = beamformer._data_cov.data
    # -------- modify covariance so it's not equal to inital -------- #
    nchans = beamformer._mne_info['nchan']
    ntimes = 100
    beamformer._update_covariance_matrix(np.random.rand(nchans, ntimes))
    # --------------------------------------------------------------- #
    filters_old = beamformer._filters.copy()
    beamformer._filters = None  # mess up the filters

    beamformer.parent.source_name = 'new_name'  # triggers reset for source
    assert not np.array_equal(filters_old, beamformer._filters)
    assert np.array_equal(beamformer._data_cov.data, data_cov_init)


def test_update(beamformer):
    beamformer._initialize()
    beamformer._update()


def test_check_value(beamformer):
    with pytest.raises(ValueError):
        beamformer.reg = -1
