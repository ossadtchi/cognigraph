import numpy as np

import pytest
from cognigraph.nodes.processors import MNE
from cognigraph.nodes.sources import FileSource
from cognigraph.nodes.tests.prepare_tests_data import (info,  # noqa
                                                       fwd_model_path,
                                                       data_path)


@pytest.fixture
def inv_model(info, fwd_model_path, data_path):  # noqa
    snr = 1
    method = 'MNE'
    inv_model = MNE(
        snr=snr, forward_model_path=fwd_model_path, method=method)
    inv_model.mne_info = info
    N_SEN = len(info['ch_names'])
    inv_model.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    inv_model.parent = parent
    return inv_model


@pytest.fixture
def inv_model_def(info):  # noqa
    inv_model_def = MNE()
    parent = FileSource()
    parent.mne_info = info
    parent.output = np.random.rand(info['nchan'], 1)
    inv_model_def.parent = parent
    return inv_model_def


def test_defaults(inv_model_def):
    assert(inv_model_def.mne_forward_model_file_path is None)
    assert(inv_model_def.mne_info is None)


def test_initialize(inv_model):
    inv_model.initialize()


def test_change_api_attributes(inv_model):
    inv_model.initialize()
    l2_old = inv_model._lambda2
    snr_old = inv_model.snr

    arbitrary_value = 1
    inv_model.snr += arbitrary_value

    assert l2_old != inv_model._lambda2
    assert inv_model._lambda2 == 1 / (snr_old + arbitrary_value) ** 2


def test_input_hist_invalidation_defined(inv_model):
    """
    Change source attribute which triggers on_upstream_change and see if
    inv_model fails

    """
    inv_model.parent.initialize()
    inv_model.initialize()

    inv_model.parent.source_name = 'new_name'  # triggers reset for source


def test_update(inv_model):
    inv_model._initialize()
    inv_model._update()


def test_check_value(inv_model):
    with pytest.raises(ValueError):
        inv_model.snr = -1
