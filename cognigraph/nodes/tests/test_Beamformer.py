import pytest
from cognigraph.nodes.processors import Beamformer
from cognigraph.nodes.sources import FileSource
from cognigraph import COGNIGRAPH_ROOT
from cognigraph.nodes.tests.prepare_inv_tests_data import info  # noqa
import os.path as op
import numpy as np

test_data_path = op.join(COGNIGRAPH_ROOT,  'tests/data')


@pytest.fixture  # noqa
def beamformer(info):
    fwd_model_path = op.join(test_data_path, 'dmalt_custom_lr-fwd.fif')
    is_adaptive = True
    beamformer = Beamformer(forward_model_path=fwd_model_path,
                            is_adaptive=is_adaptive)
    beamformer.mne_info = info
    N_SEN = len(info['ch_names'])
    beamformer.input = np.random.rand(N_SEN)
    parent = FileSource()
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    beamformer.parent = parent
    return beamformer


@pytest.fixture  # noqa
def beamformer_def(info):
    beamformer_def = Beamformer()
    parent = FileSource()
    parent.mne_info = info
    parent.output = np.random.rand(info['nchan'], 1)
    beamformer_def.parent = parent
    return beamformer_def


def test_defaults(beamformer_def):
    assert(beamformer_def.mne_forward_model_file_path is None)
    assert(beamformer_def.mne_info is None)


def test_initialize(beamformer):
    beamformer.initialize()


def test_reset(beamformer):
    out_hist = beamformer._reset()
    assert(out_hist is True)


def test_update(beamformer):
    beamformer._initialize()
    beamformer._update()


def test_check_value(beamformer):
    with pytest.raises(ValueError):
        beamformer.reg = -1
