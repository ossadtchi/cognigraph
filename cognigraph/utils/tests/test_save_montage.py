import pytest
from mne.channels import read_montage
from cognigraph.utils.channels import save_montage
import os.path as op
import numpy as np
from pathlib import Path


def test_doesnt_change_elc_montages(tmp_path):
    montage_orig = read_montage('standard_1005')
    save_montage(montage_orig, tmp_path)
    saved_path = op.join(tmp_path, 'standard_1005.elc')
    montage_saved = read_montage(saved_path)

    assert montage_orig.ch_names == montage_saved.ch_names
    assert np.allclose(montage_orig.pos, montage_saved.pos)
    assert np.allclose(montage_orig.lpa, montage_saved.lpa)
    assert np.allclose(montage_orig.rpa, montage_saved.rpa)
    assert np.allclose(montage_orig.nasion, montage_saved.nasion)
    assert montage_orig.kind == montage_saved.kind


def test_raises_FileExistsError(tmp_path):
    saved_path = op.join(tmp_path, 'standard_1005.elc')
    Path(saved_path).touch()
    with pytest.raises(FileExistsError):
        montage_orig = read_montage('standard_1005')
        save_montage(montage_orig, tmp_path)


def test_overwrite(tmp_path):
    saved_path = op.join(tmp_path, 'GSN-HydroCel-128.elc')
    Path(saved_path).touch()
    montage_orig = read_montage('GSN-HydroCel-128')
    save_montage(montage_orig, tmp_path, overwrite=True)
    montage_saved = read_montage(saved_path)

    scale = 1000  # in when loading elc mne-python converts to meters

    # Channels names should be good up to fiducials
    assert montage_orig.ch_names[3:] == montage_saved.ch_names[3:]

    assert np.allclose(montage_orig.pos, montage_saved.pos * scale)
    assert np.allclose(montage_orig.lpa, montage_saved.lpa * scale)
    assert np.allclose(montage_orig.rpa, montage_saved.rpa * scale)
    assert np.allclose(montage_orig.nasion, montage_saved.nasion * scale)
    assert montage_orig.kind == montage_saved.kind
