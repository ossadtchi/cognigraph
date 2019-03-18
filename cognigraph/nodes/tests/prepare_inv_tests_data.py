import pytest
import os.path as op
from mne.io import Raw

from cognigraph import COGNIGRAPH_ROOT

test_data_path = op.join(COGNIGRAPH_ROOT, 'tests/data')


@pytest.fixture
def info(scope='session'):
    """Get info with applied average projection"""
    info_src_path = op.join(test_data_path, 'Koleno_raw.fif')
    raw = Raw(info_src_path, preload=True)
    raw.set_eeg_reference('average', projection=True)
    return raw.info
