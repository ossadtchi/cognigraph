import pytest
import os.path as op
from mne.io import Raw

from cognigraph import COGNIGRAPH_ROOT
from cognigraph.utils.io import DataDownloader
from cognigraph.utils.channels import capitalize_chnames
import logging

test_data_path = op.join(COGNIGRAPH_ROOT, "tests/data")


@pytest.fixture
def info(scope="session"):
    """Get info with applied average projection"""
    logging.basicConfig(filename=None, level=logging.INFO)
    dloader = DataDownloader()
    info_src_path = dloader.get_file("Koleno_raw.fif")
    raw = Raw(info_src_path, preload=True)
    raw.set_eeg_reference("average", projection=True)
    capitalize_chnames(raw.info)
    return raw.info


@pytest.fixture
def data_path(scope="session"):
    logging.basicConfig(filename=None, level=logging.INFO)
    dloader = DataDownloader()
    data_path = dloader.get_file("Koleno_raw.fif")
    return data_path


@pytest.fixture(scope="session")
def fwd_model_path():
    dloader = DataDownloader()
    return dloader.get_file("dmalt_custom_lr.fif")
