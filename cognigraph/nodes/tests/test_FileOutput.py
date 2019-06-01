"""Tests for FileOutput node"""
import pytest
from cognigraph.nodes.outputs import FileOutput
from cognigraph.nodes.sources import FileSource

from cognigraph.nodes.tests.prepare_tests_data import info, data_path  # noqa
import numpy as np
import os.path as op


@pytest.fixture(scope='function')  # noqa
def file_outputter(info, data_path, tmp_path):  # noqa
    output_path = op.join(tmp_path, 'output.h5')
    file_outputter = FileOutput(output_path)
    file_outputter.mne_info = info
    N_SEN = len(info['ch_names'])
    file_outputter.input = np.random.rand(N_SEN)
    parent = FileSource(data_path)
    parent.output = np.random.rand(info['nchan'], 1)
    parent.mne_info = info
    file_outputter.parent = parent
    return file_outputter


def test_change_api_attributes(file_outputter, tmp_path):
    """
    Change output_fname and check if initialize resets out_file.

    """
    arbitrary_name = op.join(tmp_path, 'somename.h5')

    file_outputter.initialize()
    out_file_ini = file_outputter._out_file
    file_outputter.output_fname = arbitrary_name
    file_outputter.update()

    assert file_outputter._out_file is not out_file_ini


def test_input_hist_invalidation_resets_statistics(file_outputter):
    """Check that upstream history change doesn't break the node"""
    file_outputter.parent.initialize()
    file_outputter.initialize()

    file_outputter.parent.source_name = 'new_name'  # triggers reset for source
