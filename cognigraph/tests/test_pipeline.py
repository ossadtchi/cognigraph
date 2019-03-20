"""Tests for Pipeline class
author: dmalt
date: 2019-03-19

"""
import pytest
from cognigraph.pipeline import Pipeline
from cognigraph.nodes.node import SourceNode, ProcessorNode, OutputNode
from mne import create_info
import numpy as np
from numpy.testing import assert_array_equal


class ConcreteSource(SourceNode):
    """"Generates output updates of size (nchan, nsamp).
    Each element of output array on update is incremented by one
    starting with an array of all zeroes.

    """
    CHANGES_IN_THESE_REQUIRE_RESET = ()
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    def __init__(self, nsamp=50, nchan=32, sfreq=500):
        super().__init__()
        self.nsamp = nsamp  # number of time samples in one update
        self.nchan = nchan
        self.sfreq = 500
        self.n_updates = 0

    def _initialize(self):
        ch_names = [str(i).zfill(2) for i in range(self.nchan)]
        self.mne_info = create_info(ch_names, self.sfreq, ch_types='eeg')

    def _update(self):
        self.output = np.ones([self.nchan, self.nsamp]) * self.n_updates
        self.n_updates += 1

    def _check_value(self, key, value):
        pass


class ConcreteProcessor(ProcessorNode):
    """On each update increments input array by self.increment"""
    CHANGES_IN_THESE_REQUIRE_RESET = ()
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    def __init__(self, increment=0.1):
        super().__init__()
        self.increment = increment
        self.n_updates = 0

    def _update(self):
        self.output = (self.parent_node.output +
                       self.increment * (self.n_updates + 1))
        self.n_updates += 1

    def _initialize(self):
        pass

    def _check_value(self, key, value):
        pass


class ConcreteOutput(OutputNode):
    """On each update increments input array by self.increment"""
    CHANGES_IN_THESE_REQUIRE_RESET = ()
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    def __init__(self, increment=0.01):
        super().__init__()
        self.increment = increment
        self.n_updates = 0

    def _update(self):
        self.output = (self.parent_node.output +
                       self.increment * (self.n_updates + 1))
        self.n_updates += 1

    def _initialize(self):
        pass

    def _check_value(self, key, value):
        pass


@pytest.fixture(scope='function')
def pipeline():
    source = ConcreteSource()
    processor = ConcreteProcessor()
    output = ConcreteOutput()
    pipeline = Pipeline()
    pipeline.source = source
    pipeline.add_processor(processor)
    pipeline.add_output(output)
    return pipeline


def test_pipeline_initialization(pipeline):
    pipeline.initialize_all_nodes()
    assert(pipeline.source._initialized)
    assert(pipeline.source.mne_info is not None)
    assert(pipeline.source.mne_info['nchan'] == pipeline.source.nchan)
    assert(pipeline._processors[0]._initialized)
    assert(pipeline._outputs[0]._initialized)


def test_pipeline_update(pipeline):
    pipeline.initialize_all_nodes()
    pipeline.update_all_nodes()
    assert_array_equal(
        pipeline.source.output,
        np.zeros([pipeline.source.nchan, pipeline.source.nsamp]))
    assert_array_equal(
        pipeline._processors[0].output,
        np.zeros([pipeline.source.nchan, pipeline.source.nsamp]) +
        pipeline._processors[0].increment)
    assert_array_equal(
        pipeline._outputs[0].output,
        pipeline._processors[0].output +
        pipeline._outputs[0].increment)
    pipeline.update_all_nodes()
    assert_array_equal(
        pipeline.source.output,
        np.ones([pipeline.source.nchan, pipeline.source.nsamp]))
    assert_array_equal(
        pipeline._processors[0].output,
        np.ones([pipeline.source.nchan, pipeline.source.nsamp]) +
        pipeline._processors[0].increment * 2)
    assert_array_equal(
        pipeline._outputs[0].output,
        pipeline._processors[0].output +
        pipeline._outputs[0].increment * 2)
