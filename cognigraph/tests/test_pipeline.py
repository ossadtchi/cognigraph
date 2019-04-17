"""Tests for Pipeline class"""
import pytest
from cognigraph.nodes.pipeline import Pipeline
import numpy as np
from numpy.testing import assert_array_equal
from cognigraph.tests.prepare_pipeline_tests import (create_dummy_info,
                                                     ConcreteSource,
                                                     ConcreteProcessor,
                                                     ConcreteOutput)


@pytest.fixture(scope='function')
def pipeline():
    source = ConcreteSource()
    processor = ConcreteProcessor()
    output = ConcreteOutput()
    pipeline = Pipeline()
    pipeline.add_child(source)
    source.add_child(processor)
    processor.add_child(output)
    return pipeline


def test_pipeline_initialization(pipeline):
    pipeline.chain_initialize()
    source = pipeline._children[0]
    processor = source._children[0]
    output = processor._children[0]
    assert(source._initialized)
    assert(source.mne_info is not None)
    assert(source.mne_info['nchan'] == source.nchan)
    assert(processor._initialized)
    assert(output._initialized)


def test_pipeline_update(pipeline):
    """Update all pipeline nodes twice and check outputs"""
    pipeline.chain_initialize()
    src = pipeline._children[0]
    proc = src._children[0]
    out = proc._children[0]

    nch = src.nchan
    nsamp = src.nsamp
    pr_inc = proc.increment
    out_inc = out.increment

    pipeline.update()

    assert_array_equal(src.output, np.zeros([nch, nsamp]))
    assert_array_equal(proc.output, np.zeros([nch, nsamp]) + pr_inc)
    assert_array_equal(out.output, proc.output + out_inc)

    pipeline.update()

    assert_array_equal(src.output, np.ones([nch, nsamp]))
    assert_array_equal(proc.output, np.ones([nch, nsamp]) + pr_inc * 2)
    assert_array_equal(out.output, proc.output + out_inc * 2)


def test_pipeline_initialization_simplified(pipeline):
    src = pipeline._children[0]
    proc = src._children[0]
    out = proc._children[0]

    pipeline.chain_initialize()
    pipeline.update()
    new_nchan = 43
    new_info = create_dummy_info(nchan=new_nchan)

    src._mne_info = new_info
    for i in range(3):
        pipeline.update()
    pipeline.update()
    assert(np.all(out.output))
    assert(out.output.shape[0] == new_nchan)


def test_add_child_on_the_fly(pipeline):
    src = pipeline._children[0]
    pipeline.chain_initialize()
    pipeline.update()
    new_processor = ConcreteProcessor(increment=0.2)
    src.add_child(new_processor, initialize=True)
    pipeline.update()

    nch = src.nchan
    nsamp = src.nsamp
    assert_array_equal(new_processor.output, np.ones([nch, nsamp]) +
                       new_processor.increment)
    assert(new_processor._root is pipeline)
