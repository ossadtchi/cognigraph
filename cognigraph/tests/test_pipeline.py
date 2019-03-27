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


def create_dummy_info(nchan=32, sfreq=500):
    ch_names = [str(i).zfill(2) for i in range(nchan)]
    info = create_info(ch_names, sfreq, ch_types='eeg')
    return info


def exec_counter_node(cls):
    class Proxy(cls):
        def __init__(self, *pargs, **kwargs):
            cls.__init__(self, *pargs, **kwargs)
            self.n_updates = 0
            self.n_initializations = 0
            self.n_resets = 0
            self.n_messages = 0

        def _initialize(self):
            cls._initialize(self)
            self.n_initializations += 1

        def _update(self):
            cls._update(self)
            self.n_updates += 1

        def _reset(self):
            res = cls._reset(self)
            self.n_resets += 1
            return res

        def receive_a_message(self, message):
            cls.receive_a_message(self, message)
            self.n_messages += 1

    return Proxy


@exec_counter_node
class ConcreteSource(SourceNode):
    """"Generates output updates of size (nchan, nsamp).
    Each element of output array on update is incremented by one
    starting with an array of all zeroes.

    """
    CHANGES_IN_THESE_REQUIRE_RESET = ('_mne_info',)
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    def __init__(self, nsamp=50):
        SourceNode.__init__(self)
        self.nsamp = nsamp  # number of time samples in one update
        self._mne_info = create_dummy_info()
        self.nchan = self._mne_info['nchan']

    def _initialize(self):
        self.nchan = self._mne_info['nchan']
        self.mne_info = self._mne_info

    def _update(self):
        self.output = np.ones([self.nchan, self.nsamp]) * self.n_updates

    def _reset(self):
        self._should_reinitialize = True
        self.initialize()
        return False

    def _check_value(self, key, value):
        pass

    def receive_a_message(self, message):
        SourceNode.receive_a_message(self, message)

    def _on_input_history_invalidation(self):
        pass


@exec_counter_node
class ConcreteProcessor(ProcessorNode):
    """On each update increments input array by self.increment"""
    CHANGES_IN_THESE_REQUIRE_RESET = ()
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('_mne_info', )

    def __init__(self, increment=0.1):
        super().__init__()
        self.increment = increment

    def _update(self):
        self.output = (self.parent.output +
                       self.increment * (self.n_updates + 1))

    def _initialize(self):
        pass

    def _check_value(self, key, value):
        pass

    def _reset(self):
        return True

    def receive_a_message(self, message):
        super().receive_a_message(message)

    def _on_input_history_invalidation(self):
        pass


@exec_counter_node
class ConcreteOutput(OutputNode):
    """On each update increments input array by self.increment"""
    CHANGES_IN_THESE_REQUIRE_RESET = ()
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('_mne_info',)

    def __init__(self, increment=0.01):
        OutputNode.__init__(self)
        self.increment = increment

    def _update(self):
        self.output = (self.parent.output +
                       self.increment * (self.n_updates + 1))

    def _initialize(self):
        pass

    def _check_value(self, key, value):
        pass

    def receive_a_message(self, message):
        super().receive_a_message(message)

    def _reset(self):
        pass

    def _on_input_history_invalidation(self):
        pass


@pytest.fixture(scope='function')
def pipeline():
    source = ConcreteSource()
    processor = ConcreteProcessor()
    output = ConcreteOutput()
    pipeline = Pipeline()
    pipeline.source = source
    source.add_child(processor)
    processor.add_child(output)
    pipeline._processors.append(processor)
    pipeline._outputs.append(output)
    return pipeline


def test_pipeline_initialization(pipeline):
    pipeline.initialize_all_nodes()
    assert(pipeline.source._initialized)
    assert(pipeline.source.mne_info is not None)
    assert(pipeline.source.mne_info['nchan'] == pipeline.source.nchan)
    assert(pipeline._processors[0]._initialized)
    assert(pipeline._outputs[0]._initialized)


def test_pipeline_update(pipeline):
    """Update all pipeline nodes twice and check outputs"""
    pipeline.initialize_all_nodes()
    nch = pipeline.source.nchan
    nsamp = pipeline.source.nsamp
    pr_inc = pipeline._processors[0].increment
    out_inc = pipeline._outputs[0].increment
    proc = pipeline._processors[0]
    out = pipeline._outputs[0]
    src = pipeline.source

    pipeline.update_all_nodes()

    assert_array_equal(src.output, np.zeros([nch, nsamp]))
    assert_array_equal(proc.output, np.zeros([nch, nsamp]) + pr_inc)
    assert_array_equal(out.output, proc.output + out_inc)

    pipeline.update_all_nodes()

    assert_array_equal(src.output, np.ones([nch, nsamp]))
    assert_array_equal(proc.output, np.ones([nch, nsamp]) + pr_inc * 2)
    assert_array_equal(out.output, proc.output + out_inc * 2)


def test_pipeline_initialization_simplified(pipeline):
    pipeline.initialize_all_nodes()
    pipeline.update_all_nodes()
    new_nchan = 43
    new_info = create_dummy_info(nchan=new_nchan)

    pipeline.source._mne_info = new_info
    for i in range(3):
        pipeline.update_all_nodes()
    # assert_array_equal(pipeline._outputs[0].output, None)
    pipeline.update_all_nodes()
    assert(np.all(pipeline._outputs[0].output))
    assert(pipeline._outputs[0].output.shape[0] == new_nchan)


def test_add_child_on_the_fly(pipeline):
    pipeline.initialize_all_nodes()
    pipeline.update_all_nodes()
    new_processor = ConcreteProcessor(increment=0.2)
    pipeline.source.add_child(new_processor, initialize=True)
    pipeline.update_all_nodes()

    nch = pipeline.source.nchan
    nsamp = pipeline.source.nsamp
    assert_array_equal(new_processor.output, np.ones([nch, nsamp]) +
                       new_processor.increment)
    assert(new_processor._root is pipeline.source)


# def test_pipeline_reintitalization(pipeline):
#     """Check if changing critical attribute resets downstream nodes"""
#     pipeline.initialize_all_nodes()

#     nsamp = pipeline.source.nsamp
#     proc = pipeline._processors[0]
#     out = pipeline._outputs[0]
#     src = pipeline.source

#     out_inc = pipeline._outputs[0].increment
#     pr_inc = pipeline._processors[0].increment

#     pipeline.update_all_nodes()
#     new_info = create_dummy_info(nchan=33)

#     # change attribute from CHANGES_IN_THESE_REQUIRE_RESET for source node:
#     pipeline.source._mne_info = new_info
#     """
#     When attribute from CHANGES_IN_THESE_REQUIRE_RESET is changed
#     the following chain of events takes place:
#     1. __setattr__() sets _should_reset and there_has_been_a_change to True
#     2. on the next update _no_pending_changes is checked. This flag
#        is modified by _should_reset and evaluates to False; therefore
#        elif branch with _update() is skipped and we fall into the last branch.
#        output attribute which is set to None at the begining of update()
#        stays intact.
#     3. In the last (else) branch _should_reset flag is checked, triggering
#        a call to reset().
#     4. reset() calls _reset(); return value of the latter
#        modifies output_history_is_no_longer_valid flag; _should_reset is
#        set back to False. Finally, a message with
#        there_has_been_a_change=True and obtained value for
#        output_history_is_no_longer_valid is delivered to all the immediate
#        children of this node.
#     5. receive_a_message() in children nodes sets
#        _there_has_been_an_upstream_change (to True) and
#        _input_history_is_no_longer_valid
#        (to output_history_is_no_longer_valid)
#     6. All processor children set their output to None and don't call
#        Node.update() which means they don't get updated on the current
#        update iteration. This behaviour is chained down the pipeline.
#        All downstream output nodes don't change their output and also
#        don't call Node.update().
#     7. On the next pipeline update for immediate children nodes
#        _there_has_been_an_upstream_change flag is checked and
#        _the_change_requires_reinitialization() is called; its return value
#        sets _should_reinitialize flag.
#     8. _the_change_requires_reinitialization() checks for critical
#        upstream changes in items from
#        UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION tuple.
#        if any were found, it returns True (which sets _should_reinitialize).
#     9. _no_pending_changes for children nodes is checked. It evaluates to
#        False if _should_reinitialize is True
#        OR _input_history_is_no_longer_valid is True, in which case the call
#        to _update() is skipped.
#     10. If _should_reinitialize is True, a call to initialize() is made.
#        10a. _saved_from_upstream dictionary is updated with the new values
#            of critical parameters.
#        10b. _initalize() method is called and _initalized flag is set to True
#             _no_pending_changes is set to True (which sets
#             _should_reinitialize, _should_reset and
#             _input_history_is_no_longer_valid to False) and
#             _there_has_been_an_upstream_change to False.
#        10c. A message with there_has_been_a_change=True and
#             output_history_is_no_longer_valid=True is emitted.
#     11. Elif _input_history_is_no_longer_valid is True and _should_reset
#         is False a call to on_input_history_invalidation() is made.
#         11a. _on_input_history_invalidation() is called;
#              _input_history_is_no_longer_valid is set back to False;
#         11b. A message with there_has_been_a_change=True and
#              output_history_is_no_longer_valid=True is delivered to the
#              immediate children.
#     """

#     assert(pipeline.source.n_initializations == 1)
#     # Next pipeline update triggers reset (which calls initialize) for source.
#     # Actual nodes update doesn't happen.
#     out_output_prev = out.output
#     pipeline.update_all_nodes()
#     assert(pipeline.source.n_initializations == 2)
#     assert_array_equal(src.output, None)
#     assert_array_equal(proc.output, None)
#     assert_array_equal(out.output, out_output_prev)

#     # Next, update triggers reinitialization for processor node which skips
#     # update. Therefore processor and output nodes are not updated again, but
#     # the source node does get updated and the number of channels is changed.
#     pipeline.update_all_nodes()
#     assert(pipeline._processors[0].n_initializations == 2)
#     assert_array_equal(src.output, np.ones([33, nsamp]))
#     assert_array_equal(proc.output, None)
#     assert_array_equal(out.output, out_output_prev)

#     # Finally, the output node is reinitialized while source and processor are
#     # updated
#     pipeline.update_all_nodes()
#     assert(pipeline._outputs[0].n_initializations == 2)
#     assert_array_equal(out.output, None)

#     pipeline.update_all_nodes()
#     assert_array_equal(out.output, src.output + pr_inc * 3 + out_inc * 2)
#     assert(np.all(out.output))

