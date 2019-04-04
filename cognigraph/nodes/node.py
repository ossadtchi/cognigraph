import time
from typing import Tuple
from contextlib import contextmanager

import numpy as np
from mne.io.pick import channel_type

from ..utils.misc import class_name_of
import logging


class Node(object):
    """
    Any processing step (including getting and outputting data)
    is an instance of this class.
    This is an abstract class.

    """
    # Some upstream properties are mutable and thus saving them would not work
    # since update in upstrem will update a local copy as well.  Keeping a
    # local copy would trigger unnecessary reinitializations when something
    # minor has changed (i.e. for mne_info we want to reinitialize when channel
    # names or their number has changed but no reinitialization is required if
    # only the sampling rate has changed). Thus, any concrete subclass has to
    # provide a way to save only what is necessary. Keys are property names,
    # values are functions that return an appropriate tuple.

    @property
    def CHANGES_IN_THESE_REQUIRE_RESET(self) -> Tuple[str]:
        """
        A constant tuple of attributes after a change in
        which a reset should be scheduled.

        """
        msg = ('Each subclass of Node must have a '
               'CHANGES_IN_THESE_REQUIRE_RESET constant defined')
        raise NotImplementedError(msg)

    @property
    def UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION(self) -> Tuple[str]:
        """ A constant tuple of attributes after an *upstream* change in which
        an initialization should be scheduled.
        Determines what gets into self._saved_from_upstream dictionary.

        """
        msg = ('Each subclass of Node must have a '
               'UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION'
               ' constant defined')
        raise NotImplementedError(msg)

    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = dict()

    def __init__(self):
        self.initialized = False

        self._parent = None  # type: Node
        self._children = []
        self._root = self
        self.output = None  # type: np.ndarray

        self._saved_from_upstream = None  # type: dict
        self.logger = logging.getLogger(type(self).__name__)

    def __repr__(self):
        return str(self.__class__.__name__)

    def initialize(self):

        self._saved_from_upstream = {
            item: self.traverse_back_and_find(item)
            if item not in self.SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS
            else self.SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS[item](
                self.traverse_back_and_find(item))
            for item in self.UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION
        }

        with self.not_triggering_reset():
            t1 = time.time()
            self.logger.info(
                'Initializing the {} node'.format(class_name_of(self)))
            self._initialize()
            t2 = time.time()
            self.logger.info(
                'Finish initialization in {:.1f} ms'.format((t2 - t1) * 1000))
            self.initialized = True

            # Set all the resetting flags to false

    def chain_initialize(self):
        self.initialize()
        for child in self._children:
            child.chain_initialize()

    def _initialize(self):
        """
        Prepares everything for the first update.
        If called again, should remove all the traces from the past

        """
        raise NotImplementedError('_initialize should be implemented')

    def update(self) -> None:
        t1 = time.time()
        self.output = None  # Reset output in case update does not succeed
        self._update()

        t2 = time.time()
        self.logger.debug('Updated in {:.1f} ms'.format((t2 - t1) * 1000))

        for child in self._children:
            child.update()

    def _update(self):
        raise NotImplementedError('_update should be implemented')

    def reset(self, is_input_hist_invalid, is_local_attr_changed=False):
        """Take care of reinitialization and parameters reset"""
        if self._is_critical_upstream_change():
            self.initialize()
            is_output_hist_invalid = True
        elif is_local_attr_changed:  # local attribute change
            with self.not_triggering_reset():
                self.logger.info(
                    'Resetting the {} node '.format(class_name_of(self)) +
                    'because of attribute changes')
                is_output_hist_invalid = self._reset()
        else:
            is_output_hist_invalid = False

        if is_output_hist_invalid:
            self._on_input_history_invalidation()

        for child in self._children:
            child.reset(is_output_hist_invalid)

    def _reset(self) -> bool:
        """
        Does what needs to be done when one of the self.
        CHANGES_IN_THESE_REQUIRE_RESET has been changed
        Must return whether output history is no longer valid.
        True if descendants should forget about anything that
        has happened before, False if changes are strictly local.

        """
        raise NotImplementedError('_reset should be implemented')

    def add_child(self, child, initialize=False):
        """Add child node to nodes tree"""
        child.parent = self
        if initialize:
            child.initialize()
        # append to _children list is handled by the parent property setter

    def remove_child(self, child):
        """Remove child node from nodes tree"""
        child.parent = None  # the rest is handled by parent prop. setter

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, new_parent):
        if self._parent is new_parent:  # covers the case when both are None
            return

        # Tell the previous parent about disconnection
        if self._parent is not None:
            self._parent._children.remove(self)

        self._parent = new_parent

        # Tell the new input node about the connection
        if new_parent is not None:
            new_parent._children.append(self)
            self._root = new_parent._root
        else:
            self._root = self

    def _on_input_history_invalidation(self):
        """
        If the node state is dependent on previous inputs,
        reset whatever relies on them.

        """
        with self.not_triggering_reset():
            self.logger.info(
                'Resetting the {} node '.format(class_name_of(self)) +
                'because history is no longer valid')
            self._on_input_history_invalidation()

    def traverse_back_and_find(self, item: str):
        """
        This function will walk up the node tree until
        it finds a node with an attribute <item>

        """
        try:
            return getattr(self.parent, item)
        except AttributeError:
            try:
                return self.parent.traverse_back_and_find(item)
            except AttributeError:
                msg = ('None of the predecessors of a '
                       '{} node contains attribute {}'.format(
                           class_name_of(self), item))
                raise AttributeError(msg)

    # Trigger resetting chain if the change in the attribute needs it
    def __setattr__(self, key, value):
        self._check_value(key, value)
        object.__setattr__(self, key, value)
        if self.initialized:
            if key in self.CHANGES_IN_THESE_REQUIRE_RESET:
                object.__setattr__(self, '_should_reset', True)
                object.__setattr__(self, 'there_has_been_a_change', True)
                self.reset(is_input_hist_invalid=False,
                           is_local_attr_changed=True)

    @property
    def initialized(self):
        try:
            return self._initialized
        except AttributeError:
            return False

    @initialized.setter
    def initialized(self, value):
        object.__setattr__(self, '_initialized', value)

    @contextmanager
    def not_triggering_reset(self):
        """
        Change of attributes CHANGES_IN_THESE_REQUIRE_RESET
        should trigger reset() but not from within the class.
        Use this context manager to suspend reset() triggering.

        """
        backup, self.CHANGES_IN_THESE_REQUIRE_RESET =\
            self.CHANGES_IN_THESE_REQUIRE_RESET, ()
        try:
            yield
        finally:
            self.CHANGES_IN_THESE_REQUIRE_RESET = backup

    def _check_value(self, key, value):
        raise NotImplementedError('_check_value should be implemented')

    def _is_critical_upstream_change(self):
        """
        Checks if anything important changed upstream wrt
        value captured at initialization

        """
        for item, saved_value in self._saved_from_upstream.items():

            current_value = self.traverse_back_and_find(item)
            # Mutable objects are handled separately.
            # Nodes have to define a function to save only what is needed.
            if item in self.SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS:
                saver_function = self.SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS[item]
                current_value = saver_function(current_value)

            try:
                if saved_value != current_value:
                    return True
            except ValueError as e:
                exception_message = (
                        'There was a problem comparing {item} '
                        'property upstream from a {class_name} node.\n'
                        'If {value_type} is a mutable type, '
                        'then add a function to save smth immutable '
                        'from it to the SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS '
                        'dictionary property of '
                        '{class_name}'.format(
                            item=item, class_name=class_name_of(self),
                            value_type=type(current_value)))
                raise Exception(exception_message) from e

        return False  # Nothing has changed


class SourceNode(Node):
    """Objects of this class read data from a source"""

    # There is no 'upstream' for the sources
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    def __init__(self):
        Node.__init__(self)
        self.mne_info = None

    def initialize(self):
        self.mne_info = None
        Node.initialize(self)
        try:
            self._check_mne_info()
        except ValueError as e:
            self.initialized = False
            raise e

    def _check_mne_info(self):
        class_name = class_name_of(self)
        error_hint = ' Check the initialize() method'

        if self.mne_info is None:
            raise ValueError('{} node has empty mne_info '
                             'attribute.'.format(class_name) + error_hint)

        channel_count = len(self.mne_info['chs'])
        if len(self.mne_info['chs']) == 0:
            raise ValueError('{} node has 0 channels in its mne_info '
                             'attribute.'.format(class_name) + error_hint)

        channel_types = {
            channel_type(self.mne_info, i) for i in np.arange(channel_count)}
        required_channel_types = {'grad', 'mag', 'eeg'}
        if len(channel_types.intersection(required_channel_types)) == 0:
            raise ValueError('{} has no channels of types {}'.format(
                class_name, required_channel_types) + error_hint)

        try:
            self.mne_info._check_consistency()
        except RuntimeError as e:
            exception_message = ('The mne_info attribute of {} node is not '
                                 'self-consistent'.format(class_name_of(self)))
            raise Exception(exception_message) from e

    def _reset(self):
        # There is nothing to reset. Just go ahead and initialize
        self._should_reinitialize = True
        self.initialize()
        is_output_hist_invalid = True
        return is_output_hist_invalid

    def _on_input_history_invalidation(self):
        raise NotImplementedError
        # super()._on_input_history_invalidation()


class ProcessorNode(Node):
    """
    Still an abstract class.
    Initially existed for clarity of inheritance only.
    Now handles empty inputs.

    """
    def __init__(self):
        Node.__init__(self)
        with self.not_triggering_reset():
            self.disabled = False

    def update(self):
        if self.disabled is True:
            self.output = self.parent.output
            return
        if (self.parent.output is None or
                self.parent.output.size == 0):
            self.output = None
            return
        else:
            Node.update(self)


class OutputNode(Node):
    """
    Still an abstract class.
    Initially existed for clarity of inheritance only.
    Now handles empty inputs.

    """
    def update(self):
        if (self.parent.output is None or
                self.parent.output.size == 0):
            return
        else:
            Node.update(self)
