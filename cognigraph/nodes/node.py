import time
from typing import Tuple
from contextlib import contextmanager

import numpy as np
from mne.io.pick import channel_type

from ..utils.misc import class_name_of
import logging
import re


class _ReprMeta(type):
    """
    Node classes representation and printing logic for better look in GUI

    """
    def __repr__(cls):
        """
        Either use defined in class string or convert class name
        from CamelCase to spaces

        """
        if cls._GUI_STRING:
            return cls._GUI_STRING
        else:
            camel_to_spaces = re.sub(r'([A-Z]*)([A-Z])([a-z])',
                                     r'\1 \2\3', cls.__name__)
            return camel_to_spaces.lstrip()

    def __str__(cls):
        return '<' + _ReprMeta.__repr__(cls) + '>'


class Node(object, metaclass=_ReprMeta):
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

    _GUI_STRING = None

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

    @property
    def ALLOWED_CHILDREN(self) -> Tuple[str]:
        """Nodes that can be connected to the current node"""
        return tuple()

    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = dict()

    def __init__(self):
        self.initialized = False  # see __setattr__

        self._parent = None  # type: Node
        self._children = []
        self._root = self
        self.output = None  # type: np.ndarray
        self._viz_type = None

        self._saved_from_upstream = None  # type: dict
        self._logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self):
        return repr(self.__class__) + ' Node'

    def __str__(self):
        class_str = str(self.__class__)
        return class_str[:-1] + ' Node' + class_str[-1]

    def __iter__(self):
        yield self
        for child in self._children:
            yield from child

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
            self._logger.info(
                'Initializing the {} node'.format(class_name_of(self)))
            self._initialize()
            t2 = time.time()
            self._logger.info(
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
        self._logger.debug('Updated in {:.1f} ms'.format((t2 - t1) * 1000))

        for child in self._children:
            child.update()

    def _update(self):
        raise NotImplementedError('_update should be implemented')

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        """
        Does what needs to be done when one of the
        self.CHANGES_IN_THESE_REQUIRE_RESET has been changed
        Must return whether output history is no longer valid.
        True if descendants should forget about anything that
        has happened before, False if changes are strictly local.

        """
        raise NotImplementedError(
            '_on_critical_attr_change should be implemented')

    def on_upstream_change(self, is_input_hist_invalid):
        is_output_hist_invalid = is_input_hist_invalid

        if is_input_hist_invalid:
            self.on_input_history_invalidation()

        if self._is_critical_upstream_change():
            self._logger.info('Reinitializing %s' % str(self) +
                              ' due to critical upstream attribute changes')
            self.initialize()
            is_output_hist_invalid = True
        for child in self._children:
            child.on_upstream_change(is_output_hist_invalid)

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

    def on_input_history_invalidation(self):
        """
        If the node state is dependent on previous inputs,
        reset whatever relies on them.

        """
        with self.not_triggering_reset():
            self._logger.info(
                'Resetting history-dependent attributes for %s ' % str(self))
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
    def __setattr__(self, key, new_value):
        self._check_value(key, new_value)
        try:
            old_value = self.__dict__[key]
        except KeyError:  # setting attribute anew
            old_value = None
        object.__setattr__(self, key, new_value)
        if self.initialized:
            if key in self.CHANGES_IN_THESE_REQUIRE_RESET:
                self._logger.info('Resetting %s ' % str(self) +
                                  'because of local attribute change.')
                with self.not_triggering_reset():
                    is_output_hist_invalid = self._on_critical_attr_change(
                        key, old_value, new_value)
                for child in self._children:
                    child.on_upstream_change(is_output_hist_invalid)

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

    @property
    def viz_type(self) -> str:
        return self._viz_type

    @viz_type.setter
    def viz_type(self, value):
        allowed_types = ('sensor time series', 'source time series',
                         'connectivity', 'roi time series', None)
        if value in allowed_types:
            self._viz_type = value
        else:
            raise AttributeError(
                'viz_type should be one of %s; instead got %s'
                % (allowed_types, value))


class SourceNode(Node):
    """Objects of this class read data from a source"""

    # There is no 'upstream' for the sources
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()
    ALLOWED_CHILDREN = ('Preprocessing', 'LinearFilter', 'ICARejection',
                        'MNE', 'MCE', 'Beamformer', 'LSLStreamOutput')

    def __init__(self):
        Node.__init__(self)
        self.mne_info = None
        self.viz_type = 'sensor time series'

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

    def _on_critical_attr_change(self, key, old_val, new_val):
        # There is nothing to reset. Just go ahead and initialize
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
    def __init__(self):
        Node.__init__(self)
        self.viz_type = None
        with self.not_triggering_reset():
            self.disabled = False

    def update(self):
        if (self.parent.output is None or self.parent.output.size == 0
                or self.disabled is True):
            return
        else:
            Node.update(self)
