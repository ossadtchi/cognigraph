import numpy as np
from cognigraph.nodes.node import SourceNode, ProcessorNode, OutputNode
from mne import create_info


def create_dummy_info(nchan=32, sfreq=500):
    ch_names = [str(i).zfill(2) for i in range(nchan)]
    info = create_info(ch_names, sfreq, ch_types='eeg')
    return info


def count_func_runs(cls):
    class Proxy(cls):
        def __init__(self, *pargs, **kwargs):
            cls.__init__(self, *pargs, **kwargs)
            self.n_updates = 0
            self.n_initializations = 0
            self.n_resets = 0
            self.n_hist_invalidations = 0

        def _initialize(self):
            cls._initialize(self)
            self.n_initializations += 1

        def _update(self):
            cls._update(self)
            self.n_updates += 1

        def _on_critical_attr_change(self, key, old_val, new_val):
            res = cls._on_critical_attr_change(self, key, old_val, new_val)
            self.n_resets += 1
            return res

        def _on_input_history_invalidation(self):
            cls._on_input_history_invalidation(self)
            self.n_hist_invalidations += 1

        def __repr__(self):
            return repr(cls) + ' Node'

        def __str__(self):
            class_str = str(cls)
            return class_str[:-1] + ' Node' + class_str[-1]

    return Proxy


@count_func_runs
class ConcreteSource(SourceNode):
    """"Generates output updates of size (nchan, nsamp).
    Each element of output array on update is incremented by one
    starting with an array of all zeroes.

    """
    CHANGES_IN_THESE_REQUIRE_RESET = ('_mne_info',)
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    _GUI_STRING = 'Dummy Source'

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

    def _check_value(self, key, value):
        pass

    def _on_input_history_invalidation(self):
        pass


@count_func_runs
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

    def _on_input_history_invalidation(self):
        pass


@count_func_runs
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

    def _on_input_history_invalidation(self):
        pass
