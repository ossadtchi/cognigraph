from PyQt5.QtCore import QObject, pyqtSignal
import time
import scipy as sc

import math

from vendor.nfb.pynfb.protocols.ssd.topomap_selector_ica import ICADialog

import numpy as np
import mne
from numpy.linalg import svd
from scipy.optimize import linprog
from sklearn.preprocessing import normalize
from mne.preprocessing import find_outliers
from mne.minimum_norm import apply_inverse_raw  # , make_inverse_operator
from mne.minimum_norm import make_inverse_operator as mne_make_inverse_operator
from mne.minimum_norm import prepare_inverse_operator
from mne.beamformer import apply_lcmv_raw
from ..utils.make_lcmv import make_lcmv

from .node import ProcessorNode
from ..utils.matrix_functions import (make_time_dimension_second,
                                      put_time_dimension_back_from_second)
from ..utils.inverse_model import (get_default_forward_file,
                                   get_clean_forward,
                                   make_inverse_operator,
                                   get_mesh_data_from_forward_solution)

from ..utils.pynfb import (pynfb_ndarray_function_wrapper,
                           ExponentialMatrixSmoother)
from ..utils.channels import channel_labels_saver
from ..utils.aux_tools import nostdout
from .. import TIME_AXIS
from vendor.nfb.pynfb.signal_processing import filters

__all__ = ('Preprocessing', 'InverseModel', 'LinearFilter',
           'EnvelopeExtractor', 'Beamformer', 'MCE',
           'ICARejection', 'AtlasViewer', 'AmplitudeEnvelopeCorrelations',
           'Coherence', 'MneGcs')


class _Communicate(QObject):
    open_dialog = pyqtSignal()


class Preprocessing(ProcessorNode):
    CHANGES_IN_THESE_REQUIRE_RESET = ('collect_for_x_seconds', 'dsamp_freq')
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

    ALLOWED_CHILDREN = ('ICARejection', 'SignalViewer', 'MCE', 'InverseModel',
                        'Beamformer', 'EnvelopeExtractor', 'LinearFilter',
                        'LSLStreamOutput')

    def __init__(self, collect_for_x_seconds=60, dsamp_freq=None):
        ProcessorNode.__init__(self)
        self.collect_for_x_seconds = collect_for_x_seconds  # type: int

        self._samples_collected = None  # type: int
        self._samples_to_be_collected = None  # type: int
        self._enough_collected = None  # type: bool
        self._means = None  # type: np.ndarray
        self._mean_sums_of_squares = None  # type: np.ndarray
        self._bad_channel_indices = None  # type: list[int]
        self._interpolation_matrix = None  # type: np.ndarray
        self._dsamp_freq = dsamp_freq
        self.viz_type = 'sensor time series'

        self._reset_statistics()

    def _initialize(self):
        self.mne_info = self.traverse_back_and_find('mne_info')
        frequency = self.mne_info['sfreq']
        self._samples_to_be_collected = int(math.ceil(
            self.collect_for_x_seconds * frequency))

    def _update(self):
        # Have we collected enough samples without the new input?
        enough_collected = self._samples_collected >=\
                self._samples_to_be_collected
        if not enough_collected:
            if self.parent.output is not None and\
                    self.parent.output.shape[TIME_AXIS] > 0:
                self._update_statistics()

        elif not self._enough_collected:  # We just got enough samples
            self._enough_collected = True
            standard_deviations = self._calculate_standard_deviations()
            self._bad_channel_indices = find_outliers(standard_deviations)
            if any(self._bad_channel_indices):
                # message = Message(there_has_been_a_change=True,
                #                   output_history_is_no_longer_valid=True)
                # self._deliver_a_message_to_receivers(message)
                # self.mne_info['bads'].append(self._bad_channel_indices)
                # self.mne_info['bads'] = self._bad_channel_indices

                # TODO: handle emergent bad channels on the go
                pass
        if self._dsamp_freq and self._dsamp_freq < self.mne_info['sfreq']:
            raw = mne.io.RawArray(self.parent.output, self.mne_info)
            raw.resample(self._dsamp_freq)
            self.output = raw.get_data()
            self.mne_info = raw.mne_info

        else:
            self.output = self.parent.output

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        if key == 'collect_for_x_seconds':
            self._reset_statistics()
        self._input_history_is_no_longer_valid = True
        return self._input_history_is_no_longer_valid

    def _reset_statistics(self):
        self._samples_collected = 0
        self._enough_collected = False
        self._means = 0
        self._mean_sums_of_squares = 0
        self._bad_channel_indices = []

    def _update_statistics(self):
        input_array = self.parent.output.astype(np.dtype('float64'))
        # Using float64 is necessary because otherwise rounding error
        # in recursive formula accumulate
        n = self._samples_collected
        m = input_array.shape[TIME_AXIS]  # number of new samples
        self._samples_collected += m

        self._means = (
            self._means * n + np.sum(input_array, axis=TIME_AXIS)) / (n + m)
        self._mean_sums_of_squares = (
            self._mean_sums_of_squares * n +
            np.sum(input_array ** 2, axis=TIME_AXIS)) / (n + m)

    def _calculate_standard_deviations(self):
        n = self._samples_collected
        return np.sqrt(
            n / (n - 1) * (self._mean_sums_of_squares - self._means ** 2))

    def _on_input_history_invalidation(self):
        self._reset_statistics()

    def _check_value(self, key, value):
        pass


class InverseModel(ProcessorNode):
    SUPPORTED_METHODS = ['MNE', 'dSPM', 'sLORETA']
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('mne_inverse_model_file_path',
                                      'mne_forward_model_file_path',
                                      'snr', 'method')
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}
    ALLOWED_CHILDREN = ('EnvelopeExtractor', 'SignalViewer', 'BrainViewer',
                        'AtlasViewer', 'LSLStreamOutput')

    def __init__(self, forward_model_path=None, snr=1.0, method='MNE',
                 depth=None, loose=1, fixed=False):
        ProcessorNode.__init__(self)

        self.snr = snr
        self.fwd_path = forward_model_path
        self._default_forward_model_file_path = None
        self.mne_info = None
        self.fwd = None

        # self._inverse_model_matrix = None
        self.method = method
        self.loose = loose
        self.depth = depth
        self.fixed = fixed
        self.viz_type = 'source time series'

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')
        self._bad_channels = mne_info['bads']

        if self.fwd_path is None:
            self._default_forward_model_file_path =\
                get_default_forward_file(mne_info)

        self.fwd, missing_ch_names = get_clean_forward(
            self.mne_forward_model_file_path, mne_info)
        mne_info['bads'] = list(set(mne_info['bads'] + missing_ch_names))

        self.inverse_operator = make_inverse_operator(self.fwd,
                                                      mne_info,
                                                      depth=self.depth,
                                                      loose=self.loose,
                                                      fixed=self.fixed)
        self._lambda2 = 1.0 / self.snr ** 2
        self.inverse_operator = prepare_inverse_operator(
            self.inverse_operator, nave=100,
            lambda2=self._lambda2, method=self.method)
        # self._inverse_model_matrix = matrix_from_inverse_operator(
        #     inverse_operator=self.inverse_operator, mne_info=mne_info,
        #     snr=self.snr, method=self.method)

        frequency = mne_info['sfreq']
        # channel_count = self._inverse_model_matrix.shape[0]
        channel_count = self.fwd['nsource']
        channel_labels = ['vertex #{}'.format(i + 1)
                          for i in range(channel_count)]
        self.mne_info = mne.create_info(channel_labels, frequency)

    def _update(self):
        mne_info = self.traverse_back_and_find('mne_info')
        bads = mne_info['bads']
        if bads != self._bad_channels:
            self._logger.info('Found new bad channels {};'.format(bads) +
                              'updating inverse operator')
            # self.inverse_operator = make_inverse_operator(self.fwd, mne_info)
            self.inverse_operator = make_inverse_operator(self.fwd,
                                                          mne_info,
                                                          depth=self.depth,
                                                          loose=self.loose,
                                                          fixed=self.fixed)
            self.inverse_operator = prepare_inverse_operator(
                self.inverse_operator, nave=100,
                lambda2=self._lambda2, method=self.method)
            # self._inverse_model_matrix = matrix_from_inverse_operator(
            #     inverse_operator=self.inverse_operator, mne_info=mne_info,
            #     snr=self.snr, method=self.method)
            self._bad_channels = bads

        input_array = self.parent.output
        raw_array = mne.io.RawArray(input_array, mne_info, verbose='ERROR')
        raw_array.pick_types(eeg=True, meg=False, stim=False, exclude='bads')
        # data = raw_array.get_data()
        # self.output = self._apply_inverse_model_matrix(data)
        stc = apply_inverse_raw(raw_array, self.inverse_operator,
                                lambda2=self._lambda2, method=self.method,
                                prepared=True)
        self.output = stc.data

    def _on_input_history_invalidation(self):
        # The methods implemented in this node do not rely on past inputs
        pass

    def _check_value(self, key, value):
        if key == 'method':
            if value not in self.SUPPORTED_METHODS:
                raise ValueError(
                    'Method {} is not supported.'.format(value) +
                    ' Use one of: {}'.format(self.SUPPORTED_METHODS))

        if key == 'snr':
            if value <= 0:
                raise ValueError(
                    'snr (signal-to-noise ratio) must be a positive number.')

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    @property
    def mne_forward_model_file_path(self):
        return self.fwd_path or\
                self._default_forward_model_file_path

    @mne_forward_model_file_path.setter
    def mne_forward_model_file_path(self, value):
        # This setter is for public use, hence the "user_provided"
        self.fwd_path = value

    def _apply_inverse_model_matrix(self, input_array: np.ndarray):
        W = self._inverse_model_matrix  # VERTICES x CHANNELS
        output_array = W.dot(make_time_dimension_second(input_array))
        return put_time_dimension_back_from_second(output_array)


class LinearFilter(ProcessorNode):
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('lower_cutoff', 'upper_cutoff')
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info':
                                           lambda info: (info['nchan'], )}
    ALLOWED_CHILDREN = ('InverseModel', 'MCE', 'Beamformer',
                        'SignalViewer', 'EnvelopeExtractor', 'LSLStreamOutput')

    def __init__(self, lower_cutoff: float = 1, upper_cutoff: float = 50):
        ProcessorNode.__init__(self)
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff
        self._linear_filter = None  # type: filters.ButterFilter
        self.viz_type = None

    def _initialize(self):
        self.viz_type = self.parent.viz_type
        mne_info = self.traverse_back_and_find('mne_info')
        frequency = mne_info['sfreq']
        channel_count = mne_info['nchan']
        if not (self.lower_cutoff is None and self.upper_cutoff is None):
            band = (self.lower_cutoff, self.upper_cutoff)

            self._linear_filter = filters.ButterFilter(
                band, fs=frequency, n_channels=channel_count)

            self._linear_filter.apply = pynfb_ndarray_function_wrapper(
                self._linear_filter.apply)
        else:
            self._linear_filter = None

    def _update(self):
        input_data = self.parent.output
        if self._linear_filter is not None:
            self.output = self._linear_filter.apply(input_data)
        else:
            self.output = input_data

    def _check_value(self, key, value):
        if value is None:
            pass

        elif key == 'lower_cutoff':
            if (hasattr(self, 'upper_cutoff') and
                    self.upper_cutoff is not None and
                    value > self.upper_cutoff):
                raise ValueError(
                    'Lower cutoff can`t be set higher that the upper cutoff')
            if value < 0:
                raise ValueError('Lower cutoff must be a positive number')

        elif key == 'upper_cutoff':
            if (hasattr(self, 'upper_cutoff') and
                    self.lower_cutoff is not None and
                    value < self.lower_cutoff):
                raise ValueError(
                    'Upper cutoff can`t be set lower that the lower cutoff')
            if value < 0:
                raise ValueError('Upper cutoff must be a positive number')

    def _on_input_history_invalidation(self):
        # Reset filter delays
        if self._linear_filter is not None:
            self._linear_filter.reset()

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid


class EnvelopeExtractor(ProcessorNode):
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('method', 'factor')
    SUPPORTED_METHODS = ('Exponential smoothing', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info':
                                           lambda info: (info['nchan'],)}
    ALLOWED_CHILDREN = ('SignalViewer', 'LSLStreamOutput')

    def __init__(self, factor=0.9):
        ProcessorNode.__init__(self)
        self.method = 'Exponential smoothing'
        self.factor = factor
        self._envelope_extractor = None  # type: ExponentialMatrixSmoother
        self.viz_type = None

    def _initialize(self):
        channel_count = self.traverse_back_and_find('mne_info')['nchan']
        self._envelope_extractor = ExponentialMatrixSmoother(
            factor=self.factor, column_count=channel_count)
        self._envelope_extractor.apply = pynfb_ndarray_function_wrapper(
            self._envelope_extractor.apply)

        self.viz_type = self.parent.viz_type
        if self.parent.viz_type == 'source time series':
            self.ALLOWED_CHILDREN = ('BrainViewer', 'LSLStreamOutput')
        elif self.parent.viz_type == 'connectivity':
            self.ALLOWED_CHILDREN = ('ConnectivityViewer', 'LSLStreamOutput')

    def _update(self):
        input_data = self.parent.output
        self.output = self._envelope_extractor.apply(np.abs(input_data))

    def _check_value(self, key, value):
        if key == 'factor':
            if value <= 0 or value >= 1:
                raise ValueError('Factor must be a number between 0 and 1')

        if key == 'method':
            if value not in self.SUPPORTED_METHODS:
                raise ValueError(
                    'Method {} is not supported.' +
                    ' Use one of: {}'.format(value, self.SUPPORTED_METHODS))

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        self._envelope_extractor.reset()


class Beamformer(ProcessorNode):
    """Adaptive and nonadaptive beamformer"""

    SUPPORTED_OUTPUT_TYPES = ('power', 'activation')
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info',)
    CHANGES_IN_THESE_REQUIRE_RESET = ('reg', 'output_type', 'is_adaptive',
                                      'fixed_orientation',
                                      'mne_forward_model_file_path')
    ALLOWED_CHILDREN = ('EnvelopeExtractor', 'SignalViewer', 'BrainViewer',
                        'AtlasViewer', 'LSLStreamOutput')

    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

    def __init__(self, output_type='power', is_adaptive=False,
                 fixed_orientation=True, forward_model_path=None,
                 forgetting_factor_per_second=0.99, reg=0.05, whiten=True):
        ProcessorNode.__init__(self)
        self.whiten = whiten

        self.fwd_path = forward_model_path
        self._default_forward_model_file_path = None  # type: str
        self.mne_info = None  # type: mne.Info

        self.output_type = output_type  # type: np.dtype
        self.is_adaptive = is_adaptive  # type: bool
        self._initialized_as_adaptive = None  # type: bool
        self.fixed_orientation = fixed_orientation  # type: bool
        self._initialized_as_fixed = None  # type: bool

        self._channel_indices = None  # type: list
        self._gain_matrix = None  # type: np.ndarray
        self._data_cov = None  # type: np.ndarray
        self.forgetting_factor_per_second = forgetting_factor_per_second
        self._forgetting_factor_per_sample = None  # type: float
        self.reg = reg

        self.viz_type = 'source time series'

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')

        # -------------- setup forward -------------- #
        if self.fwd_path is None:
            self._default_forward_model_file_path = get_default_forward_file(
                    mne_info)

        assert self.mne_forward_model_file_path is not None,\
            'Please provide forward model file.'

        try:
            fwd, missing_ch_names = get_clean_forward(
                self.mne_forward_model_file_path, mne_info)
        except ValueError:
            raise Exception('BAD FORWARD + DATA COMBINATION!')
        self._gain_matrix = fwd['sol']['data']
        G = self._gain_matrix
        # ------------------------------------------- #

        mne_info['bads'] = list(set(mne_info['bads'] + missing_ch_names))
        Rxx = G.dot(G.T) / 1e22

        goods = mne.pick_types(mne_info, eeg=True, meg=False, exclude='bads')
        ch_names = [mne_info['ch_names'][i] for i in goods]

        self._data_cov = mne.Covariance(
            Rxx, ch_names, mne_info['bads'], mne_info['projs'], nfree=1)

        if self.whiten:
            self._noise_cov = mne.Covariance(
                G.dot(G.T), ch_names, mne_info['bads'],
                mne_info['projs'], nfree=1)

        self._mne_info = mne_info  # upstream info

        frequency = mne_info['sfreq']
        self._forgetting_factor_per_sample = np.power(
                self.forgetting_factor_per_second, 1 / frequency)

        n_vert = fwd['nsource']
        channel_labels = ['vertex #{}'.format(i + 1) for i in range(n_vert)]

        # downstream info
        self.mne_info = mne.create_info(channel_labels, frequency)

        self._initialized_as_adaptive = self.is_adaptive
        self._initialized_as_fixed = self.fixed_orientation

        self.fwd_surf = mne.convert_forward_solution(
                    fwd, surf_ori=True, force_fixed=False)
        self._compute_filters()

    def _update(self):
        t1 = time.time()
        input_array = self.parent.output
        raw_array = mne.io.RawArray(
            input_array, self._mne_info, verbose='ERROR')

        raw_array.pick_types(eeg=True, meg=False, stim=False, exclude='bads')
        raw_array.set_eeg_reference(ref_channels='average', projection=True)
        t2 = time.time()
        self._logger.debug('Prepare arrays in {:.1f} ms'.format(
                    (t2 - t1) * 1000))

        if self.is_adaptive:
            self._update_covariance_matrix(input_array)
            t1 = time.time()
            self._compute_filters()
            t2 = time.time()
            self._logger.debug('Assembled lcmv instance in {:.1f} ms'.format(
                (t2 - t1) * 1000))

        self._filters['source_nn'] = []
        t1 = time.time()
        stc = apply_lcmv_raw(raw=raw_array, filters=self._filters,
                             max_ori_out='signed')
        t2 = time.time()
        self._logger.debug('Applied lcmv inverse in {:.1f} ms'.format(
                    (t2 - t1) * 1000))

        output = stc.data
        t1 = time.time()
        if self.fixed_orientation is True:
            if self.output_type == 'power':
                output = output ** 2
        else:
            vertex_count = self.fwd_surf['nsource']
            output = np.sum(
                np.power(output, 2).reshape((vertex_count, 3, -1)), axis=1)
            if self.output_type == 'activation':
                output = np.sqrt(output)

        self.output = output
        t2 = time.time()
        self._logger.debug('Finalized in {:.1f} ms'.format((t2 - t1) * 1000))

    @property
    def mne_forward_model_file_path(self):
        # TODO: fix this
        return (self.fwd_path or
                self._default_forward_model_file_path)

    @mne_forward_model_file_path.setter
    def mne_forward_model_file_path(self, value):
        # This setter is for public use, hence the "user_provided"
        self.fwd_path = value

    def _compute_filters(self):
        self._filters = make_lcmv(info=self._mne_info, forward=self.fwd_surf,
                                  data_cov=self._data_cov, reg=self.reg,
                                  noise_cov=self._noise_cov,  # data whiten
                                  pick_ori='max-power',
                                  weight_norm='unit-noise-gain',
                                  reduce_rank=False)

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:

        # Only change adaptiveness or fixed_orientation requires reinit
        # if (self._initialized_as_adaptive is not self.is_adaptive
        #         or self._initialized_as_fixed is not self.fixed_orientation):
        # if old_val != new_val:  # we don't expect numpy arrays here
        if key in ('reg', ):
            self._compute_filters()
        else:
            self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        # Only adaptive version relies on history
        if self._initialized_as_adaptive is True:
            self.initialize()

    def _check_value(self, key, value):
        if key == 'output_type':
            if value not in self.SUPPORTED_OUTPUT_TYPES:
                raise ValueError(
                    'Method {} is not supported.' +
                    ' Use one of: {}'.format(
                        value, self.SUPPORTED_OUTPUT_TYPES))

        if key == 'reg':
            if value <= 0:
                raise ValueError('reg (covariance regularization coefficient)'
                                 ' must be a positive number')

        if key == 'is_adaptive':
            if not isinstance(value, bool):
                raise ValueError(
                    'Beamformer type (adaptive vs nonadaptive) is not set')

    def _update_covariance_matrix(self, input_array):
        t1 = time.time()
        alpha = self._forgetting_factor_per_sample
        sample_count = input_array.shape[TIME_AXIS]
        self._logger.debug('Number of samples: {}'.format(sample_count))
        new_Rxx_data = self._data_cov.data

        raw_array = mne.io.RawArray(
            input_array, self._mne_info, verbose='ERROR')
        raw_array.pick_types(eeg=True, meg=False, stim=False, exclude='bads')
        raw_array.set_eeg_reference(ref_channels='average', projection=True)
        input_array_nobads = raw_array.get_data()

        t2 = time.time()
        self._logger.debug(
            'Prepared covariance update in {:.2f} ms'.format((t2 - t1) * 1000))
        samples = make_time_dimension_second(input_array_nobads).T
        new_Rxx_data = (alpha * new_Rxx_data +
                        (1 - alpha) * samples.T.dot(samples))
        t3 = time.time()
        self._logger.debug(
            'Updated matrix data in {:.2f} ms'.format((t3 - t2) * 1000))

        self._data_cov = mne.Covariance(new_Rxx_data, self._data_cov.ch_names,
                                        raw_array.info['bads'],
                                        raw_array.info['projs'], nfree=1)
        t4 = time.time()
        self._logger.debug('Created instance of covariance' +
                           ' in {:.2f} ms'.format((t4 - t4) * 1000))


# TODO: implement this function
def pynfb_filter_based_processor_class(pynfb_filter_class):
    """
    Returns a ProcessorNode subclass with the functionality of
    pynfb_filter_class

    pynfb_filter_class: subclass of pynfb.signal_processing.filters.BaseFilter

    Sample usage 1:

    LinearFilter = pynfb_filter_based_processor_class(filters.ButterFilter)
    linear_filter = LinearFilter(band, fs, n_channels, order)

    Sample usage 2
    (this would correspond to a different implementation of this function):

    LinearFilter = pynfb_filter_based_processor_class(filters.ButterFilter)
    linear_filter = LinearFilter(band, order)

    In this case LinearFilter should provide
    fs and n_channels parameters to filters.ButterFilter automatically

    """
    class PynfbFilterBasedProcessorClass(ProcessorNode):
        def _on_input_history_invalidation(self):
            pass

        def _check_value(self, key, value):
            pass

        @property
        def CHANGES_IN_THESE_REQUIRE_RESET(self):
            pass

        @property
        def UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION(self):
            pass

        def _reset(self):
            pass

        def __init__(self):
            pass

        def _initialize(self):
            pass

        def _update(self):
            pass
    return PynfbFilterBasedProcessorClass


class MCE(ProcessorNode):
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()
    CHANGES_IN_THESE_REQUIRE_RESET = ('mne_forward_model_file_path', 'snr',
                                      'n_comp')
    ALLOWED_CHILDREN = ('BrainViewer', 'EnvelopeExtractor', 'AtlasViewer',
                        'LSLStreamOutput')

    def __init__(self, snr=1.0, forward_model_path=None, n_comp=40):
        ProcessorNode.__init__(self)
        self.snr = snr
        self.mne_forward_model_file_path = forward_model_path
        self.n_comp = n_comp
        self.mne_info = None
        self.input_data = []
        self.output = []
        self.viz_type = 'source time series'
        # pass

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')
        # mne_info['custom_ref_applied'] = True
        # -------- truncated svd for fwd_opr operator -------- #
        fwd, missing_ch_names = get_clean_forward(
            self.mne_forward_model_file_path, mne_info)
        mne_info['bads'] = list(set(mne_info['bads'] + missing_ch_names))
        fwd_fix = mne.convert_forward_solution(
                fwd, surf_ori=True, force_fixed=False)

        self._gain_matrix = fwd_fix['sol']['data']

        self._logger.info('Computing SVD of the forward operator')
        U, S, V = svd(self._gain_matrix)

        Sn = np.zeros([self.n_comp, V.shape[0]])
        Sn[:self.n_comp, :self.n_comp] = np.diag(S[:self.n_comp])

        self.Un = U[:, :self.n_comp]
        self.A_non_ori = Sn @ V
        # ---------------------------------------------------- #

        # -------- leadfield dims -------- #
        N_SEN = self._gain_matrix.shape[0]
        # -------------------------------- #

        # ------------------------ noise-covariance ------------------------ #
        cov_data = np.identity(N_SEN)
        ch_names = np.array(mne_info['ch_names'])[mne.pick_types(mne_info,
                                                                 eeg=True,
                                                                 meg=False)]
        ch_names = list(ch_names)
        noise_cov = mne.Covariance(
                cov_data, ch_names, mne_info['bads'],
                mne_info['projs'], nfree=1)
        # ------------------------------------------------------------------ #

        self.mne_inv = mne_make_inverse_operator(
                mne_info, fwd_fix, noise_cov, depth=0.8,
                loose=1, fixed=False, verbose='ERROR')
        self._mne_info = mne_info
        self.Sn = Sn
        self.V = V
        channel_count = fwd['nsource']
        channel_labels = ['vertex #{}'.format(i + 1)
                          for i in range(channel_count)]
        self.mne_info = mne.create_info(channel_labels, mne_info['sfreq'])

    def _update(self):
        input_array = self.parent.output
        # last_slice = last_sample(input_array)
        last_slice = np.mean(input_array, axis=1)
        n_src = self.mne_inv['nsource']
        n_times = input_array.shape[1]
        output_mce = np.empty([n_src, n_times])

        raw_slice = mne.io.RawArray(np.expand_dims(last_slice, axis=1),
                                    self._mne_info, verbose='ERROR')
        raw_slice.pick_types(eeg=True, meg=False, stim=False, exclude='bads')
        raw_slice.set_eeg_reference(ref_channels='average', projection=True)

        # ------------------- get dipole orientations --------------------- #
        stc_slice = apply_inverse_raw(raw_slice, self.mne_inv,
                                      pick_ori='vector',
                                      method='MNE', lambda2=1, verbose='ERROR')
        Q = normalize(stc_slice.data[:, :, 0])  # dipole orientations
        # ----------------------------------------------------------------- #

        # -------- setup linprog params -------- #
        n_sen = self.A_non_ori.shape[0]
        A_eq = np.empty([n_sen, n_src])
        for i in range(n_src):
            A_eq[:, i] = self.A_non_ori[:, i * 3: (i + 1) * 3] @ Q[i, :].T
        data_slice = raw_slice.get_data()[:, 0]
        b_eq = self.Un.T @ data_slice
        c = np.ones(A_eq.shape[1])
        # -------------------------------------- #

        with nostdout():
            sol = linprog(c, A_eq=A_eq, b_eq=b_eq,
                          method='interior-point', bounds=(0, None),
                          options={'disp': False})
        output_mce[:, :] = sol.x[:, np.newaxis]

        self.output = output_mce
        self.sol = sol
        return Q, A_eq, data_slice, b_eq, c

    def _on_input_history_invalidation(self):
        # The methods implemented in this node do not rely on past inputs
        pass

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _check_value(self, key, value):
        if key == 'snr':
            if value <= 0:
                raise ValueError(
                    'snr (signal-to-noise ratio) must be a positive number.')


class ICARejection(ProcessorNode):
    ALLOWED_CHILDREN = ('SignalViewer', 'LinearFilter', 'InverseModel',
                        'MCE', 'Beamformer', 'EnvelopeExtractor',
                        'LSLStreamOutput')

    CHANGES_IN_THESE_REQUIRE_RESET = ('collect_for_x_seconds', )

    def __init__(self, collect_for_x_seconds: int = 60):
        ProcessorNode.__init__(self)
        self.collect_for_x_seconds = collect_for_x_seconds  # type: int

        self._samples_collected = None  # type: int
        self._samples_to_be_collected = None  # type: int
        self._enough_collected = None  # type: bool

        self._reset_statistics()
        self._ica_rejector = None
        self.signal_sender = _Communicate()
        self.signal_sender.open_dialog.connect(self._on_ica_finished)

        self.viz_type = 'sensor time series'

    def _on_input_history_invalidation(self):
        self._reset_statistics()

    def _check_value(self, key, value):
        pass

    def _initialize(self):
        self._mne_info = self.traverse_back_and_find('mne_info')
        self._frequency = self._mne_info['sfreq']
        self._good_ch_inds = mne.pick_types(self._mne_info, eeg=True,
                                            meg=False, stim=False,
                                            exclude='bads')

        channels = self._mne_info['chs']
        self._ch_locs = np.array([ch['loc'] for ch in channels])

        n_ch = len(self._good_ch_inds)
        self._ica_rejector = np.eye(n_ch)
        self._samples_to_be_collected = int(math.ceil(
            self.collect_for_x_seconds * self._frequency))
        self._collected_timeseries = np.zeros(
                [n_ch, self._samples_to_be_collected])
        self._linear_filter = filters.ButterFilter(
                [1, 50], fs=self._frequency,
                n_channels=len(self._good_ch_inds))
        self._linear_filter.apply = pynfb_ndarray_function_wrapper(
                self._linear_filter.apply)

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self._reset_statistics()
        self._input_history_is_no_longer_valid = True
        return self._input_history_is_no_longer_valid

    def _reset_statistics(self):
        self._samples_collected = 0
        self._enough_collected = False

    def _update(self):
        # Have we collected enough samples without the new input?
        self.output = self.parent.output

        enough_collected = self._samples_collected >=\
            self._samples_to_be_collected
        if not enough_collected:
            if self.parent.output is not None and\
                    self.parent.output.shape[TIME_AXIS] > 0:
                self._update_statistics()

        elif not self._enough_collected:  # We just got enough samples
            self._enough_collected = True
            self._logger.info('Collected enough samples')
            self.signal_sender.open_dialog.emit()
        else:
            self.output[self._good_ch_inds, :] = np.dot(
                    self._ica_rejector,
                    self.parent.output[self._good_ch_inds, :])

    def _on_ica_finished(self):
        ica = ICADialog(
            self._collected_timeseries.T,
            list(np.array(self._mne_info['ch_names'])[self._good_ch_inds]),
            self._ch_locs[self._good_ch_inds, :], self._frequency)
        ica.exec_()
        self._ica_rejector = ica.rejection.val.T

    def _update_statistics(self):
        input_array = self.parent.output.astype(np.dtype('float64'))
        n = self._samples_collected
        m = input_array.shape[TIME_AXIS]  # number of new samples
        self._samples_collected += m
        self._collected_timeseries[:, n:n + m] = self._linear_filter.apply(
                input_array[self._good_ch_inds, :])
        # Using float64 is necessary because otherwise rounding error
        # in recursive formula accumulate

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}


class AtlasViewer(ProcessorNode):
    CHANGES_IN_THESE_REQUIRE_RESET = ('labels_info', 'parc')
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('subjects_dir',
                                                          'subject')
    ALLOWED_CHILDREN = ('EnvelopeExtractor', 'SignalViewer', 'LSLStreamOutput')

    def __init__(self, parc='aparc'):
        ProcessorNode.__init__(self)
        self.parc = parc
        self.subjects_dir = None
        self.subject = None
        self.active_labels = []

        self.viz_type = 'roi time series'

        # base, fname = os.path.split(self.annot_file)
        # self.annot_files = [
        #     os.path.join(surfaces_dir, 'label', hemi + self.annot_file)
        #     for hemi in ('lh.', 'rh.')]

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        if key == 'parc':
            self.initialize()
        else:
            self.active_labels = [l for l in self.labels if l.is_active]
            self.mne_info = {'ch_names': [a.name for a in self.active_labels],
                             'nchan': len(self.active_labels),
                             'sfreq': self.sfreq}
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        pass

    def _initialize(self):
        # Map sources for which we solve the inv problem to the dense
        # cortex which we use for plotting
        # rh_offset = number of vertices in left hemisphere; used for
        # indexing the right hemisphere sources when both hemispheres
        # are stored together intead of being split in lh and rh
        # as in freesurfer

        self.subject = self.traverse_back_and_find('subject')
        self.subjects_dir = self.traverse_back_and_find('subjects_dir')
        self._read_annotation()

        self.active_labels = [l for l in self.labels if l.is_active]

        self.sfreq = self.traverse_back_and_find('mne_info')['sfreq']
        self.mne_info = {'ch_names': [a.name for a in self.active_labels],
                         'nchan': len(self.active_labels),
                         'sfreq': self.sfreq}

    def _read_annotation(self):
        mne_forward_model_file_path = self.traverse_back_and_find(
            'mne_forward_model_file_path')
        forward_solution = mne.read_forward_solution(
            mne_forward_model_file_path, verbose='ERROR')
        sources_idx, _, _, rh_offset = get_mesh_data_from_forward_solution(
            forward_solution)
        try:
            labels = mne.read_labels_from_annot(
                self.subject, parc=self.parc, surf_name='white',
                subjects_dir=self.subjects_dir)

            for i, l in enumerate(labels):
                labels[i].mass_center = labels[i].center_of_mass(
                    subject=self.subject, subjects_dir=self.subjects_dir)
                if l.hemi == 'rh':
                    labels[i].vertices += rh_offset
                    labels[i].mass_center += rh_offset
                labels[i].forward_vertices = np.where(
                    np.isin(sources_idx, labels[i].vertices))[0]
                labels[i].is_active = True

            self.labels = labels  # label objects read from filesystem

            label_colors = np.array([l.color for l in labels])

            label_names = np.array([l.name for l in labels])
            self._logger.debug(
                'Found the following labels in annotation: {}'
                .format(label_names))

            self.labels_info = []  # extracted vital info for each label
            for i_label, label_name in enumerate(label_names):
                label_id = (i_label
                            if label_names[i_label] != 'Unknown' else -1)
                label_dict = {
                    'name': label_names[i_label],
                    'id': label_id,
                    'state': True,
                    'color': label_colors[i_label, :],
                    'vertices': labels[i_label].vertices}

                self.labels_info.append(label_dict)

        except FileNotFoundError:
            self._logger.error('Annotation files not found')
            # Open file picker dialog here

    def _update(self):
        data = self.parent.output

        n_times = data.shape[1]
        n_active_labels = len(self.active_labels)

        data_label = np.zeros([n_active_labels, n_times])
        for i, l in enumerate(self.active_labels):
            # Average inverse solution inside label
            # label_mask = self.source_labels == label['id']
            data_label[i, :] = np.mean(data[l.forward_vertices, :], axis=0)
        self.output = data_label
        self._logger.debug(data.shape)

    def _check_value(self, key, value):
        ...


class AmplitudeEnvelopeCorrelations(ProcessorNode):
    """Node computing amplitude envelopes correlation

    Parameters
    ----------
    method: str (default None)
        Method to deal with signal leakage
    factor: float
        Exponential smoothing factor
    seed: int
        Seed index

    """
    CHANGES_IN_THESE_REQUIRE_RESET = ('method', 'factor', 'seed')
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info':
                                           lambda info: (info['nchan'],)}

    def __init__(self, method=None, factor=0.9, seed=None):
        ProcessorNode.__init__(self)
        self.method = method
        self._envelope_extractor = None
        self.factor = factor
        self.seed = seed
        if seed:
            self.viz_type = 'source time series'
        else:
            self.viz_type = 'connectivity'

    def _initialize(self):
        channel_count = self.traverse_back_and_find('mne_info')['nchan']
        if self.seed:
            assert self.seed < channel_count, ('Seed index {} exceeds max'
                                               ' channel number {}'.format(
                                                   self.seed, channel_count))
        self._logger.debug('Channel count: %d' % channel_count)
        self._envelope_extractor = ExponentialMatrixSmoother(
            factor=self.factor, column_count=channel_count)
        self._envelope_extractor.apply = pynfb_ndarray_function_wrapper(
            self._envelope_extractor.apply)

    def _update(self):
        input_data = self.parent.output
        n_times = input_data.shape[1]
        ddof = 1

        self._envelopes = self._envelope_extractor.apply(np.abs(input_data))
        if self.method is None:
            if self.seed is None:
                self.output = np.corrcoef(self._envelopes)
            else:
                envs_z = self._envelopes
                envs_z -= envs_z.mean(axis=1)[:, np.newaxis]
                envs_z /= envs_z.std(axis=1, ddof=ddof)[:, np.newaxis]
                seed_env = envs_z[self.seed, :]
                self.output = (
                    seed_env.dot(envs_z.T) / (n_times - ddof))[:, np.newaxis]
        else:
            self.output = self._orthogonalized_env_corr(input_data)

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        self._envelope_extractor.reset()

    def _check_value(self, key, value):
        pass

    def _orthogonalized_env_corr(self, data):
        if self.seed is None:
            corrmat = np.empty([data.shape[0]] * 2)
        else:
            corrmat = np.empty([data.shape[0], 1])
        envs = self._envelopes - self._envelopes.mean(axis=1)[:, np.newaxis]
        n_times = envs.shape[1]
        ddof = 1
        # ddof=1 is for unbiased std estimator
        envs = envs / envs.std(axis=1, ddof=ddof)[:, np.newaxis]
        G = data.dot(data.T)  # Gramm matrix

        if self.seed is None:
            labels_iter = range(data.shape[0])
        else:
            labels_iter = [self.seed]

        for i, r in enumerate(labels_iter):
            data_orth_r = data - np.outer(G[:, r], data[r, :]) / G[r, r]
            orth_envs = self._envelope_extractor.apply(np.abs(data_orth_r))
            orth_envs -= orth_envs.mean(axis=1)[:, np.newaxis]
            orth_envs /= orth_envs.std(axis=1, ddof=ddof)[:, np.newaxis]
            corrmat[:, i] = envs[r, :].dot(orth_envs.T) / (n_times - ddof)

        if self.seed is None:
            return (corrmat + corrmat.T) / 2
        else:
            return corrmat[:, np.newaxis]


class Coherence(ProcessorNode):
    """Coherence and imaginary coherence computation for narrow-band signals

    Parameters
    ----------
    method: str (default imcoh)
        Connectivity method
    seed: int (default None)
        Seed index

    """
    CHANGES_IN_THESE_REQUIRE_RESET = ()
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    def __init__(self, method='imcoh', seed=None):
        ProcessorNode.__init__(self)
        self.method = method
        self.seed = seed
        if seed:
            self.viz_type = 'source time series'
        else:
            self.viz_type = 'connectivity'

    def _initialize(self):
        pass

    def _update(self):
        input_data = self.parent.output
        hilbert = sc.signal.hilbert(input_data, axis=1)
        if self.seed is None:
            Cp = hilbert.dot(hilbert.conj().T)
            D = np.sqrt(np.diag(Cp))
            coh = Cp / np.outer(D, D)
        else:
            seed_Cp = hilbert[self.seed, :].dot(hilbert.conj().T)
            D = np.sqrt(np.mean(hilbert * hilbert.conj(), axis=1))
            seed_outer_D = D[self.seed] * D  # coherence denominator
            coh = seed_Cp[:, np.newaxis] / seed_outer_D[:, np.newaxis]
            # coh = seed_Cp[:, np.newaxis] * 1e16

        if self.method == 'imcoh':
            self.output = coh.imag
        elif self.method == 'coh':
            self.output = np.abs(coh)

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        return False

    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        pass


class MneGcs(InverseModel):
    """
    Minimum norm fixed orientation inverse with geometric correction for
    signal leakage

    Parameters
    ----------
    seed: int
    forward_model_path: str
    snr: float

    """
    def __init__(self, seed, forward_model_path, snr=1.0):
        method = 'MNE'
        InverseModel.__init__(self, forward_model_path=forward_model_path,
                              snr=snr, method=method)
        self.seed = seed
        self.viz_type = 'source time series'
        self.depth = None
        self.loose = 0
        self.fixed = True

    def _initialize(self):
        InverseModel._initialize(self)
        self.fwd = mne.convert_forward_solution(
            self.fwd, force_fixed=True, surf_ori=True)

    def _apply_inverse_model_matrix(self, input_array: np.ndarray):
        gain = self.fwd['sol']['data']
        seed_topo = gain[:, self.seed]
        W = self._inverse_model_matrix  # VERTICES x CHANNELS
        seed_filter = W[self.seed, :]
        input_array -= (np.outer(seed_topo[:, np.newaxis],
                                 seed_filter[np.newaxis, :].dot(input_array)) /
                        seed_filter.dot(seed_topo))
        output_array = W.dot(input_array)
        return output_array
