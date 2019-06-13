"""
Definition of pipeline processor nodes

Exposed classes
---------------
Preprocessing: ProcessorNode
    Downsample and drop bad channels based on observed amplitude jumps
MNE: _InverseSolverNode
    Minimum norm source estimation + dSPM and sLORETA
LinearFilter: ProcessorNode
    Linear filtering
EnvelopeExtractor: ProcessorNode
    Envelope exctraction
Beamformer: _InverseSolverNode
    LCMV beamformer source estimation
MCE: _InverseSolverNode
    Minimum current source estimation
ICARejection: ProcessorNode
    Artefacts rejection via ICA decomposition
AtlasViewer: ProcessorNode
    Select source-level signals in regions of interest based on atlas
AmplitudeEnvelopeCorrelation: ProcessorNodeons
    Connectivity estimation via amplitude envelopes correlation
Coherence: ProcessorNode
    Connectivity estimation via coherence
MneGcs: ProcessorNode
    Inverse solver for connecivity estimation via Geometric Correction Scheme

"""
import time
import scipy as sc
from copy import deepcopy

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
from ..utils.matrix_functions import (
    make_time_dimension_second,
    put_time_dimension_back_from_second,
)
from ..utils.inverse_model import (
    get_clean_forward,
    make_inverse_operator,
    get_mesh_data_from_forward_solution,
    matrix_from_inverse_operator,
)
from ..utils.pipeline_signals import Communicate

from ..utils.pynfb import (
    pynfb_ndarray_function_wrapper,
    ExponentialMatrixSmoother,
)
from ..utils.channels import channel_labels_saver
from ..utils.aux_tools import nostdout
from .. import TIME_AXIS
from vendor.nfb.pynfb.signal_processing import filters


__all__ = (
    "Preprocessing",
    "MNE",
    "LinearFilter",
    "EnvelopeExtractor",
    "Beamformer",
    "MCE",
    "ICARejection",
    "AtlasViewer",
    "AmplitudeEnvelopeCorrelations",
    "Coherence",
    "SeedCoherence",
    "MneGcs",
)


class Preprocessing(ProcessorNode):
    CHANGES_IN_THESE_REQUIRE_RESET = (
        "collect_for_x_seconds",
        "dsamp_factor",
        "bad_channels",
    )
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {"mne_info": channel_labels_saver}

    ALLOWED_CHILDREN = (
        "ICARejection",
        "SignalViewer",
        "MCE",
        "MNE",
        "Beamformer",
        "EnvelopeExtractor",
        "LinearFilter",
        "LSLStreamOutput",
        "Coherence",
        "FileOutput",
    )

    def __init__(
        self, collect_for_x_seconds=60, dsamp_factor=1, bad_channels=[]
    ):
        ProcessorNode.__init__(self)
        self.collect_for_x_seconds = collect_for_x_seconds  # type: int

        self._samples_collected = None  # type: int
        self._enough_collected = None  # type: bool
        self._means = None  # type: np.ndarray
        self._mean_sums_of_squares = None  # type: np.ndarray
        self._bad_channel_indices = None  # type: list[int]
        self._interpolation_matrix = None  # type: np.ndarray
        self.dsamp_factor = dsamp_factor
        self.viz_type = "sensor time series"
        self.is_collecting_samples = False
        self.bad_channels = bad_channels

        self._reset_statistics()

    def _initialize(self):
        self._upstream_mne_info = self.traverse_back_and_find("mne_info")
        self.mne_info = deepcopy(self._upstream_mne_info)
        self.mne_info["bads"] += self.bad_channels
        self._signal_sender.initialized.emit()
        if self.dsamp_factor and self.dsamp_factor > 1:
            filt_freq = self.mne_info["sfreq"] / self.dsamp_factor / 2
            if self.mne_info["lowpass"] > filt_freq:
                self.mne_info["lowpass"] = filt_freq
            self._antialias_filter = filters.ButterFilter(
                band=(None, filt_freq),
                fs=self.mne_info["sfreq"],
                n_channels=self.mne_info["nchan"],
            )
            self._antialias_filter.apply = pynfb_ndarray_function_wrapper(
                self._antialias_filter.apply
            )
            self._left_n_pad = 0  # initial skip to keep decimation right
            self.mne_info["sfreq"] /= self.dsamp_factor

    def _update(self):
        # Have we collected enough samples without the new input?
        if self.is_collecting_samples:
            enough_collected = (
                self._samples_collected >= self._samples_to_be_collected
            )
            if not enough_collected:
                if (
                    self.parent.output is not None
                    and self.parent.output.shape[TIME_AXIS] > 0
                ):
                    self._update_statistics()

            elif not self._enough_collected:  # We just got enough samples
                self._enough_collected = True
                standard_deviations = self._calculate_standard_deviations()
                self._bad_channel_indices = find_outliers(standard_deviations)
                if any(self._bad_channel_indices):
                    self.mne_info["bads"] = self._upstream_mne_info["bads"] + [
                        self.mne_info["ch_names"][i]
                        for i in self._bad_channel_indices
                    ]
                    self.bad_channels = [
                        self.mne_info["ch_names"][i]
                        for i in self._bad_channel_indices
                    ]

                self._reset_statistics()
                self._signal_sender.enough_collected.emit()

        if self.dsamp_factor and self.dsamp_factor > 1:
            in_data = self.parent.output
            in_antialiased = self._antialias_filter.apply(in_data)
            self.output = in_antialiased[
                :, self._left_n_pad :: self.dsamp_factor
            ]

            timestamps = self.traverse_back_and_find("timestamps")
            self.timestamps = timestamps[self._left_n_pad :: self.dsamp_factor]
            n_samp = in_data.shape[1]
            self._left_n_pad = (n_samp - self._left_n_pad) % self.dsamp_factor
            if self.output.size == 0:
                # Empty output disables processing for children which
                # decreases update time, so the next chunk will be small
                # again and downsampled output will be zero again.
                # Wait for at leas dsamp_factor samples to avoid this
                wait_time = (
                    self.dsamp_factor / self._upstream_mne_info["sfreq"]
                )
                time.sleep(wait_time)

        else:
            self.output = self.parent.output

    def reset_bads(self):
        self.mne_info["bads"] = self._upstream_mne_info["bads"]
        self._bad_channel_indices = []
        self.bad_channels = []

    @property
    def _samples_to_be_collected(self):
        frequency = self._upstream_mne_info["sfreq"]
        return int(math.ceil(self.collect_for_x_seconds * frequency))

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        if key == "collect_for_x_seconds":
            self._reset_statistics()
            output_history_is_no_longer_valid = False
        elif key == "dsamp_factor":
            self._initialize()
            output_history_is_no_longer_valid = True
        elif key == "bad_channels":
            output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _reset_statistics(self):
        self.is_collecting_samples = False
        self._samples_collected = 0
        self._enough_collected = False
        self._means = 0
        self._mean_sums_of_squares = 0
        self._bad_channel_indices = []

    def _update_statistics(self):
        input_array = self.parent.output.astype(np.dtype("float64"))
        # Using float64 is necessary because otherwise rounding error
        # in recursive formula accumulate
        n = self._samples_collected
        m = input_array.shape[TIME_AXIS]  # number of new samples
        self._samples_collected += m

        self._means = (
            self._means * n + np.sum(input_array, axis=TIME_AXIS)
        ) / (n + m)
        self._mean_sums_of_squares = (
            self._mean_sums_of_squares * n
            + np.sum(input_array ** 2, axis=TIME_AXIS)
        ) / (n + m)

    def _calculate_standard_deviations(self):
        n = self._samples_collected
        return np.sqrt(
            n / (n - 1) * (self._mean_sums_of_squares - self._means ** 2)
        )

    def _on_input_history_invalidation(self):
        self._reset_statistics()

    def _check_value(self, key, value):
        pass


class _InverseSolverNode(ProcessorNode):
    ALLOWED_CHILDREN = (
        "EnvelopeExtractor",
        "SignalViewer",
        "BrainViewer",
        "AtlasViewer",
        "LSLStreamOutput",
        "FileOutput",
        "SeedCoherence",
    )

    def __init__(self, fwd_path=None, subject=None, subjects_dir=None):
        ProcessorNode.__init__(self)
        self.fwd_path = fwd_path
        self.subjects_dir = subjects_dir
        self.subject = subject

    def _get_forward_subject_and_subjects_dir(self):
        if not (self.fwd_path and self.subject and self.subjects_dir):
            self._signal_sender.open_fwd_dialog.emit()

    def _set_channel_locations_in_root_data_info(self):
        # bads should be set up and should include channels missing from fwd
        data_info = deepcopy(self._upstream_mne_info)
        fwd_info = self._fwd["info"]

        DATA_CHNAMES = [c.upper() for c in data_info["ch_names"]]
        DATA_BADS = [c.upper() for c in data_info["bads"]]
        FWD_CHNAMES = [c.upper() for c in fwd_info["ch_names"]]

        for i, c in enumerate(DATA_CHNAMES):
            if c not in DATA_BADS:
                try:
                    i_fwd_ch = FWD_CHNAMES.index(c)
                    data_info["chs"][i]["loc"] = fwd_info["chs"][i_fwd_ch][
                        "loc"
                    ]
                except Exception as exc:
                    self._logger.exception(exc)

        self.root.montage_info = data_info

    def _initialize(self):
        mne_info = deepcopy(self.traverse_back_and_find("mne_info"))
        self._upstream_mne_info = mne_info
        self._get_forward_subject_and_subjects_dir()

        # -------------- setup forward -------------- #
        try:
            self._fwd, self._missing_ch_names = get_clean_forward(
                self.fwd_path, mne_info
            )
        except ValueError:
            self.fwd_path = None
            self.subject = None
            self.subjects_dir = None
            self._get_forward_subject_and_subjects_dir()
            self._fwd, self._missing_ch_names = get_clean_forward(
                self.fwd_path, mne_info
            )

        self._upstream_mne_info["bads"] = list(
            set(self._upstream_mne_info["bads"] + self._missing_ch_names)
        )
        self._bad_channels = self._upstream_mne_info["bads"]
        self._set_channel_locations_in_root_data_info()


class MNE(_InverseSolverNode):
    SUPPORTED_METHODS = ["MNE", "dSPM", "sLORETA"]
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)
    CHANGES_IN_THESE_REQUIRE_RESET = (
        "fwd_path",
        "snr",
        "method",
        "subjects_dir",
        "subject",
    )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {"mne_info": channel_labels_saver}

    def __init__(
        self,
        fwd_path=None,
        snr=1.0,
        method="MNE",
        depth=None,
        loose=1,
        fixed=False,
        subjects_dir=None,
        subject=None,
    ):
        _InverseSolverNode.__init__(
            self, subjects_dir=subjects_dir, subject=subject, fwd_path=fwd_path
        )

        self.snr = snr
        self._default_forward_model_file_path = None
        self._upstream_mne_info = None
        self.mne_info = None
        self._fwd = None

        # self._inverse_model_matrix = None
        self.method = method
        self.loose = loose
        self.depth = depth
        self.fixed = fixed
        self.viz_type = "source time series"

    def _initialize(self):
        _InverseSolverNode._initialize(self)

        self.inverse_operator = make_inverse_operator(
            self._fwd,
            self._upstream_mne_info,
            depth=self.depth,
            loose=self.loose,
            fixed=self.fixed,
        )
        self._lambda2 = 1.0 / self.snr ** 2
        self.inverse_operator = prepare_inverse_operator(
            self.inverse_operator,
            nave=100,
            lambda2=self._lambda2,
            method=self.method,
        )
        self._inverse_model_matrix = matrix_from_inverse_operator(
            inverse_operator=self.inverse_operator,
            mne_info=self._upstream_mne_info,
            snr=self.snr,
            method=self.method,
        )

        frequency = self._upstream_mne_info["sfreq"]
        # channel_count = self._inverse_model_matrix.shape[0]
        channel_count = self._fwd["nsource"]
        channel_labels = [
            "vertex #{}".format(i + 1) for i in range(channel_count)
        ]
        self.mne_info = mne.create_info(channel_labels, frequency)

    def _update(self):
        mne_info = self._upstream_mne_info
        bads = mne_info["bads"]
        if bads != self._bad_channels:
            self._logger.info(
                "Found new bad channels {};".format(bads)
                + "updating inverse operator"
            )
            self.inverse_operator = make_inverse_operator(
                self._fwd,
                mne_info,
                depth=self.depth,
                loose=self.loose,
                fixed=self.fixed,
            )
            self.inverse_operator = prepare_inverse_operator(
                self.inverse_operator,
                nave=100,
                lambda2=self._lambda2,
                method=self.method,
            )
            self._bad_channels = bads

        input_array = self.parent.output
        raw_array = mne.io.RawArray(input_array, mne_info, verbose="ERROR")
        raw_array.pick_types(eeg=True, meg=False, stim=False, exclude="bads")
        data = raw_array.get_data()
        self.output = self._apply_inverse_model_matrix(data)
        # stc = apply_inverse_raw(
        #     raw_array,
        #     self.inverse_operator,
        #     lambda2=self._lambda2,
        #     method=self.method,
        #     prepared=True,
        # )
        # self.output = stc.data

    def _on_input_history_invalidation(self):
        # The methods implemented in this node do not rely on past inputs
        pass

    def _check_value(self, key, value):
        if key == "method":
            if value not in self.SUPPORTED_METHODS:
                raise ValueError(
                    "Method {} is not supported.".format(value)
                    + " Use one of: {}".format(self.SUPPORTED_METHODS)
                )

        if key == "snr":
            if value <= 0:
                raise ValueError(
                    "snr (signal-to-noise ratio) must be a positive number."
                )

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _apply_inverse_model_matrix(self, input_array: np.ndarray):
        W = self._inverse_model_matrix  # VERTICES x CHANNELS
        output_array = W.dot(make_time_dimension_second(input_array))
        return put_time_dimension_back_from_second(output_array)


class LinearFilter(ProcessorNode):
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)
    CHANGES_IN_THESE_REQUIRE_RESET = ("lower_cutoff", "upper_cutoff")
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {
        "mne_info": lambda info: (info["nchan"],)
    }
    ALLOWED_CHILDREN = (
        "MNE",
        "MCE",
        "Beamformer",
        "SignalViewer",
        "EnvelopeExtractor",
        "LSLStreamOutput",
        "FileOutput",
    )

    def __init__(self, lower_cutoff: float = 1, upper_cutoff: float = 50):
        ProcessorNode.__init__(self)
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff
        self._linear_filter = None  # type: filters.ButterFilter
        self.viz_type = None

    def _initialize(self):
        self.viz_type = self.parent.viz_type
        mne_info = self.traverse_back_and_find("mne_info")
        frequency = mne_info["sfreq"]
        channel_count = mne_info["nchan"]
        if not (self.lower_cutoff is None and self.upper_cutoff is None):
            band = (self.lower_cutoff, self.upper_cutoff)

            self._linear_filter = filters.ButterFilter(
                band, fs=frequency, n_channels=channel_count
            )

            self._linear_filter.apply = pynfb_ndarray_function_wrapper(
                self._linear_filter.apply
            )
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

        elif key == "lower_cutoff":
            if (
                hasattr(self, "upper_cutoff")
                and self.upper_cutoff is not None
                and value > self.upper_cutoff
            ):
                raise ValueError(
                    "Lower cutoff can`t be set higher that the upper cutoff"
                )
            if value < 0:
                raise ValueError("Lower cutoff must be a positive number")

        elif key == "upper_cutoff":
            if (
                hasattr(self, "upper_cutoff")
                and self.lower_cutoff is not None
                and value < self.lower_cutoff
            ):
                raise ValueError(
                    "Upper cutoff can`t be set lower that the lower cutoff"
                )
            if value < 0:
                raise ValueError("Upper cutoff must be a positive number")

    def _on_input_history_invalidation(self):
        # Reset filter delays
        if self._linear_filter is not None:
            self._linear_filter.reset()

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid


class EnvelopeExtractor(ProcessorNode):
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)
    CHANGES_IN_THESE_REQUIRE_RESET = ("method", "factor")
    SUPPORTED_METHODS = ("Exponential smoothing",)
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {
        "mne_info": lambda info: (info["nchan"],)
    }
    ALLOWED_CHILDREN = ("SignalViewer", "LSLStreamOutput", "FileOutput")

    def __init__(self, factor=0.9, method="Exponential smoothing"):
        ProcessorNode.__init__(self)
        self.method = method
        self.factor = factor
        self._envelope_extractor = None  # type: ExponentialMatrixSmoother
        self.viz_type = None

    def _initialize(self):
        channel_count = self.traverse_back_and_find("mne_info")["nchan"]
        self._envelope_extractor = ExponentialMatrixSmoother(
            factor=self.factor, column_count=channel_count
        )
        self._envelope_extractor.apply = pynfb_ndarray_function_wrapper(
            self._envelope_extractor.apply
        )

        self.viz_type = self.parent.viz_type
        if self.parent.viz_type == "source time series":
            self.ALLOWED_CHILDREN = (
                "BrainViewer",
                "LSLStreamOutput",
                "FileOutput",
            )
        elif self.parent.viz_type == "connectivity":
            self.ALLOWED_CHILDREN = (
                "ConnectivityViewer",
                "LSLStreamOutput",
                "FileOutput",
            )

    def _update(self):
        input_data = self.parent.output
        self.output = self._envelope_extractor.apply(np.abs(input_data))

    def _check_value(self, key, value):
        if key == "factor":
            if value <= 0 or value >= 1:
                raise ValueError("Factor must be a number between 0 and 1")

        if key == "method":
            if value not in self.SUPPORTED_METHODS:
                raise ValueError(
                    "Method {} is not supported."
                    + " Use one of: {}".format(value, self.SUPPORTED_METHODS)
                )

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        self._envelope_extractor.reset()


class Beamformer(_InverseSolverNode):
    """Adaptive and nonadaptive beamformer"""

    SUPPORTED_OUTPUT_TYPES = ("power", "activation")
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)
    CHANGES_IN_THESE_REQUIRE_RESET = (
        "reg",
        "output_type",
        "is_adaptive",
        "fixed_orientation",
        "fwd_path",
        "whiten",
        "subject",
        "subjects_dir",
        "forgetting_factor_per_second",
    )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {"mne_info": channel_labels_saver}

    def __init__(
        self,
        output_type="power",
        is_adaptive=False,
        fixed_orientation=True,
        forgetting_factor_per_second=0.99,
        reg=0.05,
        whiten=False,
        fwd_path=None,
        subject=None,
        subjects_dir=None,
    ):
        _InverseSolverNode.__init__(
            self, subjects_dir=subjects_dir, subject=subject
        )
        self.output_type = output_type  # type: np.dtype
        self.is_adaptive = is_adaptive  # type: bool
        self.fixed_orientation = fixed_orientation  # type: bool
        self.mne_info = None  # type: mne.Info
        self.whiten = whiten
        self.reg = reg
        self.forgetting_factor_per_second = forgetting_factor_per_second

        self.fwd_path = fwd_path
        self._default_forward_model_file_path = None  # type: str

        self._channel_indices = None  # type: list
        self._gain_matrix = None  # type: np.ndarray
        self._data_cov = None  # type: np.ndarray
        self._forgetting_factor_per_sample = None  # type: float

        self.viz_type = "source time series"
        self._noise_cov = None

    def _initialize(self):
        # self.fwd_dialog_signal_sender.open_dialog.emit()
        # raise Exception("BAD FORWARD + DATA COMBINATION!")
        # raise Exception
        _InverseSolverNode._initialize(self)
        self._gain_matrix = self._fwd["sol"]["data"]
        G = self._gain_matrix
        # ------------------------------------------- #

        Rxx = G.dot(G.T) / 1e22

        goods = mne.pick_types(
            self._upstream_mne_info, eeg=True, meg=False, exclude="bads"
        )
        ch_names = [self._upstream_mne_info["ch_names"][i] for i in goods]

        self._data_cov = mne.Covariance(
            Rxx,
            ch_names,
            self._upstream_mne_info["bads"],
            self._upstream_mne_info["projs"],
            nfree=1,
        )

        if self.whiten:
            self._noise_cov = mne.Covariance(
                G.dot(G.T),
                ch_names,
                self._upstream_mne_info["bads"],
                self._upstream_mne_info["projs"],
                nfree=1,
            )
        else:
            self._noise_cov = None

        frequency = self._upstream_mne_info["sfreq"]
        self._forgetting_factor_per_sample = np.power(
            self.forgetting_factor_per_second, 1 / frequency
        )

        n_vert = self._fwd["nsource"]
        channel_labels = ["vertex #{}".format(i + 1) for i in range(n_vert)]

        # downstream info
        self.mne_info = mne.create_info(channel_labels, frequency)

        self._initialized_as_adaptive = self.is_adaptive
        self._initialized_as_fixed = self.fixed_orientation

        self.fwd_surf = mne.convert_forward_solution(
            self._fwd, surf_ori=True, force_fixed=False
        )
        self._compute_filters(self._upstream_mne_info)

    def _update(self):
        t1 = time.time()
        input_array = self.parent.output
        raw_array = mne.io.RawArray(
            input_array, self._upstream_mne_info, verbose="ERROR"
        )

        raw_array.pick_types(eeg=True, meg=False, stim=False, exclude="bads")
        raw_array.set_eeg_reference(ref_channels="average", projection=True)
        t2 = time.time()
        self._logger.timing(
            "Prepare arrays in {:.1f} ms".format((t2 - t1) * 1000)
        )

        if self.is_adaptive:
            self._update_covariance_matrix(input_array)
            t1 = time.time()
            self._compute_filters(raw_array.info)
            t2 = time.time()
            self._logger.timing(
                "Assembled lcmv instance in {:.1f} ms".format((t2 - t1) * 1000)
            )

        self._filters["source_nn"] = []
        t1 = time.time()
        stc = apply_lcmv_raw(
            raw=raw_array, filters=self._filters, max_ori_out="signed"
        )
        t2 = time.time()
        self._logger.timing(
            "Applied lcmv inverse in {:.1f} ms".format((t2 - t1) * 1000)
        )

        output = stc.data
        t1 = time.time()
        if self.fixed_orientation is True:
            if self.output_type == "power":
                output = output ** 2
        else:
            vertex_count = self.fwd_surf["nsource"]
            output = np.sum(
                np.power(output, 2).reshape((vertex_count, 3, -1)), axis=1
            )
            if self.output_type == "activation":
                output = np.sqrt(output)

        self.output = output
        t2 = time.time()
        self._logger.timing("Finalized in {:.1f} ms".format((t2 - t1) * 1000))

    def _compute_filters(self, info):
        self._filters = make_lcmv(
            info=info,
            forward=self.fwd_surf,
            data_cov=self._data_cov,
            reg=self.reg,
            noise_cov=self._noise_cov,  # data whiten
            pick_ori="max-power",
            weight_norm="unit-noise-gain",
            reduce_rank=False,
        )

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:

        # Only change adaptiveness or fixed_orientation requires reinit
        # if (self._initialized_as_adaptive is not self.is_adaptive
        #         or self._initialized_as_fixed is not self.fixed_orientation):
        # if old_val != new_val:  # we don't expect numpy arrays here
        if key in ("reg",):
            self._compute_filters(self._upstream_mne_info)
        else:
            self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        # Only adaptive version relies on history
        if self._initialized_as_adaptive is True:
            self.initialize()

    def _check_value(self, key, value):
        if key == "output_type":
            if value not in self.SUPPORTED_OUTPUT_TYPES:
                raise ValueError(
                    "Method {} is not supported."
                    + " Use one of: {}".format(
                        value, self.SUPPORTED_OUTPUT_TYPES
                    )
                )

        if key == "reg":
            if value <= 0:
                raise ValueError(
                    "reg (covariance regularization coefficient)"
                    " must be a positive number"
                )

        if key == "is_adaptive":
            if not isinstance(value, bool):
                raise ValueError(
                    "Beamformer type (adaptive vs nonadaptive) is not set"
                )

    def _update_covariance_matrix(self, input_array):
        t1 = time.time()
        alpha = self._forgetting_factor_per_sample
        sample_count = input_array.shape[TIME_AXIS]
        self._logger.timing("Number of samples: {}".format(sample_count))
        new_Rxx_data = self._data_cov.data

        raw_array = mne.io.RawArray(
            input_array, self._upstream_mne_info, verbose="ERROR"
        )
        raw_array.pick_types(eeg=True, meg=False, stim=False, exclude="bads")
        raw_array.set_eeg_reference(ref_channels="average", projection=True)
        input_array_nobads = raw_array.get_data()

        t2 = time.time()
        self._logger.timing(
            "Prepared covariance update in {:.2f} ms".format((t2 - t1) * 1000)
        )
        samples = make_time_dimension_second(input_array_nobads).T
        new_Rxx_data = alpha * new_Rxx_data + (1 - alpha) * samples.T.dot(
            samples
        )
        t3 = time.time()
        self._logger.timing(
            "Updated matrix data in {:.2f} ms".format((t3 - t2) * 1000)
        )

        self._data_cov = mne.Covariance(
            new_Rxx_data,
            self._data_cov.ch_names,
            raw_array.info["bads"],
            raw_array.info["projs"],
            nfree=1,
        )
        t4 = time.time()
        self._logger.timing(
            "Created instance of covariance"
            + " in {:.2f} ms".format((t4 - t4) * 1000)
        )


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


class MCE(_InverseSolverNode):
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()
    CHANGES_IN_THESE_REQUIRE_RESET = (
        "fwd_path",
        "snr",
        "n_comp",
        "subjects_dir",
        "subject",
    )

    def __init__(
        self,
        snr=1.0,
        fwd_path=None,
        n_comp=40,
        subjects_dir=None,
        subject=None,
    ):
        _InverseSolverNode.__init__(
            self, subjects_dir=subjects_dir, subject=subject, fwd_path=fwd_path
        )
        self.snr = snr
        self.fwd_path = fwd_path
        self.n_comp = n_comp
        self.mne_info = None
        self._upstream_mne_info = None
        self.input_data = []
        self.output = []
        self.viz_type = "source time series"
        # pass

    def _initialize(self):
        # self.fwd_dialog_signal_sender.open_dialog.emit()
        _InverseSolverNode._initialize(self)

        fwd_fix = mne.convert_forward_solution(
            self._fwd, surf_ori=True, force_fixed=False
        )

        self._gain_matrix = fwd_fix["sol"]["data"]

        self._logger.info("Computing SVD of the forward operator")
        U, S, V = svd(self._gain_matrix)

        Sn = np.zeros([self.n_comp, V.shape[0]])
        Sn[: self.n_comp, : self.n_comp] = np.diag(S[: self.n_comp])

        self.Un = U[:, : self.n_comp]
        self.A_non_ori = Sn @ V
        # ---------------------------------------------------- #

        # -------- leadfield dims -------- #
        N_SEN = self._gain_matrix.shape[0]
        # -------------------------------- #

        # ------------------------ noise-covariance ------------------------ #
        cov_data = np.identity(N_SEN)
        ch_names = np.array(self._upstream_mne_info["ch_names"])[
            mne.pick_types(self._upstream_mne_info, eeg=True, meg=False)
        ]
        ch_names = list(ch_names)
        noise_cov = mne.Covariance(
            cov_data,
            ch_names,
            self._upstream_mne_info["bads"],
            self._upstream_mne_info["projs"],
            nfree=1,
        )
        # ------------------------------------------------------------------ #

        self.mne_inv = mne_make_inverse_operator(
            self._upstream_mne_info,
            fwd_fix,
            noise_cov,
            depth=0.8,
            loose=1,
            fixed=False,
            verbose="ERROR",
        )
        self.Sn = Sn
        self.V = V
        channel_count = self._fwd["nsource"]
        channel_labels = [
            "vertex #{}".format(i + 1) for i in range(channel_count)
        ]
        self.mne_info = mne.create_info(
            channel_labels, self._upstream_mne_info["sfreq"]
        )
        self._upstream_mne_info = self._upstream_mne_info

    def _update(self):
        input_array = self.parent.output
        # last_slice = last_sample(input_array)
        last_slice = np.mean(input_array, axis=1)
        n_src = self.mne_inv["nsource"]
        n_times = input_array.shape[1]
        output_mce = np.empty([n_src, n_times])

        raw_slice = mne.io.RawArray(
            np.expand_dims(last_slice, axis=1),
            self._upstream_mne_info,
            verbose="ERROR",
        )
        raw_slice.pick_types(eeg=True, meg=False, stim=False, exclude="bads")
        raw_slice.set_eeg_reference(ref_channels="average", projection=True)

        # ------------------- get dipole orientations --------------------- #
        stc_slice = apply_inverse_raw(
            raw_slice,
            self.mne_inv,
            pick_ori="vector",
            method="MNE",
            lambda2=1,
            verbose="ERROR",
        )
        Q = normalize(stc_slice.data[:, :, 0])  # dipole orientations
        # ----------------------------------------------------------------- #

        # -------- setup linprog params -------- #
        n_sen = self.A_non_ori.shape[0]
        A_eq = np.empty([n_sen, n_src])
        for i in range(n_src):
            A_eq[:, i] = self.A_non_ori[:, i * 3 : (i + 1) * 3] @ Q[i, :].T
        data_slice = raw_slice.get_data()[:, 0]
        b_eq = self.Un.T @ data_slice
        c = np.ones(A_eq.shape[1])
        # -------------------------------------- #

        with nostdout():
            sol = linprog(
                c,
                A_eq=A_eq,
                b_eq=b_eq,
                method="interior-point",
                bounds=(0, None),
                options={"disp": False},
            )
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
        if key == "snr":
            if value <= 0:
                raise ValueError(
                    "snr (signal-to-noise ratio) must be a positive number."
                )


class ICARejection(ProcessorNode):
    ALLOWED_CHILDREN = (
        "SignalViewer",
        "LinearFilter",
        "MNE",
        "MCE",
        "Beamformer",
        "EnvelopeExtractor",
        "LSLStreamOutput",
        "FileOutput",
    )

    CHANGES_IN_THESE_REQUIRE_RESET = ("collect_for_x_seconds",)

    def __init__(self, collect_for_x_seconds: int = 10):
        ProcessorNode.__init__(self)
        self.collect_for_x_seconds = collect_for_x_seconds  # type: int

        self._samples_collected = None  # type: int
        self._enough_collected = None  # type: bool

        self._reset_statistics()
        self._ica_rejector = None
        self._ica_hfreq = 50
        self._ica_lfreq = 1
        self._signal_sender = Communicate()
        self._signal_sender.open_dialog.connect(self._on_ica_finished)
        self.is_collecting_samples = False
        self.ica_dialog = None

        self.viz_type = "sensor time series"

    def _on_input_history_invalidation(self):
        self._reset_statistics()

    def _check_value(self, key, value):
        pass

    def _initialize(self):
        self._upstream_mne_info = self.traverse_back_and_find("mne_info")
        self._frequency = self._upstream_mne_info["sfreq"]
        self._good_ch_inds = mne.pick_types(
            self._upstream_mne_info,
            eeg=True,
            meg=False,
            stim=False,
            exclude="bads",
        )

        channels = self._upstream_mne_info["chs"]
        self._ch_locs = np.array([ch["loc"] for ch in channels])

        n_ch = len(self._good_ch_inds)
        self._collected_timeseries = np.zeros(
            [n_ch, self._samples_to_be_collected]
        )
        mne_info = self.traverse_back_and_find("mne_info")

        lowpass = mne_info["lowpass"]
        if lowpass and lowpass < self._ica_hfreq:
            self._ica_hfreq = None
            self._logger.debug("Setting lowpass for ICA filter to None")
        highpass = mne_info["highpass"]
        if highpass and highpass > self._ica_lfreq:
            self._logger.debug("Setting highpass for ICA filter to None")
            self._ica_lfreq = None

    @property
    def _samples_to_be_collected(self):
        return int(math.ceil(self.collect_for_x_seconds * self._frequency))

    def reset_rejector(self):
        if self._ica_rejector is not None:
            self._logger.info("ICA artefacts rejection is stopped.")
            self._ica_rejector = None
            # trigger reset
            self._reset_buffer.append(("dummy", "dummy", "dummy"))
        else:
            self._logger.info("ICA artefacts rejection is already inactive")
            self.root._signal_sender.request_message.emit(
                "ICA artefacts rejection is already inactive", "", "info"
            )
        if hasattr(self, "ica_dialog") and self.ica_dialog is not None:
            self.ica_dialog.deleteLater()
            self.ica_dialog = None

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self._reset_statistics()
        if self.has_ica:
            output_history_is_no_longer_valid = True
        else:
            output_history_is_no_longer_valid = False
        if key == "collect_for_x_seconds":
            n_ch = len(self._good_ch_inds)
            self._collected_timeseries = np.zeros(
                [n_ch, self._samples_to_be_collected]
            )
        return output_history_is_no_longer_valid

    def _reset_statistics(self):
        self._samples_collected = 0
        self._enough_collected = False
        self.is_collecting_samples = False

    def _update(self):
        input_array = self.parent.output
        if self.is_collecting_samples:
            self.output = input_array

            # Have we collected enough samples without the new input?
            enough_collected = (
                self._samples_collected >= self._samples_to_be_collected
            )
            if not enough_collected:
                if (
                    self.parent.output is not None
                    and self.parent.output.shape[TIME_AXIS] > 0
                ):
                    self._update_statistics()

            elif not self._enough_collected:  # We just got enough samples
                try:
                    self._upstream_mne_info = self.root.montage_info
                    self._good_ch_inds = mne.pick_types(
                        self._upstream_mne_info,
                        eeg=True,
                        meg=False,
                        stim=False,
                        exclude="bads",
                    )
                    channels = self._upstream_mne_info["chs"]
                    self._ch_locs = np.array([ch["loc"] for ch in channels])

                    self._enough_collected = True
                    self._logger.info("Collected enough samples")
                    self._signal_sender.open_dialog.emit()
                    self._reset_statistics()
                    self._signal_sender.enough_collected.emit()
                except AttributeError as exc:
                    self._logger.exception(exc)

        else:
            if self.has_ica:
                self.output = np.dot(
                    self._ica_rejector, input_array[self._good_ch_inds, :]
                )
            else:
                self.output = self.parent.output

    def _on_ica_finished(self):
        # executed on the main thread
        self.ica_dialog = ICADialog(
            self._collected_timeseries.T,
            list(
                np.array(self._upstream_mne_info["ch_names"])[
                    self._good_ch_inds
                ]
            ),
            self._ch_locs[self._good_ch_inds, :],
            self._frequency,
            band=(self._ica_lfreq, self._ica_hfreq),
        )
        self.ica_dialog.spatial_button.hide()
        self.ica_dialog.sliders.hide()
        self.ica_dialog.add_to_all_checkbox.hide()
        self.ica_dialog.update_band_checkbox.hide()
        self.ica_dialog.exec_()
        if self.ica_dialog.result():
            self._ica_rejector = self.ica_dialog.rejection.val.T
            # Hack to trigger reset after since we start to apply ica rejector
            self._reset_buffer.append(("dummy", "dummy", "dummy"))

    @property
    def has_ica(self):
        return (
            hasattr(self, "_ica_rejector") and self._ica_rejector is not None
        )

    def _update_statistics(self):
        input_array = self.parent.output.astype(np.dtype("float64"))
        n = self._samples_collected
        m = input_array.shape[TIME_AXIS]  # number of new samples
        n_samp_remain = self._collected_timeseries.shape[1] - n
        if n_samp_remain < m:
            m = n_samp_remain
        self._samples_collected += m
        self._collected_timeseries[:, n : n + m] = input_array[
            self._good_ch_inds, :m
        ]
        # Using float64 is necessary because otherwise rounding error
        # in recursive formula accumulate

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {"mne_info": channel_labels_saver}


class AtlasViewer(ProcessorNode):
    CHANGES_IN_THESE_REQUIRE_RESET = ("active_label_names", "parc")
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = (
        "subjects_dir",
        "subject",
    )
    ALLOWED_CHILDREN = (
        "EnvelopeExtractor",
        "SignalViewer",
        "LSLStreamOutput",
        "Coherence",
        "FileOutput",
    )

    def __init__(self, parc="aparc", active_label_names=[]):
        ProcessorNode.__init__(self)
        self.parc = parc
        self.subjects_dir = None
        self.subject = None
        self.active_label_names = active_label_names

        self.viz_type = "roi time series"

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        if key == "parc":
            self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        pass

    def _initialize(self):

        self.subject = self.traverse_back_and_find("subject")
        self.subjects_dir = self.traverse_back_and_find("subjects_dir")
        self._read_annotation()

        self.sfreq = self.traverse_back_and_find("mne_info")["sfreq"]
        self._signal_sender.initialized.emit()

    @property
    def mne_info(self):
        return {
            "ch_names": self.active_label_names,
            "nchan": len(self.active_label_names),
            "sfreq": self.sfreq,
            "bads": [],
        }

    def _read_annotation(self):
        """
        Read freesurfer annotation files.

        Map sources for which we solve the inv problem to the dense cortex
        which we use for plotting.

        """
        fwd_path = self.traverse_back_and_find("fwd_path")
        forward_solution = mne.read_forward_solution(fwd_path, verbose="ERROR")
        sources_idx, vert, _, rh_offset = get_mesh_data_from_forward_solution(
            forward_solution
        )
        self._fwd_vertices = vert
        # rh_offset = number of vertices in left hemisphere;
        # used for indexing the right hemisphere sources when both
        # hemispheres are stored together intead of being split in lh and rh as
        # in freesurfer
        try:
            labels = mne.read_labels_from_annot(
                self.subject,
                parc=self.parc,
                surf_name="white",
                subjects_dir=self.subjects_dir,
            )

            label_names = {l.name for l in labels}
            if not set(self.active_label_names).intersection(label_names):
                self.active_label_names = []

            for i, l in enumerate(labels):
                l.mass_center = l.center_of_mass(
                    subject=self.subject, subjects_dir=self.subjects_dir
                )
                if l.hemi == "rh":
                    l.vertices += rh_offset
                    l.mass_center += rh_offset
                l.forward_vertices = np.where(
                    np.isin(sources_idx, l.vertices)
                )[0]
                l.is_active = l.name in self.active_label_names
            self.labels = labels  # label objects read from filesystem

        except Exception as e:
            self._logger.exception(e)
            raise e

    def _update(self):
        data = self.parent.output

        n_times = data.shape[1]
        n_active_labels = len(self.active_label_names)

        data_label = np.zeros([n_active_labels, n_times])
        active_labels = [
            l for l in self.labels if l.name in self.active_label_names
        ]
        for i, l in enumerate(active_labels):
            # Average inverse solution inside label
            # label_mask = self.source_labels == label['id']
            data_label[i, :] = np.mean(data[l.forward_vertices, :], axis=0)
        self.output = data_label
        # self._logger.debug("Output data shape is %s" % str(data.shape))

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

    CHANGES_IN_THESE_REQUIRE_RESET = ("method", "factor", "seed")
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {
        "mne_info": lambda info: (info["nchan"],)
    }

    def __init__(self, method=None, factor=0.9, seed=None):
        ProcessorNode.__init__(self)
        self.method = method
        self._envelope_extractor = None
        self.factor = factor
        self.seed = seed
        if seed:
            self.viz_type = "source time series"
        else:
            self.viz_type = "connectivity"

    def _initialize(self):
        channel_count = self.traverse_back_and_find("mne_info")["nchan"]
        if self.seed:
            assert self.seed < channel_count, (
                "Seed index {} exceeds max"
                " channel number {}".format(self.seed, channel_count)
            )
        self._logger.debug("Channel count: %d" % channel_count)
        self._envelope_extractor = ExponentialMatrixSmoother(
            factor=self.factor, column_count=channel_count
        )
        self._envelope_extractor.apply = pynfb_ndarray_function_wrapper(
            self._envelope_extractor.apply
        )

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
                self.output = (seed_env.dot(envs_z.T) / (n_times - ddof))[
                    :, np.newaxis
                ]
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

    CHANGES_IN_THESE_REQUIRE_RESET = ("method", "seed")
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()
    ALLOWED_CHILDREN = ("FileOutput",)

    def __init__(self, method="coh", seed=None):
        ProcessorNode.__init__(self)
        self.method = method
        self.viz_type = "connectivity"
        self.ALLOWED_CHILDREN = self.ALLOWED_CHILDREN + ("ConnectivityViewer",)

    def _initialize(self):
        pass

    def _update(self):
        input_data = self.parent.output
        hilbert = sc.signal.hilbert(input_data, axis=1)

        Cp = hilbert.dot(hilbert.conj().T)
        D = np.sqrt(np.diag(Cp))
        coh = Cp / np.outer(D, D)

        if self.method == "imcoh":
            self.output = coh.imag
        elif self.method == "coh":
            self.output = np.abs(coh)

        timestamps = self.traverse_back_and_find("timestamps")
        self.timestamps = timestamps[-2:-1]  # we want to get an array

        # self.output = np.zeros_like(self.output)
        # self.output[:, 6] = 0.5
        # self.output[6, :] = 0.5

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        return True

    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        pass


class SeedCoherence(AtlasViewer):
    CHANGES_IN_THESE_REQUIRE_RESET = ("seed", "parc")
    ALLOWED_CHILDREN = ("EnvelopeExtractor", "BrainViewer", "FileOutput")

    def __init__(self, seed=None, parc="aparc", method="coh"):
        if seed:
            active_label_names = [seed]
        else:
            active_label_names = []
        AtlasViewer.__init__(
            self, parc=parc, active_label_names=active_label_names
        )
        self.seed = seed
        self.method = method
        self.viz_type = "source time series"
        self._seed_ind = None

    def _initialize(self):
        self.subject = self.traverse_back_and_find("subject")
        self.subjects_dir = self.traverse_back_and_find("subjects_dir")
        self._read_annotation()
        self.sfreq = self.traverse_back_and_find("mne_info")["sfreq"]
        self._signal_sender.initialized.emit()
        self._get_seed_ind()

    @property
    def mne_info(self):
        return self.traverse_back_and_find("mne_info")

    def _update(self):
        input_data = self.parent.output
        self._get_seed_ind()

        if self._seed_ind is not None:
            hilbert = sc.signal.hilbert(input_data, axis=1)
            seed_Cp = hilbert[self._seed_ind, :].dot(hilbert.conj().T)
            D = np.sqrt(np.mean(hilbert * hilbert.conj(), axis=1))
            seed_outer_D = D[self._seed_ind] * D  # coherence denominator
            coh = seed_Cp[:, np.newaxis] / seed_outer_D[:, np.newaxis]
            if self.method == "imcoh":
                self.output = coh.imag
            elif self.method == "coh":
                self.output = np.abs(coh)
        else:
            self.output = input_data[:, -1, np.newaxis]
        timestamps = self.traverse_back_and_find("timestamps")
        self.timestamps = timestamps[-2:-1]  # we want to get an array

    def _get_seed_ind(self):
        seed_label = None
        for l in self.labels:
            if l.is_active and l.name == self.seed:
                seed_label = l
                break
        if seed_label:
            seed_fwd_vert = self._fwd_vertices[seed_label.forward_vertices, :]
            seed_label_center_xyz = seed_fwd_vert.mean(axis=0)
            cent_fwd_ind = np.argmin(
                ((seed_fwd_vert - seed_label_center_xyz) ** 2).sum(axis=1)
            )
            self._seed_ind = l.forward_vertices[cent_fwd_ind]
        else:
            self._seed_ind = None


class MneGcs(MNE):
    """
    Minimum norm fixed orientation inverse with geometric correction for
    signal leakage

    Parameters
    ----------
    seed: int
    fwd_path: str
    snr: float

    """

    CHANGES_IN_THESE_REQUIRE_RESET = MNE.CHANGES_IN_THESE_REQUIRE_RESET + (
        "seed",
    )

    def __init__(
        self, seed, snr=1.0, fwd_path=None, subjects_dir=None, subject=None
    ):
        method = "MNE"
        MNE.__init__(
            self,
            fwd_path=fwd_path,
            snr=snr,
            method=method,
            subjects_dir=subjects_dir,
            subject=subject,
        )
        self.seed = seed
        self.viz_type = "source time series"
        self.depth = None
        self.loose = 0
        self.fixed = True

    def _initialize(self):
        MNE._initialize(self)
        self._fwd_fixed = mne.convert_forward_solution(
            self._fwd, force_fixed=True, surf_ori=True
        )

    def _apply_inverse_model_matrix(self, input_array: np.ndarray):
        gain = self._fwd["sol"]["data"]
        seed_topo = gain[:, self.seed]
        W = self._inverse_model_matrix  # VERTICES x CHANNELS
        seed_filter = W[self.seed, :]
        input_array -= np.outer(
            seed_topo[:, np.newaxis],
            seed_filter[np.newaxis, :].dot(input_array),
        ) / seed_filter.dot(seed_topo)
        output_array = W.dot(input_array)
        return output_array
