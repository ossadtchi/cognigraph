"""
Definition of pipeline source nodes

Exposed classes
---------------
LSLStreamSource: SourceNode
    Input from an LSL stream
FileSource: SourceNode
    Input from a file

"""
import os
import time

import pylsl as lsl
import numpy as np
import mne

from cognigraph.utils.matrix_functions import get_a_time_slice
from .. import TIME_AXIS, DTYPE
from .node import SourceNode
from ..utils.lsl import (
    convert_lsl_chunk_to_numpy_array,
    read_channel_labels_from_info,
)
from ..utils.brainvision import (
    read_brain_vision_data,
    read_fif_data,
    read_edf_data,
)

__all__ = ("LSLStreamSource", "FileSource")


class _FixedStreamInfo(lsl.StreamInfo):
    def as_xml(self):
        return lsl.pylsl.lib.lsl_get_xml(self.obj).decode("utf-8", "ignore")


class _FixedStreamInlet(lsl.StreamInlet):
    def info(self, timeout=lsl.pylsl.FOREVER):
        errcode = lsl.pylsl.c_int()
        result = lsl.pylsl.lib.lsl_get_fullinfo(
            self.obj, lsl.pylsl.c_double(timeout), lsl.pylsl.byref(errcode)
        )
        lsl.pylsl.handle_error(errcode)
        return _FixedStreamInfo(handle=result)  # StreamInfo(handle=result)


class LSLStreamSource(SourceNode):
    """ Class for reading data from an LSL stream defined by its name """

    CHANGES_IN_THESE_REQUIRE_RESET = ("source_name",)
    MAX_SAMPLES_IN_CHUNK = 1024

    def _check_value(self, key, value):
        pass

    SECONDS_TO_WAIT_FOR_THE_STREAM = 0.5
    _GUI_STRING = "LSL stream"

    def __init__(self, source_name=None):
        super().__init__()
        self.source_name = source_name
        self._inlet = None  # type: lsl.StreamInlet
        self.timestamps = None

    @property
    def stream_name(self):
        return self.source_name

    @property
    def frequency(self):
        return self.mne_info["sfreq"]

    @stream_name.setter
    def stream_name(self, stream_name):
        self.source_name = stream_name

    def _initialize(self):
        if not self.source_name:
            raise ValueError("Please set LSL stream name.")

        stream_infos = lsl.resolve_byprop(
            "name",
            self.source_name,
            timeout=self.SECONDS_TO_WAIT_FOR_THE_STREAM,
        )
        if len(stream_infos) == 0:
            raise ValueError(
                "Cannot find LSL stream with name {}".format(self.source_name)
            )
        elif len(stream_infos) > 1:
            raise ValueError(
                "Multiple LSL streams with name {}.".format(self.source_name)
            )
        else:
            info = stream_infos[0]
            self._inlet = _FixedStreamInlet(info)
            self._inlet.open_stream()
            frequency = info.nominal_srate()
            self.dtype = DTYPE
            channel_labels, channel_types = read_channel_labels_from_info(
                self._inlet.info()
            )
            self.mne_info = mne.create_info(
                channel_labels, frequency, ch_types=channel_types
            )
            self.timestamps = []

    def _update(self):
        lsl_chunk, timestamps = self._inlet.pull_chunk()
        if len(timestamps) > self.MAX_SAMPLES_IN_CHUNK:
            timestamps = timestamps[: self.MAX_SAMPLES_IN_CHUNK]
            lsl_chunk = lsl_chunk[: self.MAX_SAMPLES_IN_CHUNK]
        self.output = convert_lsl_chunk_to_numpy_array(
            lsl_chunk, dtype=self.dtype
        )
        self.timestamps = timestamps


class FileSource(SourceNode):
    SUPPORTED_EXTENSIONS = {
        "Brainvision": (".vhdr", ".eeg", ".vmrk"),
        "MNE-python": (".fif",),
        "European Data Format": (".edf",),
    }

    CHANGES_IN_THESE_REQUIRE_RESET = ("file_path", "loop_the_file")
    _GUI_STRING = "File Source"

    MAX_SAMPLES_IN_CHUNK = 1024

    def __init__(self, file_path=None, loop_the_file=True):
        super().__init__()
        self.source_name = None
        self._file_path = None
        self.file_path = file_path  # This will also populate self.source_name
        self.data = None  # type: np.ndarray
        self.timestamps = None
        self.loop_the_file = loop_the_file
        self.is_alive = True

        self._time_of_the_last_update = None
        self._n_samples_already_read = None

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        if file_path in (None, ""):
            self._file_path = None
            self.source_name = None
        else:
            basename = os.path.basename(file_path)
            file_name, extension = os.path.splitext(basename)

            all_ext = ()
            for ext_group in self.SUPPORTED_EXTENSIONS.keys():
                all_ext = all_ext + self.SUPPORTED_EXTENSIONS[ext_group]

            if extension not in all_ext:
                raise ValueError(
                    "Cannot read {}.".format(basename)
                    + "Extension must be one of: {}".format(all_ext)
                )
            else:
                self._file_path = file_path
                self.source_name = file_name

    def _initialize(self):
        self._time_of_the_last_update = None
        self._n_samples_already_read = 0

        if self.file_path is not None:
            basename = os.path.basename(self.file_path)
            _, ext = os.path.splitext(basename)

            if ext in self.SUPPORTED_EXTENSIONS["Brainvision"]:
                self.data, self.mne_info, self.times = read_brain_vision_data(
                    file_path=self.file_path, time_axis=TIME_AXIS
                )

            elif ext in self.SUPPORTED_EXTENSIONS["MNE-python"]:
                self.data, self.mne_info, self.times = read_fif_data(
                    file_path=self.file_path, time_axis=TIME_AXIS
                )

            elif ext in self.SUPPORTED_EXTENSIONS["European Data Format"]:
                self.data, self.mne_info, self.times = read_edf_data(
                    file_path=self.file_path, time_axis=TIME_AXIS
                )

            else:
                raise ValueError(
                    "Cannot read {}.".format(basename)
                    + "Extension must be one of the following: {}".format(
                        self.SUPPORTED_EXTENSIONS.values()
                    )
                )

            self.dtype = DTYPE
            self.data = self.data.astype(self.dtype)
            self.timestamps = []
        else:
            raise ValueError("File path is not set.")

    def _update(self):
        if self.data is None:
            return

        current_time = time.time()

        if self._time_of_the_last_update is not None:

            seconds_since_last_update = (
                current_time - self._time_of_the_last_update
            )
            self._time_of_the_last_update = current_time
            frequency = self.mne_info["sfreq"]

            # How many sample we would like to read
            max_samples_in_chunk = np.int64(
                seconds_since_last_update * frequency
            )
            # Lower it to amount we can process in a reasonable amount of time
            max_samples_in_chunk = min(
                max_samples_in_chunk, self.MAX_SAMPLES_IN_CHUNK
            )

            # have to read max_samples_in_chunk samples unless we hit the end
            samples_in_data = self.data.shape[TIME_AXIS]
            stop_idx = self._n_samples_already_read + max_samples_in_chunk
            self.output = get_a_time_slice(
                self.data,
                start_idx=self._n_samples_already_read,
                stop_idx=stop_idx,
            )
            self.timestamps = self.times[
                self._n_samples_already_read : stop_idx
            ]
            actual_samples_in_chunk = self.output.shape[TIME_AXIS]
            self._n_samples_already_read = (
                self._n_samples_already_read + actual_samples_in_chunk
            )

            # If we do hit the end we need to either start again or
            # stop completely depending on loop_the_file
            if self._n_samples_already_read == samples_in_data:
                if self.loop_the_file is True:
                    self._n_samples_already_read = 0
                else:
                    self.is_alive = False

        else:
            self._time_of_the_last_update = current_time

    def _check_value(self, key, value):
        pass
