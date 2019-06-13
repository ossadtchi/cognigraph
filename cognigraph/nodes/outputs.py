"""
Definition of pipeline output nodes

Exposed classes
---------------
LSLStreamOutput: OutputNode
    Output signal to LSL stream
BrainViewer: _WidgetOutput
    Plot heatmap on a 3d brain
SignalViewer: _WidgetOutput
    Plot signals
FileOutput: OutputNode
    Output signal to file
TorchOutput: OutputNode
    Wrap signal in Torch tensors
ConnectivityViewer: _WidgetOutput
    Plot connectivity

"""
import os
import time
from types import SimpleNamespace

import tables
from PyQt5.QtWidgets import QApplication

import mne
import numpy as np
from scipy import sparse

from ..utils.pysurfer.smoothing_matrix import smoothing_matrix, mesh_edges
from .node import OutputNode
from .. import CHANNEL_AXIS, TIME_AXIS, PYNFB_TIME_AXIS
from ..utils.lsl import (
    convert_numpy_format_to_lsl,
    convert_numpy_array_to_lsl_chunk,
    create_lsl_outlet,
)
from ..utils.matrix_functions import last_sample, make_time_dimension_second
from ..utils.ring_buffer import RingBuffer
from ..utils.channels import read_channel_types, channel_labels_saver
from ..utils.inverse_model import get_mesh_data_from_forward_solution
from ..utils.brain_visualization import get_mesh_data_from_surfaces_dir
from vendor.nfb.pynfb.widgets.signal_viewers import RawSignalViewer

from ..gui.connect_obj import ConnectObj
from ..gui.source_obj import SourceObj
from vispy import scene


# -------- gif recorder -------- #
from PIL import Image as im

# ------------------------------ #

__all__ = (
    "LSLStreamOutput",
    "BrainViewer",
    "SignalViewer",
    "FileOutput",
    "TorchOutput",
    "ConnectivityViewer",
)


class _WidgetOutput(OutputNode):
    """Abstract class for widget initialization logic with qt signals"""

    def __init__(self, *pargs, **kwargs):
        OutputNode.__init__(self, *pargs, **kwargs)
        self._signal_sender.init_widget_sig.connect(self._init_widget)
        self._signal_sender.draw_sig.connect(self.on_draw)

    def _init_widget(self):
        if self.widget and self.widget.parent():
            parent = self.widget.parent()
            old_widget = self.widget
        else:
            parent = None
        self.widget = self._create_widget()
        if parent:
            parent.setWidget(self.widget)
            old_widget.deleteLater()
        else:
            self.root._signal_sender.node_widget_added.emit(
                self.widget, repr(self)
            )
        self.widget.pipeline_node = self

    def _create_widget(self):
        raise NotImplementedError

    def on_draw(self):
        raise NotImplementedError


class LSLStreamOutput(OutputNode):
    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        pass  # TODO: check that value as a string usable as a stream name

    CHANGES_IN_THESE_REQUIRE_RESET = ("stream_name",)

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = (
        "source_name",
        "mne_info",
        "dtype",
    )

    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {
        "mne_info": lambda info: (info["sfreq"],) + channel_labels_saver(info)
    }

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        # It is impossible to change then name of an already
        # started stream so we have to initialize again
        self.initialize()

    def __init__(self, stream_name=None):
        super().__init__()
        self._provided_stream_name = stream_name
        self.stream_name = None
        self._outlet = None

    def _initialize(self):
        # If no name was supplied use a modified
        # version of the source name (a file or a stream name)
        source_name = self.traverse_back_and_find("source_name")
        if not self.stream_name:
            self.stream_name = source_name + "_output"

        # Get other info from somewhere down the predecessor chain
        dtype = self.traverse_back_and_find("dtype")
        channel_format = convert_numpy_format_to_lsl(dtype)
        mne_info = self.traverse_back_and_find("mne_info")
        frequency = mne_info["sfreq"]
        channel_labels = mne_info["ch_names"]
        channel_types = read_channel_types(mne_info)

        self._outlet = create_lsl_outlet(
            name=self.stream_name,
            frequency=frequency,
            channel_format=channel_format,
            channel_labels=channel_labels,
            channel_types=channel_types,
        )

    def _update(self):
        chunk = self.parent.output
        lsl_chunk = convert_numpy_array_to_lsl_chunk(chunk)
        self._outlet.push_chunk(lsl_chunk)


class BrainViewer(_WidgetOutput):

    CHANGES_IN_THESE_REQUIRE_RESET = (
        "buffer_length",
        "take_abs",
        "limits_mode",
        "threshold_pct",
    )
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = (
        "fwd_path",
        "mne_info",
    )

    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {"mne_info": channel_labels_saver}

    LIMITS_MODES = SimpleNamespace(
        GLOBAL="Global", LOCAL="Local", MANUAL="Manual"
    )

    def __init__(
        self,
        take_abs=True,
        limits_mode=LIMITS_MODES.LOCAL,
        buffer_length=1,
        threshold_pct=50,
    ):
        super().__init__()

        self.limits_mode = limits_mode
        self.lock_limits = False
        self.buffer_length = buffer_length
        self.take_abs = take_abs
        self.colormap_limits = SimpleNamespace(lower=None, upper=None)
        self.threshold_pct = threshold_pct

        self._limits_buffer = None
        self.surfaces_dir = None
        self._mesh = None
        self._smoothing_matrix = None
        self.widget = None
        self.output = None

        # -------- gif recorder -------- #
        self.is_recording = False
        self.sector = None

        self._start_time = None
        self._display_time = None  # Time in ms between switching images

        self._images = []
        self._signal_sender.screenshot_sig.connect(self._append_screenshot)
        # ------------------------------ #

    def _initialize(self):
        fwd_path = self.traverse_back_and_find("fwd_path")
        subject = self.traverse_back_and_find("subject")
        subjects_dir = self.traverse_back_and_find("subjects_dir")
        self.surfaces_dir = os.path.join(subjects_dir, subject)

        frequency = self.traverse_back_and_find("mne_info")["sfreq"]
        buffer_sample_count = np.int(self.buffer_length * frequency)
        self._limits_buffer = RingBuffer(row_cnt=2, maxlen=buffer_sample_count)

        self.forward_solution = mne.read_forward_solution(
            fwd_path, verbose="ERROR"
        )
        self._mesh = get_mesh_data_from_surfaces_dir(self.surfaces_dir)
        self._signal_sender.init_widget_sig.emit()
        self._smoothing_matrix = self._get_smoothing_matrix(fwd_path)

    def _on_input_history_invalidation(self):
        # TODO: change min-max buffer values
        pass

    def _check_value(self, key, value):
        pass

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        self._limits_buffer.clear()

    def _update(self):
        sources = self.parent.output
        self.output = sources
        if self.take_abs:
            sources = np.abs(sources)
        self._update_colormap_limits(sources)
        normalized_sources = self._normalize_sources(last_sample(sources))
        self._signal_sender.draw_sig.emit(normalized_sources)

        if self.is_recording:
            self._signal_sender.screenshot_sig.emit()

    def _update_colormap_limits(self, sources):
        self._limits_buffer.extend(
            np.array(
                [
                    make_time_dimension_second(
                        np.min(sources, axis=CHANNEL_AXIS)
                    ),
                    make_time_dimension_second(
                        np.max(sources, axis=CHANNEL_AXIS)
                    ),
                ]
            )
        )

        if self.limits_mode == self.LIMITS_MODES.GLOBAL:
            mins, maxs = self._limits_buffer.data
            self.colormap_limits.lower = np.percentile(mins, q=5)
            self.colormap_limits.upper = np.percentile(maxs, q=95)
        elif self.limits_mode == self.LIMITS_MODES.LOCAL:
            sources = last_sample(sources)
            self.colormap_limits.lower = np.min(sources)
            self.colormap_limits.upper = np.max(sources)
        elif self.limits_mode == self.LIMITS_MODES.MANUAL:
            pass

    def _normalize_sources(self, last_sources):
        minimum = self.colormap_limits.lower
        maximum = self.colormap_limits.upper
        if minimum == maximum:
            return last_sources * 0
        else:
            return (last_sources - minimum) / (maximum - minimum)

    def on_draw(self, normalized_values):
        QApplication.processEvents()
        if self._smoothing_matrix is not None:
            sources_smoothed = self._smoothing_matrix.dot(normalized_values)
        else:
            self._logger.debug("Draw without smoothing")
            sources_smoothed = normalized_values
        threshold = self.threshold_pct / 100
        mask = sources_smoothed <= threshold

        # reset colors to white
        self._mesh._alphas[:, :] = 0.0
        self._mesh._alphas_buffer.set_data(self._mesh._alphas)

        if np.any(~mask):
            self._mesh.add_overlay(
                sources_smoothed[~mask],
                vertices=np.where(~mask)[0],
                to_overlay=1,
            )

        self._mesh.update()
        # if self._logger.getEffectiveLevel() == 20:  # INFO level
        self.canvas.measure_fps(
            window=10, callback=self._signal_sender.fps_updated.emit
        )

    def _create_widget(self):
        canvas = scene.SceneCanvas(keys="interactive", show=False)
        self.canvas = canvas

        # Add a ViewBox to let the user zoom/rotate
        view = canvas.central_widget.add_view()
        view.camera = "turntable"
        view.camera.fov = 50
        view.camera.distance = 400
        # Make light follow the camera
        self._mesh.shared_program.frag["camtf"] = view.camera.transform
        view.add(self._mesh)
        return canvas.native

    def _get_smoothing_matrix(self, fwd_path):
        """
        Creates or loads a smoothing matrix that lets us
        interpolate source values onto all mesh vertices

        """
        # Not all the vertices in the forward solution mesh are sources.
        # sources_idx actually indexes into the union of
        # high-definition meshes for left and right hemispheres.
        # The smoothing matrix then lets us assign a color to each vertex.
        # If in future we decide to use low-definition mesh from
        # the forward model for drawing, we should index into that.
        # Shorter: the coordinates of the jth source are
        # in self._mesh.vertexes()[sources_idx[j], :]
        smoothing_matrix_file_path = (
            os.path.splitext(fwd_path)[0] + "-smoothing-matrix.npz"
        )
        try:
            return sparse.load_npz(smoothing_matrix_file_path)
        except FileNotFoundError:
            self._logger.info(
                "Calculating smoothing matrix."
                + " This might take a while the first time."
            )
            sources_idx, *_ = get_mesh_data_from_forward_solution(
                self.forward_solution
            )
            adj_mat = mesh_edges(self._mesh._faces)
            smoothing_mat = smoothing_matrix(sources_idx, adj_mat)
            sparse.save_npz(smoothing_matrix_file_path, smoothing_mat)
            return smoothing_mat

    def _start_gif(self):
        self._images = []
        self._gif_times = []
        self._gif_start_time = time.time()

        self.is_recording = True

    def _stop_gif(self):
        self.is_recording = False

        duration = time.time() - self._gif_start_time
        self._display_time = (duration * 1000) / len(self._images)

    def _save_gif(self, path):
        try:
            self._images[0].save(
                path,
                save_all=True,
                append_images=self._images[1:],
                duration=self._display_time,
                loop=0,
            )

            base, ext = os.path.splitext(path)
            times_savepath = base + "_gif_times.txt"
            with open(times_savepath, "w") as f:
                for t in self._gif_times:
                    f.write("%1.3f\n" % t)
        except Exception as e:
            self._logger.exception(e)
            self._root._signal_sender.request_message.emit(
                "Saving gif to %s failed!" % path, str(e), "error"
            )

    def _append_screenshot(self):
        last_sample_time = self.traverse_back_and_find("timestamps")[-1]
        self._gif_times.append(last_sample_time)
        self._images.append(im.fromarray(self.canvas.render()))


class SignalViewer(_WidgetOutput):
    CHANGES_IN_THESE_REQUIRE_RESET = ()

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {"mne_info": channel_labels_saver}

    def __init__(self):
        super().__init__()
        self.widget = None

    def _initialize(self):
        self._signal_sender.init_widget_sig.emit()

    def _create_widget(self):
        mne_info = self.traverse_back_and_find("mne_info")
        if mne_info["nchan"]:
            return RawSignalViewer(
                fs=mne_info["sfreq"],
                names=mne_info["ch_names"],
                seconds_to_plot=10,
            )
        else:
            return RawSignalViewer(
                fs=mne_info["sfreq"], names=[""], seconds_to_plot=10
            )

    def _update(self):
        chunk = self.parent.output
        self._signal_sender.draw_sig.emit(chunk)

    def on_draw(self, chunk):
        QApplication.processEvents()
        if chunk.size:
            if TIME_AXIS == PYNFB_TIME_AXIS:
                self.widget.update(chunk)
            else:
                self.widget.update(chunk.T)

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        # Nothing to reset, really
        pass

    def _on_input_history_invalidation(self):
        # Doesn't really care, will draw anything
        pass

    def _check_value(self, key, value):
        # Nothing to be set
        pass


class FileOutput(OutputNode):

    CHANGES_IN_THESE_REQUIRE_RESET = ("output_path",)

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {
        "mne_info": lambda info: (info["sfreq"],) + channel_labels_saver(info)
    }

    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        pass  # TODO: check that value as a string usable as a stream name

    def _on_critical_attr_change(self, key, old_val, new_val):
        self.initialize()

    def __init__(self, output_path="cognigraph_output.h5"):
        OutputNode.__init__(self)
        self.output_path = output_path
        self._out_file = None

    def _initialize(self):
        if self._out_file:  # for resets
            self._out_file.close()

        info = self.traverse_back_and_find("mne_info")
        col_size = info["nchan"]
        self._out_file = tables.open_file(self.output_path, mode="w")
        atom = tables.Float64Atom()

        self.output_array = self._out_file.create_earray(
            self._out_file.root, "data", atom, (col_size, 0)
        )
        self.timestamps_array = self._out_file.create_earray(
            self._out_file.root, "timestamps", atom, (1, 0)
        )
        self.ch_names = self._out_file.create_array(
            self._out_file.root,
            "ch_names",
            np.array(info["ch_names"]),
            "Channel names in data",
        )
        self._out_file.root.data.attrs.sfreq = info["sfreq"]
        try:
            fwd = self.traverse_back_and_find("_fwd")
            self._out_file.create_array(
                "src_xyz",
                fwd['source_rr'],
                "Source space coordinates",
            )
        except Exception:
            pass

    def toggle(self):
        if self.disabled:
            self._start()
        else:
            self._stop()

    def _stop(self):
        self._out_file.close()
        self.disabled = True

    def _start(self):
        self.disabled = False
        self._initialize()

    def _update(self):
        data_chunk = self.parent.output
        timestamps = np.array(self.traverse_back_and_find("timestamps"))[
            np.newaxis, :
        ]
        self.output_array.append(data_chunk)
        self.timestamps_array.append(timestamps)


class TorchOutput(OutputNode):

    CHANGES_IN_THESE_REQUIRE_RESET = ()
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        pass  # TODO: check that value as a string usable as a stream name

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        pass

    def _initialize(self):
        pass

    def _update(self):
        import torch

        self.output = torch.from_numpy(self.parent.output)


class ConnectivityViewer(_WidgetOutput):
    """Plot connectivity matrix on glass brain"""

    CHANGES_IN_THESE_REQUIRE_RESET = ("n_lines",)
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ("mne_info",)

    def __init__(self, n_lines=30):
        super().__init__()
        self._mesh = None
        self.widget = None
        self.s_obj = None
        self.c_obj = None
        self.view = None
        self.n_lines = n_lines

    def _initialize(self):
        self.mne_info = self.traverse_back_and_find("mne_info")
        subject = self.traverse_back_and_find("subject")
        subjects_dir = self.traverse_back_and_find("subjects_dir")
        self.surfaces_dir = os.path.join(subjects_dir, subject)

        self._mesh = get_mesh_data_from_surfaces_dir(
            self.surfaces_dir, translucent=True
        )
        self._signal_sender.init_widget_sig.emit()

    def _update(self):
        input_data = np.abs(self.parent.output)  # connectivity matrix
        # 1. Get n_lines stronges connections indices (i, j)
        # get only off-diagonal elements
        l_triang = np.tril(input_data, k=-1)
        nl = self.n_lines
        n_ch = input_data.shape[0]
        nl_max = int(n_ch * (n_ch - 1) / 2)
        if nl > nl_max:
            nl = nl_max
        ii, jj = np.unravel_index(
            np.argpartition(-l_triang, nl, axis=None)[:nl], l_triang.shape
        )
        # 2. Get corresponding vertices indices
        nodes_inds = np.unique(np.r_[ii, jj])
        labels = self.traverse_back_and_find("labels")
        active_labels = [l for l in labels if l.is_active]
        nodes_inds_surf = np.array(
            [active_labels[i].mass_center for i in nodes_inds]
        )
        # 3. Get nodes = xyz of these vertices
        nodes = self._mesh._vertices[nodes_inds_surf]
        # 4. Edges are input data restricted to best n_lines nodes
        edges = input_data[nodes_inds[:, None], nodes_inds]  # None needed
        # 5. Select = mask matrix with True in (i,j)-th positions
        select = np.zeros_like(input_data, dtype=bool)
        select[ii, jj] = True
        select = select[nodes_inds[:, None], nodes_inds]
        select += select.T
        nchan = self.mne_info["nchan"]
        assert input_data.shape == (
            nchan,
            nchan,
        ), "Number of channels doesnt conform to input data shape"
        try:
            self.s_obj._sources.visible = False
        except Exception:
            pass
        try:
            self.c_obj._connect.visible = False
        except Exception:
            pass

        self.s_obj = SourceObj(
            "sources", nodes, color="#ab4642", radius_min=20.0
        )

        self.c_obj = ConnectObj(
            "default",
            nodes,
            edges,
            select=select,
            line_width=2.0,
            cmap="Spectral_r",
            color_by="strength",
        )
        self._signal_sender.draw_sig.emit(None)

    def on_draw(self):
        self.view.add(self.s_obj._sources)
        self.view.add(self.c_obj._connect)

    def _on_critical_attr_change(self, key, old_val, new_val) -> bool:
        pass

    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        pass

    def _create_widget(self):
        canvas = scene.SceneCanvas(keys="interactive", show=False)
        self.canvas = canvas

        # Add a ViewBox to let the user zoom/rotate
        self.view = canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.fov = 50
        self.view.camera.distance = 400
        # Make light follow the camera
        self._mesh.shared_program.frag["camtf"] = self.view.camera.transform
        self.view.add(self._mesh)
        return canvas.native
