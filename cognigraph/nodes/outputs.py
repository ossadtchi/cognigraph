import os
import time
from types import SimpleNamespace

import tables
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication

import mne
import numpy as np
from scipy import sparse

from ..utils.pysurfer.smoothing_matrix import smoothing_matrix, mesh_edges
from .node import OutputNode
from .. import CHANNEL_AXIS, TIME_AXIS, PYNFB_TIME_AXIS
from ..utils.lsl import (convert_numpy_format_to_lsl,
                         convert_numpy_array_to_lsl_chunk,
                         create_lsl_outlet)
from ..utils.matrix_functions import last_sample, make_time_dimension_second
from ..utils.ring_buffer import RingBuffer
from ..utils.channels import read_channel_types, channel_labels_saver
from ..utils.inverse_model import get_mesh_data_from_forward_solution
from ..utils.brain_visualization import get_mesh_data_from_surfaces_dir
from vendor.nfb.pynfb.widgets.signal_viewers import RawSignalViewer

# visbrain visualization imports
# from ..gui.brain_visual import BrainMesh
from ..gui.connect_obj import ConnectObj
from ..gui.source_obj import SourceObj
from vispy import scene
# from vispy.app import Canvas

# import logging

# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
# from matplotlib.figure import Figure

# -------- gif recorder -------- #
from vispy.gloo.util import _screenshot
from PIL import Image as im
# ------------------------------ #


class Communicate(QObject):
    init_widget_sig = pyqtSignal()
    draw_sig = pyqtSignal('PyQt_PyObject')
    screenshot_sig = pyqtSignal()


class WidgetOutput(OutputNode):
    """Abstract class for widget initialization logic with qt signals"""
    def __init__(self, *pargs, **kwargs):
        OutputNode.__init__(self, *pargs, **kwargs)
        self.signal_sender = Communicate()
        self.signal_sender.init_widget_sig.connect(self._init_widget)
        self.signal_sender.draw_sig.connect(self.on_draw)

    def _init_widget(self):
        if self.widget is not None:
            parent = self.widget.parent()
            ind = parent.indexOf(self.widget)
            cur_width = self.widget.size().width()
            cur_height = self.widget.size().height()
            self.widget.deleteLater()
            self.widget = self._create_widget()
            parent.insertWidget(ind, self.widget)
            self.widget.resize(cur_width, cur_height)
        else:
            self.widget = self._create_widget()
            self.widget.setMinimumWidth(50)

    def _create_widget(self):
        raise NotImplementedError

    def on_draw(self):
        raise NotImplementedError


class LSLStreamOutput(OutputNode):

    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        pass  # TODO: check that value as a string usable as a stream name

    CHANGES_IN_THESE_REQUIRE_RESET = ('stream_name', )

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = (
        'source_name', 'mne_info', 'dtype')

    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = (
        {'mne_info': lambda info: (info['sfreq'], ) +
         channel_labels_saver(info)})

    def _reset(self):
        # It is impossible to change then name of an already
        # started stream so we have to initialize again
        self._should_reinitialize = True
        self.initialize()

    def __init__(self, stream_name=None):
        super().__init__()
        self._provided_stream_name = stream_name
        self.stream_name = None
        self._outlet = None

    def _initialize(self):
        # If no name was supplied we will use a modified
        # version of the source name (a file or a stream name)
        source_name = self.traverse_back_and_find('source_name')
        self.stream_name = (self._provided_stream_name or
                            (source_name + '_output'))

        # Get other info from somewhere down the predecessor chain
        dtype = self.traverse_back_and_find('dtype')
        channel_format = convert_numpy_format_to_lsl(dtype)
        mne_info = self.traverse_back_and_find('mne_info')
        frequency = mne_info['sfreq']
        channel_labels = mne_info['ch_names']
        channel_types = read_channel_types(mne_info)

        self._outlet = create_lsl_outlet(
            name=self.stream_name, frequency=frequency,
            channel_format=channel_format, channel_labels=channel_labels,
            channel_types=channel_types)

    def _update(self):
        chunk = self.parent.output
        lsl_chunk = convert_numpy_array_to_lsl_chunk(chunk)
        self._outlet.push_chunk(lsl_chunk)


class BrainViewer(WidgetOutput):

    CHANGES_IN_THESE_REQUIRE_RESET = ('buffer_length', 'take_abs', )
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = (
        'mne_forward_model_file_path', 'mne_info')

    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

    LIMITS_MODES = SimpleNamespace(GLOBAL='Global', LOCAL='Local',
                                   MANUAL='Manual')

    def __init__(self, take_abs=True, limits_mode=LIMITS_MODES.LOCAL,
                 buffer_length=1, threshold_pct=50, surfaces_dir=None):
        super().__init__()

        self.limits_mode = limits_mode
        self.lock_limits = False
        self.buffer_length = buffer_length
        self.take_abs = take_abs
        self.colormap_limits = SimpleNamespace(lower=None, upper=None)
        self._threshold_pct = threshold_pct

        self._limits_buffer = None
        self.surfaces_dir = surfaces_dir
        self.mesh_data = None
        self.smoothing_matrix = None
        self.widget = None
        self.output = None

        # -------- gif recorder -------- #
        self.is_recording = False
        self.sector = None

        self._start_time = None
        self._display_time = None  # Time in ms between switching images

        self._images = []
        self.signal_sender.screenshot_sig.connect(self._append_screenshot)
        # ------------------------------ #

    def _initialize(self):
        mne_forward_model_file_path = self.traverse_back_and_find(
            'mne_forward_model_file_path')

        frequency = self.traverse_back_and_find('mne_info')['sfreq']
        buffer_sample_count = np.int(self.buffer_length * frequency)
        self._limits_buffer = RingBuffer(row_cnt=2, maxlen=buffer_sample_count)

        self.forward_solution = mne.read_forward_solution(
            mne_forward_model_file_path, verbose='ERROR')
        self.mesh_data = get_mesh_data_from_surfaces_dir(self.surfaces_dir)
        self.signal_sender.init_widget_sig.emit()
        self.smoothing_matrix = self._get_smoothing_matrix(
            mne_forward_model_file_path)

    def _on_input_history_invalidation(self):
        self._should_reset = True
        self.reset()

    def _check_value(self, key, value):
        pass

    def _reset(self):
        self._limits_buffer.clear()

    @property
    def threshold_pct(self):
        return self._threshold_pct

    @threshold_pct.setter
    def threshold_pct(self, value):
        self._threshold_pct = value

    def _update(self):
        sources = self.parent.output
        self.output = sources
        if self.take_abs:
            sources = np.abs(sources)
        self._update_colormap_limits(sources)
        normalized_sources = self._normalize_sources(last_sample(sources))
        self.signal_sender.draw_sig.emit(normalized_sources)

        if self.is_recording:
            self.signal_sender.screenshot_sig.emit()

    def _update_colormap_limits(self, sources):
        self._limits_buffer.extend(np.array([
            make_time_dimension_second(np.min(sources, axis=CHANNEL_AXIS)),
            make_time_dimension_second(np.max(sources, axis=CHANNEL_AXIS)),
        ]))

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
        if self.smoothing_matrix is not None:
            sources_smoothed = self.smoothing_matrix.dot(normalized_values)
        else:
            self.logger.debug('Draw without smoothing')
            sources_smoothed = normalized_values
        threshold = self.threshold_pct / 100
        mask = sources_smoothed <= threshold

        # reset colors to white
        self.mesh_data._alphas[:, :] = 0.
        self.mesh_data._alphas_buffer.set_data(self.mesh_data._alphas)

        if np.any(~mask):
            self.mesh_data.add_overlay(sources_smoothed[~mask],
                                       vertices=np.where(~mask)[0],
                                       to_overlay=1)

        self.mesh_data.update()
        if self.logger.getEffectiveLevel() == 20:  # INFO level
            self.canvas.measure_fps(
                window=10,
                callback=(lambda x:
                          self.logger.info('Updating at %1.1f FPS' % x)))

    def _create_widget(self):
        canvas = scene.SceneCanvas(keys='interactive', show=False)
        self.canvas = canvas

        # Add a ViewBox to let the user zoom/rotate
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 50
        view.camera.distance = 400
        # Make light follow the camera
        self.mesh_data.shared_program.frag['camtf'] = view.camera.transform
        view.add(self.mesh_data)
        return canvas.native

    def _get_smoothing_matrix(self, mne_forward_model_file_path):
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
        # in self.mesh_data.vertexes()[sources_idx[j], :]
        smoothing_matrix_file_path = (
            os.path.splitext(mne_forward_model_file_path)[0] +
            '-smoothing-matrix.npz')
        try:
            return sparse.load_npz(smoothing_matrix_file_path)
        except FileNotFoundError:
            self.logger.info('Calculating smoothing matrix.' +
                             ' This might take a while the first time.')
            sources_idx, *_ = get_mesh_data_from_forward_solution(
                self.forward_solution)
            adj_mat = mesh_edges(self.mesh_data._faces)
            smoothing_mat = smoothing_matrix(sources_idx, adj_mat)
            sparse.save_npz(smoothing_matrix_file_path, smoothing_mat)
            return smoothing_mat

    def _start_gif(self):
        self._images = []
        self._start_time = time.time()

        self.is_recording = True

    def _stop_gif(self):
        self.is_recording = False
        # self._timer.stop()

        duration = time.time() - self._start_time
        self._display_time = (duration * 1000) / len(self._images)

    def _save_gif(self, path):
        self._images[0].save(
            path,
            save_all=True,
            append_images=self._images[1:],
            duration=self._display_time,
            loop=0)

    def _append_screenshot(self):
        if self.sector is None:
            # self._images.append(ImageGrab.grab())
            self._images.append(im.fromarray(_screenshot()))
        else:
            # self._images.append(ImageGrab.grab(bbox=self.sector))
            self._images.append(im.fromarray(_screenshot()))


class SignalViewer(WidgetOutput):
    CHANGES_IN_THESE_REQUIRE_RESET = ()

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info',)
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

    def __init__(self):
        super().__init__()
        self.widget = None

    def _initialize(self):
        self.signal_sender.init_widget_sig.emit()

    def _create_widget(self):
        mne_info = self.traverse_back_and_find('mne_info')
        if mne_info['nchan']:
            return RawSignalViewer(fs=mne_info['sfreq'],
                                   names=mne_info['ch_names'],
                                   seconds_to_plot=10)
        else:
            return RawSignalViewer(fs=mne_info['sfreq'],
                                   names=[''],
                                   seconds_to_plot=10)

    def _update(self):
        chunk = self.parent.output
        self.signal_sender.draw_sig.emit(chunk)

    def on_draw(self, chunk):
        QApplication.processEvents()
        if chunk.size:
            if TIME_AXIS == PYNFB_TIME_AXIS:
                self.widget.update(chunk)
            else:
                self.widget.update(chunk.T)

    def _reset(self) -> bool:
        # Nothing to reset, really
        pass

    def _on_input_history_invalidation(self):
        # Don't really care, will draw whatever
        pass

    def _check_value(self, key, value):
        # Nothing to be set
        pass


class FileOutput(OutputNode):

    CHANGES_IN_THESE_REQUIRE_RESET = ('stream_name', )

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info':
                                           lambda info: (info['sfreq'], ) +
                                           channel_labels_saver(info)}

    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        pass  # TODO: check that value as a string usable as a stream name

    def _reset(self):
        self._should_reinitialize = True
        self.initialize()

    def __init__(self, output_fname='output.h5'):
        super().__init__()
        self.output_fname = output_fname
        self.out_file = None

    def _initialize(self):
        if self.out_file:  # for resets
            self.out_file.close()

        info = self.traverse_back_and_find('mne_info')
        col_size = info['nchan']
        self.out_file = tables.open_file(self.output_fname, mode='w')
        atom = tables.Float64Atom()

        self.output_array = self.out_file.create_earray(
            self.out_file.root, 'data', atom, (col_size, 0))

    def _update(self):
        chunk = self.parent.output
        self.output_array.append(chunk)


class TorchOutput(OutputNode):

    CHANGES_IN_THESE_REQUIRE_RESET = ()
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    def _on_input_history_invalidation(self):
        pass

    def _check_value(self, key, value):
        pass  # TODO: check that value as a string usable as a stream name

    def _reset(self):
        pass

    def _initialize(self):
        pass

    def _update(self):
        import torch
        self.output = torch.from_numpy(self.parent.output)


class ConnectivityViewer(WidgetOutput):
    """Plot connectivity matrix on circular graph"""
    CHANGES_IN_THESE_REQUIRE_RESET = ()
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info',)

    def __init__(self, surfaces_dir, n_lines=30):
        super().__init__()
        self.mesh = None
        self.widget = None
        self.s_obj = None
        self.c_obj = None
        self.view = None
        self.n_lines = n_lines
        self.surfaces_dir = surfaces_dir

    def _initialize(self):
        self.mne_info = self.traverse_back_and_find('mne_info')
        self.mesh = get_mesh_data_from_surfaces_dir(self.surfaces_dir,
                                                    translucent=True)
        self.signal_sender.init_widget_sig.emit()

    def _update(self):
        input_data = np.abs(self.parent.output)  # connectivity matrix
        # 1. Get n_lines stronges connections indices (i, j)
        # get only off-diagonal elements
        l_triang = np.tril(input_data, k=-1)
        nl = self.n_lines
        ii, jj = np.unravel_index(
             np.argpartition(-l_triang, nl, axis=None)[:nl], l_triang.shape)
        # 2. Get corresponding vertices indices
        nodes_inds = np.unique(np.r_[ii, jj])
        labels = self.traverse_back_and_find('labels')
        nodes_inds_surf = np.array([labels[i].mass_center for i in nodes_inds])
        # 3. Get nodes = xyz of these vertices
        nodes = self.mesh._vertices[nodes_inds_surf]
        # 4. Edges are input data restricted to best n_lines nodes
        edges = input_data[nodes_inds[:, None], nodes_inds]  # None needed
        # 5. Select = mask matrix with True in (i,j)-th positions
        select = np.zeros_like(input_data, dtype=bool)
        select[ii, jj] = True
        select = select[nodes_inds[:, None], nodes_inds]
        select += select.T
        nchan = self.mne_info['nchan']
        assert input_data.shape == (nchan, nchan), ('Number of channels doesnt'
                                                    ' conform to input data'
                                                    ' shape')
        try:
            self.s_obj._sources.visible = False
        except Exception:
            pass
        try:
            self.c_obj._connect.visible = False
        except Exception:
            pass

        self.s_obj = SourceObj(
            'sources', nodes, color='#ab4642', radius_min=10.)

        self.c_obj = ConnectObj(
            'default', nodes, edges, select=select, line_width=2.,
            cmap='Spectral_r', color_by='strength')

        self.view.add(self.s_obj._sources)
        self.view.add(self.c_obj._connect)

    def _reset(self):
        ...

    def _on_input_history_invalidation(self):
        ...

    def _check_value(self, key, value):
        ...

    def _create_widget(self):
        canvas = scene.SceneCanvas(keys='interactive', show=False)
        self.canvas = canvas

        # Add a ViewBox to let the user zoom/rotate
        self.view = canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 50
        self.view.camera.distance = 400
        # Make light follow the camera
        self.mesh.shared_program.frag['camtf'] = self.view.camera.transform
        self.view.add(self.mesh)
        return canvas.native
