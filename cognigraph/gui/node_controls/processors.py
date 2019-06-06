import os.path as op
from PyQt5.QtCore import Qt
import logging
from ...nodes import processors
from ...utils.pyqtgraph import MyGroupParameter, parameterTypes
from ..widgets import RoiSelectionDialog
from ...utils.brain_visualization import get_mesh_data_from_surfaces_dir
from ..forward_dialog import FwdSetupDialog

__all__ = (
    "PreprocessingControls",
    "LinearFilterControls",
    "MNEControls",
    "EnvelopeExtractorControls",
    "BeamformerControls",
    "MCEControls",
    "ICARejectionControls",
    "AtlasViewerControls",
    "AmplitudeEnvelopeCorrelationsControls",
    "CoherenceControls",
)


class _ProcessorNodeControls(MyGroupParameter):

    DISABLED_NAME = "Disable: "

    @property
    def PROCESSOR_CLASS(self):
        raise NotImplementedError

    @property
    def CONTROLS_LABEL(self):
        raise NotImplementedError

    def __init__(self, processor_node: PROCESSOR_CLASS = None, **kwargs):
        super().__init__(name=self.CONTROLS_LABEL, **kwargs)

        if processor_node is None:
            raise ValueError(
                "Right now we can create controls only"
                " for an already existing node"
            )

        self._processor_node = processor_node  # type: self.PROCESSOR_CLASS
        self._create_parameters()
        self._add_disable_parameter()

        self._logger = logging.getLogger(type(self).__name__)
        self._logger.debug("Constructor called")

    def _create_parameters(self):
        raise NotImplementedError

    def _add_disable_parameter(self):
        disabled_value = False  # TODO: change once disabling is implemented
        disabled = parameterTypes.SimpleParameter(
            type="bool",
            name=self.DISABLED_NAME,
            value=disabled_value,
            readonly=False,
        )
        disabled.sigValueChanged.connect(self._on_disabled_changed)
        self.disabled = self.addChild(disabled)
        self._processor_node._signal_sender.disabled_changed.connect(
            self.disabled.setValue
        )

    def _on_disabled_changed(self, param, value):
        self._processor_node.disabled = value


class PreprocessingControls(_ProcessorNodeControls):
    PROCESSOR_CLASS = processors.Preprocessing
    CONTROLS_LABEL = "Preprocessing"

    DURATION_NAME = "Baseline duration: "
    DSAMP_FREQ_NAME = "Downsample factor: "

    BUTTON_START_STR = "Find bad channels"
    BUTTON_ABORT_STR = "Abort data collection"
    RESET_BADS_BUTTON_NAME = "Reset bad channels"

    def _create_parameters(self):

        duration_value = self._processor_node.collect_for_x_seconds
        duration = parameterTypes.SimpleParameter(
            type="float",
            name=self.DURATION_NAME,
            suffix="s",
            limits=(5, 180),
            value=duration_value,
        )
        self.duration = self.addChild(duration)
        self.duration.sigValueChanged.connect(self._on_duration_changed)

        # max_sfreq = self._processor_node.traverse_back_and_find("info")
        dsamp_factor_combo = parameterTypes.SimpleParameter(
            type="int",
            name=self.DSAMP_FREQ_NAME,
            suffix="Hz",
            limits=(1, 4),
            value=1,
        )
        self.dsamp_factor_combo = self.addChild(dsamp_factor_combo)
        self.dsamp_factor_combo.sigValueChanged.connect(
            self._on_dsamp_factor_changed
        )
        self._processor_node._signal_sender.initialized.connect(
            self._reset_combo
        )

        find_bads_button = parameterTypes.ActionParameter(
            type="action", name=self.BUTTON_START_STR
        )
        self.find_bads_button = self.addChild(find_bads_button)
        self.find_bads_button.sigActivated.connect(self._on_find_bads_clicked)

        bads = parameterTypes.ListParameter(
            name="Bad channels",
            values=["Waiting for initialization"],
            value=None,
        )
        self.bads = self.addChild(bads)
        self._processor_node._signal_sender.enough_collected.connect(
            self._on_enough_collected
        )

        reset_bads_button = parameterTypes.ActionParameter(
            type="action", name=self.RESET_BADS_BUTTON_NAME
        )
        reset_bads_button.sigActivated.connect(self._processor_node.reset_bads)
        reset_bads_button.sigActivated.connect(self._reset_combo)
        self._reset_bads_button = self.addChild(reset_bads_button)

    def _on_duration_changed(self, param, value):
        self._processor_node.collect_for_x_seconds = value

    def _on_dsamp_factor_changed(self, param, value):
        self._processor_node.dsamp_factor = value

    def _on_find_bads_clicked(self):
        self.find_bads_button.setName(self.BUTTON_ABORT_STR)
        self.find_bads_button.setReadonly()
        self._processor_node.is_collecting_samples = True

    def _reset_combo(self):
        if not self._processor_node._enough_collected:
            self.removeChild(self.bads)
            bads = parameterTypes.ListParameter(
                name="Bad channels",
                values=self._processor_node.bad_channels,
            )
            self.bads = self.addChild(bads)

    def _on_enough_collected(self):
        self.removeChild(self.bads)
        bads = parameterTypes.ListParameter(
            name="Bad channels", values=self._processor_node.mne_info["bads"]
        )
        self.bads = self.addChild(bads)
        self.find_bads_button.setName(self.BUTTON_START_STR)


class LinearFilterControls(_ProcessorNodeControls):
    PROCESSOR_CLASS = processors.LinearFilter
    CONTROLS_LABEL = "Linear filter"

    LOWER_CUTOFF_NAME = "Lower cutoff: "
    UPPER_CUTOFF_NAME = "Upper cutoff: "

    def _create_parameters(self):

        lower_cutoff_value = self._processor_node.lower_cutoff
        upper_cutoff_value = self._processor_node.upper_cutoff

        lower_cutoff = parameterTypes.SimpleParameter(
            type="float",
            name=self.LOWER_CUTOFF_NAME,
            suffix="Hz",
            limits=(0, upper_cutoff_value - 0.01),
            value=lower_cutoff_value,
        )
        upper_cutoff = parameterTypes.SimpleParameter(
            type="float",
            name=self.UPPER_CUTOFF_NAME,
            suffix="Hz",
            limits=(lower_cutoff_value, 100),
            value=upper_cutoff_value,
        )

        self.lower_cutoff = self.addChild(lower_cutoff)
        self.upper_cutoff = self.addChild(upper_cutoff)

        lower_cutoff.sigValueChanged.connect(self._on_lower_cutoff_changed)
        upper_cutoff.sigValueChanged.connect(self._on_upper_cutoff_changed)

    def _on_lower_cutoff_changed(self, param, value):
        # Update the node
        if value == 0.0:
            self._processor_node.lower_cutoff = None
        else:
            self._processor_node.lower_cutoff = value
            # TODO: implement on the filter side
        # Update the upper cutoff so that it is not lower that the lower one
        if self.upper_cutoff.value() != 0.0:
            self.upper_cutoff.setLimits((value, 100))

    def _on_upper_cutoff_changed(self, param, value):
        # Update the node
        if value == 0.0:
            self._processor_node.upper_cutoff = None
            value = 100
        else:
            self._processor_node.upper_cutoff = value
            # TODO: implement on the filter side

        if self.lower_cutoff.value() != 0:
            # Update lower cutoff so that it is not higher that the upper one
            self.lower_cutoff.setLimits((0, value))


class _InverseSolverNodeControls(_ProcessorNodeControls):
    def __init__(self, pipeline, **kwargs):
        _ProcessorNodeControls.__init__(self, pipeline, **kwargs)
        file_path_button = parameterTypes.ActionParameter(
            type="action", name="Setup forward model"
        )

        file_path_button.sigActivated.connect(self._choose_file)
        self.file_path_button = self.addChild(file_path_button)
        self._processor_node._signal_sender.open_fwd_dialog.connect(
            self._choose_file, type=Qt.BlockingQueuedConnection
        )

    def _choose_file(self):
        subject = self._processor_node.subject
        subjects_dir = self._processor_node.subjects_dir
        data_chnames = self._processor_node._upstream_mne_info["ch_names"]
        fwd_dialog = FwdSetupDialog(
            subjects_dir=subjects_dir,
            subject=subject,
            data_chnames=data_chnames,
        )
        fwd_dialog.exec()

        if fwd_dialog.result():
            fwd_path = fwd_dialog.fwd_path
            subject = fwd_dialog.subject
            subjects_dir = fwd_dialog.subjects_dir
            self._processor_node.fwd_path = fwd_path
            self._logger.info("Forward path is set to %s" % fwd_path)
            self._processor_node.subject = subject
            self._logger.info("Subject is set to %s" % subject)
            self._processor_node.subjects_dir = subjects_dir
            self._logger.info("Subjects directory is set to %s" % subjects_dir)


class MNEControls(_InverseSolverNodeControls):
    CONTROLS_LABEL = "Inverse modelling"
    PROCESSOR_CLASS = processors.MNE
    METHODS_COMBO_NAME = "Method: "
    FILE_PATH_STR_NAME = "Path to forward solution: "

    def _create_parameters(self):

        method_values = self.PROCESSOR_CLASS.SUPPORTED_METHODS
        method_value = self._processor_node.method
        methods_combo = parameterTypes.ListParameter(
            name=self.METHODS_COMBO_NAME,
            values=method_values,
            value=method_value,
        )
        methods_combo.sigValueChanged.connect(self._on_method_changed)
        self.methods_combo = self.addChild(methods_combo)

    def _on_method_changed(self, param, value):
        self._processor_node.method = value

    def _on_file_path_changed(self, param, value):
        self._processor_node.fwd_path = value


class EnvelopeExtractorControls(_ProcessorNodeControls):
    PROCESSOR_CLASS = processors.EnvelopeExtractor
    CONTROLS_LABEL = "Extract envelope: "

    FACTOR_NAME = "Factor: "
    METHODS_COMBO_NAME = "Method: "

    def _create_parameters(self):

        method_values = ["Exponential smoothing"]
        # TODO: change once we support more methods
        method_value = self._processor_node.method
        methods_combo = parameterTypes.ListParameter(
            name=self.METHODS_COMBO_NAME,
            values=method_values,
            value=method_value,
        )
        methods_combo.sigValueChanged.connect(self._on_method_changed)
        self.methods_combo = self.addChild(methods_combo)

        factor_value = self._processor_node.factor
        factor_spin_box = parameterTypes.SimpleParameter(
            type="float",
            name=self.FACTOR_NAME,
            decimals=3,
            limits=(0.5, 1),
            value=factor_value,
            step=0.001,
        )
        factor_spin_box.sigValueChanged.connect(self._on_factor_changed)
        self.factor_spin_box = self.addChild(factor_spin_box)

    def _on_method_changed(self):
        pass  # TODO: implement

    def _on_factor_changed(self, param, value):
        self._processor_node.factor = value


class BeamformerControls(_InverseSolverNodeControls):
    PROCESSOR_CLASS = processors.Beamformer
    CONTROLS_LABEL = "Beamformer"

    ADAPTIVENESS_NAME = "Use adaptive version: "
    SNR_NAME = "Regularization: "
    OUTPUT_TYPE_COMBO_NAME = "Output type: "
    FORGETTING_FACTOR_NAME = "Forgetting factor (per second): "
    FILE_PATH_STR_NAME = "Path to forward solution: "

    def __init__(self, pipeline, **kwargs):
        kwargs["title"] = "Forward solution file"

        # Add PushButton for choosing file
        _InverseSolverNodeControls.__init__(self, pipeline, **kwargs)

    def _create_parameters(self):
        # snr: float = 3.0, output_type: str = 'power',
        # is_adaptive: bool = False, forgetting_factor_per_second = 0.99
        is_adaptive = self._processor_node.is_adaptive
        adaptiveness_check = parameterTypes.SimpleParameter(
            type="bool",
            name=self.ADAPTIVENESS_NAME,
            value=is_adaptive,
            readonly=False,
        )
        adaptiveness_check.sigValueChanged.connect(
            self._on_adaptiveness_changed
        )
        self.adaptiveness_check = self.addChild(adaptiveness_check)

        reg_value = self._processor_node.reg
        snr_spin_box = parameterTypes.SimpleParameter(
            type="float",
            name=self.SNR_NAME,
            decimals=2,
            limits=(0, 100.0),
            value=reg_value,
        )
        snr_spin_box.sigValueChanged.connect(self._on_snr_changed)
        self.snr_spin_box = self.addChild(snr_spin_box)

        output_type_value = self._processor_node.output_type
        output_type_values = self.PROCESSOR_CLASS.SUPPORTED_OUTPUT_TYPES
        output_type_combo = parameterTypes.ListParameter(
            name=self.OUTPUT_TYPE_COMBO_NAME,
            values=output_type_values,
            value=output_type_value,
        )
        output_type_combo.sigValueChanged.connect(self._on_output_type_changed)
        self.output_type_combo = self.addChild(output_type_combo)

        forgetting_factor_value = (
            self._processor_node.forgetting_factor_per_second
        )
        forgetting_factor_spin_box = parameterTypes.SimpleParameter(
            type="float",
            name=self.FORGETTING_FACTOR_NAME,
            decimals=2,
            limits=(0.90, 0.99),
            value=forgetting_factor_value,
        )
        forgetting_factor_spin_box.sigValueChanged.connect(
            self._on_forgetting_factor_changed
        )
        self.forgetting_factor_spin_box = self.addChild(
            forgetting_factor_spin_box
        )

    def _on_adaptiveness_changed(self, param, value):
        self.forgetting_factor_spin_box.show(value)
        self._processor_node.is_adaptive = value

    def _on_snr_changed(self, param, value):
        self._processor_node.reg = value

    def _on_output_type_changed(self, param, value):
        self._processor_node.output_type = value

    def _on_forgetting_factor_changed(self, param, value):
        self._processor_node.forgetting_factor_per_second = value

    def _on_file_path_changed(self, param, value):
        self._processor_node.fwd_path = value


class MCEControls(_InverseSolverNodeControls):
    CONTROLS_LABEL = "MCE Inverse modelling"
    PROCESSOR_CLASS = processors.MCE

    FILE_PATH_STR_NAME = "Path to forward solution: "

    def _create_parameters(self):
        # method_values = self.PROCESSOR_CLASS.SUPPORTED_METHODS
        # method_value = self._processor_node.method
        # methods_combo = parameterTypes.ListParameter(
        # name=self.METHODS_COMBO_NAME, values=method_values,
        # value=method_value)
        # methods_combo.sigValueChanged.connect(self._on_method_changed)
        # self.methods_combo = self.addChild(methods_combo)
        pass

    def _on_method_changed(self, param, value):
        # self._processor_node.method = value
        pass

    def _on_file_path_changed(self, param, value):
        self._processor_node.fwd_path = value


class ICARejectionControls(_ProcessorNodeControls):
    CONTROLS_LABEL = "ICA rejection"
    PROCESSOR_CLASS = processors.MCE

    METHODS_COMBO_NAME = "Method: "

    def _create_parameters(self):

        # method_values = self.PROCESSOR_CLASS.SUPPORTED_METHODS
        # method_value = self._processor_node.method
        # methods_combo = parameterTypes.ListParameter(
        # name=self.METHODS_COMBO_NAME, values=method_values,
        # value=method_value)
        # methods_combo.sigValueChanged.connect(self._on_method_changed)
        # self.methods_combo = self.addChild(methods_combo)
        pass

    def _on_method_changed(self, param, value):
        # self._processor_node.method = value
        pass


class AtlasViewerControls(_ProcessorNodeControls):
    OUTPUT_CLASS = processors.AtlasViewer
    CONTROLS_LABEL = "Atlas Viewer"

    def __init__(self, processor_node, **kwargs):
        _ProcessorNodeControls.__init__(self, processor_node, **kwargs)
        self._processor_node._signal_sender.initialized.connect(
            self._on_initialize
        )

    def _create_parameters(self):
        roi_selection_button = parameterTypes.ActionParameter(
            type="action", name="Select ROI"
        )
        roi_selection_button.sigActivated.connect(self._choose_roi)
        self.roi_selection_button = self.addChild(roi_selection_button)

    def _on_initialize(self):
        self._logger.info("Initializing mesh for atlas viewer")
        self._mesh = get_mesh_data_from_surfaces_dir(
            op.join(
                self._processor_node.subjects_dir, self._processor_node.subject
            )
        )
        self.dialog = RoiSelectionDialog(
            self._processor_node.labels, self._mesh
        )

    def _choose_roi(self):
        self.dialog.exec_()

        if self.dialog.result():
            self._processor_node.labels = self.dialog.table.labels
            self._processor_node.active_label_names = [
                l.name for l in self.dialog.table.labels if l.is_active
            ]
        self._logger.debug("ROI selection button was clicked")

    def _on_label_state_changed(self, i, val):
        self._processor_node.labels_info[i]["state"] = val.value()
        self._processor_node.labels_info = self._processor_node.labels_info


class AmplitudeEnvelopeCorrelationsControls(_ProcessorNodeControls):
    """Controls class for AEC node"""

    CONTROLS_LABEL = "AmplitudeEnvelopeCorrelations controls"
    PROCESSOR_CLASS = processors.AmplitudeEnvelopeCorrelations

    def _create_parameters(self):
        ...


class CoherenceControls(_ProcessorNodeControls):
    """Coherence node controls"""

    CONTROLS_LABEL = "Coherence controls"
    PROCESSOR_CLASS = processors.Coherence

    def _create_parameters(self):
        ...
