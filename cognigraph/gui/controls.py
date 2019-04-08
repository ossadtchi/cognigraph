from collections import namedtuple, OrderedDict

from pyqtgraph.parametertree import parameterTypes

from ..pipeline import Pipeline
from .. import nodes

from . import node_controls

from ..utils.pyqtgraph import MyGroupParameter
from ..utils.misc import class_name_of


node_controls_map = namedtuple('node_controls_map',
                               ['node_class', 'controls_class'])

node_to_controls_map = {
    'LinearFilter': 'LinearFilterControls',
    'InverseModel': 'InverseModelControls',
    'EnvelopeExtractor': 'EnvelopeExtractorControls',
    'Preprocessing': 'PreprocessingControls',
    'Beamformer': 'BeamformerControls',
    'MCE': 'MCEControls',
    'ICARejection': 'ICARejectionControls',
    'AtlasViewer': 'AtlasViewerControls',
    'AmplitudeEnvelopeCorrelations': 'AmplitudeEnvelopeCorrelationsControls',
    'Coherence': 'CoherenceControls',
    'LSLStreamOutput': 'LSLStreamOutputControls',
    'BrainViewer': 'BrainViewerControls',
    'SignalViewer': 'SignalViewerControls',
    'FileOutput': 'FileOutputControls',
    'TorchOutput': 'TorchOutputControls',
    'ConnectivityViewer': 'ConnectivityViewerControls',
    'LSLStreamSource': 'LSLStreamSourceControls',
    'FileSource': 'FileSourceControls'
}


class MultipleNodeControls(MyGroupParameter):
    """
    Base class for grouping of node settings (processors or outputs).
    Source is supported by a separate class.

    """

    @property
    def SUPPORTED_NODES(self):
        raise NotImplementedError

    def __init__(self, nodes, **kwargs):
        self._nodes = nodes
        super().__init__(**kwargs)

        for node in nodes:
            controls_class = self._find_controls_class_for_a_node(node)
            self.addChild(controls_class(node), autoIncrementName=True)

    @classmethod
    def _find_controls_class_for_a_node(cls, processor_node):
        for node_control_classes in cls.SUPPORTED_NODES:
            if isinstance(processor_node, node_control_classes.node_class):
                return node_control_classes.controls_class

        # Raise an error if processor node is not supported
        msg = ("Node of class {0} is not supported by {1}.\n"
               "Add node_controls_map(node_class, controls_class) to"
               " {1}.SUPPORTED_NODES").format(
                class_name_of(processor_node), cls.__name__)
        raise ValueError(msg)


class ProcessorsControls(MultipleNodeControls):
    SUPPORTED_NODES = [
        node_controls_map(
            nodes.LinearFilter,
            node_controls.LinearFilterControls),
        node_controls_map(
            nodes.InverseModel,
            node_controls.InverseModelControls),
        node_controls_map(
            nodes.EnvelopeExtractor,
            node_controls.EnvelopeExtractorControls),
        node_controls_map(
            nodes.Preprocessing,
            node_controls.PreprocessingControls),
        node_controls_map(
            nodes.Beamformer,
            node_controls.BeamformerControls),
        node_controls_map(
            nodes.MCE,
            node_controls.MCEControls),
        node_controls_map(
            nodes.ICARejection,
            node_controls.ICARejectionControls),
        node_controls_map(
            nodes.AtlasViewer,
            node_controls.AtlasViewerControls),
        node_controls_map(
            nodes.AmplitudeEnvelopeCorrelations,
            node_controls.AmplitudeEnvelopeCorrelationsControls),
        node_controls_map(nodes.Coherence,
                          node_controls.CoherenceControls)
    ]


class OutputsControls(MultipleNodeControls):
    SUPPORTED_NODES = [
        node_controls_map(nodes.LSLStreamOutput,
                          node_controls.LSLStreamOutputControls),
        node_controls_map(nodes.BrainViewer,
                          node_controls.BrainViewerControls),
        node_controls_map(nodes.SignalViewer,
                          node_controls.SignalViewerControls),
        node_controls_map(nodes.FileOutput,
                          node_controls.FileOutputControls),
        node_controls_map(nodes.TorchOutput,
                          node_controls.TorchOutputControls),
        node_controls_map(nodes.ConnectivityViewer,
                          node_controls.ConnectivityViewerControls)
    ]


class BaseControls(MyGroupParameter):
    def __init__(self, pipeline):
        super().__init__(name='Base controls', type='BaseControls')
        self._pipeline = pipeline

        # TODO: Change names to delineate source_controls as defined here and
        # source_controls - gui.node_controls.source
        source_controls = SourceControls(pipeline=pipeline, name='Source')
        node_controls = ProcessorsControls(nodes=pipeline._processors,
                                           name='Processors')
        outputs_controls = OutputsControls(nodes=pipeline._outputs,
                                           name='Outputs')

        self.source_controls = self.addChild(source_controls)
        self.node_controls = self.addChild(node_controls)
        self.outputs_controls = self.addChild(outputs_controls)


class SourceControls(MyGroupParameter):
    """
    Represents a drop-down list with the names of supported source types.
    Selecting a type creates controls for that type below the drop-down.

    """

    # Order is important.
    # Entries with node subclasses must precede entries with the parent class
    SOURCE_OPTIONS = OrderedDict((
        ('LSL stream',
         node_controls_map(nodes.LSLStreamSource,
                           node_controls.LSLStreamSourceControls)),
        ('File data',
         node_controls_map(nodes.FileSource,
                           node_controls.FileSourceControls)),
    ))

    SOURCE_TYPE_COMBO_NAME = 'Source type: '
    SOURCE_TYPE_PLACEHOLDER = ''
    SOURCE_CONTROLS_NAME = 'source controls'

    def __init__(self, pipeline: Pipeline, **kwargs):
        self._pipeline = pipeline
        super().__init__(**kwargs)

        labels = ([self.SOURCE_TYPE_PLACEHOLDER] +
                  [label for label in self.SOURCE_OPTIONS])

        source_type_combo = parameterTypes.ListParameter(
            name=self.SOURCE_TYPE_COMBO_NAME, values=labels, value=labels[0])

        source_type_combo.sigValueChanged.connect(self._on_source_type_changed)
        self.source_type_combo = self.addChild(source_type_combo)

        if pipeline.source is not None:
            for source_option, classes in self.SOURCE_OPTIONS.items():
                if isinstance(pipeline.source, classes.node_class):
                    self.source_type_combo.setValue(source_option)

    def _on_source_type_changed(self, param, value):
        try:
            source_controls = self.source_controls
            self.removeChild(source_controls)
        except AttributeError:  # No source type has been chosen
            pass
        if value != self.SOURCE_TYPE_PLACEHOLDER:
            # Update source controls
            source_classes = self.SOURCE_OPTIONS[value]
            controls = source_classes.controls_class(
                pipeline=self._pipeline, name=self.SOURCE_CONTROLS_NAME)
            self.source_controls = self.addChild(controls)

            # Update source
            if not isinstance(self._pipeline.source,
                              source_classes.node_class):
                self._pipeline.source = self.source_controls.create_node()


class Controls(object):
    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self._pipeline = pipeline  # type: Pipeline
        self._base_controls = BaseControls(pipeline=self._pipeline)
        self.widget = self._base_controls.create_widget()
