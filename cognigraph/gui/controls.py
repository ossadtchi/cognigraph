from PyQt5.QtWidgets import QVBoxLayout, QWidget
from PyQt5.Qt import QSizePolicy
from collections import namedtuple, OrderedDict

from pyqtgraph.parametertree import parameterTypes, ParameterTree

from cognigraph.pipeline import Pipeline
from cognigraph import nodes

from cognigraph.gui import node_controls

from cognigraph.utils.pyqtgraph import MyGroupParameter
from cognigraph.utils.misc import class_name_of

from PyQt5.QtWidgets import (QApplication, QTreeWidget, QTreeWidgetItem,
                             QMenu, QAction, QDialog, QTreeWidgetItemIterator)
from PyQt5.QtCore import Qt
from cognigraph.nodes.node import Node
from functools import partial


node_controls_map = namedtuple('node_controls_map',
                               ['node_class', 'controls_class'])


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

    def __init__(self, source_node, **kwargs):
        self._source_node = source_node
        name = repr(source_node)
        super().__init__(name=name)

        labels = ([self.SOURCE_TYPE_PLACEHOLDER] +
                  [label for label in self.SOURCE_OPTIONS])

        source_type_combo = parameterTypes.ListParameter(
            name=self.SOURCE_TYPE_COMBO_NAME, values=labels, value=labels[0])

        source_type_combo.sigValueChanged.connect(self._on_source_type_changed)
        self.source_type_combo = self.addChild(source_type_combo)

        if source_node is not None:
            for source_option, classes in self.SOURCE_OPTIONS.items():
                if isinstance(source_node, classes.node_class):
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
                source_node=self._source_node, name=self.SOURCE_CONTROLS_NAME)
            self.source_controls = self.addChild(controls)

            # Update source
            if not isinstance(self._source_node,
                              source_classes.node_class):
                self._source_node = self.source_controls.create_node()


node_to_controls_map = {
    'LinearFilter': node_controls.LinearFilterControls,
    'InverseModel': node_controls.InverseModelControls,
    'EnvelopeExtractor': node_controls.EnvelopeExtractorControls,
    'Preprocessing': node_controls.PreprocessingControls,
    'Beamformer': node_controls.BeamformerControls,
    'MCE': node_controls.MCEControls,
    'ICARejection': node_controls.ICARejectionControls,
    'AtlasViewer': node_controls.AtlasViewerControls,
    'AmplitudeEnvelopeCorrelations': node_controls.AmplitudeEnvelopeCorrelationsControls,  # noqa
    'Coherence': node_controls.CoherenceControls,
    'LSLStreamOutput': node_controls.LSLStreamOutputControls,
    'BrainViewer': node_controls.BrainViewerControls,
    'SignalViewer': node_controls.SignalViewerControls,
    'FileOutput': node_controls.FileOutputControls,
    'TorchOutput': node_controls.TorchOutputControls,
    'ConnectivityViewer': node_controls.ConnectivityViewerControls,
    'LSLStreamSource': SourceControls,  # node_controls.LSLStreamSourceControls
    'FileSource': SourceControls  # node_controls.FileSourceControls
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


class BaseControls(QWidget):
    def __init__(self, pipeline, name='BaseControls', type='BaseControls'):
        super().__init__()
        self._pipeline = pipeline

        # TODO: Change names to delineate source_controls as defined here and
        # source_controls - gui.node_controls.source

        # self.source_controls = self.addChild(source_controls)
        layout = QVBoxLayout()
        for node in pipeline.all_nodes:
            widget = self._create_node_controls_widget(node)
            widget.hide()
            layout.addWidget(widget)
        self.setLayout(layout)

class _CreateNodeDialog(QDialog):
    def __init__(self, node_cls, parent=None):
        QDialog.__init__(self, parent)

        self.widget = ParameterTree(showHeader=False)
        layout = QVBoxLayout()
        layout.addWidget(self.widget)
        self.setLayout(layout)
        params = parameterTypes.GroupParameter(name='test')
        child_node = node_cls()
        controls_cls = node_to_controls_map[node_cls.__name__]
        controls = controls_cls(child_node)
        params.addChild(controls)
        self.widget.setParameters(params)


class PipelineTreeWidget(QTreeWidget):
    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.clear()
        self.setColumnCount(1)
        self.setItemsExpandable(True)

        self.controls_layout = QVBoxLayout()
        self.resizeColumnToContents(0)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(
            self._on_context_menu_requiested)
        self.itemSelectionChanged.connect(self._on_tree_item_selection_changed)

    def _on_context_menu_requiested(self, pos):
        menu = QMenu(self)
        submenu = QMenu(menu)
        submenu.setTitle('Add node')
        item = self.itemAt(pos)
        menu.addMenu(submenu)

        allowed_children = item.node.ALLOWED_CHILDREN
        actions = []
        for c in allowed_children:
            child_cls = getattr(nodes, c)
            action = QAction(repr(child_cls), submenu)
            action.triggered.connect(
                partial(self._on_adding_node, parent=item.node,
                        child_cls=child_cls))
            actions.append(action)

        submenu.addActions(actions)
        menu.exec(self.viewport().mapToGlobal(pos))

    def _on_adding_node(self, t, parent: Node, child_cls):
        """Add node to pipeline
        Parameters
        ----------
        t
            Something that action.triggered signal passes.
        parent: Node
            Parent Node instance
        chld_cls
            Class of node we're about to add

        """
        print('Adding %s to %s' % (child_cls, parent))
        self.create_node_dialog = _CreateNodeDialog(child_cls, parent=self)
        self.create_node_dialog.show()
        self.create_node_dialog.widget.setSizeAdjustPolicy(1)
        self.create_node_dialog.widget.setSizePolicy(QSizePolicy.Expanding,
                                                     QSizePolicy.Expanding)
        self.create_node_dialog.adjustSize()

    def _create_node_controls_widget(self, node):
        controls_cls = node_to_controls_map[node.__class__.__name__]
        controls = controls_cls(node)
        params = parameterTypes.GroupParameter(name=repr(node))
        params.addChild(controls)
        widget = ParameterTree(showHeader=False)
        widget.setParameters(params)
        widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        widget.setSizeAdjustPolicy(1)
        return widget

    def _on_tree_item_selection_changed(self):
        iterator = QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            item.widget.hide()
            iterator += 1
        for item in self.selectedItems():
            item.widget.show()


class NodeTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, parent_item, node, widget):
        QTreeWidgetItem.__init__(self, parent_item, [repr(node)])
        self.node = node
        self.widget = widget


class Controls(QWidget):
    def __init__(self, pipeline: Pipeline, parent=None):
        QWidget.__init__(self, parent)
        self._pipeline = pipeline  # type: Pipeline
        layout = QVBoxLayout()
        self.tree_widget = PipelineTreeWidget(pipeline=self._pipeline)
        # self.tree_widget.itemSelectionChanged.connect(
        #     self._on_tree_item_selection_changed)
        layout.addWidget(self.tree_widget)
        # self.params_widget = BaseControls(pipeline=self._pipeline)
        # layout.addWidget(self.params_widget)
        self.params_layout = QVBoxLayout()
        self.add_nodes(pipeline.source, self.tree_widget)
        layout.addLayout(self.params_layout)
        self.setLayout(layout)

    def add_nodes(self, node, parent_item):
        widget = self._create_node_controls_widget(node)
        this_item = NodeTreeWidgetItem(parent_item, node, widget)
        self.tree_widget.controls_layout.addWidget(widget)
        self.params_layout.addWidget(widget)
        widget.hide()
        self.tree_widget.expandItem(this_item)
        for child in node._children:
            self.add_nodes(child, this_item)

    def _create_node_controls_widget(self, node):
        controls_cls = node_to_controls_map[node.__class__.__name__]
        controls = controls_cls(node)
        params = parameterTypes.GroupParameter(name=repr(node))
        params.addChild(controls)
        widget = ParameterTree(showHeader=False)
        widget.setParameters(params)
        widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        widget.setSizeAdjustPolicy(1)
        return widget


if __name__ == '__main__':
    import sys
    from cognigraph.tests.test_pipeline import (ConcreteSource,
                                                ConcreteProcessor,
                                                ConcreteOutput)

    pipeline = Pipeline()
    src = ConcreteSource()
    proc = ConcreteProcessor()
    out = ConcreteOutput()
    src.add_child(proc)
    proc.add_child(out)
    pipeline.source = src
    app = QApplication(sys.argv)
    tree_widget = PipelineTreeWidget(pipeline)
    tree_widget.show()
    sys.exit(app.exec_())  # dont need this: tree_widget has event_loop
