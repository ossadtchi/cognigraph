from PyQt5.QtWidgets import (QApplication, QTreeWidget, QTreeWidgetItem,
                             QMenu, QAction, QDialog)
from PyQt5.QtCore import Qt
from PyQt5.Qt import QSizePolicy
from PyQt5.QtWidgets import QVBoxLayout
from cognigraph.pipeline import Pipeline
from cognigraph.nodes.node import Node
from cognigraph import nodes
from functools import partial
from pyqtgraph.parametertree import parameterTypes, ParameterTree
from cognigraph.gui.controls import node_to_controls_map
from cognigraph.gui import node_controls


class _CreateNodeDialog(QDialog):
    def __init__(self, node_cls, parent=None):
        QDialog.__init__(self, parent)

        self.widget = ParameterTree(showHeader=True)
        layout = QVBoxLayout()
        layout.addWidget(self.widget)
        self.setLayout(layout)
        params = parameterTypes.GroupParameter(name='test')
        child_node = node_cls()
        controls_cls_name = node_to_controls_map[node_cls.__name__]
        controls_cls = getattr(node_controls, controls_cls_name)
        controls = controls_cls(child_node)
        params.addChild(controls)
        self.widget.setParameters(params)


class PipelineTreeWidget(QTreeWidget):
    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.clear()
        self.setColumnCount(1)
        self.setItemsExpandable(True)

        self.add_nodes(pipeline.source, self)
        self.resizeColumnToContents(0)
        pipeline._source

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(
            self._on_context_menu_requiested)

    def add_nodes(self, node, parent_item):
        this_item = QTreeWidgetItem(parent_item, [repr(node)])
        this_item.setData(0, 1, node)
        self.expandItem(this_item)
        for child in node._children:
            self.add_nodes(child, this_item)

    def _on_context_menu_requiested(self, pos):
        menu = QMenu(self)
        submenu = QMenu(menu)
        submenu.setTitle('Add node')
        item = self.itemAt(pos)
        menu.addMenu(submenu)

        allowed_children = item.data(0, 1).ALLOWED_CHILDREN
        actions = []
        for c in allowed_children:
            child_cls = getattr(nodes, c)
            action = QAction(repr(child_cls), submenu)
            action.triggered.connect(
                partial(self._on_adding_node, parent=item.data(0, 1),
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
    # app = QApplication(sys.argv)
    tree_widget = PipelineTreeWidget(pipeline)
    tree_widget.show()
    # sys.exit(app.exec_())  # dont need this: tree_widget has event_loop
