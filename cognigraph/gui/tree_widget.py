from PyQt5.QtWidgets import (QApplication, QTreeWidget, QTreeWidgetItem,
                             QMenu, QAction, QDialog)
from PyQt5.QtCore import Qt
from cognigraph.pipeline import Pipeline
from cognigraph.nodes.node import Node
from cognigraph import nodes


class _AddNodeDialog(QDialog):
    def __init__(self, node_cls):
        _AddNodeDialog.__init__(self)


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
        this_item = QTreeWidgetItem(parent_item, [str(node)])
        this_item.setData(0, 1, node)
        self.expandItem(this_item)
        # this_item.setChildIndicatorPolicy(QTreeWidgetItem.DontShowIndicator)
        for child in node._children:
            self.add_nodes(child, this_item)

    def _on_context_menu_requiested(self, pos):
        menu = QMenu(self)
        submenu = QMenu(menu)
        submenu.setTitle('Add node')
        item = self.itemAt(pos)
        menu.addMenu(submenu)

        allowed_children = item.data(0, 1).ALLOWED_CHILDREN
        actions = [QAction(c, submenu) for c in allowed_children]

        for action in actions:
            action.triggered.connect(
                lambda t, p=item.data(0, 1), c=getattr(nodes, action.text()):
                    self._on_adding_node(t, p, c))

        submenu.addActions(actions)

        menu.exec(self.viewport().mapToGlobal(pos))
        # print('hi')

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
    sys.exit(app.exec_())
