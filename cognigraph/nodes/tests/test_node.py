import pytest
from cognigraph.nodes.node import Node


class NodeConcrete(Node):
    """Redefine methods of Node which raise NotImplementedError
    for testing purposes

    """
    CHANGES_IN_THESE_REQUIRE_RESET = ()

    def _check_value(self, key, value):
        return True


@pytest.fixture
def node():
    return NodeConcrete()


def test_init(node):
    assert(node._parent_node is None)
    assert(node.output is None)
    iter(node._child_nodes)  # check if children_nodes is iterable
    assert(len(node._child_nodes) == 0)


def test_initialize(node):
    pass

