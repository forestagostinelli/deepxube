import pytest  # type: ignore

from typing import Type
from deepxube.base.pathfinding import Node
from deepxube.domains.cube3 import Cube3
from deepxube.pathfinding.beam_search import InstanceBeam, InstanceNodeBeam, InstanceEdgeBeam
from deepxube.pathfinding.graph_search import InstanceGraph, InstanceNodeGraph, InstanceEdgeGraph


@pytest.fixture
def root_node() -> Node:
    states, goals = Cube3().sample_problem_instances([1])
    return Node(states[0], goals[0], 0.0, 0.0, None, None, None, None, None)


@pytest.mark.parametrize("instance_cls", [InstanceNodeBeam, InstanceEdgeBeam], ids=["node_beam", "edge_beam"],)
def test_instance_beam_defaults(instance_cls: Type[InstanceBeam], root_node: Node,) -> None:
    instance = instance_cls(root_node, None)

    assert instance.beam_size == 1
    assert instance.temp == 0.0
    assert instance.eps == 0.0
    assert instance.rollout is False


@pytest.mark.parametrize("instance_cls", [InstanceNodeGraph, InstanceEdgeGraph], ids=["node_graph", "edge_graph"],)
def test_instance_graph_defaults(instance_cls: Type[InstanceGraph], root_node: Node,) -> None:
    instance = instance_cls(root_node, None)

    assert instance.batch_size == 1
    assert instance.weight == 1.0
    assert instance.eps == 0.0
