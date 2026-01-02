from deepxube.base.factory import Factory
from deepxube.base.pathfinding import PathFind


pathfinding_factory: Factory[PathFind] = Factory[PathFind]("PathFind")
