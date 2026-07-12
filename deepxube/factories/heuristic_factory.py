from deepxube.base.heuristic import DeepXubeNNet
from deepxube.base.factory import Factory

deepxube_nnet_factory: Factory[DeepXubeNNet] = Factory[DeepXubeNNet]("DeepXubeNNet")
