from deepxube.base.nnet import DeepXubeNNet
from deepxube.base.factory import Factory

deepxube_nnet_factory: Factory[DeepXubeNNet] = Factory[DeepXubeNNet]("DeepXubeNNet")
