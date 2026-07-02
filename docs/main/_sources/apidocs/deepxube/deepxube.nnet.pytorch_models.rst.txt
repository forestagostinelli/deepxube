:py:mod:`deepxube.nnet.pytorch_models`
======================================

.. py:module:: deepxube.nnet.pytorch_models

.. autodoc2-docstring:: deepxube.nnet.pytorch_models
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`OneHot <deepxube.nnet.pytorch_models.OneHot>`
     -
   * - :py:obj:`SPLASH <deepxube.nnet.pytorch_models.SPLASH>`
     -
   * - :py:obj:`SPLASH1 <deepxube.nnet.pytorch_models.SPLASH1>`
     -
   * - :py:obj:`LinearAct <deepxube.nnet.pytorch_models.LinearAct>`
     -
   * - :py:obj:`ReLU2 <deepxube.nnet.pytorch_models.ReLU2>`
     -
   * - :py:obj:`ResnetModel <deepxube.nnet.pytorch_models.ResnetModel>`
     -
   * - :py:obj:`FullyConnectedModel <deepxube.nnet.pytorch_models.FullyConnectedModel>`
     -
   * - :py:obj:`Conv2dModel <deepxube.nnet.pytorch_models.Conv2dModel>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_act_fn <deepxube.nnet.pytorch_models.get_act_fn>`
     - .. autodoc2-docstring:: deepxube.nnet.pytorch_models.get_act_fn
          :summary:
   * - :py:obj:`make_onehots <deepxube.nnet.pytorch_models.make_onehots>`
     - .. autodoc2-docstring:: deepxube.nnet.pytorch_models.make_onehots
          :summary:

API
~~~

.. py:class:: OneHot(one_hot_depth: int, flatten_oh: bool)
   :canonical: deepxube.nnet.pytorch_models.OneHot

   Bases: :py:obj:`torch.nn.Module`

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor
      :canonical: deepxube.nnet.pytorch_models.OneHot.forward

      .. autodoc2-docstring:: deepxube.nnet.pytorch_models.OneHot.forward

.. py:class:: SPLASH(num_hinges: int = 5, init: str = 'RELU')
   :canonical: deepxube.nnet.pytorch_models.SPLASH

   Bases: :py:obj:`torch.nn.Module`

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor
      :canonical: deepxube.nnet.pytorch_models.SPLASH.forward

      .. autodoc2-docstring:: deepxube.nnet.pytorch_models.SPLASH.forward

.. py:class:: SPLASH1(init: str = 'RELU')
   :canonical: deepxube.nnet.pytorch_models.SPLASH1

   Bases: :py:obj:`torch.nn.Module`

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor
      :canonical: deepxube.nnet.pytorch_models.SPLASH1.forward

      .. autodoc2-docstring:: deepxube.nnet.pytorch_models.SPLASH1.forward

.. py:class:: LinearAct()
   :canonical: deepxube.nnet.pytorch_models.LinearAct

   Bases: :py:obj:`torch.nn.Module`

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor
      :canonical: deepxube.nnet.pytorch_models.LinearAct.forward

      .. autodoc2-docstring:: deepxube.nnet.pytorch_models.LinearAct.forward

.. py:class:: ReLU2()
   :canonical: deepxube.nnet.pytorch_models.ReLU2

   Bases: :py:obj:`torch.nn.Module`

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor
      :canonical: deepxube.nnet.pytorch_models.ReLU2.forward

      .. autodoc2-docstring:: deepxube.nnet.pytorch_models.ReLU2.forward

.. py:function:: get_act_fn(act: str) -> torch.nn.Module
   :canonical: deepxube.nnet.pytorch_models.get_act_fn

   .. autodoc2-docstring:: deepxube.nnet.pytorch_models.get_act_fn

.. py:class:: ResnetModel(block_init: typing.Callable[[], torch.nn.Module], num_resnet_blocks: int, act_fn: str)
   :canonical: deepxube.nnet.pytorch_models.ResnetModel

   Bases: :py:obj:`torch.nn.Module`

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor
      :canonical: deepxube.nnet.pytorch_models.ResnetModel.forward

      .. autodoc2-docstring:: deepxube.nnet.pytorch_models.ResnetModel.forward

.. py:class:: FullyConnectedModel(input_dim: int, dims: typing.List[int], acts: typing.List[str], batch_norms: typing.Optional[typing.List[bool]] = None, weight_norms: typing.Optional[typing.List[bool]] = None, group_norms: typing.Optional[typing.List[int]] = None)
   :canonical: deepxube.nnet.pytorch_models.FullyConnectedModel

   Bases: :py:obj:`torch.nn.Module`

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor
      :canonical: deepxube.nnet.pytorch_models.FullyConnectedModel.forward

      .. autodoc2-docstring:: deepxube.nnet.pytorch_models.FullyConnectedModel.forward

.. py:class:: Conv2dModel(chan_in: int, channel_sizes: typing.List[int], kernel_sizes: typing.List[int], paddings: typing.List[int], layer_acts: typing.List[str], batch_norms: typing.Optional[typing.List[bool]] = None, strides: typing.Optional[typing.List[int]] = None, transpose: bool = False, weight_norms: typing.Optional[typing.List[bool]] = None, dropouts: typing.Optional[typing.List[float]] = None)
   :canonical: deepxube.nnet.pytorch_models.Conv2dModel

   Bases: :py:obj:`torch.nn.Module`

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor
      :canonical: deepxube.nnet.pytorch_models.Conv2dModel.forward

      .. autodoc2-docstring:: deepxube.nnet.pytorch_models.Conv2dModel.forward

.. py:function:: make_onehots(input_dims: typing.List[int], one_hot_depths: typing.List[int]) -> typing.Tuple[torch.nn.ModuleList, int]
   :canonical: deepxube.nnet.pytorch_models.make_onehots

   .. autodoc2-docstring:: deepxube.nnet.pytorch_models.make_onehots
