:py:mod:`deepxube.nnets.resnet_2d`
==================================

.. py:module:: deepxube.nnets.resnet_2d

.. autodoc2-docstring:: deepxube.nnets.resnet_2d
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Resnet2D <deepxube.nnets.resnet_2d.Resnet2D>`
     -
   * - :py:obj:`ResnetFCParser <deepxube.nnets.resnet_2d.ResnetFCParser>`
     -

API
~~~

.. py:class:: Resnet2D(nnet_input: deepxube.base.nnet_input.TwoDIn, out_dim: int, q_fix: bool, num_chan: int = 64, num_blocks: int = 4, batch_norm: bool = False, weight_norm: bool = False, act_fn: str = 'RELU')
   :canonical: deepxube.nnets.resnet_2d.Resnet2D

   Bases: :py:obj:`deepxube.base.nnet.HeurNNet`\ [\ :py:obj:`deepxube.base.nnet_input.TwoDIn`\ ]

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet_input.TwoDIn]
      :canonical: deepxube.nnets.resnet_2d.Resnet2D.nnet_input_type
      :staticmethod:

   .. py:method:: _forward(inputs: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.nnets.resnet_2d.Resnet2D._forward

      .. autodoc2-docstring:: deepxube.nnets.resnet_2d.Resnet2D._forward

.. py:class:: ResnetFCParser
   :canonical: deepxube.nnets.resnet_2d.ResnetFCParser

   Bases: :py:obj:`deepxube.base.factory.Parser`

   .. py:method:: parse(args_str: str) -> typing.Dict[str, typing.Any]
      :canonical: deepxube.nnets.resnet_2d.ResnetFCParser.parse

      .. autodoc2-docstring:: deepxube.nnets.resnet_2d.ResnetFCParser.parse

   .. py:method:: help() -> str
      :canonical: deepxube.nnets.resnet_2d.ResnetFCParser.help

      .. autodoc2-docstring:: deepxube.nnets.resnet_2d.ResnetFCParser.help
