:py:mod:`deepxube.heuristics.resnet_fc`
=======================================

.. py:module:: deepxube.heuristics.resnet_fc

.. autodoc2-docstring:: deepxube.heuristics.resnet_fc
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ResnetFCHeur <deepxube.heuristics.resnet_fc.ResnetFCHeur>`
     -
   * - :py:obj:`ResnetFCPolicy <deepxube.heuristics.resnet_fc.ResnetFCPolicy>`
     -
   * - :py:obj:`ResnetFCParser <deepxube.heuristics.resnet_fc.ResnetFCParser>`
     -
   * - :py:obj:`ResnetFCParserHeur <deepxube.heuristics.resnet_fc.ResnetFCParserHeur>`
     -
   * - :py:obj:`ResnetFCParserPolicy <deepxube.heuristics.resnet_fc.ResnetFCParserPolicy>`
     -

API
~~~

.. py:class:: ResnetFCHeur(nnet_input: deepxube.base.nnet_input.FlatIn, out_dim: int, q_fix: bool, res_dim: int = 1000, num_blocks: int = 4, batch_norm: bool = False, weight_norm: bool = False, layer_norm: bool = False, act_fn: str = 'RELU')
   :canonical: deepxube.heuristics.resnet_fc.ResnetFCHeur

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNet`\ [\ :py:obj:`deepxube.base.nnet_input.FlatIn`\ ]

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet_input.FlatIn]
      :canonical: deepxube.heuristics.resnet_fc.ResnetFCHeur.nnet_input_type
      :staticmethod:

   .. py:method:: _forward(inputs: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.heuristics.resnet_fc.ResnetFCHeur._forward

      .. autodoc2-docstring:: deepxube.heuristics.resnet_fc.ResnetFCHeur._forward

.. py:class:: ResnetFCPolicy(nnet_input: deepxube.base.nnet_input.FlatInPolicy, num_samp: int, kl_weight: float, enc_dim: int = 10, res_dim: int = 1000, num_blocks: int = 4, batch_norm: bool = False, weight_norm: bool = False, layer_norm: bool = False, act_fn: str = 'RELU')
   :canonical: deepxube.heuristics.resnet_fc.ResnetFCPolicy

   Bases: :py:obj:`deepxube.base.heuristic.PolicyVAE`\ [\ :py:obj:`deepxube.base.nnet_input.FlatInPolicy`\ ]

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet_input.FlatInPolicy]
      :canonical: deepxube.heuristics.resnet_fc.ResnetFCPolicy.nnet_input_type
      :staticmethod:

   .. py:method:: latent_shape() -> typing.Tuple[int, ...]
      :canonical: deepxube.heuristics.resnet_fc.ResnetFCPolicy.latent_shape

   .. py:method:: encode(states_goals: typing.List[torch.Tensor], actions: typing.List[torch.Tensor]) -> typing.Tuple[typing.List[torch.Tensor], torch.Tensor, torch.Tensor]
      :canonical: deepxube.heuristics.resnet_fc.ResnetFCPolicy.encode

   .. py:method:: decode(states_goals: typing.List[torch.Tensor], z: torch.Tensor) -> typing.List[torch.Tensor]
      :canonical: deepxube.heuristics.resnet_fc.ResnetFCPolicy.decode

.. py:class:: ResnetFCParser()
   :canonical: deepxube.heuristics.resnet_fc.ResnetFCParser

   Bases: :py:obj:`deepxube.base.factory.DelimParser`

   .. py:property:: delim
      :canonical: deepxube.heuristics.resnet_fc.ResnetFCParser.delim
      :type: str

      .. autodoc2-docstring:: deepxube.heuristics.resnet_fc.ResnetFCParser.delim

.. py:class:: ResnetFCParserHeur()
   :canonical: deepxube.heuristics.resnet_fc.ResnetFCParserHeur

   Bases: :py:obj:`deepxube.heuristics.resnet_fc.ResnetFCParser`

.. py:class:: ResnetFCParserPolicy()
   :canonical: deepxube.heuristics.resnet_fc.ResnetFCParserPolicy

   Bases: :py:obj:`deepxube.heuristics.resnet_fc.ResnetFCParser`
