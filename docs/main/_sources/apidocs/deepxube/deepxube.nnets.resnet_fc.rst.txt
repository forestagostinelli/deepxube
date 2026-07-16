:py:mod:`deepxube.nnets.resnet_fc`
==================================

.. py:module:: deepxube.nnets.resnet_fc

.. autodoc2-docstring:: deepxube.nnets.resnet_fc
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ResnetFCHeur <deepxube.nnets.resnet_fc.ResnetFCHeur>`
     -
   * - :py:obj:`ResnetFCPolicy <deepxube.nnets.resnet_fc.ResnetFCPolicy>`
     -
   * - :py:obj:`ResnetFCParser <deepxube.nnets.resnet_fc.ResnetFCParser>`
     -
   * - :py:obj:`ResnetFCParserHeur <deepxube.nnets.resnet_fc.ResnetFCParserHeur>`
     -
   * - :py:obj:`ResnetFCParserPolicy <deepxube.nnets.resnet_fc.ResnetFCParserPolicy>`
     -

API
~~~

.. py:class:: ResnetFCHeur(nnet_input: deepxube.base.nnet_input.FlatIn, out_dim: int, q_fix: bool, res_dim: int = 1000, num_blocks: int = 4, batch_norm: bool = False, weight_norm: bool = False, layer_norm: bool = False, act_fn: str = 'RELU')
   :canonical: deepxube.nnets.resnet_fc.ResnetFCHeur

   Bases: :py:obj:`deepxube.base.nnet.HeurNNet`\ [\ :py:obj:`deepxube.base.nnet_input.FlatIn`\ ]

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet_input.FlatIn]
      :canonical: deepxube.nnets.resnet_fc.ResnetFCHeur.nnet_input_type
      :staticmethod:

   .. py:method:: _forward(inputs: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.nnets.resnet_fc.ResnetFCHeur._forward

      .. autodoc2-docstring:: deepxube.nnets.resnet_fc.ResnetFCHeur._forward

.. py:class:: ResnetFCPolicy(nnet_input: deepxube.base.nnet_input.FlatInPolicy, num_samp: int, kl_weight: float, enc_dim: int = 10, res_dim: int = 1000, num_blocks: int = 4, batch_norm: bool = False, weight_norm: bool = False, layer_norm: bool = False, act_fn: str = 'RELU')
   :canonical: deepxube.nnets.resnet_fc.ResnetFCPolicy

   Bases: :py:obj:`deepxube.base.nnet.PolicyVAE`\ [\ :py:obj:`deepxube.base.nnet_input.FlatInPolicy`\ ]

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet_input.FlatInPolicy]
      :canonical: deepxube.nnets.resnet_fc.ResnetFCPolicy.nnet_input_type
      :staticmethod:

   .. py:method:: latent_shape() -> typing.Tuple[int, ...]
      :canonical: deepxube.nnets.resnet_fc.ResnetFCPolicy.latent_shape

   .. py:method:: encode(states_goals: typing.List[torch.Tensor], actions: typing.List[torch.Tensor]) -> typing.Tuple[typing.List[torch.Tensor], torch.Tensor, torch.Tensor]
      :canonical: deepxube.nnets.resnet_fc.ResnetFCPolicy.encode

   .. py:method:: decode(states_goals: typing.List[torch.Tensor], z: torch.Tensor) -> typing.List[torch.Tensor]
      :canonical: deepxube.nnets.resnet_fc.ResnetFCPolicy.decode

.. py:class:: ResnetFCParser()
   :canonical: deepxube.nnets.resnet_fc.ResnetFCParser

   Bases: :py:obj:`deepxube.base.factory.DelimParser`

   .. py:property:: delim
      :canonical: deepxube.nnets.resnet_fc.ResnetFCParser.delim
      :type: str

      .. autodoc2-docstring:: deepxube.nnets.resnet_fc.ResnetFCParser.delim

.. py:class:: ResnetFCParserHeur()
   :canonical: deepxube.nnets.resnet_fc.ResnetFCParserHeur

   Bases: :py:obj:`deepxube.nnets.resnet_fc.ResnetFCParser`

.. py:class:: ResnetFCParserPolicy()
   :canonical: deepxube.nnets.resnet_fc.ResnetFCParserPolicy

   Bases: :py:obj:`deepxube.nnets.resnet_fc.ResnetFCParser`
