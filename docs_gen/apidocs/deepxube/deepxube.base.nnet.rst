:py:mod:`deepxube.base.nnet`
============================

.. py:module:: deepxube.base.nnet

.. autodoc2-docstring:: deepxube.base.nnet
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DeepXubeNNet <deepxube.base.nnet.DeepXubeNNet>`
     - .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet
          :summary:
   * - :py:obj:`HeurNNet <deepxube.base.nnet.HeurNNet>`
     -
   * - :py:obj:`PolicyNNet <deepxube.base.nnet.PolicyNNet>`
     - .. autodoc2-docstring:: deepxube.base.nnet.PolicyNNet
          :summary:
   * - :py:obj:`PolicyVAE <deepxube.base.nnet.PolicyVAE>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_flatten_list <deepxube.base.nnet._flatten_list>`
     - .. autodoc2-docstring:: deepxube.base.nnet._flatten_list
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`In <deepxube.base.nnet.In>`
     - .. autodoc2-docstring:: deepxube.base.nnet.In
          :summary:
   * - :py:obj:`PNNetIn <deepxube.base.nnet.PNNetIn>`
     - .. autodoc2-docstring:: deepxube.base.nnet.PNNetIn
          :summary:

API
~~~

.. py:data:: In
   :canonical: deepxube.base.nnet.In
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.nnet.In

.. py:class:: DeepXubeNNet(nnet_input: deepxube.base.nnet.In)
   :canonical: deepxube.base.nnet.DeepXubeNNet

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.nnet.In`\ ], :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet.__init__

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet.In]
      :canonical: deepxube.base.nnet.DeepXubeNNet.nnet_input_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet.nnet_input_type

   .. py:method:: forward(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.nnet.DeepXubeNNet.forward

      .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet.forward

   .. py:method:: get_optimizer() -> torch.optim.Optimizer
      :canonical: deepxube.base.nnet.DeepXubeNNet.get_optimizer

      .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet.get_optimizer

   .. py:method:: update_optimizer(optimizer: torch.optim.Optimizer, train_itr: int) -> None
      :canonical: deepxube.base.nnet.DeepXubeNNet.update_optimizer

      .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet.update_optimizer

   .. py:method:: get_loss_and_info(fwd_tr_tensors: typing.List[torch.Tensor], get_info: bool) -> typing.Tuple[torch.Tensor, typing.Optional[str]]
      :canonical: deepxube.base.nnet.DeepXubeNNet.get_loss_and_info
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet.get_loss_and_info

   .. py:method:: _forward_eval(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.nnet.DeepXubeNNet._forward_eval
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet._forward_eval

   .. py:method:: _forward_train(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.nnet.DeepXubeNNet._forward_train
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet.DeepXubeNNet._forward_train

.. py:class:: HeurNNet(nnet_input: deepxube.base.nnet.In, out_dim: int, q_fix: bool, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet.HeurNNet

   Bases: :py:obj:`deepxube.base.nnet.DeepXubeNNet`\ [\ :py:obj:`deepxube.base.nnet.In`\ ]

   .. py:method:: get_loss_and_info(fwd_tr_tensors: typing.List[torch.Tensor], get_info: bool) -> typing.Tuple[torch.Tensor, typing.Optional[str]]
      :canonical: deepxube.base.nnet.HeurNNet.get_loss_and_info

   .. py:method:: _forward_train(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.nnet.HeurNNet._forward_train

   .. py:method:: _forward_eval(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.nnet.HeurNNet._forward_eval

   .. py:method:: _forward_heur(inputs: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.base.nnet.HeurNNet._forward_heur

      .. autodoc2-docstring:: deepxube.base.nnet.HeurNNet._forward_heur

   .. py:method:: _forward(inputs: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.base.nnet.HeurNNet._forward
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet.HeurNNet._forward

.. py:data:: PNNetIn
   :canonical: deepxube.base.nnet.PNNetIn
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.nnet.PNNetIn

.. py:class:: PolicyNNet(nnet_input: deepxube.base.nnet.PNNetIn, num_samp: int, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet.PolicyNNet

   Bases: :py:obj:`deepxube.base.nnet.DeepXubeNNet`\ [\ :py:obj:`deepxube.base.nnet.PNNetIn`\ ], :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.nnet.PolicyNNet

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.nnet.PolicyNNet.__init__

.. py:function:: _flatten_list(data_l: typing.List[torch.Tensor]) -> torch.Tensor
   :canonical: deepxube.base.nnet._flatten_list

   .. autodoc2-docstring:: deepxube.base.nnet._flatten_list

.. py:class:: PolicyVAE(nnet_input: deepxube.base.nnet.PNNetIn, num_samp: int, kl_weight: float, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet.PolicyVAE

   Bases: :py:obj:`deepxube.base.nnet.PolicyNNet`\ [\ :py:obj:`deepxube.base.nnet.PNNetIn`\ ]

   .. py:method:: _compute_recon_loss(action_proc: typing.List[torch.Tensor], actions_recon: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.base.nnet.PolicyVAE._compute_recon_loss
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.nnet.PolicyVAE._compute_recon_loss

   .. py:method:: get_loss_and_info(fwd_tr_tensors: typing.List[torch.Tensor], get_info: bool) -> typing.Tuple[torch.Tensor, typing.Optional[str]]
      :canonical: deepxube.base.nnet.PolicyVAE.get_loss_and_info

   .. py:method:: _forward_eval(states_goals: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.nnet.PolicyVAE._forward_eval

   .. py:method:: _forward_train(states_goals_actions: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.nnet.PolicyVAE._forward_train

   .. py:method:: latent_shape() -> typing.Tuple[int, ...]
      :canonical: deepxube.base.nnet.PolicyVAE.latent_shape
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet.PolicyVAE.latent_shape

   .. py:method:: encode(states_goals: typing.List[torch.Tensor], actions: typing.List[torch.Tensor]) -> typing.Tuple[typing.List[torch.Tensor], torch.Tensor, torch.Tensor]
      :canonical: deepxube.base.nnet.PolicyVAE.encode
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet.PolicyVAE.encode

   .. py:method:: decode(states_goals: typing.List[torch.Tensor], z: torch.Tensor) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.nnet.PolicyVAE.decode
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet.PolicyVAE.decode

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.nnet.PolicyVAE.__repr__
