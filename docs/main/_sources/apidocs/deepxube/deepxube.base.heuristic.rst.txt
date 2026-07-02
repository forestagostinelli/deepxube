:py:mod:`deepxube.base.heuristic`
=================================

.. py:module:: deepxube.base.heuristic

.. autodoc2-docstring:: deepxube.base.heuristic
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DeepXubeNNet <deepxube.base.heuristic.DeepXubeNNet>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet
          :summary:
   * - :py:obj:`HeurNNet <deepxube.base.heuristic.HeurNNet>`
     -
   * - :py:obj:`PolicyNNet <deepxube.base.heuristic.PolicyNNet>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.PolicyNNet
          :summary:
   * - :py:obj:`PolicyVAE <deepxube.base.heuristic.PolicyVAE>`
     -
   * - :py:obj:`HeurFnV <deepxube.base.heuristic.HeurFnV>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.HeurFnV
          :summary:
   * - :py:obj:`HeurFnQ <deepxube.base.heuristic.HeurFnQ>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.HeurFnQ
          :summary:
   * - :py:obj:`PolicyFn <deepxube.base.heuristic.PolicyFn>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.PolicyFn
          :summary:
   * - :py:obj:`HeurNNetPar <deepxube.base.heuristic.HeurNNetPar>`
     -
   * - :py:obj:`HeurNNetParV <deepxube.base.heuristic.HeurNNetParV>`
     -
   * - :py:obj:`HeurNNetParQ <deepxube.base.heuristic.HeurNNetParQ>`
     -
   * - :py:obj:`HeurNNetParQFixOut <deepxube.base.heuristic.HeurNNetParQFixOut>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQFixOut
          :summary:
   * - :py:obj:`HeurNNetParQIn <deepxube.base.heuristic.HeurNNetParQIn>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQIn
          :summary:
   * - :py:obj:`PolicyNNetPar <deepxube.base.heuristic.PolicyNNetPar>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_flatten_list <deepxube.base.heuristic._flatten_list>`
     - .. autodoc2-docstring:: deepxube.base.heuristic._flatten_list
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`In <deepxube.base.heuristic.In>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.In
          :summary:
   * - :py:obj:`PNNetIn <deepxube.base.heuristic.PNNetIn>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.PNNetIn
          :summary:
   * - :py:obj:`HeurFn <deepxube.base.heuristic.HeurFn>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.HeurFn
          :summary:
   * - :py:obj:`H <deepxube.base.heuristic.H>`
     - .. autodoc2-docstring:: deepxube.base.heuristic.H
          :summary:

API
~~~

.. py:data:: In
   :canonical: deepxube.base.heuristic.In
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.heuristic.In

.. py:class:: DeepXubeNNet(nnet_input: deepxube.base.heuristic.In)
   :canonical: deepxube.base.heuristic.DeepXubeNNet

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.heuristic.In`\ ], :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet.__init__

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.heuristic.In]
      :canonical: deepxube.base.heuristic.DeepXubeNNet.nnet_input_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet.nnet_input_type

   .. py:method:: forward(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.heuristic.DeepXubeNNet.forward

      .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet.forward

   .. py:method:: get_optimizer() -> torch.optim.optimizer.Optimizer
      :canonical: deepxube.base.heuristic.DeepXubeNNet.get_optimizer

      .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet.get_optimizer

   .. py:method:: update_optimizer(optimizer: torch.optim.optimizer.Optimizer, train_itr: int) -> None
      :canonical: deepxube.base.heuristic.DeepXubeNNet.update_optimizer

      .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet.update_optimizer

   .. py:method:: get_loss_and_info(fwd_tr_tensors: typing.List[torch.Tensor], get_info: bool) -> typing.Tuple[torch.Tensor, typing.Optional[str]]
      :canonical: deepxube.base.heuristic.DeepXubeNNet.get_loss_and_info
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet.get_loss_and_info

   .. py:method:: _forward_eval(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.heuristic.DeepXubeNNet._forward_eval
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet._forward_eval

   .. py:method:: _forward_train(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.heuristic.DeepXubeNNet._forward_train
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.DeepXubeNNet._forward_train

.. py:class:: HeurNNet(nnet_input: deepxube.base.heuristic.In, out_dim: int, q_fix: bool, **kwargs: typing.Any)
   :canonical: deepxube.base.heuristic.HeurNNet

   Bases: :py:obj:`deepxube.base.heuristic.DeepXubeNNet`\ [\ :py:obj:`deepxube.base.heuristic.In`\ ]

   .. py:method:: get_loss_and_info(fwd_tr_tensors: typing.List[torch.Tensor], get_info: bool) -> typing.Tuple[torch.Tensor, typing.Optional[str]]
      :canonical: deepxube.base.heuristic.HeurNNet.get_loss_and_info

   .. py:method:: _forward_train(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.heuristic.HeurNNet._forward_train

   .. py:method:: _forward_eval(inputs: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.heuristic.HeurNNet._forward_eval

   .. py:method:: _forward_heur(inputs: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.base.heuristic.HeurNNet._forward_heur

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNet._forward_heur

   .. py:method:: _forward(inputs: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.base.heuristic.HeurNNet._forward
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNet._forward

.. py:data:: PNNetIn
   :canonical: deepxube.base.heuristic.PNNetIn
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.heuristic.PNNetIn

.. py:class:: PolicyNNet(nnet_input: deepxube.base.heuristic.PNNetIn, num_samp: int, **kwargs: typing.Any)
   :canonical: deepxube.base.heuristic.PolicyNNet

   Bases: :py:obj:`deepxube.base.heuristic.DeepXubeNNet`\ [\ :py:obj:`deepxube.base.heuristic.PNNetIn`\ ], :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.heuristic.PolicyNNet

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.heuristic.PolicyNNet.__init__

.. py:function:: _flatten_list(data_l: typing.List[torch.Tensor]) -> torch.Tensor
   :canonical: deepxube.base.heuristic._flatten_list

   .. autodoc2-docstring:: deepxube.base.heuristic._flatten_list

.. py:class:: PolicyVAE(nnet_input: deepxube.base.heuristic.PNNetIn, num_samp: int, kl_weight: float, **kwargs: typing.Any)
   :canonical: deepxube.base.heuristic.PolicyVAE

   Bases: :py:obj:`deepxube.base.heuristic.PolicyNNet`\ [\ :py:obj:`deepxube.base.heuristic.PNNetIn`\ ]

   .. py:method:: get_loss_and_info(fwd_tr_tensors: typing.List[torch.Tensor], get_info: bool) -> typing.Tuple[torch.Tensor, typing.Optional[str]]
      :canonical: deepxube.base.heuristic.PolicyVAE.get_loss_and_info

   .. py:method:: _forward_eval(states_goals: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.heuristic.PolicyVAE._forward_eval

   .. py:method:: _forward_train(states_goals_actions: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.heuristic.PolicyVAE._forward_train

   .. py:method:: latent_shape() -> typing.Tuple[int, ...]
      :canonical: deepxube.base.heuristic.PolicyVAE.latent_shape
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.PolicyVAE.latent_shape

   .. py:method:: encode(states_goals: typing.List[torch.Tensor], actions: typing.List[torch.Tensor]) -> typing.Tuple[typing.List[torch.Tensor], torch.Tensor, torch.Tensor]
      :canonical: deepxube.base.heuristic.PolicyVAE.encode
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.PolicyVAE.encode

   .. py:method:: decode(states_goals: typing.List[torch.Tensor], z: torch.Tensor) -> typing.List[torch.Tensor]
      :canonical: deepxube.base.heuristic.PolicyVAE.decode
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.PolicyVAE.decode

   .. py:method:: _compute_recon_loss(action_proc: typing.List[torch.Tensor], actions_recon: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.base.heuristic.PolicyVAE._compute_recon_loss

      .. autodoc2-docstring:: deepxube.base.heuristic.PolicyVAE._compute_recon_loss

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.heuristic.PolicyVAE.__repr__

.. py:class:: HeurFnV
   :canonical: deepxube.base.heuristic.HeurFnV

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: deepxube.base.heuristic.HeurFnV

   .. py:method:: __call__(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[float]
      :canonical: deepxube.base.heuristic.HeurFnV.__call__

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurFnV.__call__

.. py:class:: HeurFnQ
   :canonical: deepxube.base.heuristic.HeurFnQ

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: deepxube.base.heuristic.HeurFnQ

   .. py:method:: __call__(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.base.heuristic.HeurFnQ.__call__

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurFnQ.__call__

.. py:data:: HeurFn
   :canonical: deepxube.base.heuristic.HeurFn
   :value: None

   .. autodoc2-docstring:: deepxube.base.heuristic.HeurFn

.. py:class:: PolicyFn
   :canonical: deepxube.base.heuristic.PolicyFn

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: deepxube.base.heuristic.PolicyFn

   .. py:method:: __call__(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
      :canonical: deepxube.base.heuristic.PolicyFn.__call__

      .. autodoc2-docstring:: deepxube.base.heuristic.PolicyFn.__call__

.. py:data:: H
   :canonical: deepxube.base.heuristic.H
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.heuristic.H

.. py:class:: HeurNNetPar
   :canonical: deepxube.base.heuristic.HeurNNetPar

   Bases: :py:obj:`deepxube.nnet.nnet_utils.NNetPar`\ [\ :py:obj:`deepxube.base.heuristic.H`\ ]

   .. py:method:: get_nnet() -> deepxube.base.heuristic.HeurNNet
      :canonical: deepxube.base.heuristic.HeurNNetPar.get_nnet
      :abstractmethod:

   .. py:method:: get_nnet_fn(nnet: torch.nn.Module, batch_size: typing.Optional[int], device: torch.device, update_num: typing.Optional[int]) -> deepxube.base.heuristic.H
      :canonical: deepxube.base.heuristic.HeurNNetPar.get_nnet_fn
      :abstractmethod:

   .. py:method:: get_nnet_par_fn(nnet_par_info: deepxube.nnet.nnet_utils.NNetParInfo, update_num: typing.Optional[int]) -> deepxube.base.heuristic.H
      :canonical: deepxube.base.heuristic.HeurNNetPar.get_nnet_par_fn
      :abstractmethod:

.. py:class:: HeurNNetParV
   :canonical: deepxube.base.heuristic.HeurNNetParV

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNetPar`\ [\ :py:obj:`deepxube.base.heuristic.HeurFnV`\ ]

   .. py:method:: _get_output(heurs: numpy.typing.NDArray[numpy.float64], update_num: typing.Optional[int]) -> typing.List[float]
      :canonical: deepxube.base.heuristic.HeurNNetParV._get_output
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParV._get_output

   .. py:method:: get_nnet_fn(nnet: torch.nn.Module, batch_size: typing.Optional[int], device: torch.device, update_num: typing.Optional[int]) -> deepxube.base.heuristic.HeurFnV
      :canonical: deepxube.base.heuristic.HeurNNetParV.get_nnet_fn

   .. py:method:: get_nnet_par_fn(nnet_par_info: deepxube.nnet.nnet_utils.NNetParInfo, update_num: typing.Optional[int]) -> deepxube.base.heuristic.HeurFnV
      :canonical: deepxube.base.heuristic.HeurNNetParV.get_nnet_par_fn

   .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.base.heuristic.HeurNNetParV.to_np
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParV.to_np

.. py:class:: HeurNNetParQ
   :canonical: deepxube.base.heuristic.HeurNNetParQ

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNetPar`\ [\ :py:obj:`deepxube.base.heuristic.HeurFnQ`\ ]

   .. py:method:: get_nnet_fn(nnet: torch.nn.Module, batch_size: typing.Optional[int], device: torch.device, update_num: typing.Optional[int]) -> deepxube.base.heuristic.HeurFnQ
      :canonical: deepxube.base.heuristic.HeurNNetParQ.get_nnet_fn
      :abstractmethod:

   .. py:method:: get_nnet_par_fn(nnet_par_info: deepxube.nnet.nnet_utils.NNetParInfo, update_num: typing.Optional[int]) -> deepxube.base.heuristic.HeurFnQ
      :canonical: deepxube.base.heuristic.HeurNNetParQ.get_nnet_par_fn
      :abstractmethod:

   .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.base.heuristic.HeurNNetParQ.to_np
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQ.to_np

.. py:class:: HeurNNetParQFixOut
   :canonical: deepxube.base.heuristic.HeurNNetParQFixOut

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNetParQ`, :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQFixOut

   .. py:method:: get_nnet_fn(nnet: torch.nn.Module, batch_size: typing.Optional[int], device: torch.device, update_num: typing.Optional[int]) -> deepxube.base.heuristic.HeurFnQ
      :canonical: deepxube.base.heuristic.HeurNNetParQFixOut.get_nnet_fn

   .. py:method:: get_nnet_par_fn(nnet_par_info: deepxube.nnet.nnet_utils.NNetParInfo, update_num: typing.Optional[int]) -> deepxube.base.heuristic.HeurFnQ
      :canonical: deepxube.base.heuristic.HeurNNetParQFixOut.get_nnet_par_fn

   .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.base.heuristic.HeurNNetParQFixOut.to_np

   .. py:method:: _to_np_fixed_acts(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.base.heuristic.HeurNNetParQFixOut._to_np_fixed_acts
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQFixOut._to_np_fixed_acts

   .. py:method:: _get_input(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.heuristic.HeurNNetParQFixOut._get_input

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQFixOut._get_input

   .. py:method:: _get_output(states: typing.List[deepxube.base.domain.State], q_vals_np: numpy.typing.NDArray[numpy.float64], update_num: typing.Optional[int]) -> typing.List[typing.List[float]]
      :canonical: deepxube.base.heuristic.HeurNNetParQFixOut._get_output
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQFixOut._get_output

   .. py:method:: _check_same_num_acts(actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> None
      :canonical: deepxube.base.heuristic.HeurNNetParQFixOut._check_same_num_acts
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQFixOut._check_same_num_acts

.. py:class:: HeurNNetParQIn
   :canonical: deepxube.base.heuristic.HeurNNetParQIn

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNetParQ`, :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQIn

   .. py:method:: get_nnet_fn(nnet: torch.nn.Module, batch_size: typing.Optional[int], device: torch.device, update_num: typing.Optional[int]) -> deepxube.base.heuristic.HeurFnQ
      :canonical: deepxube.base.heuristic.HeurNNetParQIn.get_nnet_fn

   .. py:method:: get_nnet_par_fn(nnet_par_info: deepxube.nnet.nnet_utils.NNetParInfo, update_num: typing.Optional[int]) -> deepxube.base.heuristic.HeurFnQ
      :canonical: deepxube.base.heuristic.HeurNNetParQIn.get_nnet_par_fn

   .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.base.heuristic.HeurNNetParQIn.to_np

   .. py:method:: _to_np_one_act(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.base.heuristic.HeurNNetParQIn._to_np_one_act
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQIn._to_np_one_act

   .. py:method:: _get_input(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.Tuple[typing.List[numpy.typing.NDArray], typing.List[deepxube.base.domain.State], typing.List[int]]
      :canonical: deepxube.base.heuristic.HeurNNetParQIn._get_input

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQIn._get_input

   .. py:method:: _get_output(states_rep: typing.List[deepxube.base.domain.State], q_vals_np: numpy.typing.NDArray[numpy.float64], split_idxs: typing.List[int], update_num: typing.Optional[int]) -> typing.List[typing.List[float]]
      :canonical: deepxube.base.heuristic.HeurNNetParQIn._get_output
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.HeurNNetParQIn._get_output

.. py:class:: PolicyNNetPar
   :canonical: deepxube.base.heuristic.PolicyNNetPar

   Bases: :py:obj:`deepxube.nnet.nnet_utils.NNetPar`\ [\ :py:obj:`deepxube.base.heuristic.PolicyFn`\ ]

   .. py:method:: get_nnet_fn(nnet: torch.nn.Module, batch_size: typing.Optional[int], device: torch.device, update_num: typing.Optional[int]) -> deepxube.base.heuristic.PolicyFn
      :canonical: deepxube.base.heuristic.PolicyNNetPar.get_nnet_fn

   .. py:method:: get_nnet_par_fn(nnet_par_info: deepxube.nnet.nnet_utils.NNetParInfo, update_num: typing.Optional[int]) -> deepxube.base.heuristic.PolicyFn
      :canonical: deepxube.base.heuristic.PolicyNNetPar.get_nnet_par_fn

   .. py:method:: get_nnet() -> deepxube.base.heuristic.PolicyNNet
      :canonical: deepxube.base.heuristic.PolicyNNetPar.get_nnet
      :abstractmethod:

   .. py:method:: to_np_fn(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.base.heuristic.PolicyNNetPar.to_np_fn
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.PolicyNNetPar.to_np_fn

   .. py:method:: to_np_train(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.base.heuristic.PolicyNNetPar.to_np_train
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.PolicyNNetPar.to_np_train

   .. py:method:: _nnet_out_to_actions(nnet_out: typing.List[numpy.typing.NDArray[numpy.float64]]) -> typing.List[deepxube.base.domain.Action]
      :canonical: deepxube.base.heuristic.PolicyNNetPar._nnet_out_to_actions
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.heuristic.PolicyNNetPar._nnet_out_to_actions

   .. py:method:: _np_to_acts_and_pdfs(actions_np: typing.List[numpy.typing.NDArray[numpy.float64]], pdfs_np: numpy.typing.NDArray[numpy.float64], num_states: int) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
      :canonical: deepxube.base.heuristic.PolicyNNetPar._np_to_acts_and_pdfs

      .. autodoc2-docstring:: deepxube.base.heuristic.PolicyNNetPar._np_to_acts_and_pdfs

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.heuristic.PolicyNNetPar.__repr__
