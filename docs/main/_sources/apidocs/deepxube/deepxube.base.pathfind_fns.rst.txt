:py:mod:`deepxube.base.pathfind_fns`
====================================

.. py:module:: deepxube.base.pathfind_fns

.. autodoc2-docstring:: deepxube.base.pathfind_fns
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HeurVFn <deepxube.base.pathfind_fns.HeurVFn>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVFn
          :summary:
   * - :py:obj:`HeurQFn <deepxube.base.pathfind_fns.HeurQFn>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurQFn
          :summary:
   * - :py:obj:`PolicyFn <deepxube.base.pathfind_fns.PolicyFn>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyFn
          :summary:
   * - :py:obj:`PFNs <deepxube.base.pathfind_fns.PFNs>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNs
          :summary:
   * - :py:obj:`PFNsHeurV <deepxube.base.pathfind_fns.PFNsHeurV>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurV
          :summary:
   * - :py:obj:`PFNsHeurQ <deepxube.base.pathfind_fns.PFNsHeurQ>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurQ
          :summary:
   * - :py:obj:`PFNsPolicy <deepxube.base.pathfind_fns.PFNsPolicy>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsPolicy
          :summary:
   * - :py:obj:`PFNsHeurVPolicy <deepxube.base.pathfind_fns.PFNsHeurVPolicy>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurVPolicy
          :summary:
   * - :py:obj:`PFNsHeurQPolicy <deepxube.base.pathfind_fns.PFNsHeurQPolicy>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurQPolicy
          :summary:
   * - :py:obj:`DeepXubeNNetPar <deepxube.base.pathfind_fns.DeepXubeNNetPar>`
     -
   * - :py:obj:`HeurNNetPar <deepxube.base.pathfind_fns.HeurNNetPar>`
     -
   * - :py:obj:`HeurVNNetPar <deepxube.base.pathfind_fns.HeurVNNetPar>`
     -
   * - :py:obj:`HeurQNNetPar <deepxube.base.pathfind_fns.HeurQNNetPar>`
     -
   * - :py:obj:`PolicyCtx <deepxube.base.pathfind_fns.PolicyCtx>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyCtx
          :summary:
   * - :py:obj:`PolicyNNetPar <deepxube.base.pathfind_fns.PolicyNNetPar>`
     -
   * - :py:obj:`UFNs <deepxube.base.pathfind_fns.UFNs>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNs
          :summary:
   * - :py:obj:`UFNsHeurV <deepxube.base.pathfind_fns.UFNsHeurV>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurV
          :summary:
   * - :py:obj:`UFNsHeurQ <deepxube.base.pathfind_fns.UFNsHeurQ>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurQ
          :summary:
   * - :py:obj:`UFNsPolicy <deepxube.base.pathfind_fns.UFNsPolicy>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsPolicy
          :summary:
   * - :py:obj:`UFNsHeurVPolicy <deepxube.base.pathfind_fns.UFNsHeurVPolicy>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurVPolicy
          :summary:
   * - :py:obj:`UFNsHeurQPolicy <deepxube.base.pathfind_fns.UFNsHeurQPolicy>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurQPolicy
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`policy_fn_rand <deepxube.base.pathfind_fns.policy_fn_rand>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.policy_fn_rand
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HeurFn <deepxube.base.pathfind_fns.HeurFn>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurFn
          :summary:
   * - :py:obj:`D <deepxube.base.pathfind_fns.D>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.D
          :summary:
   * - :py:obj:`NNInP <deepxube.base.pathfind_fns.NNInP>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.NNInP
          :summary:
   * - :py:obj:`DXNNet <deepxube.base.pathfind_fns.DXNNet>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.DXNNet
          :summary:
   * - :py:obj:`H <deepxube.base.pathfind_fns.H>`
     - .. autodoc2-docstring:: deepxube.base.pathfind_fns.H
          :summary:

API
~~~

.. py:class:: HeurVFn
   :canonical: deepxube.base.pathfind_fns.HeurVFn

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVFn

   .. py:method:: __call__(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[float]
      :canonical: deepxube.base.pathfind_fns.HeurVFn.__call__

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVFn.__call__

.. py:class:: HeurQFn
   :canonical: deepxube.base.pathfind_fns.HeurQFn

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurQFn

   .. py:method:: __call__(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.base.pathfind_fns.HeurQFn.__call__

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurQFn.__call__

.. py:data:: HeurFn
   :canonical: deepxube.base.pathfind_fns.HeurFn
   :value: None

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurFn

.. py:class:: PolicyFn
   :canonical: deepxube.base.pathfind_fns.PolicyFn

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyFn

   .. py:method:: __call__(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
      :canonical: deepxube.base.pathfind_fns.PolicyFn.__call__

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyFn.__call__

.. py:class:: PFNs
   :canonical: deepxube.base.pathfind_fns.PFNs

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNs

.. py:class:: PFNsHeurV
   :canonical: deepxube.base.pathfind_fns.PFNsHeurV

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNs`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurV

   .. py:attribute:: heurv
      :canonical: deepxube.base.pathfind_fns.PFNsHeurV.heurv
      :type: deepxube.base.pathfind_fns.HeurVFn
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurV.heurv

.. py:class:: PFNsHeurQ
   :canonical: deepxube.base.pathfind_fns.PFNsHeurQ

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNs`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurQ

   .. py:attribute:: heurq
      :canonical: deepxube.base.pathfind_fns.PFNsHeurQ.heurq
      :type: deepxube.base.pathfind_fns.HeurQFn
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurQ.heurq

.. py:class:: PFNsPolicy
   :canonical: deepxube.base.pathfind_fns.PFNsPolicy

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNs`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsPolicy

   .. py:attribute:: policy
      :canonical: deepxube.base.pathfind_fns.PFNsPolicy.policy
      :type: deepxube.base.pathfind_fns.PolicyFn
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsPolicy.policy

.. py:class:: PFNsHeurVPolicy
   :canonical: deepxube.base.pathfind_fns.PFNsHeurVPolicy

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNsPolicy`, :py:obj:`deepxube.base.pathfind_fns.PFNsHeurV`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurVPolicy

.. py:class:: PFNsHeurQPolicy
   :canonical: deepxube.base.pathfind_fns.PFNsHeurQPolicy

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNsPolicy`, :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQ`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.PFNsHeurQPolicy

.. py:data:: D
   :canonical: deepxube.base.pathfind_fns.D
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.D

.. py:data:: NNInP
   :canonical: deepxube.base.pathfind_fns.NNInP
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.NNInP

.. py:data:: DXNNet
   :canonical: deepxube.base.pathfind_fns.DXNNet
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.DXNNet

.. py:class:: DeepXubeNNetPar(domain: deepxube.base.pathfind_fns.D, nnet_input_name: typing.Optional[typing.Tuple[str, str]], nnet_name_args: typing.Optional[str], **kwargs: typing.Any)
   :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar

   Bases: :py:obj:`deepxube.pytorch.nnet_utils.NNetPar`\ [\ :py:obj:`deepxube.pytorch.nnet_utils.NNF_T`\ , :py:obj:`deepxube.pytorch.nnet_utils.CTX_T`\ ], :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.pytorch.nnet_utils.NNF_T`\ , :py:obj:`deepxube.pytorch.nnet_utils.CTX_T`\ , :py:obj:`deepxube.base.pathfind_fns.D`\ , :py:obj:`deepxube.base.pathfind_fns.NNInP`\ , :py:obj:`deepxube.base.pathfind_fns.DXNNet`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.pathfind_fns.D]
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar.domain_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.DeepXubeNNetPar.domain_type

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.pathfind_fns.NNInP]
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar.nnet_input_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.DeepXubeNNetPar.nnet_input_type

   .. py:method:: nnet_type() -> typing.Type[deepxube.base.pathfind_fns.DXNNet]
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar.nnet_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.DeepXubeNNetPar.nnet_type

   .. py:method:: get_incompat_reason(domain: deepxube.base.domain.Domain, nnet_input_t: typing.Optional[typing.Type[deepxube.base.nnet_input.NNetInput]], nnet_t: typing.Optional[typing.Type[deepxube.base.nnet.DeepXubeNNet]]) -> typing.Optional[str]
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar.get_incompat_reason
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.DeepXubeNNetPar.get_incompat_reason

   .. py:method:: get_field_name() -> str
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar.get_field_name
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.DeepXubeNNetPar.get_field_name

   .. py:method:: get_nnet() -> deepxube.base.pathfind_fns.DXNNet
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar.get_nnet

   .. py:method:: _add_nnet_kwargs(nnet_kwargs: typing.Dict) -> None
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar._add_nnet_kwargs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.DeepXubeNNetPar._add_nnet_kwargs

   .. py:method:: _get_nnet_input() -> deepxube.base.pathfind_fns.NNInP
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar._get_nnet_input

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.DeepXubeNNetPar._get_nnet_input

   .. py:method:: __getstate__() -> typing.Dict
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar.__getstate__

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.DeepXubeNNetPar.__getstate__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.pathfind_fns.DeepXubeNNetPar.__repr__

.. py:data:: H
   :canonical: deepxube.base.pathfind_fns.H
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.H

.. py:class:: HeurNNetPar(domain: deepxube.base.pathfind_fns.D, nnet_input_name: typing.Optional[typing.Tuple[str, str]], nnet_name_args: typing.Optional[str], **kwargs: typing.Any)
   :canonical: deepxube.base.pathfind_fns.HeurNNetPar

   Bases: :py:obj:`deepxube.base.pathfind_fns.DeepXubeNNetPar`\ [\ :py:obj:`deepxube.base.pathfind_fns.H`\ , :py:obj:`deepxube.pytorch.nnet_utils.CTX_T`\ , :py:obj:`deepxube.base.pathfind_fns.D`\ , :py:obj:`deepxube.base.pathfind_fns.NNInP`\ , :py:obj:`deepxube.base.nnet.HeurNNet`\ ], :py:obj:`abc.ABC`

   .. py:method:: nnet_type() -> typing.Type[deepxube.base.nnet.HeurNNet]
      :canonical: deepxube.base.pathfind_fns.HeurNNetPar.nnet_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurNNetPar.nnet_type

   .. py:method:: _qfix() -> bool
      :canonical: deepxube.base.pathfind_fns.HeurNNetPar._qfix
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurNNetPar._qfix

   .. py:method:: _out_dim() -> int
      :canonical: deepxube.base.pathfind_fns.HeurNNetPar._out_dim
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurNNetPar._out_dim

   .. py:method:: _add_nnet_kwargs(nnet_kwargs: typing.Dict) -> None
      :canonical: deepxube.base.pathfind_fns.HeurNNetPar._add_nnet_kwargs

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurNNetPar._add_nnet_kwargs

.. py:class:: HeurVNNetPar(domain: deepxube.base.pathfind_fns.D, nnet_input_name: typing.Optional[typing.Tuple[str, str]], nnet_name_args: typing.Optional[str], **kwargs: typing.Any)
   :canonical: deepxube.base.pathfind_fns.HeurVNNetPar

   Bases: :py:obj:`deepxube.base.pathfind_fns.HeurNNetPar`\ [\ :py:obj:`deepxube.base.pathfind_fns.HeurVFn`\ , :py:obj:`None`\ , :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.nnet_input.StateGoalIn`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.base.pathfind_fns.HeurVNNetPar.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVNNetPar.domain_type

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet_input.StateGoalIn]
      :canonical: deepxube.base.pathfind_fns.HeurVNNetPar.nnet_input_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVNNetPar.nnet_input_type

   .. py:method:: get_default_fn() -> deepxube.base.pathfind_fns.HeurVFn
      :canonical: deepxube.base.pathfind_fns.HeurVNNetPar.get_default_fn

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVNNetPar.get_default_fn

   .. py:method:: get_field_name() -> str
      :canonical: deepxube.base.pathfind_fns.HeurVNNetPar.get_field_name

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVNNetPar.get_field_name

   .. py:method:: process_inputs(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> deepxube.pytorch.nnet_utils.ProcessedInput[None]
      :canonical: deepxube.base.pathfind_fns.HeurVNNetPar.process_inputs

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVNNetPar.process_inputs

   .. py:method:: process_outputs(outs: typing.List[numpy.typing.NDArray], ctx: None) -> typing.List[float]
      :canonical: deepxube.base.pathfind_fns.HeurVNNetPar.process_outputs

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVNNetPar.process_outputs

   .. py:method:: _qfix() -> bool
      :canonical: deepxube.base.pathfind_fns.HeurVNNetPar._qfix

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVNNetPar._qfix

   .. py:method:: _out_dim() -> int
      :canonical: deepxube.base.pathfind_fns.HeurVNNetPar._out_dim

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurVNNetPar._out_dim

.. py:class:: HeurQNNetPar(domain: deepxube.base.pathfind_fns.D, nnet_input_name: typing.Optional[typing.Tuple[str, str]], nnet_name_args: typing.Optional[str], **kwargs: typing.Any)
   :canonical: deepxube.base.pathfind_fns.HeurQNNetPar

   Bases: :py:obj:`deepxube.base.pathfind_fns.HeurNNetPar`\ [\ :py:obj:`deepxube.base.pathfind_fns.HeurQFn`\ , :py:obj:`deepxube.pytorch.nnet_utils.CTX_T`\ , :py:obj:`deepxube.base.pathfind_fns.D`\ , :py:obj:`deepxube.base.pathfind_fns.NNInP`\ ], :py:obj:`abc.ABC`

   .. py:method:: process_inputs(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> deepxube.pytorch.nnet_utils.ProcessedInput[deepxube.pytorch.nnet_utils.CTX_T]
      :canonical: deepxube.base.pathfind_fns.HeurQNNetPar.process_inputs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurQNNetPar.process_inputs

   .. py:method:: get_default_fn() -> deepxube.base.pathfind_fns.HeurQFn
      :canonical: deepxube.base.pathfind_fns.HeurQNNetPar.get_default_fn

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurQNNetPar.get_default_fn

   .. py:method:: get_field_name() -> str
      :canonical: deepxube.base.pathfind_fns.HeurQNNetPar.get_field_name

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.HeurQNNetPar.get_field_name

.. py:class:: PolicyCtx
   :canonical: deepxube.base.pathfind_fns.PolicyCtx

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyCtx

   .. py:attribute:: num_states
      :canonical: deepxube.base.pathfind_fns.PolicyCtx.num_states
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyCtx.num_states

.. py:function:: policy_fn_rand(domain: deepxube.base.domain.Domain, states: typing.List[deepxube.base.domain.State], num_rand: int) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
   :canonical: deepxube.base.pathfind_fns.policy_fn_rand

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.policy_fn_rand

.. py:class:: PolicyNNetPar(*args: typing.Any, num_samp: int = 0, **kwargs: typing.Any)
   :canonical: deepxube.base.pathfind_fns.PolicyNNetPar

   Bases: :py:obj:`deepxube.base.pathfind_fns.DeepXubeNNetPar`\ [\ :py:obj:`deepxube.base.pathfind_fns.PolicyFn`\ , :py:obj:`deepxube.base.pathfind_fns.PolicyCtx`\ , :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.nnet_input.PolicyNNetIn`\ , :py:obj:`deepxube.base.nnet.PolicyNNet`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyNNetPar.domain_type

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet_input.PolicyNNetIn]
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar.nnet_input_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyNNetPar.nnet_input_type

   .. py:method:: nnet_type() -> typing.Type[deepxube.base.nnet.PolicyNNet]
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar.nnet_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyNNetPar.nnet_type

   .. py:method:: get_default_fn() -> deepxube.base.pathfind_fns.PolicyFn
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar.get_default_fn

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyNNetPar.get_default_fn

   .. py:method:: get_field_name() -> str
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar.get_field_name

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyNNetPar.get_field_name

   .. py:method:: to_np_train(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar.to_np_train

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyNNetPar.to_np_train

   .. py:method:: process_inputs(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> deepxube.pytorch.nnet_utils.ProcessedInput[deepxube.base.pathfind_fns.PolicyCtx]
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar.process_inputs

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyNNetPar.process_inputs

   .. py:method:: process_outputs(outs: typing.List[numpy.typing.NDArray], ctx: deepxube.base.pathfind_fns.PolicyCtx) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar.process_outputs

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyNNetPar.process_outputs

   .. py:method:: _add_nnet_kwargs(nnet_kwargs: typing.Dict) -> None
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar._add_nnet_kwargs

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.PolicyNNetPar._add_nnet_kwargs

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.pathfind_fns.PolicyNNetPar.__repr__

.. py:class:: UFNs
   :canonical: deepxube.base.pathfind_fns.UFNs

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNs

   .. py:method:: get_field_names() -> typing.List[str]
      :canonical: deepxube.base.pathfind_fns.UFNs.get_field_names

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNs.get_field_names

   .. py:method:: get_up_fns() -> typing.List[deepxube.base.pathfind_fns.DeepXubeNNetPar]
      :canonical: deepxube.base.pathfind_fns.UFNs.get_up_fns

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNs.get_up_fns

   .. py:method:: get_up_fn(field_name: str) -> deepxube.base.pathfind_fns.DeepXubeNNetPar
      :canonical: deepxube.base.pathfind_fns.UFNs.get_up_fn

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNs.get_up_fn

.. py:class:: UFNsHeurV
   :canonical: deepxube.base.pathfind_fns.UFNsHeurV

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNs`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurV

   .. py:attribute:: heurv
      :canonical: deepxube.base.pathfind_fns.UFNsHeurV.heurv
      :type: deepxube.base.pathfind_fns.HeurVNNetPar
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurV.heurv

.. py:class:: UFNsHeurQ
   :canonical: deepxube.base.pathfind_fns.UFNsHeurQ

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNs`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurQ

   .. py:attribute:: heurq
      :canonical: deepxube.base.pathfind_fns.UFNsHeurQ.heurq
      :type: deepxube.base.pathfind_fns.HeurQNNetPar
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurQ.heurq

.. py:class:: UFNsPolicy
   :canonical: deepxube.base.pathfind_fns.UFNsPolicy

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNs`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsPolicy

   .. py:attribute:: policy
      :canonical: deepxube.base.pathfind_fns.UFNsPolicy.policy
      :type: deepxube.base.pathfind_fns.PolicyNNetPar
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsPolicy.policy

.. py:class:: UFNsHeurVPolicy
   :canonical: deepxube.base.pathfind_fns.UFNsHeurVPolicy

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNsPolicy`, :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurVPolicy

.. py:class:: UFNsHeurQPolicy
   :canonical: deepxube.base.pathfind_fns.UFNsHeurQPolicy

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNsPolicy`, :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQ`

   .. autodoc2-docstring:: deepxube.base.pathfind_fns.UFNsHeurQPolicy
