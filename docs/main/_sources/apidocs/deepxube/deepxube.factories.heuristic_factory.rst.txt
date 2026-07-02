:py:mod:`deepxube.factories.heuristic_factory`
==============================================

.. py:module:: deepxube.factories.heuristic_factory

.. autodoc2-docstring:: deepxube.factories.heuristic_factory
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HeurNNetParFacClass <deepxube.factories.heuristic_factory.HeurNNetParFacClass>`
     -
   * - :py:obj:`PolicyNNetParFacClass <deepxube.factories.heuristic_factory.PolicyNNetParFacClass>`
     -
   * - :py:obj:`HeurNNetParVConcrete <deepxube.factories.heuristic_factory.HeurNNetParVConcrete>`
     -
   * - :py:obj:`HeurNNetParQFixOutConcrete <deepxube.factories.heuristic_factory.HeurNNetParQFixOutConcrete>`
     -
   * - :py:obj:`HeurNNetParQActInConcrete <deepxube.factories.heuristic_factory.HeurNNetParQActInConcrete>`
     -
   * - :py:obj:`PolicyNNetParConcrete <deepxube.factories.heuristic_factory.PolicyNNetParConcrete>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`build_heur_nnet_par <deepxube.factories.heuristic_factory.build_heur_nnet_par>`
     - .. autodoc2-docstring:: deepxube.factories.heuristic_factory.build_heur_nnet_par
          :summary:
   * - :py:obj:`build_policy_nnet_par <deepxube.factories.heuristic_factory.build_policy_nnet_par>`
     - .. autodoc2-docstring:: deepxube.factories.heuristic_factory.build_policy_nnet_par
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`heuristic_factory <deepxube.factories.heuristic_factory.heuristic_factory>`
     - .. autodoc2-docstring:: deepxube.factories.heuristic_factory.heuristic_factory
          :summary:
   * - :py:obj:`policy_factory <deepxube.factories.heuristic_factory.policy_factory>`
     - .. autodoc2-docstring:: deepxube.factories.heuristic_factory.policy_factory
          :summary:

API
~~~

.. py:data:: heuristic_factory
   :canonical: deepxube.factories.heuristic_factory.heuristic_factory
   :type: deepxube.base.factory.Factory[deepxube.base.heuristic.HeurNNet]
   :value: '(...)'

   .. autodoc2-docstring:: deepxube.factories.heuristic_factory.heuristic_factory

.. py:data:: policy_factory
   :canonical: deepxube.factories.heuristic_factory.policy_factory
   :type: deepxube.base.factory.Factory[deepxube.base.heuristic.PolicyNNet]
   :value: '(...)'

   .. autodoc2-docstring:: deepxube.factories.heuristic_factory.policy_factory

.. py:function:: build_heur_nnet_par(domain: deepxube.base.domain.Domain, domain_name: str, nnet_name: str, nnet_kwargs: typing.Dict[str, typing.Any], heur_type: str) -> deepxube.base.heuristic.HeurNNetPar
   :canonical: deepxube.factories.heuristic_factory.build_heur_nnet_par

   .. autodoc2-docstring:: deepxube.factories.heuristic_factory.build_heur_nnet_par

.. py:function:: build_policy_nnet_par(domain: deepxube.base.domain.Domain, domain_name: str, nnet_name: str, nnet_kwargs: typing.Dict[str, typing.Any], num_samp: int) -> deepxube.base.heuristic.PolicyNNetPar
   :canonical: deepxube.factories.heuristic_factory.build_policy_nnet_par

   .. autodoc2-docstring:: deepxube.factories.heuristic_factory.build_policy_nnet_par

.. py:class:: HeurNNetParFacClass(domain: deepxube.base.domain.Domain, nnet_input_name: typing.Tuple[str, str], nnet_name: str, nnet_kwargs: typing.Dict[str, typing.Any], q_fix: bool, out_dim: int)
   :canonical: deepxube.factories.heuristic_factory.HeurNNetParFacClass

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNetPar`, :py:obj:`abc.ABC`

   .. py:method:: get_nnet() -> deepxube.base.heuristic.HeurNNet
      :canonical: deepxube.factories.heuristic_factory.HeurNNetParFacClass.get_nnet

   .. py:method:: _get_nnet_input() -> deepxube.base.nnet_input.NNetInput
      :canonical: deepxube.factories.heuristic_factory.HeurNNetParFacClass._get_nnet_input

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.HeurNNetParFacClass._get_nnet_input

   .. py:method:: __getstate__() -> typing.Dict
      :canonical: deepxube.factories.heuristic_factory.HeurNNetParFacClass.__getstate__

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.HeurNNetParFacClass.__getstate__

.. py:class:: PolicyNNetParFacClass(domain: deepxube.base.domain.Domain, nnet_input_name: typing.Tuple[str, str], nnet_name: str, nnet_kwargs: typing.Dict[str, typing.Any], num_samp: int)
   :canonical: deepxube.factories.heuristic_factory.PolicyNNetParFacClass

   Bases: :py:obj:`deepxube.base.heuristic.PolicyNNetPar`, :py:obj:`abc.ABC`

   .. py:method:: get_nnet() -> deepxube.base.heuristic.PolicyNNet
      :canonical: deepxube.factories.heuristic_factory.PolicyNNetParFacClass.get_nnet

   .. py:method:: _get_nnet_input() -> deepxube.base.nnet_input.PolicyNNetIn
      :canonical: deepxube.factories.heuristic_factory.PolicyNNetParFacClass._get_nnet_input

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.PolicyNNetParFacClass._get_nnet_input

   .. py:method:: __getstate__() -> typing.Dict
      :canonical: deepxube.factories.heuristic_factory.PolicyNNetParFacClass.__getstate__

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.PolicyNNetParFacClass.__getstate__

.. py:class:: HeurNNetParVConcrete(domain: deepxube.base.domain.Domain, nnet_input_name: typing.Tuple[str, str], nnet_name: str, nnet_kwargs: typing.Dict[str, typing.Any])
   :canonical: deepxube.factories.heuristic_factory.HeurNNetParVConcrete

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNetParV`, :py:obj:`deepxube.factories.heuristic_factory.HeurNNetParFacClass`

   .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.factories.heuristic_factory.HeurNNetParVConcrete.to_np

   .. py:method:: _get_nnet_input() -> deepxube.base.nnet_input.StateGoalIn
      :canonical: deepxube.factories.heuristic_factory.HeurNNetParVConcrete._get_nnet_input

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.HeurNNetParVConcrete._get_nnet_input

.. py:class:: HeurNNetParQFixOutConcrete(domain: deepxube.base.domain.Domain, nnet_input_name: typing.Tuple[str, str], nnet_name: str, nnet_kwargs: typing.Dict[str, typing.Any], out_dim: int)
   :canonical: deepxube.factories.heuristic_factory.HeurNNetParQFixOutConcrete

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNetParQFixOut`, :py:obj:`deepxube.factories.heuristic_factory.HeurNNetParFacClass`

   .. py:method:: _to_np_fixed_acts(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.factories.heuristic_factory.HeurNNetParQFixOutConcrete._to_np_fixed_acts

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.HeurNNetParQFixOutConcrete._to_np_fixed_acts

   .. py:method:: _get_nnet_input() -> deepxube.base.nnet_input.StateGoalActFixIn
      :canonical: deepxube.factories.heuristic_factory.HeurNNetParQFixOutConcrete._get_nnet_input

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.HeurNNetParQFixOutConcrete._get_nnet_input

.. py:class:: HeurNNetParQActInConcrete(domain: deepxube.base.domain.Domain, nnet_input_name: typing.Tuple[str, str], nnet_name: str, nnet_kwargs: typing.Dict[str, typing.Any])
   :canonical: deepxube.factories.heuristic_factory.HeurNNetParQActInConcrete

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNetParQIn`, :py:obj:`deepxube.factories.heuristic_factory.HeurNNetParFacClass`

   .. py:method:: _to_np_one_act(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.factories.heuristic_factory.HeurNNetParQActInConcrete._to_np_one_act

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.HeurNNetParQActInConcrete._to_np_one_act

   .. py:method:: _get_nnet_input() -> deepxube.base.nnet_input.StateGoalActIn
      :canonical: deepxube.factories.heuristic_factory.HeurNNetParQActInConcrete._get_nnet_input

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.HeurNNetParQActInConcrete._get_nnet_input

.. py:class:: PolicyNNetParConcrete(domain: deepxube.base.domain.Domain, nnet_input_name: typing.Tuple[str, str], nnet_name: str, nnet_kwargs: typing.Dict[str, typing.Any], num_samp: int)
   :canonical: deepxube.factories.heuristic_factory.PolicyNNetParConcrete

   Bases: :py:obj:`deepxube.factories.heuristic_factory.PolicyNNetParFacClass`

   .. py:method:: to_np_fn(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.factories.heuristic_factory.PolicyNNetParConcrete.to_np_fn

   .. py:method:: to_np_train(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action]) -> typing.List[numpy.typing.NDArray[typing.Any]]
      :canonical: deepxube.factories.heuristic_factory.PolicyNNetParConcrete.to_np_train

   .. py:method:: _nnet_out_to_actions(nnet_out: typing.List[numpy.typing.NDArray[numpy.float64]]) -> typing.List[deepxube.base.domain.Action]
      :canonical: deepxube.factories.heuristic_factory.PolicyNNetParConcrete._nnet_out_to_actions

      .. autodoc2-docstring:: deepxube.factories.heuristic_factory.PolicyNNetParConcrete._nnet_out_to_actions
