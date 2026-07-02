:py:mod:`deepxube.tests.time_tests`
===================================

.. py:module:: deepxube.tests.time_tests

.. autodoc2-docstring:: deepxube.tests.time_tests
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`data_runner <deepxube.tests.time_tests.data_runner>`
     - .. autodoc2-docstring:: deepxube.tests.time_tests.data_runner
          :summary:
   * - :py:obj:`test_env <deepxube.tests.time_tests.test_env>`
     - .. autodoc2-docstring:: deepxube.tests.time_tests.test_env
          :summary:
   * - :py:obj:`test_envstartgoalrw <deepxube.tests.time_tests.test_envstartgoalrw>`
     - .. autodoc2-docstring:: deepxube.tests.time_tests.test_envstartgoalrw
          :summary:
   * - :py:obj:`test_envenumerableacts <deepxube.tests.time_tests.test_envenumerableacts>`
     - .. autodoc2-docstring:: deepxube.tests.time_tests.test_envenumerableacts
          :summary:
   * - :py:obj:`init_nnet <deepxube.tests.time_tests.init_nnet>`
     - .. autodoc2-docstring:: deepxube.tests.time_tests.init_nnet
          :summary:
   * - :py:obj:`heur_fn_out <deepxube.tests.time_tests.heur_fn_out>`
     - .. autodoc2-docstring:: deepxube.tests.time_tests.heur_fn_out
          :summary:
   * - :py:obj:`test_heur_nnet_par <deepxube.tests.time_tests.test_heur_nnet_par>`
     - .. autodoc2-docstring:: deepxube.tests.time_tests.test_heur_nnet_par
          :summary:
   * - :py:obj:`test_policy_nnet_par <deepxube.tests.time_tests.test_policy_nnet_par>`
     - .. autodoc2-docstring:: deepxube.tests.time_tests.test_policy_nnet_par
          :summary:
   * - :py:obj:`time_test <deepxube.tests.time_tests.time_test>`
     - .. autodoc2-docstring:: deepxube.tests.time_tests.time_test
          :summary:

API
~~~

.. py:function:: data_runner(queue1: torch.multiprocessing.Queue, queue2: torch.multiprocessing.Queue) -> None
   :canonical: deepxube.tests.time_tests.data_runner

   .. autodoc2-docstring:: deepxube.tests.time_tests.data_runner

.. py:function:: test_env(env: deepxube.base.domain.Domain, num_states: int, step_min: int, step_max: int) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[deepxube.base.domain.Action]]
   :canonical: deepxube.tests.time_tests.test_env

   .. autodoc2-docstring:: deepxube.tests.time_tests.test_env

.. py:function:: test_envstartgoalrw(env: deepxube.base.domain.StartGoalWalkable, num_states: int) -> None
   :canonical: deepxube.tests.time_tests.test_envstartgoalrw

   .. autodoc2-docstring:: deepxube.tests.time_tests.test_envstartgoalrw

.. py:function:: test_envenumerableacts(env: deepxube.base.domain.ActsEnum, states: typing.List[deepxube.base.domain.State]) -> None
   :canonical: deepxube.tests.time_tests.test_envenumerableacts

   .. autodoc2-docstring:: deepxube.tests.time_tests.test_envenumerableacts

.. py:function:: init_nnet(nnet_par: deepxube.nnet.nnet_utils.NNetPar) -> typing.Tuple[torch.nn.Module, torch.device]
   :canonical: deepxube.tests.time_tests.init_nnet

   .. autodoc2-docstring:: deepxube.tests.time_tests.init_nnet

.. py:function:: heur_fn_out(heur_nnet: deepxube.base.heuristic.HeurNNetPar, heur_fn: deepxube.nnet.nnet_utils.NNetCallable, states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action]) -> None
   :canonical: deepxube.tests.time_tests.heur_fn_out

   .. autodoc2-docstring:: deepxube.tests.time_tests.heur_fn_out

.. py:function:: test_heur_nnet_par(heur_nnet_par: deepxube.base.heuristic.HeurNNetPar, states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action]) -> None
   :canonical: deepxube.tests.time_tests.test_heur_nnet_par

   .. autodoc2-docstring:: deepxube.tests.time_tests.test_heur_nnet_par

.. py:function:: test_policy_nnet_par(policy_nnet_par: deepxube.base.heuristic.PolicyNNetPar, states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action]) -> None
   :canonical: deepxube.tests.time_tests.test_policy_nnet_par

   .. autodoc2-docstring:: deepxube.tests.time_tests.test_policy_nnet_par

.. py:function:: time_test(domain: deepxube.base.domain.Domain, heur_nnet_par: typing.Optional[deepxube.base.heuristic.HeurNNetPar], policy_nnet_par: typing.Optional[deepxube.base.heuristic.PolicyNNetPar], num_states: int, step_min: int, step_max: int) -> None
   :canonical: deepxube.tests.time_tests.time_test

   .. autodoc2-docstring:: deepxube.tests.time_tests.time_test
