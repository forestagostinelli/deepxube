:py:mod:`deepxube.updaters.updater_policy_rl`
=============================================

.. py:module:: deepxube.updaters.updater_policy_rl

.. autodoc2-docstring:: deepxube.updaters.updater_policy_rl
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdatePolicyRL <deepxube.updaters.updater_policy_rl.UpdatePolicyRL>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoalABC <deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC>`
     -
   * - :py:obj:`UpdatePolicyRLHERABC <deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERABC>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoal <deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoal>`
     -
   * - :py:obj:`UpdatePolicyRLHER <deepxube.updaters.updater_policy_rl.UpdatePolicyRLHER>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoalHeurV <deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurV>`
     -
   * - :py:obj:`UpdatePolicyRLHERHeurV <deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurV>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoalHeurQ <deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurQ>`
     -
   * - :py:obj:`UpdatePolicyRLHERHeurQ <deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurQ>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_pathfind_step <deepxube.updaters.updater_policy_rl._pathfind_step>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl._pathfind_step
          :summary:
   * - :py:obj:`_get_edge_popped_data <deepxube.updaters.updater_policy_rl._get_edge_popped_data>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl._get_edge_popped_data
          :summary:

API
~~~

.. py:function:: _pathfind_step(pathfind: deepxube.base.pathfinding.PathFind) -> typing.List[deepxube.base.pathfinding.EdgeQ]
   :canonical: deepxube.updaters.updater_policy_rl._pathfind_step

   .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl._pathfind_step

.. py:function:: _get_edge_popped_data(edges_popped: typing.List[deepxube.base.pathfinding.EdgeQ], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[deepxube.base.domain.Action]]
   :canonical: deepxube.updaters.updater_policy_rl._get_edge_popped_data

   .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl._get_edge_popped_data

.. py:class:: UpdatePolicyRL(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRL

   Bases: :py:obj:`deepxube.base.updater.UpdatePolicy`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsP`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsP`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindActsPolicy]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRL.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRL.pathfind_type

   .. py:method:: _step(pathfind: deepxube.base.pathfinding.PathFindActsPolicy, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._step

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._step

   .. py:method:: _inputs_ctgs_to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._inputs_ctgs_to_np

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._inputs_ctgs_to_np

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._init_replay_buffer

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._init_replay_buffer

   .. py:method:: _rb_add(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action], times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._rb_add

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._rb_add

   .. py:method:: _sample_rb(num: int, times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[deepxube.base.domain.Action]]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._sample_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRL._sample_rb

.. py:class:: UpdatePolicyRLKeepGoalABC(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_policy_rl.UpdatePolicyRL`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsP`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC.domain_type

   .. py:method:: _step_sync_main(pathfind: deepxube.base.pathfinding.PathFindActsPolicy, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC._step_sync_main

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC._step_sync_main

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC._get_instance_data_norb

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC._get_instance_data_rb

.. py:class:: UpdatePolicyRLHERABC(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERABC

   Bases: :py:obj:`deepxube.updaters.updater_policy_rl.UpdatePolicyRL`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.FNsP`\ ], :py:obj:`deepxube.base.updater.UpdateHER`\ [\ :py:obj:`deepxube.base.pathfinding.FNsP`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.GoalSampleableFromState]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERABC.domain_type

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERABC._get_instance_data_rb

.. py:class:: UpdatePolicyRLKeepGoal(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsPolicy`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsPolicy]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoal.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoal.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsPolicy
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoal._get_pathfind_functions

.. py:class:: UpdatePolicyRLHER(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHER

   Bases: :py:obj:`deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsPolicy`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsPolicy]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHER.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHER.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsPolicy
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHER._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHER._get_pathfind_functions

.. py:class:: UpdatePolicyRLKeepGoalHeurV(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurV

   Bases: :py:obj:`deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeur`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.heuristic.HeurNNetParV`\ , :py:obj:`deepxube.base.heuristic.HeurFnV`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurV.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurV.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurVPolicy
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurV._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurV._get_pathfind_functions

.. py:class:: UpdatePolicyRLHERHeurV(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurV

   Bases: :py:obj:`deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeur`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.heuristic.HeurNNetParV`\ , :py:obj:`deepxube.base.heuristic.HeurFnV`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurV.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurV.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurVPolicy
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurV._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurV._get_pathfind_functions

.. py:class:: UpdatePolicyRLKeepGoalHeurQ(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurQ

   Bases: :py:obj:`deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeur`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.heuristic.HeurNNetParQ`\ , :py:obj:`deepxube.base.heuristic.HeurFnQ`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurQ.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurQ.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurQPolicy
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurQ._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLKeepGoalHeurQ._get_pathfind_functions

.. py:class:: UpdatePolicyRLHERHeurQ(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurQ

   Bases: :py:obj:`deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeur`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.heuristic.HeurNNetParQ`\ , :py:obj:`deepxube.base.heuristic.HeurFnQ`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurQ.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurQ.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurQPolicy
      :canonical: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurQ._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_policy_rl.UpdatePolicyRLHERHeurQ._get_pathfind_functions
