:py:mod:`deepxube.updaters.updater_q_rl`
========================================

.. py:module:: deepxube.updaters.updater_q_rl

.. autodoc2-docstring:: deepxube.updaters.updater_q_rl
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdateHeurQRL <deepxube.updaters.updater_q_rl.UpdateHeurQRL>`
     -
   * - :py:obj:`UpdateHeurQRLKeepGoalABC <deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC>`
     -
   * - :py:obj:`UpdateHeurQRLHERABC <deepxube.updaters.updater_q_rl.UpdateHeurQRLHERABC>`
     -
   * - :py:obj:`UpdateHeurQRLKeepGoal <deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoal>`
     -
   * - :py:obj:`UpdateHeurQRLHER <deepxube.updaters.updater_q_rl.UpdateHeurQRLHER>`
     -
   * - :py:obj:`UpdateHeurQRLKeepGoalPolicy <deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalPolicy>`
     -
   * - :py:obj:`UpdateHeurQRLHERPolicy <deepxube.updaters.updater_q_rl.UpdateHeurQRLHERPolicy>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_pathfind_q_step <deepxube.updaters.updater_q_rl._pathfind_q_step>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_q_rl._pathfind_q_step
          :summary:
   * - :py:obj:`_get_edge_popped_data <deepxube.updaters.updater_q_rl._get_edge_popped_data>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_q_rl._get_edge_popped_data
          :summary:

API
~~~

.. py:function:: _pathfind_q_step(pathfind: deepxube.base.pathfinding.PathFindSetHeurQ) -> typing.List[deepxube.base.pathfinding.EdgeQ]
   :canonical: deepxube.updaters.updater_q_rl._pathfind_q_step

   .. autodoc2-docstring:: deepxube.updaters.updater_q_rl._pathfind_q_step

.. py:function:: _get_edge_popped_data(edges_popped: typing.List[deepxube.base.pathfinding.EdgeQ], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[bool], typing.List[deepxube.base.domain.Action], typing.List[float], typing.List[deepxube.base.domain.State]]
   :canonical: deepxube.updaters.updater_q_rl._get_edge_popped_data

   .. autodoc2-docstring:: deepxube.updaters.updater_q_rl._get_edge_popped_data

.. py:class:: UpdateHeurQRL(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRL

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurQ`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHQ`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHQ`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.InstanceEdge`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetHeurQ]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRL.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRL.pathfind_type

   .. py:method:: _step(pathfind: deepxube.base.pathfinding.PathFindSetHeurQ, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRL._step

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRL._step

   .. py:method:: _q_learning_target(goals: typing.List[deepxube.base.domain.Goal], is_solved_l: typing.List[bool], tcs: typing.List[float], states_next: typing.List[deepxube.base.domain.State]) -> typing.List[float]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRL._q_learning_target

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRL._q_learning_target

   .. py:method:: _inputs_ctgs_to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action], ctgs_backup: typing.List[float], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRL._inputs_ctgs_to_np

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRL._inputs_ctgs_to_np

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRL._init_replay_buffer

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRL._init_replay_buffer

   .. py:method:: _rb_add(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], is_solved_l: typing.List[bool], actions: typing.List[deepxube.base.domain.Action], tcs: typing.List[float], states_next: typing.List[deepxube.base.domain.State], times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRL._rb_add

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRL._rb_add

   .. py:method:: _sample_rb_qlearn_target(num: int, times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[deepxube.base.domain.Action], typing.List[float]]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRL._sample_rb_qlearn_target

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRL._sample_rb_qlearn_target

.. py:class:: UpdateHeurQRLKeepGoalABC(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_q_rl.UpdateHeurQRL`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHQ`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC.domain_type

   .. py:method:: _step_sync_main(pathfind: deepxube.base.pathfinding.PathFindSetHeurQ, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC._step_sync_main

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC._step_sync_main

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.InstanceEdge], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC._get_instance_data_norb

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.InstanceEdge], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC._get_instance_data_rb

.. py:class:: UpdateHeurQRLHERABC(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERABC

   Bases: :py:obj:`deepxube.updaters.updater_q_rl.UpdateHeurQRL`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.FNsHQ`\ ], :py:obj:`deepxube.base.updater.UpdateHER`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHQ`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.InstanceEdge`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.GoalSampleableFromState]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERABC.domain_type

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.InstanceEdge], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERABC._get_instance_data_rb

.. py:class:: UpdateHeurQRLKeepGoal(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurQ`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurQ]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoal.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoal.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurQ
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoal._get_pathfind_functions

.. py:class:: UpdateHeurQRLHER(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLHER

   Bases: :py:obj:`deepxube.updaters.updater_q_rl.UpdateHeurQRLHERABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurQ`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurQ]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLHER.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLHER.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurQ
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLHER._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLHER._get_pathfind_functions

.. py:class:: UpdateHeurQRLKeepGoalPolicy(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalPolicy

   Bases: :py:obj:`deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.InstanceEdge`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalPolicy.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalPolicy.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurQPolicy
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLKeepGoalPolicy._get_pathfind_functions

.. py:class:: UpdateHeurQRLHERPolicy(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERPolicy

   Bases: :py:obj:`deepxube.updaters.updater_q_rl.UpdateHeurQRLHERABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.InstanceEdge`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERPolicy.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERPolicy.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurQPolicy
      :canonical: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_q_rl.UpdateHeurQRLHERPolicy._get_pathfind_functions
