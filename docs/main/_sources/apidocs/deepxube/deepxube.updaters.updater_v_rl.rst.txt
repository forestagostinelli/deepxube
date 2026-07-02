:py:mod:`deepxube.updaters.updater_v_rl`
========================================

.. py:module:: deepxube.updaters.updater_v_rl

.. autodoc2-docstring:: deepxube.updaters.updater_v_rl
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdateHeurVRL <deepxube.updaters.updater_v_rl.UpdateHeurVRL>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoalABC <deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC>`
     -
   * - :py:obj:`UpdateHeurVRLHERABC <deepxube.updaters.updater_v_rl.UpdateHeurVRLHERABC>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoal <deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoal>`
     -
   * - :py:obj:`UpdateHeurVRLHER <deepxube.updaters.updater_v_rl.UpdateHeurVRLHER>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoalPolicy <deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalPolicy>`
     -
   * - :py:obj:`UpdateHeurVRLHERPolicy <deepxube.updaters.updater_v_rl.UpdateHeurVRLHERPolicy>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_pathfind_v_step <deepxube.updaters.updater_v_rl._pathfind_v_step>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_v_rl._pathfind_v_step
          :summary:
   * - :py:obj:`_get_nodes_popped_data <deepxube.updaters.updater_v_rl._get_nodes_popped_data>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_v_rl._get_nodes_popped_data
          :summary:

API
~~~

.. py:function:: _pathfind_v_step(pathfind: deepxube.base.pathfinding.PathFindSetHeurV) -> typing.List[deepxube.base.pathfinding.Node]
   :canonical: deepxube.updaters.updater_v_rl._pathfind_v_step

   .. autodoc2-docstring:: deepxube.updaters.updater_v_rl._pathfind_v_step

.. py:function:: _get_nodes_popped_data(nodes_popped: typing.List[deepxube.base.pathfinding.Node], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[bool]]
   :canonical: deepxube.updaters.updater_v_rl._get_nodes_popped_data

   .. autodoc2-docstring:: deepxube.updaters.updater_v_rl._get_nodes_popped_data

.. py:class:: UpdateHeurVRL(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRL

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurV`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHV`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHV`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.InstanceNode`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetHeurV]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRL.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRL.pathfind_type

   .. py:method:: _step(pathfind: deepxube.base.pathfinding.PathFindSetHeurV, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRL._step

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRL._step

   .. py:method:: _value_iteration_target(goals: typing.List[deepxube.base.domain.Goal], is_solved_l: typing.List[bool], tcs_l: typing.List[typing.List[float]], states_exp: typing.List[typing.List[deepxube.base.domain.State]]) -> typing.List[float]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRL._value_iteration_target

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRL._value_iteration_target

   .. py:method:: _inputs_ctgs_to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], ctgs_backup: typing.List[float], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRL._inputs_ctgs_to_np

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRL._inputs_ctgs_to_np

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRL._init_replay_buffer

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRL._init_replay_buffer

   .. py:method:: _rb_add(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], is_solved_l: typing.List[bool], times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRL._rb_add

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRL._rb_add

   .. py:method:: _sample_rb_vi_target(num: int, times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[float]]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRL._sample_rb_vi_target

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRL._sample_rb_vi_target

.. py:class:: UpdateHeurVRLKeepGoalABC(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_v_rl.UpdateHeurVRL`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHV`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC.domain_type

   .. py:method:: _step_sync_main(pathfind: deepxube.base.pathfinding.PathFindSetHeurV, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC._step_sync_main

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC._step_sync_main

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.InstanceNode], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC._get_instance_data_norb

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.InstanceNode], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC._get_instance_data_rb

.. py:class:: UpdateHeurVRLHERABC(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERABC

   Bases: :py:obj:`deepxube.updaters.updater_v_rl.UpdateHeurVRL`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.FNsHV`\ ], :py:obj:`deepxube.base.updater.UpdateHER`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHV`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.InstanceNode`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.GoalSampleableFromState]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERABC.domain_type

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.InstanceNode], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERABC._get_instance_data_rb

.. py:class:: UpdateHeurVRLKeepGoal(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurV`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurV]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoal.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoal.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurV
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoal._get_pathfind_functions

.. py:class:: UpdateHeurVRLHER(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLHER

   Bases: :py:obj:`deepxube.updaters.updater_v_rl.UpdateHeurVRLHERABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurV`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurV]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLHER.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLHER.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurV
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLHER._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLHER._get_pathfind_functions

.. py:class:: UpdateHeurVRLKeepGoalPolicy(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalPolicy

   Bases: :py:obj:`deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.InstanceNode`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalPolicy.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalPolicy.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurVPolicy
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLKeepGoalPolicy._get_pathfind_functions

.. py:class:: UpdateHeurVRLHERPolicy(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERPolicy

   Bases: :py:obj:`deepxube.updaters.updater_v_rl.UpdateHeurVRLHERABC`\ [\ :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.InstanceNode`\ ]

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERPolicy.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERPolicy.functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNsHeurVPolicy
      :canonical: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_v_rl.UpdateHeurVRLHERPolicy._get_pathfind_functions
