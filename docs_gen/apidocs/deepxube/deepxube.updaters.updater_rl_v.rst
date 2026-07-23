:py:mod:`deepxube.updaters.updater_rl_v`
========================================

.. py:module:: deepxube.updaters.updater_rl_v

.. autodoc2-docstring:: deepxube.updaters.updater_rl_v
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdateHeurVRL <deepxube.updaters.updater_rl_v.UpdateHeurVRL>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoalABC <deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC>`
     -
   * - :py:obj:`UpdateHeurVRLHERABC <deepxube.updaters.updater_rl_v.UpdateHeurVRLHERABC>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoal <deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoal>`
     -
   * - :py:obj:`UpdateHeurVRLHER <deepxube.updaters.updater_rl_v.UpdateHeurVRLHER>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoalPolicy <deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalPolicy>`
     -
   * - :py:obj:`UpdateHeurVRLHERPolicy <deepxube.updaters.updater_rl_v.UpdateHeurVRLHERPolicy>`
     -
   * - :py:obj:`UpdateVRL <deepxube.updaters.updater_rl_v.UpdateVRL>`
     -
   * - :py:obj:`UpdateVRLHER <deepxube.updaters.updater_rl_v.UpdateVRLHER>`
     -
   * - :py:obj:`UpdateVPRL <deepxube.updaters.updater_rl_v.UpdateVPRL>`
     -
   * - :py:obj:`UpdateVPRLHER <deepxube.updaters.updater_rl_v.UpdateVPRLHER>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_pathfind_v_step <deepxube.updaters.updater_rl_v._pathfind_v_step>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_rl_v._pathfind_v_step
          :summary:
   * - :py:obj:`_get_nodes_popped_data <deepxube.updaters.updater_rl_v._get_nodes_popped_data>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_rl_v._get_nodes_popped_data
          :summary:

API
~~~

.. py:function:: _pathfind_v_step(pathfind: deepxube.base.pathfinding.PathFindSetHeurV) -> typing.List[deepxube.base.pathfinding.Node]
   :canonical: deepxube.updaters.updater_rl_v._pathfind_v_step

   .. autodoc2-docstring:: deepxube.updaters.updater_rl_v._pathfind_v_step

.. py:function:: _get_nodes_popped_data(nodes_popped: typing.List[deepxube.base.pathfinding.Node], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[bool]]
   :canonical: deepxube.updaters.updater_rl_v._get_nodes_popped_data

   .. autodoc2-docstring:: deepxube.updaters.updater_rl_v._get_nodes_popped_data

.. py:class:: UpdateHeurVRL(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRL

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurV`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetHeurV]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRL.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRL.pathfind_type

   .. py:method:: _step(pathfind: deepxube.base.pathfinding.PathFindSetHeurV, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRL._step

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRL._step

   .. py:method:: _value_iteration_target(goals: typing.List[deepxube.base.domain.Goal], is_solved_l: typing.List[bool], tcs_l: typing.List[typing.List[float]], states_exp: typing.List[typing.List[deepxube.base.domain.State]]) -> typing.List[float]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRL._value_iteration_target

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRL._value_iteration_target

   .. py:method:: _inputs_ctgs_to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], ctgs_backup: typing.List[float], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRL._inputs_ctgs_to_np

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRL._inputs_ctgs_to_np

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRL._init_replay_buffer

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRL._init_replay_buffer

   .. py:method:: _rb_add(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], is_solved_l: typing.List[bool], times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRL._rb_add

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRL._rb_add

   .. py:method:: _sample_rb_vi_target(num: int, times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[float]]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRL._sample_rb_vi_target

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRL._sample_rb_vi_target

.. py:class:: UpdateHeurVRLKeepGoalABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_rl_v.UpdateHeurVRL`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC.domain_type

   .. py:method:: _step_sync_main(pathfind: deepxube.base.pathfinding.PathFindSetHeurV, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC._step_sync_main

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC._step_sync_main

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC._get_instance_data_norb

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC._get_instance_data_rb

.. py:class:: UpdateHeurVRLHERABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERABC

   Bases: :py:obj:`deepxube.updaters.updater_rl_v.UpdateHeurVRL`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`deepxube.base.updater.UpdateHER`\ [\ :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.GoalSampleableFromState]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERABC.domain_type

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERABC._get_instance_data_rb

.. py:class:: UpdateHeurVRLKeepGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurV`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurV]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoal.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoal.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurV]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoal.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoal.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurV
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoal._get_pathfind_functions

.. py:class:: UpdateHeurVRLHER(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHER

   Bases: :py:obj:`deepxube.updaters.updater_rl_v.UpdateHeurVRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurV`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurV]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHER.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLHER.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurV]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHER.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLHER.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurV
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHER._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLHER._get_pathfind_functions

.. py:class:: UpdateHeurVRLKeepGoalPolicy(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalPolicy

   Bases: :py:obj:`deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalPolicy.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalPolicy.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalPolicy.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalPolicy.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurVPolicy
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLKeepGoalPolicy._get_pathfind_functions

.. py:class:: UpdateHeurVRLHERPolicy(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERPolicy

   Bases: :py:obj:`deepxube.updaters.updater_rl_v.UpdateHeurVRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERPolicy.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERPolicy.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERPolicy.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERPolicy.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurVPolicy
      :canonical: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_v.UpdateHeurVRLHERPolicy._get_pathfind_functions

.. py:class:: UpdateVRL()
   :canonical: deepxube.updaters.updater_rl_v.UpdateVRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateVRLHER()
   :canonical: deepxube.updaters.updater_rl_v.UpdateVRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateVPRL()
   :canonical: deepxube.updaters.updater_rl_v.UpdateVPRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateVPRLHER()
   :canonical: deepxube.updaters.updater_rl_v.UpdateVPRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`
