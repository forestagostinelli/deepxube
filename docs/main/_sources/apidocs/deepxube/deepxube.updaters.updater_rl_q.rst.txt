:py:mod:`deepxube.updaters.updater_rl_q`
========================================

.. py:module:: deepxube.updaters.updater_rl_q

.. autodoc2-docstring:: deepxube.updaters.updater_rl_q
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdateHeurQRL <deepxube.updaters.updater_rl_q.UpdateHeurQRL>`
     -
   * - :py:obj:`UpdateHeurQRLKeepGoalABC <deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC>`
     -
   * - :py:obj:`UpdateHeurQRLHERABC <deepxube.updaters.updater_rl_q.UpdateHeurQRLHERABC>`
     -
   * - :py:obj:`UpdateHeurQRLKeepGoal <deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoal>`
     -
   * - :py:obj:`UpdateHeurQRLHER <deepxube.updaters.updater_rl_q.UpdateHeurQRLHER>`
     -
   * - :py:obj:`UpdateHeurQRLKeepGoalPolicy <deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalPolicy>`
     -
   * - :py:obj:`UpdateHeurQRLHERPolicy <deepxube.updaters.updater_rl_q.UpdateHeurQRLHERPolicy>`
     -
   * - :py:obj:`UpdateVRL <deepxube.updaters.updater_rl_q.UpdateVRL>`
     -
   * - :py:obj:`UpdateVRLHER <deepxube.updaters.updater_rl_q.UpdateVRLHER>`
     -
   * - :py:obj:`UpdateVPRL <deepxube.updaters.updater_rl_q.UpdateVPRL>`
     -
   * - :py:obj:`UpdateVPRLHER <deepxube.updaters.updater_rl_q.UpdateVPRLHER>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_pathfind_q_step <deepxube.updaters.updater_rl_q._pathfind_q_step>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_rl_q._pathfind_q_step
          :summary:
   * - :py:obj:`_get_edge_popped_data <deepxube.updaters.updater_rl_q._get_edge_popped_data>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_rl_q._get_edge_popped_data
          :summary:

API
~~~

.. py:function:: _pathfind_q_step(pathfind: deepxube.base.pathfinding.PathFindSetHeurQ) -> typing.List[deepxube.base.pathfinding.EdgeQ]
   :canonical: deepxube.updaters.updater_rl_q._pathfind_q_step

   .. autodoc2-docstring:: deepxube.updaters.updater_rl_q._pathfind_q_step

.. py:function:: _get_edge_popped_data(edges_popped: typing.List[deepxube.base.pathfinding.EdgeQ], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[bool], typing.List[deepxube.base.domain.Action], typing.List[float], typing.List[deepxube.base.domain.State]]
   :canonical: deepxube.updaters.updater_rl_q._get_edge_popped_data

   .. autodoc2-docstring:: deepxube.updaters.updater_rl_q._get_edge_popped_data

.. py:class:: UpdateHeurQRL(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRL

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurQ`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetHeurQ]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRL.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRL.pathfind_type

   .. py:method:: _step(pathfind: deepxube.base.pathfinding.PathFindSetHeurQ, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRL._step

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRL._step

   .. py:method:: _q_learning_target(goals: typing.List[deepxube.base.domain.Goal], is_solved_l: typing.List[bool], tcs: typing.List[float], states_next: typing.List[deepxube.base.domain.State]) -> typing.List[float]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRL._q_learning_target

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRL._q_learning_target

   .. py:method:: _inputs_ctgs_to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action], ctgs_backup: typing.List[float], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRL._inputs_ctgs_to_np

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRL._inputs_ctgs_to_np

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRL._init_replay_buffer

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRL._init_replay_buffer

   .. py:method:: _rb_add(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], is_solved_l: typing.List[bool], actions: typing.List[deepxube.base.domain.Action], tcs: typing.List[float], states_next: typing.List[deepxube.base.domain.State], times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRL._rb_add

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRL._rb_add

   .. py:method:: _sample_rb_qlearn_target(num: int, times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[deepxube.base.domain.Action], typing.List[float]]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRL._sample_rb_qlearn_target

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRL._sample_rb_qlearn_target

.. py:class:: UpdateHeurQRLKeepGoalABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_rl_q.UpdateHeurQRL`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC.domain_type

   .. py:method:: _step_sync_main(pathfind: deepxube.base.pathfinding.PathFindSetHeurQ, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC._step_sync_main

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC._step_sync_main

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC._get_instance_data_norb

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC._get_instance_data_rb

.. py:class:: UpdateHeurQRLHERABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERABC

   Bases: :py:obj:`deepxube.updaters.updater_rl_q.UpdateHeurQRL`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`deepxube.base.updater.UpdateHER`\ [\ :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.GoalSampleableFromState]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERABC.domain_type

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERABC._get_instance_data_rb

.. py:class:: UpdateHeurQRLKeepGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQ`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQ`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQ]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoal.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoal.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQ]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoal.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoal.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQ
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoal._get_pathfind_functions

.. py:class:: UpdateHeurQRLHER(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHER

   Bases: :py:obj:`deepxube.updaters.updater_rl_q.UpdateHeurQRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQ`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQ`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQ]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHER.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLHER.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQ]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHER.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLHER.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQ
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHER._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLHER._get_pathfind_functions

.. py:class:: UpdateHeurQRLKeepGoalPolicy(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalPolicy

   Bases: :py:obj:`deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalPolicy.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalPolicy.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalPolicy.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalPolicy.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQPolicy
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLKeepGoalPolicy._get_pathfind_functions

.. py:class:: UpdateHeurQRLHERPolicy(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERPolicy

   Bases: :py:obj:`deepxube.updaters.updater_rl_q.UpdateHeurQRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERPolicy.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERPolicy.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERPolicy.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERPolicy.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQPolicy
      :canonical: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_q.UpdateHeurQRLHERPolicy._get_pathfind_functions

.. py:class:: UpdateVRL()
   :canonical: deepxube.updaters.updater_rl_q.UpdateVRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateVRLHER()
   :canonical: deepxube.updaters.updater_rl_q.UpdateVRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateVPRL()
   :canonical: deepxube.updaters.updater_rl_q.UpdateVPRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateVPRLHER()
   :canonical: deepxube.updaters.updater_rl_q.UpdateVPRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`
