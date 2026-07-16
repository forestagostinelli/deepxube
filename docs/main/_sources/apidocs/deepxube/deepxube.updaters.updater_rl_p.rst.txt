:py:mod:`deepxube.updaters.updater_rl_p`
========================================

.. py:module:: deepxube.updaters.updater_rl_p

.. autodoc2-docstring:: deepxube.updaters.updater_rl_p
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdatePolicyRL <deepxube.updaters.updater_rl_p.UpdatePolicyRL>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoalABC <deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC>`
     -
   * - :py:obj:`UpdatePolicyRLHERABC <deepxube.updaters.updater_rl_p.UpdatePolicyRLHERABC>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoal <deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoal>`
     -
   * - :py:obj:`UpdatePolicyRLHER <deepxube.updaters.updater_rl_p.UpdatePolicyRLHER>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoalHeurV <deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurV>`
     -
   * - :py:obj:`UpdatePolicyRLHERHeurV <deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurV>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoalHeurQ <deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurQ>`
     -
   * - :py:obj:`UpdatePolicyRLHERHeurQ <deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurQ>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_pathfind_step <deepxube.updaters.updater_rl_p._pathfind_step>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_rl_p._pathfind_step
          :summary:
   * - :py:obj:`_get_edge_popped_data <deepxube.updaters.updater_rl_p._get_edge_popped_data>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_rl_p._get_edge_popped_data
          :summary:

API
~~~

.. py:function:: _pathfind_step(pathfind: deepxube.base.pathfinding.PathFind) -> typing.List[deepxube.base.pathfinding.EdgeQ]
   :canonical: deepxube.updaters.updater_rl_p._pathfind_step

   .. autodoc2-docstring:: deepxube.updaters.updater_rl_p._pathfind_step

.. py:function:: _get_edge_popped_data(edges_popped: typing.List[deepxube.base.pathfinding.EdgeQ], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[deepxube.base.domain.Action]]
   :canonical: deepxube.updaters.updater_rl_p._get_edge_popped_data

   .. autodoc2-docstring:: deepxube.updaters.updater_rl_p._get_edge_popped_data

.. py:class:: UpdatePolicyRL(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRL

   Bases: :py:obj:`deepxube.base.updater.UpdatePolicy`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindActsPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRL.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRL.pathfind_type

   .. py:method:: _step(pathfind: deepxube.base.pathfinding.PathFindActsPolicy, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRL._step

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRL._step

   .. py:method:: _inputs_ctgs_to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRL._inputs_ctgs_to_np

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRL._inputs_ctgs_to_np

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRL._init_replay_buffer

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRL._init_replay_buffer

   .. py:method:: _rb_add(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action], times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRL._rb_add

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRL._rb_add

   .. py:method:: _sample_rb(num: int, times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Goal], typing.List[deepxube.base.domain.Action]]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRL._sample_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRL._sample_rb

.. py:class:: UpdatePolicyRLKeepGoalABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_rl_p.UpdatePolicyRL`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC.domain_type

   .. py:method:: _step_sync_main(pathfind: deepxube.base.pathfinding.PathFindActsPolicy, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC._step_sync_main

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC._step_sync_main

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC._get_instance_data_norb

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC._get_instance_data_rb

.. py:class:: UpdatePolicyRLHERABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERABC

   Bases: :py:obj:`deepxube.updaters.updater_rl_p.UpdatePolicyRL`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`deepxube.base.updater.UpdateHER`\ [\ :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.GoalSampleableFromState]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERABC.domain_type

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERABC._get_instance_data_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERABC._get_instance_data_rb

.. py:class:: UpdatePolicyRLKeepGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoal.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoal.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoal.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoal.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsPolicy
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoal._get_pathfind_functions

.. py:class:: UpdatePolicyRLHER(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHER

   Bases: :py:obj:`deepxube.updaters.updater_rl_p.UpdatePolicyRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHER.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHER.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHER.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHER.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsPolicy
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHER._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHER._get_pathfind_functions

.. py:class:: UpdatePolicyRLKeepGoalHeurV(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurV

   Bases: :py:obj:`deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurV`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurV.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurV.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurV.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurV.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurVPolicy
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurV._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurV._get_pathfind_functions

.. py:class:: UpdatePolicyRLHERHeurV(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurV

   Bases: :py:obj:`deepxube.updaters.updater_rl_p.UpdatePolicyRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurV`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurV.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurV.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurV.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurV.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurVPolicy
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurV._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurV._get_pathfind_functions

.. py:class:: UpdatePolicyRLKeepGoalHeurQ(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurQ

   Bases: :py:obj:`deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurQ`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurQ.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurQ.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurQ.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurQ.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQPolicy
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurQ._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLKeepGoalHeurQ._get_pathfind_functions

.. py:class:: UpdatePolicyRLHERHeurQ(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurQ

   Bases: :py:obj:`deepxube.updaters.updater_rl_p.UpdatePolicyRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurQ`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurQ.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurQ.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurQ.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurQ.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQPolicy
      :canonical: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurQ._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl_p.UpdatePolicyRLHERHeurQ._get_pathfind_functions
