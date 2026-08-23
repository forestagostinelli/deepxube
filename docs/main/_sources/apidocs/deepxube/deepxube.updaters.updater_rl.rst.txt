:py:mod:`deepxube.updaters.updater_rl`
======================================

.. py:module:: deepxube.updaters.updater_rl

.. autodoc2-docstring:: deepxube.updaters.updater_rl
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdateHeurVRL <deepxube.updaters.updater_rl.UpdateHeurVRL>`
     -
   * - :py:obj:`UpdateHeurQRL <deepxube.updaters.updater_rl.UpdateHeurQRL>`
     -
   * - :py:obj:`UpdatePolicyRL <deepxube.updaters.updater_rl.UpdatePolicyRL>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoalABC <deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalABC>`
     -
   * - :py:obj:`UpdateHeurQRLKeepGoalABC <deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalABC>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoalABC <deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalABC>`
     -
   * - :py:obj:`UpdateHeurVRLHERABC <deepxube.updaters.updater_rl.UpdateHeurVRLHERABC>`
     -
   * - :py:obj:`UpdateHeurQRLHERABC <deepxube.updaters.updater_rl.UpdateHeurQRLHERABC>`
     -
   * - :py:obj:`UpdatePolicyRLHERABC <deepxube.updaters.updater_rl.UpdatePolicyRLHERABC>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoal <deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoal>`
     -
   * - :py:obj:`UpdateHeurVRLHER <deepxube.updaters.updater_rl.UpdateHeurVRLHER>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoalPolicy <deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalPolicy>`
     -
   * - :py:obj:`UpdateHeurVRLHERPolicy <deepxube.updaters.updater_rl.UpdateHeurVRLHERPolicy>`
     -
   * - :py:obj:`UpdateHeurQRLKeepGoal <deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoal>`
     -
   * - :py:obj:`UpdateHeurQRLHER <deepxube.updaters.updater_rl.UpdateHeurQRLHER>`
     -
   * - :py:obj:`UpdateHeurQRLKeepGoalPolicy <deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalPolicy>`
     -
   * - :py:obj:`UpdateHeurQRLHERPolicy <deepxube.updaters.updater_rl.UpdateHeurQRLHERPolicy>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoal <deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoal>`
     -
   * - :py:obj:`UpdatePolicyRLHER <deepxube.updaters.updater_rl.UpdatePolicyRLHER>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoalHeurV <deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurV>`
     -
   * - :py:obj:`UpdatePolicyRLHERHeurV <deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurV>`
     -
   * - :py:obj:`UpdatePolicyRLKeepGoalHeurQ <deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurQ>`
     -
   * - :py:obj:`UpdatePolicyRLHERHeurQ <deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurQ>`
     -
   * - :py:obj:`UpdateVRL <deepxube.updaters.updater_rl.UpdateVRL>`
     -
   * - :py:obj:`UpdateVRLHER <deepxube.updaters.updater_rl.UpdateVRLHER>`
     -
   * - :py:obj:`UpdateVPRL <deepxube.updaters.updater_rl.UpdateVPRL>`
     -
   * - :py:obj:`UpdateVPRLHER <deepxube.updaters.updater_rl.UpdateVPRLHER>`
     -
   * - :py:obj:`UpdateQRL <deepxube.updaters.updater_rl.UpdateQRL>`
     -
   * - :py:obj:`UpdateQRLHER <deepxube.updaters.updater_rl.UpdateQRLHER>`
     -
   * - :py:obj:`UpdateQPRL <deepxube.updaters.updater_rl.UpdateQPRL>`
     -
   * - :py:obj:`UpdateQPRLHER <deepxube.updaters.updater_rl.UpdateQPRLHER>`
     -
   * - :py:obj:`UpdatePRL <deepxube.updaters.updater_rl.UpdatePRL>`
     -
   * - :py:obj:`UpdatePRLHER <deepxube.updaters.updater_rl.UpdatePRLHER>`
     -
   * - :py:obj:`UpdatePQRL <deepxube.updaters.updater_rl.UpdatePQRL>`
     -
   * - :py:obj:`UpdatePQRLHER <deepxube.updaters.updater_rl.UpdatePQRLHER>`
     -
   * - :py:obj:`UpdatePVRL <deepxube.updaters.updater_rl.UpdatePVRL>`
     -
   * - :py:obj:`UpdatePVRLHER <deepxube.updaters.updater_rl.UpdatePVRLHER>`
     -

API
~~~

.. py:class:: UpdateHeurVRL(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurVRL

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurVPathFind`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferV`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayV`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetHeurV]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRL.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRL.pathfind_type

   .. py:method:: _get_rb(max_size: int) -> deepxube.utils.replay_buffer_utils.ReplayBufferV
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRL._get_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRL._get_rb

   .. py:method:: _get_rb_data(popped: typing.List[deepxube.base.pathfinding.Node], times: deepxube.utils.timing_utils.Times) -> deepxube.utils.replay_buffer_utils.ReplayV
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRL._get_rb_data

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRL._get_rb_data

   .. py:method:: _get_labels_rb(input_data: deepxube.base.updater.InDataNode, replay_data: deepxube.utils.replay_buffer_utils.ReplayV, times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRL._get_labels_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRL._get_labels_rb

.. py:class:: UpdateHeurQRL(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurQRL

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurQPathFind`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferQ`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayQ`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetHeurQ]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRL.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRL.pathfind_type

   .. py:method:: _get_rb(max_size: int) -> deepxube.utils.replay_buffer_utils.ReplayBufferQ
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRL._get_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRL._get_rb

   .. py:method:: _get_rb_data(popped: typing.List[deepxube.base.pathfinding.EdgeQ], times: deepxube.utils.timing_utils.Times) -> deepxube.utils.replay_buffer_utils.ReplayQ
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRL._get_rb_data

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRL._get_rb_data

   .. py:method:: _get_labels_rb(input_data: deepxube.base.updater.InDataEdge, replay_data: deepxube.utils.replay_buffer_utils.ReplayQ, times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRL._get_labels_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRL._get_labels_rb

.. py:class:: UpdatePolicyRL(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdatePolicyRL

   Bases: :py:obj:`deepxube.base.updater.UpdatePolicyPathFind`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferP`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayP`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRL.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRL.pathfind_type

   .. py:method:: _get_rb(max_size: int) -> deepxube.utils.replay_buffer_utils.ReplayBufferP
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRL._get_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRL._get_rb

   .. py:method:: _get_rb_data(popped: typing.List[deepxube.base.pathfinding.EdgeQ], times: deepxube.utils.timing_utils.Times) -> deepxube.utils.replay_buffer_utils.ReplayP
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRL._get_rb_data

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRL._get_rb_data

   .. py:method:: _get_labels_rb(input_data: deepxube.base.updater.InDataEdge, replay_data: deepxube.utils.replay_buffer_utils.ReplayP, times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRL._get_labels_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRL._get_labels_rb

.. py:class:: UpdateHeurVRLKeepGoalABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurVRL`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`deepxube.base.updater.UpdatePathFindKeepGoal`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.Node`\ , :py:obj:`deepxube.base.updater.InDataNode`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferV`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayV`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalABC.domain_type

   .. py:method:: _get_labels_no_rb(popped: typing.List[deepxube.base.pathfinding.Node], instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalABC._get_labels_no_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalABC._get_labels_no_rb

.. py:class:: UpdateHeurQRLKeepGoalABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurQRL`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`deepxube.base.updater.UpdatePathFindKeepGoal`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.EdgeQ`\ , :py:obj:`deepxube.base.updater.InDataEdge`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferQ`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayQ`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalABC.domain_type

   .. py:method:: _get_labels_no_rb(popped: typing.List[deepxube.base.pathfinding.EdgeQ], instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalABC._get_labels_no_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalABC._get_labels_no_rb

.. py:class:: UpdatePolicyRLKeepGoalABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdatePolicyRL`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`deepxube.base.updater.UpdatePathFindKeepGoal`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ , :py:obj:`deepxube.base.pathfinding.EdgeQ`\ , :py:obj:`deepxube.base.updater.InDataEdge`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferP`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayP`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalABC.domain_type

   .. py:method:: _get_labels_no_rb(popped: typing.List[deepxube.base.pathfinding.EdgeQ], instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalABC._get_labels_no_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalABC._get_labels_no_rb

.. py:class:: UpdateHeurVRLHERABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHERABC

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurVRL`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`deepxube.base.updater.UpdatePathFindHER`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.Node`\ , :py:obj:`deepxube.base.updater.InDataNode`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferV`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayV`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.GoalSampleableFromState]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHERABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLHERABC.domain_type

   .. py:method:: _get_her_data(instances: typing.List[deepxube.base.pathfinding.Instance], goals_inst_her: typing.List[deepxube.base.domain.Goal], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[deepxube.base.updater.InDataNode, deepxube.utils.replay_buffer_utils.ReplayV, int]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHERABC._get_her_data

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLHERABC._get_her_data

.. py:class:: UpdateHeurQRLHERABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHERABC

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurQRL`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`deepxube.base.updater.UpdatePathFindHER`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.EdgeQ`\ , :py:obj:`deepxube.base.updater.InDataEdge`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferQ`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayQ`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.GoalSampleableFromState]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHERABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLHERABC.domain_type

   .. py:method:: _get_her_data(instances: typing.List[deepxube.base.pathfinding.Instance], goals_inst_her: typing.List[deepxube.base.domain.Goal], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[deepxube.base.updater.InDataEdge, deepxube.utils.replay_buffer_utils.ReplayQ, int]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHERABC._get_her_data

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLHERABC._get_her_data

.. py:class:: UpdatePolicyRLHERABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERABC

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdatePolicyRL`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`deepxube.base.updater.UpdatePathFindHER`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ , :py:obj:`deepxube.base.pathfinding.EdgeQ`\ , :py:obj:`deepxube.base.updater.InDataEdge`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferP`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayP`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.GoalSampleableFromState]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHERABC.domain_type

   .. py:method:: _get_her_data(instances: typing.List[deepxube.base.pathfinding.Instance], goals_inst_her: typing.List[deepxube.base.domain.Goal], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[deepxube.base.updater.InDataEdge, deepxube.utils.replay_buffer_utils.ReplayP, int]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERABC._get_her_data

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHERABC._get_her_data

.. py:class:: UpdateHeurVRLKeepGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurV`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurV]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoal.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoal.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurV]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoal.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoal.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurV
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoal._get_pathfind_functions

.. py:class:: UpdateHeurVRLHER(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHER

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurVRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurV`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurV]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHER.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLHER.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurV]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHER.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLHER.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurV
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHER._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLHER._get_pathfind_functions

.. py:class:: UpdateHeurVRLKeepGoalPolicy(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalPolicy

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalPolicy.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalPolicy.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalPolicy.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalPolicy.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurVPolicy
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLKeepGoalPolicy._get_pathfind_functions

.. py:class:: UpdateHeurVRLHERPolicy(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHERPolicy

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurVRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHERPolicy.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLHERPolicy.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHERPolicy.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLHERPolicy.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurVPolicy
      :canonical: deepxube.updaters.updater_rl.UpdateHeurVRLHERPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurVRLHERPolicy._get_pathfind_functions

.. py:class:: UpdateHeurQRLKeepGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQ`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQ`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQ]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoal.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoal.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQ]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoal.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoal.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQ
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoal._get_pathfind_functions

.. py:class:: UpdateHeurQRLHER(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHER

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurQRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQ`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQ`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQ]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHER.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLHER.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQ]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHER.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLHER.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQ
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHER._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLHER._get_pathfind_functions

.. py:class:: UpdateHeurQRLKeepGoalPolicy(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalPolicy

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalPolicy.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalPolicy.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalPolicy.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalPolicy.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQPolicy
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLKeepGoalPolicy._get_pathfind_functions

.. py:class:: UpdateHeurQRLHERPolicy(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHERPolicy

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdateHeurQRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHERPolicy.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLHERPolicy.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHERPolicy.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLHERPolicy.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQPolicy
      :canonical: deepxube.updaters.updater_rl.UpdateHeurQRLHERPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdateHeurQRLHERPolicy._get_pathfind_functions

.. py:class:: UpdatePolicyRLKeepGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoal.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoal.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoal.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoal.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsPolicy
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoal._get_pathfind_functions

.. py:class:: UpdatePolicyRLHER(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHER

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdatePolicyRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHER.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHER.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHER.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHER.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsPolicy
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHER._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHER._get_pathfind_functions

.. py:class:: UpdatePolicyRLKeepGoalHeurV(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurV

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurV`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurV.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurV.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurV.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurV.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurVPolicy
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurV._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurV._get_pathfind_functions

.. py:class:: UpdatePolicyRLHERHeurV(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurV

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdatePolicyRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurV`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurV.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurV.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurV.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurV.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurVPolicy
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurV._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurV._get_pathfind_functions

.. py:class:: UpdatePolicyRLKeepGoalHeurQ(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurQ

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurQ`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurQ.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurQ.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurQ.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurQ.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQPolicy
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurQ._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLKeepGoalHeurQ._get_pathfind_functions

.. py:class:: UpdatePolicyRLHERHeurQ(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurQ

   Bases: :py:obj:`deepxube.updaters.updater_rl.UpdatePolicyRLHERABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurQ`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurQ.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurQ.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQPolicy]
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurQ.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurQ.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurQPolicy
      :canonical: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurQ._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_rl.UpdatePolicyRLHERHeurQ._get_pathfind_functions

.. py:class:: UpdateVRL()
   :canonical: deepxube.updaters.updater_rl.UpdateVRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateVRLHER()
   :canonical: deepxube.updaters.updater_rl.UpdateVRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateVPRL()
   :canonical: deepxube.updaters.updater_rl.UpdateVPRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateVPRLHER()
   :canonical: deepxube.updaters.updater_rl.UpdateVPRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateQRL()
   :canonical: deepxube.updaters.updater_rl.UpdateQRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateQRLHER()
   :canonical: deepxube.updaters.updater_rl.UpdateQRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateQPRL()
   :canonical: deepxube.updaters.updater_rl.UpdateQPRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdateQPRLHER()
   :canonical: deepxube.updaters.updater_rl.UpdateQPRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdatePRL()
   :canonical: deepxube.updaters.updater_rl.UpdatePRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdatePRLHER()
   :canonical: deepxube.updaters.updater_rl.UpdatePRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdatePQRL()
   :canonical: deepxube.updaters.updater_rl.UpdatePQRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdatePQRLHER()
   :canonical: deepxube.updaters.updater_rl.UpdatePQRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdatePVRL()
   :canonical: deepxube.updaters.updater_rl.UpdatePVRL

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`

.. py:class:: UpdatePVRLHER()
   :canonical: deepxube.updaters.updater_rl.UpdatePVRLHER

   Bases: :py:obj:`deepxube.base.updater.UpdateRLParser`
