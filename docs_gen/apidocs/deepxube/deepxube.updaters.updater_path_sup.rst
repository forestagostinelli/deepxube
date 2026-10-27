:py:mod:`deepxube.updaters.updater_path_sup`
============================================

.. py:module:: deepxube.updaters.updater_path_sup

.. autodoc2-docstring:: deepxube.updaters.updater_path_sup
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdateHeurVPathSup <deepxube.updaters.updater_path_sup.UpdateHeurVPathSup>`
     -
   * - :py:obj:`UpdateHeurQPathSup <deepxube.updaters.updater_path_sup.UpdateHeurQPathSup>`
     -
   * - :py:obj:`UpdateHeurVPathSupKeepGoalABC <deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoalABC>`
     -
   * - :py:obj:`UpdateHeurQPathSupKeepGoalABC <deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoalABC>`
     -
   * - :py:obj:`UpdateHeurVPathSupKeepGoal <deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoal>`
     -
   * - :py:obj:`UpdateHeurQPathSupKeepGoal <deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoal>`
     -
   * - :py:obj:`UpdateVPathSupParser <deepxube.updaters.updater_path_sup.UpdateVPathSupParser>`
     -
   * - :py:obj:`UpdateQPathSupParser <deepxube.updaters.updater_path_sup.UpdateQPathSupParser>`
     -

API
~~~

.. py:class:: UpdateHeurVPathSup(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSup

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurVPathFind`\ [\ :py:obj:`deepxube.base.domain.NodesLabelable`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferVLab`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayVLab`\ ], :py:obj:`deepxube.base.updater.UpdateLabel`\ [\ :py:obj:`deepxube.base.domain.NodesLabelable`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetHeurV]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSup.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurVPathSup.pathfind_type

   .. py:method:: _get_rb(max_size: int) -> deepxube.utils.replay_buffer_utils.ReplayBufferVLab
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSup._get_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurVPathSup._get_rb

   .. py:method:: _get_rb_data(popped: typing.List[deepxube.base.pathfinding.Node], times: deepxube.utils.timing_utils.Times) -> deepxube.utils.replay_buffer_utils.ReplayVLab
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSup._get_rb_data

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurVPathSup._get_rb_data

   .. py:method:: _get_labels_rb(input_data: deepxube.base.updater.InDataNode, replay_data: deepxube.utils.replay_buffer_utils.ReplayVLab, times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSup._get_labels_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurVPathSup._get_labels_rb

.. py:class:: UpdateHeurQPathSup(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSup

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurQPathFind`\ [\ :py:obj:`deepxube.base.domain.EdgesLabelable`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferQLab`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayQLab`\ ], :py:obj:`deepxube.base.updater.UpdateLabel`\ [\ :py:obj:`deepxube.base.domain.EdgesLabelable`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetHeurQ]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSup.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurQPathSup.pathfind_type

   .. py:method:: _get_rb(max_size: int) -> deepxube.utils.replay_buffer_utils.ReplayBufferQLab
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSup._get_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurQPathSup._get_rb

   .. py:method:: _get_rb_data(popped: typing.List[deepxube.base.pathfinding.EdgeQ], times: deepxube.utils.timing_utils.Times) -> deepxube.utils.replay_buffer_utils.ReplayQLab
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSup._get_rb_data

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurQPathSup._get_rb_data

   .. py:method:: _get_labels_rb(input_data: deepxube.base.updater.InDataEdge, replay_data: deepxube.utils.replay_buffer_utils.ReplayQLab, times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSup._get_labels_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurQPathSup._get_labels_rb

.. py:class:: UpdateHeurVPathSupKeepGoalABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_path_sup.UpdateHeurVPathSup`\ [\ :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`deepxube.base.updater.UpdatePathFindKeepGoal`\ [\ :py:obj:`deepxube.base.domain.NodesLabelable`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.Node`\ , :py:obj:`deepxube.base.updater.InDataNode`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferVLab`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayVLab`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.NodesLabelable]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoalABC.domain_type

   .. py:method:: _get_labels_no_rb(popped: typing.List[deepxube.base.pathfinding.Node], instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoalABC._get_labels_no_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoalABC._get_labels_no_rb

.. py:class:: UpdateHeurQPathSupKeepGoalABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_path_sup.UpdateHeurQPathSup`\ [\ :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`deepxube.base.updater.UpdatePathFindKeepGoal`\ [\ :py:obj:`deepxube.base.domain.EdgesLabelable`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ , :py:obj:`deepxube.base.pathfinding.EdgeQ`\ , :py:obj:`deepxube.base.updater.InDataEdge`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferQLab`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayQLab`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.EdgesLabelable]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoalABC.domain_type

   .. py:method:: _get_labels_no_rb(popped: typing.List[deepxube.base.pathfinding.EdgeQ], instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoalABC._get_labels_no_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoalABC._get_labels_no_rb

.. py:class:: UpdateHeurVPathSupKeepGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`\ ]

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurV]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoal.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurVPathSupKeepGoal.updater_functions_type

.. py:class:: UpdateHeurQPathSupKeepGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQ`\ ]

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQ]
      :canonical: deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoal.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_path_sup.UpdateHeurQPathSupKeepGoal.updater_functions_type

.. py:class:: UpdateVPathSupParser()
   :canonical: deepxube.updaters.updater_path_sup.UpdateVPathSupParser

   Bases: :py:obj:`deepxube.base.updater.UpdateParser`

.. py:class:: UpdateQPathSupParser()
   :canonical: deepxube.updaters.updater_path_sup.UpdateQPathSupParser

   Bases: :py:obj:`deepxube.base.updater.UpdateParser`
