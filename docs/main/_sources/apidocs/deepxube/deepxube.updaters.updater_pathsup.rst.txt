:py:mod:`deepxube.updaters.updater_pathsup`
===========================================

.. py:module:: deepxube.updaters.updater_pathsup

.. autodoc2-docstring:: deepxube.updaters.updater_pathsup
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdateHeurVPathSup <deepxube.updaters.updater_pathsup.UpdateHeurVPathSup>`
     -
   * - :py:obj:`UpdateHeurVPathSupKeepGoalABC <deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoalABC>`
     -
   * - :py:obj:`UpdateHeurVPathSupKeepGoal <deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoal>`
     -
   * - :py:obj:`UpdateHeurVRLKeepGoalPolicy <deepxube.updaters.updater_pathsup.UpdateHeurVRLKeepGoalPolicy>`
     -
   * - :py:obj:`UpdateVPathSupParser <deepxube.updaters.updater_pathsup.UpdateVPathSupParser>`
     -
   * - :py:obj:`UpdateVPPathSupParser <deepxube.updaters.updater_pathsup.UpdateVPPathSupParser>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`D_NL_T <deepxube.updaters.updater_pathsup.D_NL_T>`
     - .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.D_NL_T
          :summary:

API
~~~

.. py:data:: D_NL_T
   :canonical: deepxube.updaters.updater_pathsup.D_NL_T
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.D_NL_T

.. py:class:: UpdateHeurVPathSup(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSup

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurVPathFind`\ [\ :py:obj:`deepxube.updaters.updater_pathsup.D_NL_T`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferVLab`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayVLab`\ ], :py:obj:`deepxube.base.updater.UpdateRL`\ [\ :py:obj:`deepxube.updaters.updater_pathsup.D_NL_T`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.pathfinding.PathFindSetHeurV]
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSup.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVPathSup.pathfind_type

   .. py:method:: _get_rb(max_size: int) -> deepxube.utils.replay_buffer_utils.ReplayBufferVLab
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSup._get_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVPathSup._get_rb

   .. py:method:: _get_rb_data(popped: typing.List[deepxube.base.pathfinding.Node], times: deepxube.utils.timing_utils.Times) -> deepxube.utils.replay_buffer_utils.ReplayVLab
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSup._get_rb_data

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVPathSup._get_rb_data

   .. py:method:: _get_labels_rb(input_data: deepxube.base.updater.InDataNode, replay_data: deepxube.utils.replay_buffer_utils.ReplayVLab, times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSup._get_labels_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVPathSup._get_labels_rb

.. py:class:: UpdateHeurVPathSupKeepGoalABC(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoalABC

   Bases: :py:obj:`deepxube.updaters.updater_pathsup.UpdateHeurVPathSup`\ [\ :py:obj:`deepxube.base.domain.NodesLabelable`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`deepxube.base.updater.UpdatePathFindKeepGoal`\ [\ :py:obj:`deepxube.base.domain.NodesLabelable`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ , :py:obj:`deepxube.base.pathfinding.Node`\ , :py:obj:`deepxube.base.updater.InDataNode`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayBufferVLab`\ , :py:obj:`deepxube.utils.replay_buffer_utils.ReplayVLab`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.NodesLabelable]
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoalABC.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoalABC.domain_type

   .. py:method:: _get_labels_no_rb(popped: typing.List[deepxube.base.pathfinding.Node], instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[float]
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoalABC._get_labels_no_rb

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoalABC._get_labels_no_rb

.. py:class:: UpdateHeurVPathSupKeepGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoal

   Bases: :py:obj:`deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurV`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurV]
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoal.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoal.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurV]
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoal.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoal.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurV
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoal._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoal._get_pathfind_functions

.. py:class:: UpdateHeurVRLKeepGoalPolicy(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVRLKeepGoalPolicy

   Bases: :py:obj:`deepxube.updaters.updater_pathsup.UpdateHeurVPathSupKeepGoalABC`\ [\ :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`\ , :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`\ ]

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVRLKeepGoalPolicy.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVRLKeepGoalPolicy.pathfind_functions_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurVPolicy]
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVRLKeepGoalPolicy.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVRLKeepGoalPolicy.updater_functions_type

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNsHeurVPolicy
      :canonical: deepxube.updaters.updater_pathsup.UpdateHeurVRLKeepGoalPolicy._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.updaters.updater_pathsup.UpdateHeurVRLKeepGoalPolicy._get_pathfind_functions

.. py:class:: UpdateVPathSupParser()
   :canonical: deepxube.updaters.updater_pathsup.UpdateVPathSupParser

   Bases: :py:obj:`deepxube.base.updater.UpdateParser`

.. py:class:: UpdateVPPathSupParser()
   :canonical: deepxube.updaters.updater_pathsup.UpdateVPPathSupParser

   Bases: :py:obj:`deepxube.base.updater.UpdateParser`
