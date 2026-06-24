:py:mod:`deepxube.domains.lightsout`
====================================

.. py:module:: deepxube.domains.lightsout

.. autodoc2-docstring:: deepxube.domains.lightsout
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOState <deepxube.domains.lightsout.LOState>`
     -
   * - :py:obj:`LOGoal <deepxube.domains.lightsout.LOGoal>`
     -
   * - :py:obj:`LOAction <deepxube.domains.lightsout.LOAction>`
     -
   * - :py:obj:`LightsOut <deepxube.domains.lightsout.LightsOut>`
     -
   * - :py:obj:`LightsOutParser <deepxube.domains.lightsout.LightsOutParser>`
     -

API
~~~

.. py:class:: LOState(tiles: numpy.typing.NDArray[numpy.uint8])
   :canonical: deepxube.domains.lightsout.LOState

   Bases: :py:obj:`deepxube.base.domain.State`

   .. py:attribute:: __slots__
      :canonical: deepxube.domains.lightsout.LOState.__slots__
      :value: ['tiles', 'hash']

      .. autodoc2-docstring:: deepxube.domains.lightsout.LOState.__slots__

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.lightsout.LOState.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.lightsout.LOState.__eq__

.. py:class:: LOGoal(tiles: numpy.typing.NDArray[numpy.uint8])
   :canonical: deepxube.domains.lightsout.LOGoal

   Bases: :py:obj:`deepxube.base.domain.Goal`

.. py:class:: LOAction(action: int, dim: int)
   :canonical: deepxube.domains.lightsout.LOAction

   Bases: :py:obj:`deepxube.base.domain.Action`

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.lightsout.LOAction.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.lightsout.LOAction.__eq__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.domains.lightsout.LOAction.__repr__

.. py:class:: LightsOut(dim: int = 7)
   :canonical: deepxube.domains.lightsout.LightsOut

   Bases: :py:obj:`deepxube.base.domain.NextStateNPActsEnumFixed`\ [\ :py:obj:`deepxube.domains.lightsout.LOState`\ , :py:obj:`deepxube.domains.lightsout.LOAction`\ , :py:obj:`deepxube.domains.lightsout.LOGoal`\ ], :py:obj:`deepxube.base.domain.GoalStartRevWalkableActsRev`\ [\ :py:obj:`deepxube.domains.lightsout.LOState`\ , :py:obj:`deepxube.domains.lightsout.LOAction`\ , :py:obj:`deepxube.domains.lightsout.LOGoal`\ ], :py:obj:`deepxube.base.domain.StateGoalVizable`\ [\ :py:obj:`deepxube.domains.lightsout.LOState`\ , :py:obj:`deepxube.domains.lightsout.LOAction`\ , :py:obj:`deepxube.domains.lightsout.LOGoal`\ ], :py:obj:`deepxube.base.domain.StringToAct`\ [\ :py:obj:`deepxube.domains.lightsout.LOState`\ , :py:obj:`deepxube.domains.lightsout.LOAction`\ , :py:obj:`deepxube.domains.lightsout.LOGoal`\ ], :py:obj:`deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn`\ [\ :py:obj:`deepxube.domains.lightsout.LOState`\ , :py:obj:`deepxube.domains.lightsout.LOAction`\ , :py:obj:`deepxube.domains.lightsout.LOGoal`\ ], :py:obj:`deepxube.base.nnet_input.HasFlatSGAIn`\ [\ :py:obj:`deepxube.domains.lightsout.LOState`\ , :py:obj:`deepxube.domains.lightsout.LOAction`\ , :py:obj:`deepxube.domains.lightsout.LOGoal`\ ], :py:obj:`deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn`\ [\ :py:obj:`deepxube.domains.lightsout.LOState`\ , :py:obj:`deepxube.domains.lightsout.LOAction`\ , :py:obj:`deepxube.domains.lightsout.LOGoal`\ ]

   .. py:method:: is_solved(states: typing.List[deepxube.domains.lightsout.LOState], goals: typing.List[deepxube.domains.lightsout.LOGoal]) -> typing.List[bool]
      :canonical: deepxube.domains.lightsout.LightsOut.is_solved

   .. py:method:: sample_goalstate_goal_pairs(num: int) -> typing.Tuple[typing.List[deepxube.domains.lightsout.LOState], typing.List[deepxube.domains.lightsout.LOGoal]]
      :canonical: deepxube.domains.lightsout.LightsOut.sample_goalstate_goal_pairs

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.sample_goalstate_goal_pairs

   .. py:method:: sample_rev_state(states: typing.List[deepxube.domains.lightsout.LOState]) -> typing.Tuple[typing.List[deepxube.domains.lightsout.LOState], typing.List[deepxube.domains.lightsout.LOAction], typing.List[float]]
      :canonical: deepxube.domains.lightsout.LightsOut.sample_rev_state

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.sample_rev_state

   .. py:method:: get_input_info_flat_sg() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.domains.lightsout.LightsOut.get_input_info_flat_sg

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.get_input_info_flat_sg

   .. py:method:: get_input_info_flat_sga() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.domains.lightsout.LightsOut.get_input_info_flat_sga

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.get_input_info_flat_sga

   .. py:method:: to_np_flat_sg(states: typing.List[deepxube.domains.lightsout.LOState], goals: typing.List[deepxube.domains.lightsout.LOGoal]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.lightsout.LightsOut.to_np_flat_sg

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.to_np_flat_sg

   .. py:method:: to_np_flat_sga(states: typing.List[deepxube.domains.lightsout.LOState], goals: typing.List[deepxube.domains.lightsout.LOGoal], actions: typing.List[deepxube.domains.lightsout.LOAction]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.lightsout.LightsOut.to_np_flat_sga

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.to_np_flat_sga

   .. py:method:: get_input_info_2d_sg() -> typing.Tuple[typing.List[int], typing.Tuple[int, int], typing.List[int], typing.Optional[int]]
      :canonical: deepxube.domains.lightsout.LightsOut.get_input_info_2d_sg

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.get_input_info_2d_sg

   .. py:method:: to_np_2d_sg(states: typing.List[deepxube.domains.lightsout.LOState], goals: typing.List[deepxube.domains.lightsout.LOGoal]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.lightsout.LightsOut.to_np_2d_sg

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.to_np_2d_sg

   .. py:method:: actions_to_indices(actions: typing.List[deepxube.domains.lightsout.LOAction]) -> typing.List[int]
      :canonical: deepxube.domains.lightsout.LightsOut.actions_to_indices

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.actions_to_indices

   .. py:method:: get_actions_fixed() -> typing.List[deepxube.domains.lightsout.LOAction]
      :canonical: deepxube.domains.lightsout.LightsOut.get_actions_fixed

   .. py:method:: visualize_state_goal(state: deepxube.domains.lightsout.LOState, goal: deepxube.domains.lightsout.LOGoal, fig: matplotlib.figure.Figure) -> None
      :canonical: deepxube.domains.lightsout.LightsOut.visualize_state_goal

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.visualize_state_goal

   .. py:method:: string_to_action_help() -> str
      :canonical: deepxube.domains.lightsout.LightsOut.string_to_action_help

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.string_to_action_help

   .. py:method:: string_to_action(act_str: str) -> typing.Optional[deepxube.domains.lightsout.LOAction]
      :canonical: deepxube.domains.lightsout.LightsOut.string_to_action

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut.string_to_action

   .. py:method:: _make_ax(grid: numpy.typing.NDArray, ax: matplotlib.axes.Axes) -> None
      :canonical: deepxube.domains.lightsout.LightsOut._make_ax

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut._make_ax

   .. py:method:: _states_to_np(states: typing.List[deepxube.domains.lightsout.LOState]) -> typing.List[numpy.typing.NDArray[numpy.uint8]]
      :canonical: deepxube.domains.lightsout.LightsOut._states_to_np

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut._states_to_np

   .. py:method:: _np_to_states(states_np: typing.List[numpy.typing.NDArray]) -> typing.List[deepxube.domains.lightsout.LOState]
      :canonical: deepxube.domains.lightsout.LightsOut._np_to_states

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOut._np_to_states

   .. py:method:: _next_state_np(states_np_l: typing.List[numpy.typing.NDArray], actions: typing.List[deepxube.domains.lightsout.LOAction]) -> typing.Tuple[typing.List[numpy.typing.NDArray], typing.List[float]]
      :canonical: deepxube.domains.lightsout.LightsOut._next_state_np

   .. py:method:: __repr__() -> str
      :canonical: deepxube.domains.lightsout.LightsOut.__repr__

.. py:class:: LightsOutParser
   :canonical: deepxube.domains.lightsout.LightsOutParser

   Bases: :py:obj:`deepxube.base.factory.Parser`

   .. py:method:: parse(args_str: str) -> typing.Dict[str, typing.Any]
      :canonical: deepxube.domains.lightsout.LightsOutParser.parse

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOutParser.parse

   .. py:method:: help() -> str
      :canonical: deepxube.domains.lightsout.LightsOutParser.help

      .. autodoc2-docstring:: deepxube.domains.lightsout.LightsOutParser.help
