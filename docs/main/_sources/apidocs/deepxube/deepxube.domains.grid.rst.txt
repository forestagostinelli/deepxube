:py:mod:`deepxube.domains.grid`
===============================

.. py:module:: deepxube.domains.grid

.. autodoc2-docstring:: deepxube.domains.grid
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`GridState <deepxube.domains.grid.GridState>`
     -
   * - :py:obj:`GridGoal <deepxube.domains.grid.GridGoal>`
     -
   * - :py:obj:`GridAction <deepxube.domains.grid.GridAction>`
     -
   * - :py:obj:`Grid <deepxube.domains.grid.Grid>`
     -
   * - :py:obj:`GridParser <deepxube.domains.grid.GridParser>`
     -
   * - :py:obj:`GridFlatIn <deepxube.domains.grid.GridFlatIn>`
     -
   * - :py:obj:`GridFlatInQFix <deepxube.domains.grid.GridFlatInQFix>`
     -
   * - :py:obj:`GridFlatInActIn <deepxube.domains.grid.GridFlatInActIn>`
     -
   * - :py:obj:`GridNNetInput <deepxube.domains.grid.GridNNetInput>`
     -
   * - :py:obj:`GridNet <deepxube.domains.grid.GridNet>`
     -
   * - :py:obj:`GridNetParser <deepxube.domains.grid.GridNetParser>`
     -

API
~~~

.. py:class:: GridState(robot_x: int, robot_y: int)
   :canonical: deepxube.domains.grid.GridState

   Bases: :py:obj:`deepxube.base.domain.State`

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.grid.GridState.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.grid.GridState.__eq__

.. py:class:: GridGoal(robot_x: int, robot_y: int)
   :canonical: deepxube.domains.grid.GridGoal

   Bases: :py:obj:`deepxube.base.domain.Goal`

.. py:class:: GridAction(action: int)
   :canonical: deepxube.domains.grid.GridAction

   Bases: :py:obj:`deepxube.base.domain.Action`

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.grid.GridAction.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.grid.GridAction.__eq__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.domains.grid.GridAction.__repr__

.. py:class:: Grid(dim: int = 7)
   :canonical: deepxube.domains.grid.Grid

   Bases: :py:obj:`deepxube.base.domain.ActsEnumFixed`\ [\ :py:obj:`deepxube.domains.grid.GridState`\ , :py:obj:`deepxube.domains.grid.GridAction`\ , :py:obj:`deepxube.domains.grid.GridGoal`\ ], :py:obj:`deepxube.base.domain.StartGoalWalkable`\ [\ :py:obj:`deepxube.domains.grid.GridState`\ , :py:obj:`deepxube.domains.grid.GridAction`\ , :py:obj:`deepxube.domains.grid.GridGoal`\ ], :py:obj:`deepxube.base.domain.StateGoalVizable`\ [\ :py:obj:`deepxube.domains.grid.GridState`\ , :py:obj:`deepxube.domains.grid.GridAction`\ , :py:obj:`deepxube.domains.grid.GridGoal`\ ], :py:obj:`deepxube.base.domain.StringToAct`\ [\ :py:obj:`deepxube.domains.grid.GridState`\ , :py:obj:`deepxube.domains.grid.GridAction`\ , :py:obj:`deepxube.domains.grid.GridGoal`\ ]

   .. py:method:: is_solved(states: typing.List[deepxube.domains.grid.GridState], goals: typing.List[deepxube.domains.grid.GridGoal]) -> typing.List[bool]
      :canonical: deepxube.domains.grid.Grid.is_solved

   .. py:method:: sample_start_states(num_states: int) -> typing.List[deepxube.domains.grid.GridState]
      :canonical: deepxube.domains.grid.Grid.sample_start_states

   .. py:method:: next_state(states: typing.List[deepxube.domains.grid.GridState], actions: typing.List[deepxube.domains.grid.GridAction]) -> typing.Tuple[typing.List[deepxube.domains.grid.GridState], typing.List[float]]
      :canonical: deepxube.domains.grid.Grid.next_state

   .. py:method:: sample_goal_from_state(states_start: typing.Optional[typing.List[deepxube.domains.grid.GridState]], states_goal: typing.List[deepxube.domains.grid.GridState]) -> typing.List[deepxube.domains.grid.GridGoal]
      :canonical: deepxube.domains.grid.Grid.sample_goal_from_state

   .. py:method:: visualize_state_goal(state: deepxube.domains.grid.GridState, goal: deepxube.domains.grid.GridGoal, fig: matplotlib.figure.Figure) -> None
      :canonical: deepxube.domains.grid.Grid.visualize_state_goal

   .. py:method:: string_to_action(act_str: str) -> typing.Optional[deepxube.domains.grid.GridAction]
      :canonical: deepxube.domains.grid.Grid.string_to_action

      .. autodoc2-docstring:: deepxube.domains.grid.Grid.string_to_action

   .. py:method:: string_to_action_help() -> str
      :canonical: deepxube.domains.grid.Grid.string_to_action_help

      .. autodoc2-docstring:: deepxube.domains.grid.Grid.string_to_action_help

   .. py:method:: get_actions_fixed() -> typing.List[deepxube.domains.grid.GridAction]
      :canonical: deepxube.domains.grid.Grid.get_actions_fixed

   .. py:method:: __repr__() -> str
      :canonical: deepxube.domains.grid.Grid.__repr__

.. py:class:: GridParser()
   :canonical: deepxube.domains.grid.GridParser

   Bases: :py:obj:`deepxube.base.factory.DelimParser`

   .. py:property:: delim
      :canonical: deepxube.domains.grid.GridParser.delim
      :type: str

      .. autodoc2-docstring:: deepxube.domains.grid.GridParser.delim

.. py:class:: GridFlatIn(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.domains.grid.GridFlatIn

   Bases: :py:obj:`deepxube.base.nnet_input.StateGoalIn`\ [\ :py:obj:`deepxube.domains.grid.Grid`\ , :py:obj:`deepxube.domains.grid.GridState`\ , :py:obj:`deepxube.domains.grid.GridGoal`\ ], :py:obj:`deepxube.base.nnet_input.FlatIn`\ [\ :py:obj:`deepxube.domains.grid.Grid`\ ]

   .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.domains.grid.GridFlatIn.get_input_info

   .. py:method:: to_np(states: typing.List[deepxube.domains.grid.GridState], goals: typing.List[deepxube.domains.grid.GridGoal]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.grid.GridFlatIn.to_np

.. py:class:: GridFlatInQFix(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.domains.grid.GridFlatInQFix

   Bases: :py:obj:`deepxube.base.nnet_input.StateGoalActFixIn`\ [\ :py:obj:`deepxube.domains.grid.Grid`\ , :py:obj:`deepxube.domains.grid.GridState`\ , :py:obj:`deepxube.domains.grid.GridGoal`\ , :py:obj:`deepxube.domains.grid.GridAction`\ ], :py:obj:`deepxube.base.nnet_input.FlatIn`\ [\ :py:obj:`deepxube.domains.grid.Grid`\ ]

   .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.domains.grid.GridFlatInQFix.get_input_info

   .. py:method:: to_np(states: typing.List[deepxube.domains.grid.GridState], goals: typing.List[deepxube.domains.grid.GridGoal], actions_l: typing.List[typing.List[deepxube.domains.grid.GridAction]]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.grid.GridFlatInQFix.to_np

.. py:class:: GridFlatInActIn(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.domains.grid.GridFlatInActIn

   Bases: :py:obj:`deepxube.base.nnet_input.StateGoalActIn`\ [\ :py:obj:`deepxube.domains.grid.Grid`\ , :py:obj:`deepxube.domains.grid.GridState`\ , :py:obj:`deepxube.domains.grid.GridGoal`\ , :py:obj:`deepxube.domains.grid.GridAction`\ ], :py:obj:`deepxube.base.nnet_input.FlatIn`\ [\ :py:obj:`deepxube.domains.grid.Grid`\ ]

   .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.domains.grid.GridFlatInActIn.get_input_info

   .. py:method:: to_np(states: typing.List[deepxube.domains.grid.GridState], goals: typing.List[deepxube.domains.grid.GridGoal], actions: typing.List[deepxube.domains.grid.GridAction]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.grid.GridFlatInActIn.to_np

.. py:class:: GridNNetInput(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.domains.grid.GridNNetInput

   Bases: :py:obj:`deepxube.base.nnet_input.StateGoalIn`\ [\ :py:obj:`deepxube.domains.grid.Grid`\ , :py:obj:`deepxube.domains.grid.GridState`\ , :py:obj:`deepxube.domains.grid.GridGoal`\ ]

   .. py:method:: get_input_info() -> int
      :canonical: deepxube.domains.grid.GridNNetInput.get_input_info

   .. py:method:: to_np(states: typing.List[deepxube.domains.grid.GridState], goals: typing.List[deepxube.domains.grid.GridGoal]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.grid.GridNNetInput.to_np

.. py:class:: GridNet(nnet_input: deepxube.domains.grid.GridNNetInput, out_dim: int, q_fix: bool, chan_size: int = 8, fc_size: int = 100)
   :canonical: deepxube.domains.grid.GridNet

   Bases: :py:obj:`deepxube.base.heuristic.HeurNNet`\ [\ :py:obj:`deepxube.domains.grid.GridNNetInput`\ ]

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.domains.grid.GridNNetInput]
      :canonical: deepxube.domains.grid.GridNet.nnet_input_type
      :staticmethod:

   .. py:method:: _forward(inputs: typing.List[torch.Tensor]) -> torch.Tensor
      :canonical: deepxube.domains.grid.GridNet._forward

      .. autodoc2-docstring:: deepxube.domains.grid.GridNet._forward

.. py:class:: GridNetParser()
   :canonical: deepxube.domains.grid.GridNetParser

   Bases: :py:obj:`deepxube.base.factory.DelimParser`

   .. py:property:: delim
      :canonical: deepxube.domains.grid.GridNetParser.delim
      :type: str

      .. autodoc2-docstring:: deepxube.domains.grid.GridNetParser.delim
