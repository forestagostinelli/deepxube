:py:mod:`deepxube.domains.npuzzle`
==================================

.. py:module:: deepxube.domains.npuzzle

.. autodoc2-docstring:: deepxube.domains.npuzzle
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NPState <deepxube.domains.npuzzle.NPState>`
     -
   * - :py:obj:`NPGoal <deepxube.domains.npuzzle.NPGoal>`
     -
   * - :py:obj:`NPAction <deepxube.domains.npuzzle.NPAction>`
     -
   * - :py:obj:`NPuzzle <deepxube.domains.npuzzle.NPuzzle>`
     -
   * - :py:obj:`GridParser <deepxube.domains.npuzzle.GridParser>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`int_t <deepxube.domains.npuzzle.int_t>`
     - .. autodoc2-docstring:: deepxube.domains.npuzzle.int_t
          :summary:

API
~~~

.. py:data:: int_t
   :canonical: deepxube.domains.npuzzle.int_t
   :value: None

   .. autodoc2-docstring:: deepxube.domains.npuzzle.int_t

.. py:class:: NPState(tiles: numpy.typing.NDArray[deepxube.domains.npuzzle.int_t])
   :canonical: deepxube.domains.npuzzle.NPState

   Bases: :py:obj:`deepxube.base.domain.State`

   .. py:attribute:: __slots__
      :canonical: deepxube.domains.npuzzle.NPState.__slots__
      :value: ['tiles', 'hash']

      .. autodoc2-docstring:: deepxube.domains.npuzzle.NPState.__slots__

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.npuzzle.NPState.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.npuzzle.NPState.__eq__

.. py:class:: NPGoal(tiles: numpy.typing.NDArray[deepxube.domains.npuzzle.int_t])
   :canonical: deepxube.domains.npuzzle.NPGoal

   Bases: :py:obj:`deepxube.base.domain.Goal`

.. py:class:: NPAction(action: int)
   :canonical: deepxube.domains.npuzzle.NPAction

   Bases: :py:obj:`deepxube.base.domain.Action`

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.npuzzle.NPAction.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.npuzzle.NPAction.__eq__

.. py:class:: NPuzzle(dim: int = 4)
   :canonical: deepxube.domains.npuzzle.NPuzzle

   Bases: :py:obj:`deepxube.base.domain.ActsEnumFixed`\ [\ :py:obj:`deepxube.domains.npuzzle.NPState`\ , :py:obj:`deepxube.domains.npuzzle.NPAction`\ , :py:obj:`deepxube.domains.npuzzle.NPGoal`\ ], :py:obj:`deepxube.base.domain.GoalStartRevWalkable`\ [\ :py:obj:`deepxube.domains.npuzzle.NPState`\ , :py:obj:`deepxube.domains.npuzzle.NPAction`\ , :py:obj:`deepxube.domains.npuzzle.NPGoal`\ ], :py:obj:`deepxube.base.nnet_input.HasFlatSGIn`\ [\ :py:obj:`deepxube.domains.npuzzle.NPState`\ , :py:obj:`deepxube.domains.npuzzle.NPAction`\ , :py:obj:`deepxube.domains.npuzzle.NPGoal`\ ], :py:obj:`deepxube.base.domain.StateGoalVizable`\ [\ :py:obj:`deepxube.domains.npuzzle.NPState`\ , :py:obj:`deepxube.domains.npuzzle.NPAction`\ , :py:obj:`deepxube.domains.npuzzle.NPGoal`\ ], :py:obj:`deepxube.base.domain.StringToAct`\ [\ :py:obj:`deepxube.domains.npuzzle.NPState`\ , :py:obj:`deepxube.domains.npuzzle.NPAction`\ , :py:obj:`deepxube.domains.npuzzle.NPGoal`\ ]

   .. py:attribute:: moves
      :canonical: deepxube.domains.npuzzle.NPuzzle.moves
      :type: typing.List[str]
      :value: ['U', 'D', 'L', 'R']

      .. autodoc2-docstring:: deepxube.domains.npuzzle.NPuzzle.moves

   .. py:attribute:: moves_rev
      :canonical: deepxube.domains.npuzzle.NPuzzle.moves_rev
      :type: typing.List[str]
      :value: ['D', 'U', 'R', 'L']

      .. autodoc2-docstring:: deepxube.domains.npuzzle.NPuzzle.moves_rev

   .. py:method:: sample_goalstate_goal_pairs(num: int) -> typing.Tuple[typing.List[deepxube.domains.npuzzle.NPState], typing.List[deepxube.domains.npuzzle.NPGoal]]
      :canonical: deepxube.domains.npuzzle.NPuzzle.sample_goalstate_goal_pairs

   .. py:method:: next_state(states: typing.List[deepxube.domains.npuzzle.NPState], actions: typing.List[deepxube.domains.npuzzle.NPAction]) -> typing.Tuple[typing.List[deepxube.domains.npuzzle.NPState], typing.List[float]]
      :canonical: deepxube.domains.npuzzle.NPuzzle.next_state

   .. py:method:: expand(states: typing.List[deepxube.domains.npuzzle.NPState]) -> typing.Tuple[typing.List[typing.List[deepxube.domains.npuzzle.NPState]], typing.List[typing.List[deepxube.domains.npuzzle.NPAction]], typing.List[typing.List[float]]]
      :canonical: deepxube.domains.npuzzle.NPuzzle.expand

   .. py:method:: get_actions_fixed() -> typing.List[deepxube.domains.npuzzle.NPAction]
      :canonical: deepxube.domains.npuzzle.NPuzzle.get_actions_fixed

   .. py:method:: random_walk_rev_no_path_cost(states: typing.List[deepxube.domains.npuzzle.NPState], num_steps_l: typing.List[int]) -> typing.List[deepxube.domains.npuzzle.NPState]
      :canonical: deepxube.domains.npuzzle.NPuzzle.random_walk_rev_no_path_cost

   .. py:method:: is_solved(states: typing.List[deepxube.domains.npuzzle.NPState], goals: typing.List[deepxube.domains.npuzzle.NPGoal]) -> typing.List[bool]
      :canonical: deepxube.domains.npuzzle.NPuzzle.is_solved

   .. py:method:: get_input_info_flat_sg() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.domains.npuzzle.NPuzzle.get_input_info_flat_sg

   .. py:method:: to_np_flat_sg(states: typing.List[deepxube.domains.npuzzle.NPState], goals: typing.List[deepxube.domains.npuzzle.NPGoal]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.npuzzle.NPuzzle.to_np_flat_sg

   .. py:method:: visualize_state_goal(state: deepxube.domains.npuzzle.NPState, goal: deepxube.domains.npuzzle.NPGoal, fig: matplotlib.figure.Figure) -> None
      :canonical: deepxube.domains.npuzzle.NPuzzle.visualize_state_goal

   .. py:method:: string_to_action(act_str: str) -> typing.Optional[deepxube.domains.npuzzle.NPAction]
      :canonical: deepxube.domains.npuzzle.NPuzzle.string_to_action

   .. py:method:: string_to_action_help() -> str
      :canonical: deepxube.domains.npuzzle.NPuzzle.string_to_action_help

   .. py:method:: _is_solvable(states_np: numpy.typing.NDArray[deepxube.domains.npuzzle.int_t]) -> numpy.typing.NDArray[numpy.bool_]
      :canonical: deepxube.domains.npuzzle.NPuzzle._is_solvable

      .. autodoc2-docstring:: deepxube.domains.npuzzle.NPuzzle._is_solvable

   .. py:method:: _get_num_inversions(states_np: numpy.typing.NDArray[deepxube.domains.npuzzle.int_t]) -> numpy.typing.NDArray[numpy.int_]
      :canonical: deepxube.domains.npuzzle.NPuzzle._get_num_inversions

      .. autodoc2-docstring:: deepxube.domains.npuzzle.NPuzzle._get_num_inversions

   .. py:method:: random_walk(states: typing.List[deepxube.domains.npuzzle.NPState], num_steps_l: typing.List[int]) -> typing.Tuple[typing.List[deepxube.domains.npuzzle.NPState], typing.List[typing.List[deepxube.domains.npuzzle.NPAction]], typing.List[float]]
      :canonical: deepxube.domains.npuzzle.NPuzzle.random_walk

   .. py:method:: _get_swap_zero_idxs(n: int) -> numpy.typing.NDArray[deepxube.domains.npuzzle.int_t]
      :canonical: deepxube.domains.npuzzle.NPuzzle._get_swap_zero_idxs

      .. autodoc2-docstring:: deepxube.domains.npuzzle.NPuzzle._get_swap_zero_idxs

   .. py:method:: _move_np(states_np: numpy.typing.NDArray[deepxube.domains.npuzzle.int_t], z_idxs: numpy.typing.NDArray[numpy.int_], action: int) -> typing.Tuple[numpy.typing.NDArray[deepxube.domains.npuzzle.int_t], numpy.typing.NDArray[deepxube.domains.npuzzle.int_t], typing.List[float]]
      :canonical: deepxube.domains.npuzzle.NPuzzle._move_np

      .. autodoc2-docstring:: deepxube.domains.npuzzle.NPuzzle._move_np

   .. py:method:: __repr__() -> str
      :canonical: deepxube.domains.npuzzle.NPuzzle.__repr__

.. py:class:: GridParser
   :canonical: deepxube.domains.npuzzle.GridParser

   Bases: :py:obj:`deepxube.base.factory.Parser`

   .. py:method:: parse(args_str: str) -> typing.Dict[str, typing.Any]
      :canonical: deepxube.domains.npuzzle.GridParser.parse

      .. autodoc2-docstring:: deepxube.domains.npuzzle.GridParser.parse

   .. py:method:: help() -> str
      :canonical: deepxube.domains.npuzzle.GridParser.help

      .. autodoc2-docstring:: deepxube.domains.npuzzle.GridParser.help
