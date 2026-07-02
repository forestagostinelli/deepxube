:py:mod:`deepxube.domains.sokoban`
==================================

.. py:module:: deepxube.domains.sokoban

.. autodoc2-docstring:: deepxube.domains.sokoban
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SkState <deepxube.domains.sokoban.SkState>`
     -
   * - :py:obj:`SkGoal <deepxube.domains.sokoban.SkGoal>`
     -
   * - :py:obj:`SkAction <deepxube.domains.sokoban.SkAction>`
     -
   * - :py:obj:`Sokoban <deepxube.domains.sokoban.Sokoban>`
     -
   * - :py:obj:`SkNNetInput <deepxube.domains.sokoban.SkNNetInput>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`load_states <deepxube.domains.sokoban.load_states>`
     - .. autodoc2-docstring:: deepxube.domains.sokoban.load_states
          :summary:
   * - :py:obj:`_get_surfaces <deepxube.domains.sokoban._get_surfaces>`
     - .. autodoc2-docstring:: deepxube.domains.sokoban._get_surfaces
          :summary:
   * - :py:obj:`_get_train_states <deepxube.domains.sokoban._get_train_states>`
     - .. autodoc2-docstring:: deepxube.domains.sokoban._get_train_states
          :summary:
   * - :py:obj:`get_data_dir <deepxube.domains.sokoban.get_data_dir>`
     - .. autodoc2-docstring:: deepxube.domains.sokoban.get_data_dir
          :summary:

API
~~~

.. py:class:: SkState(agent: numpy.typing.NDArray[numpy.int_], boxes: numpy.typing.NDArray[numpy.uint8], walls: numpy.typing.NDArray[numpy.uint8])
   :canonical: deepxube.domains.sokoban.SkState

   Bases: :py:obj:`deepxube.base.domain.State`

   .. py:attribute:: __slots__
      :canonical: deepxube.domains.sokoban.SkState.__slots__
      :value: ['agent', 'boxes', 'walls', 'hash']

      .. autodoc2-docstring:: deepxube.domains.sokoban.SkState.__slots__

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.sokoban.SkState.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.sokoban.SkState.__eq__

.. py:class:: SkGoal(boxes: numpy.typing.NDArray[numpy.uint8])
   :canonical: deepxube.domains.sokoban.SkGoal

   Bases: :py:obj:`deepxube.base.domain.Goal`

   .. py:attribute:: __slots__
      :canonical: deepxube.domains.sokoban.SkGoal.__slots__
      :value: ['boxes']

      .. autodoc2-docstring:: deepxube.domains.sokoban.SkGoal.__slots__

.. py:class:: SkAction(action: int)
   :canonical: deepxube.domains.sokoban.SkAction

   Bases: :py:obj:`deepxube.base.domain.Action`

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.sokoban.SkAction.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.sokoban.SkAction.__eq__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.domains.sokoban.SkAction.__repr__

.. py:function:: load_states(data_dir: str) -> typing.List[deepxube.domains.sokoban.SkState]
   :canonical: deepxube.domains.sokoban.load_states

   .. autodoc2-docstring:: deepxube.domains.sokoban.load_states

.. py:function:: _get_surfaces() -> typing.Dict[str, numpy.typing.NDArray]
   :canonical: deepxube.domains.sokoban._get_surfaces

   .. autodoc2-docstring:: deepxube.domains.sokoban._get_surfaces

.. py:function:: _get_train_states() -> typing.List[deepxube.domains.sokoban.SkState]
   :canonical: deepxube.domains.sokoban._get_train_states

   .. autodoc2-docstring:: deepxube.domains.sokoban._get_train_states

.. py:function:: get_data_dir() -> str
   :canonical: deepxube.domains.sokoban.get_data_dir

   .. autodoc2-docstring:: deepxube.domains.sokoban.get_data_dir

.. py:class:: Sokoban()
   :canonical: deepxube.domains.sokoban.Sokoban

   Bases: :py:obj:`deepxube.base.domain.ActsEnumFixed`\ [\ :py:obj:`deepxube.domains.sokoban.SkState`\ , :py:obj:`deepxube.domains.sokoban.SkAction`\ , :py:obj:`deepxube.domains.sokoban.SkGoal`\ ], :py:obj:`deepxube.base.domain.StartGoalWalkable`\ [\ :py:obj:`deepxube.domains.sokoban.SkState`\ , :py:obj:`deepxube.domains.sokoban.SkAction`\ , :py:obj:`deepxube.domains.sokoban.SkGoal`\ ], :py:obj:`deepxube.base.domain.StateGoalVizable`\ [\ :py:obj:`deepxube.domains.sokoban.SkState`\ , :py:obj:`deepxube.domains.sokoban.SkAction`\ , :py:obj:`deepxube.domains.sokoban.SkGoal`\ ], :py:obj:`deepxube.base.domain.StringToAct`\ [\ :py:obj:`deepxube.domains.sokoban.SkState`\ , :py:obj:`deepxube.domains.sokoban.SkAction`\ , :py:obj:`deepxube.domains.sokoban.SkGoal`\ ]

   .. py:method:: next_state(states: typing.List[deepxube.domains.sokoban.SkState], actions: typing.List[deepxube.domains.sokoban.SkAction]) -> typing.Tuple[typing.List[deepxube.domains.sokoban.SkState], typing.List[float]]
      :canonical: deepxube.domains.sokoban.Sokoban.next_state

   .. py:method:: get_actions_fixed() -> typing.List[deepxube.domains.sokoban.SkAction]
      :canonical: deepxube.domains.sokoban.Sokoban.get_actions_fixed

   .. py:method:: is_solved(states: typing.List[deepxube.domains.sokoban.SkState], goals: typing.List[deepxube.domains.sokoban.SkGoal]) -> typing.List[bool]
      :canonical: deepxube.domains.sokoban.Sokoban.is_solved

   .. py:method:: sample_start_states(num_states: int) -> typing.List[deepxube.domains.sokoban.SkState]
      :canonical: deepxube.domains.sokoban.Sokoban.sample_start_states

   .. py:method:: sample_goal_from_state(states_start: typing.Optional[typing.List[deepxube.domains.sokoban.SkState]], states_goal: typing.List[deepxube.domains.sokoban.SkState]) -> typing.List[deepxube.domains.sokoban.SkGoal]
      :canonical: deepxube.domains.sokoban.Sokoban.sample_goal_from_state

   .. py:method:: string_to_action(act_str: str) -> typing.Optional[deepxube.domains.sokoban.SkAction]
      :canonical: deepxube.domains.sokoban.Sokoban.string_to_action

      .. autodoc2-docstring:: deepxube.domains.sokoban.Sokoban.string_to_action

   .. py:method:: string_to_action_help() -> str
      :canonical: deepxube.domains.sokoban.Sokoban.string_to_action_help

      .. autodoc2-docstring:: deepxube.domains.sokoban.Sokoban.string_to_action_help

   .. py:method:: visualize_state_goal(state: deepxube.domains.sokoban.SkState, goal: deepxube.domains.sokoban.SkGoal, fig: matplotlib.figure.Figure) -> None
      :canonical: deepxube.domains.sokoban.Sokoban.visualize_state_goal

   .. py:method:: to_img(state: deepxube.domains.sokoban.SkState, goal: deepxube.domains.sokoban.SkGoal) -> numpy.typing.NDArray
      :canonical: deepxube.domains.sokoban.Sokoban.to_img

      .. autodoc2-docstring:: deepxube.domains.sokoban.Sokoban.to_img

   .. py:method:: _get_next_idx(curr_idxs: numpy.typing.NDArray[numpy.int_], actions: typing.List[deepxube.domains.sokoban.SkAction]) -> numpy.typing.NDArray[numpy.int_]
      :canonical: deepxube.domains.sokoban.Sokoban._get_next_idx

      .. autodoc2-docstring:: deepxube.domains.sokoban.Sokoban._get_next_idx

   .. py:method:: __getstate__() -> typing.Dict
      :canonical: deepxube.domains.sokoban.Sokoban.__getstate__

      .. autodoc2-docstring:: deepxube.domains.sokoban.Sokoban.__getstate__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.domains.sokoban.Sokoban.__repr__

.. py:class:: SkNNetInput(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.domains.sokoban.SkNNetInput

   Bases: :py:obj:`deepxube.base.nnet_input.FlatIn`\ [\ :py:obj:`deepxube.domains.sokoban.Sokoban`\ ], :py:obj:`deepxube.base.nnet_input.StateGoalIn`\ [\ :py:obj:`deepxube.domains.sokoban.Sokoban`\ , :py:obj:`deepxube.domains.sokoban.SkState`\ , :py:obj:`deepxube.domains.sokoban.SkGoal`\ ]

   .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.domains.sokoban.SkNNetInput.get_input_info

   .. py:method:: to_np(states: typing.List[deepxube.domains.sokoban.SkState], goals: typing.List[deepxube.domains.sokoban.SkGoal]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.sokoban.SkNNetInput.to_np
