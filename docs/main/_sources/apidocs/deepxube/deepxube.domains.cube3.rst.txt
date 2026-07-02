:py:mod:`deepxube.domains.cube3`
================================

.. py:module:: deepxube.domains.cube3

.. autodoc2-docstring:: deepxube.domains.cube3
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Cube3State <deepxube.domains.cube3.Cube3State>`
     -
   * - :py:obj:`Cube3Goal <deepxube.domains.cube3.Cube3Goal>`
     -
   * - :py:obj:`Cube3Action <deepxube.domains.cube3.Cube3Action>`
     -
   * - :py:obj:`Quaternion <deepxube.domains.cube3.Quaternion>`
     - .. autodoc2-docstring:: deepxube.domains.cube3.Quaternion
          :summary:
   * - :py:obj:`InteractiveCube <deepxube.domains.cube3.InteractiveCube>`
     -
   * - :py:obj:`Cube3 <deepxube.domains.cube3.Cube3>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_get_adj <deepxube.domains.cube3._get_adj>`
     - .. autodoc2-docstring:: deepxube.domains.cube3._get_adj
          :summary:
   * - :py:obj:`project_points <deepxube.domains.cube3.project_points>`
     - .. autodoc2-docstring:: deepxube.domains.cube3.project_points
          :summary:

API
~~~

.. py:class:: Cube3State(colors: numpy.typing.NDArray[numpy.uint8])
   :canonical: deepxube.domains.cube3.Cube3State

   Bases: :py:obj:`deepxube.base.domain.State`

   .. py:attribute:: __slots__
      :canonical: deepxube.domains.cube3.Cube3State.__slots__
      :value: ['colors', 'hash']

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3State.__slots__

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.cube3.Cube3State.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.cube3.Cube3State.__eq__

.. py:class:: Cube3Goal(colors: numpy.typing.NDArray[numpy.uint8])
   :canonical: deepxube.domains.cube3.Cube3Goal

   Bases: :py:obj:`deepxube.base.domain.Goal`

.. py:class:: Cube3Action(action: int)
   :canonical: deepxube.domains.cube3.Cube3Action

   Bases: :py:obj:`deepxube.base.domain.Action`

   .. py:method:: __hash__() -> int
      :canonical: deepxube.domains.cube3.Cube3Action.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.domains.cube3.Cube3Action.__eq__

.. py:function:: _get_adj() -> typing.Dict[int, numpy.typing.NDArray[numpy.int_]]
   :canonical: deepxube.domains.cube3._get_adj

   .. autodoc2-docstring:: deepxube.domains.cube3._get_adj

.. py:class:: Quaternion(x: numpy.typing.NDArray)
   :canonical: deepxube.domains.cube3.Quaternion

   .. autodoc2-docstring:: deepxube.domains.cube3.Quaternion

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.domains.cube3.Quaternion.__init__

   .. py:method:: from_v_theta(v, theta) -> deepxube.domains.cube3.Quaternion
      :canonical: deepxube.domains.cube3.Quaternion.from_v_theta
      :classmethod:

      .. autodoc2-docstring:: deepxube.domains.cube3.Quaternion.from_v_theta

   .. py:method:: __repr__() -> str
      :canonical: deepxube.domains.cube3.Quaternion.__repr__

   .. py:method:: __mul__(other: deepxube.domains.cube3.Quaternion) -> deepxube.domains.cube3.Quaternion
      :canonical: deepxube.domains.cube3.Quaternion.__mul__

      .. autodoc2-docstring:: deepxube.domains.cube3.Quaternion.__mul__

   .. py:method:: as_v_theta() -> typing.Tuple[numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: deepxube.domains.cube3.Quaternion.as_v_theta

      .. autodoc2-docstring:: deepxube.domains.cube3.Quaternion.as_v_theta

   .. py:method:: as_rotation_matrix() -> numpy.typing.NDArray
      :canonical: deepxube.domains.cube3.Quaternion.as_rotation_matrix

      .. autodoc2-docstring:: deepxube.domains.cube3.Quaternion.as_rotation_matrix

   .. py:method:: rotate(points)
      :canonical: deepxube.domains.cube3.Quaternion.rotate

      .. autodoc2-docstring:: deepxube.domains.cube3.Quaternion.rotate

.. py:function:: project_points(points, q: deepxube.domains.cube3.Quaternion, view, vertical) -> numpy.typing.NDArray
   :canonical: deepxube.domains.cube3.project_points

   .. autodoc2-docstring:: deepxube.domains.cube3.project_points

.. py:class:: InteractiveCube(n, colors: numpy.typing.NDArray, view=(0, 0, 10), fig=None, **kwargs)
   :canonical: deepxube.domains.cube3.InteractiveCube

   Bases: :py:obj:`matplotlib.pyplot.Axes`

   .. py:attribute:: base_face
      :canonical: deepxube.domains.cube3.InteractiveCube.base_face
      :value: 'array(...)'

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube.base_face

   .. py:attribute:: stickerwidth
      :canonical: deepxube.domains.cube3.InteractiveCube.stickerwidth
      :value: 0.9

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube.stickerwidth

   .. py:attribute:: stickermargin
      :canonical: deepxube.domains.cube3.InteractiveCube.stickermargin
      :value: None

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube.stickermargin

   .. py:attribute:: stickerthickness
      :canonical: deepxube.domains.cube3.InteractiveCube.stickerthickness
      :value: 0.001

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube.stickerthickness

   .. py:attribute:: base_sticker
      :canonical: deepxube.domains.cube3.InteractiveCube.base_sticker
      :value: 'array(...)'

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube.base_sticker

   .. py:attribute:: base_face_centroid
      :canonical: deepxube.domains.cube3.InteractiveCube.base_face_centroid
      :value: 'array(...)'

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube.base_face_centroid

   .. py:attribute:: base_sticker_centroid
      :canonical: deepxube.domains.cube3.InteractiveCube.base_sticker_centroid
      :value: 'array(...)'

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube.base_sticker_centroid

   .. py:method:: set_rot(rot: int) -> None
      :canonical: deepxube.domains.cube3.InteractiveCube.set_rot

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube.set_rot

   .. py:method:: _initialize_arrays() -> None
      :canonical: deepxube.domains.cube3.InteractiveCube._initialize_arrays

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube._initialize_arrays

   .. py:method:: rotate(rot) -> None
      :canonical: deepxube.domains.cube3.InteractiveCube.rotate

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube.rotate

   .. py:method:: _project(pts)
      :canonical: deepxube.domains.cube3.InteractiveCube._project

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube._project

   .. py:method:: _draw_cube() -> None
      :canonical: deepxube.domains.cube3.InteractiveCube._draw_cube

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube._draw_cube

   .. py:method:: _mouse_press(event, event_x=None, event_y=None)
      :canonical: deepxube.domains.cube3.InteractiveCube._mouse_press

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube._mouse_press

   .. py:method:: _mouse_release(event)
      :canonical: deepxube.domains.cube3.InteractiveCube._mouse_release

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube._mouse_release

   .. py:method:: _mouse_motion(event, event_x=None, event_y=None)
      :canonical: deepxube.domains.cube3.InteractiveCube._mouse_motion

      .. autodoc2-docstring:: deepxube.domains.cube3.InteractiveCube._mouse_motion

.. py:class:: Cube3()
   :canonical: deepxube.domains.cube3.Cube3

   Bases: :py:obj:`deepxube.base.domain.NextStateNPActsEnumFixed`\ [\ :py:obj:`deepxube.domains.cube3.Cube3State`\ , :py:obj:`deepxube.domains.cube3.Cube3Action`\ , :py:obj:`deepxube.domains.cube3.Cube3Goal`\ ], :py:obj:`deepxube.base.domain.GoalStartRevWalkableActsRev`\ [\ :py:obj:`deepxube.domains.cube3.Cube3State`\ , :py:obj:`deepxube.domains.cube3.Cube3Action`\ , :py:obj:`deepxube.domains.cube3.Cube3Goal`\ ], :py:obj:`deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn`\ [\ :py:obj:`deepxube.domains.cube3.Cube3State`\ , :py:obj:`deepxube.domains.cube3.Cube3Action`\ , :py:obj:`deepxube.domains.cube3.Cube3Goal`\ ], :py:obj:`deepxube.base.nnet_input.HasFlatSGAIn`\ [\ :py:obj:`deepxube.domains.cube3.Cube3State`\ , :py:obj:`deepxube.domains.cube3.Cube3Action`\ , :py:obj:`deepxube.domains.cube3.Cube3Goal`\ ], :py:obj:`deepxube.base.domain.StateGoalVizable`\ [\ :py:obj:`deepxube.domains.cube3.Cube3State`\ , :py:obj:`deepxube.domains.cube3.Cube3Action`\ , :py:obj:`deepxube.domains.cube3.Cube3Goal`\ ], :py:obj:`deepxube.base.domain.StringToAct`\ [\ :py:obj:`deepxube.domains.cube3.Cube3State`\ , :py:obj:`deepxube.domains.cube3.Cube3Action`\ , :py:obj:`deepxube.domains.cube3.Cube3Goal`\ ]

   .. py:attribute:: atomic_actions
      :canonical: deepxube.domains.cube3.Cube3.atomic_actions
      :type: typing.List[str]
      :value: None

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.atomic_actions

   .. py:method:: is_solved(states: typing.List[deepxube.domains.cube3.Cube3State], goals: typing.List[deepxube.domains.cube3.Cube3Goal]) -> typing.List[bool]
      :canonical: deepxube.domains.cube3.Cube3.is_solved

   .. py:method:: get_goal_states(num_states: int) -> typing.List[deepxube.domains.cube3.Cube3State]
      :canonical: deepxube.domains.cube3.Cube3.get_goal_states

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.get_goal_states

   .. py:method:: sample_goalstate_goal_pairs(num: int) -> typing.Tuple[typing.List[deepxube.domains.cube3.Cube3State], typing.List[deepxube.domains.cube3.Cube3Goal]]
      :canonical: deepxube.domains.cube3.Cube3.sample_goalstate_goal_pairs

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.sample_goalstate_goal_pairs

   .. py:method:: get_input_info_flat_sg() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.domains.cube3.Cube3.get_input_info_flat_sg

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.get_input_info_flat_sg

   .. py:method:: get_input_info_flat_sga() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.domains.cube3.Cube3.get_input_info_flat_sga

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.get_input_info_flat_sga

   .. py:method:: to_np_flat_sg(states: typing.List[deepxube.domains.cube3.Cube3State], goals: typing.List[deepxube.domains.cube3.Cube3Goal]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.cube3.Cube3.to_np_flat_sg

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.to_np_flat_sg

   .. py:method:: to_np_flat_sga(states: typing.List[deepxube.domains.cube3.Cube3State], goals: typing.List[deepxube.domains.cube3.Cube3Goal], actions: typing.List[deepxube.domains.cube3.Cube3Action]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.domains.cube3.Cube3.to_np_flat_sga

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.to_np_flat_sga

   .. py:method:: actions_to_indices(actions: typing.List[deepxube.domains.cube3.Cube3Action]) -> typing.List[int]
      :canonical: deepxube.domains.cube3.Cube3.actions_to_indices

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.actions_to_indices

   .. py:method:: visualize_state_goal(state: deepxube.domains.cube3.Cube3State, goal: deepxube.domains.cube3.Cube3Goal, fig: matplotlib.figure.Figure) -> None
      :canonical: deepxube.domains.cube3.Cube3.visualize_state_goal

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.visualize_state_goal

   .. py:method:: string_to_action(act_str: str) -> typing.Optional[deepxube.domains.cube3.Cube3Action]
      :canonical: deepxube.domains.cube3.Cube3.string_to_action

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.string_to_action

   .. py:method:: string_to_action_help() -> str
      :canonical: deepxube.domains.cube3.Cube3.string_to_action_help

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.string_to_action_help

   .. py:method:: get_actions_fixed() -> typing.List[deepxube.domains.cube3.Cube3Action]
      :canonical: deepxube.domains.cube3.Cube3.get_actions_fixed

   .. py:method:: sample_rev_state(states: typing.List[deepxube.domains.cube3.Cube3State]) -> typing.Tuple[typing.List[deepxube.domains.cube3.Cube3State], typing.List[deepxube.domains.cube3.Cube3Action], typing.List[float]]
      :canonical: deepxube.domains.cube3.Cube3.sample_rev_state

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3.sample_rev_state

   .. py:method:: _states_to_np(states: typing.List[deepxube.domains.cube3.Cube3State]) -> typing.List[numpy.typing.NDArray[numpy.uint8]]
      :canonical: deepxube.domains.cube3.Cube3._states_to_np

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3._states_to_np

   .. py:method:: _np_to_states(states_np: typing.List[numpy.typing.NDArray]) -> typing.List[deepxube.domains.cube3.Cube3State]
      :canonical: deepxube.domains.cube3.Cube3._np_to_states

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3._np_to_states

   .. py:method:: _next_state_np(states_np_l: typing.List[numpy.typing.NDArray[numpy.uint8]], actions: typing.List[deepxube.domains.cube3.Cube3Action]) -> typing.Tuple[typing.List[numpy.typing.NDArray], typing.List[float]]
      :canonical: deepxube.domains.cube3.Cube3._next_state_np

   .. py:method:: _compute_rotation_idxs(cube_len: int, moves: typing.List[str]) -> typing.Tuple[typing.Dict[str, numpy.typing.NDArray[numpy.int_]], typing.Dict[str, numpy.typing.NDArray[numpy.int_]]]
      :canonical: deepxube.domains.cube3.Cube3._compute_rotation_idxs

      .. autodoc2-docstring:: deepxube.domains.cube3.Cube3._compute_rotation_idxs

   .. py:method:: __repr__() -> str
      :canonical: deepxube.domains.cube3.Cube3.__repr__
