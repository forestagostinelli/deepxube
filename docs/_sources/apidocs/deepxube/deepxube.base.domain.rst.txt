:py:mod:`deepxube.base.domain`
==============================

.. py:module:: deepxube.base.domain

.. autodoc2-docstring:: deepxube.base.domain
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`State <deepxube.base.domain.State>`
     - .. autodoc2-docstring:: deepxube.base.domain.State
          :summary:
   * - :py:obj:`Action <deepxube.base.domain.Action>`
     - .. autodoc2-docstring:: deepxube.base.domain.Action
          :summary:
   * - :py:obj:`Goal <deepxube.base.domain.Goal>`
     - .. autodoc2-docstring:: deepxube.base.domain.Goal
          :summary:
   * - :py:obj:`Domain <deepxube.base.domain.Domain>`
     - .. autodoc2-docstring:: deepxube.base.domain.Domain
          :summary:
   * - :py:obj:`StateGoalVizable <deepxube.base.domain.StateGoalVizable>`
     - .. autodoc2-docstring:: deepxube.base.domain.StateGoalVizable
          :summary:
   * - :py:obj:`StringToAct <deepxube.base.domain.StringToAct>`
     - .. autodoc2-docstring:: deepxube.base.domain.StringToAct
          :summary:
   * - :py:obj:`ActsFixed <deepxube.base.domain.ActsFixed>`
     -
   * - :py:obj:`ActsRev <deepxube.base.domain.ActsRev>`
     - .. autodoc2-docstring:: deepxube.base.domain.ActsRev
          :summary:
   * - :py:obj:`ActsEnum <deepxube.base.domain.ActsEnum>`
     -
   * - :py:obj:`ActsEnumFixed <deepxube.base.domain.ActsEnumFixed>`
     -
   * - :py:obj:`NodesSupervisable <deepxube.base.domain.NodesSupervisable>`
     -
   * - :py:obj:`EdgesSupervisable <deepxube.base.domain.EdgesSupervisable>`
     -
   * - :py:obj:`EdgesSampleable <deepxube.base.domain.EdgesSampleable>`
     -
   * - :py:obj:`GoalSampleable <deepxube.base.domain.GoalSampleable>`
     - .. autodoc2-docstring:: deepxube.base.domain.GoalSampleable
          :summary:
   * - :py:obj:`GoalStateSampleable <deepxube.base.domain.GoalStateSampleable>`
     - .. autodoc2-docstring:: deepxube.base.domain.GoalStateSampleable
          :summary:
   * - :py:obj:`GoalSampleableFromState <deepxube.base.domain.GoalSampleableFromState>`
     - .. autodoc2-docstring:: deepxube.base.domain.GoalSampleableFromState
          :summary:
   * - :py:obj:`StateSampleableFromGoal <deepxube.base.domain.StateSampleableFromGoal>`
     - .. autodoc2-docstring:: deepxube.base.domain.StateSampleableFromGoal
          :summary:
   * - :py:obj:`GoalFixed <deepxube.base.domain.GoalFixed>`
     - .. autodoc2-docstring:: deepxube.base.domain.GoalFixed
          :summary:
   * - :py:obj:`GoalStateGoalPairSampleable <deepxube.base.domain.GoalStateGoalPairSampleable>`
     -
   * - :py:obj:`GoalStateSampGoalSamp <deepxube.base.domain.GoalStateSampGoalSamp>`
     - .. autodoc2-docstring:: deepxube.base.domain.GoalStateSampGoalSamp
          :summary:
   * - :py:obj:`GoalSampGoalStateSamp <deepxube.base.domain.GoalSampGoalStateSamp>`
     - .. autodoc2-docstring:: deepxube.base.domain.GoalSampGoalStateSamp
          :summary:
   * - :py:obj:`StartGoalWalkable <deepxube.base.domain.StartGoalWalkable>`
     - .. autodoc2-docstring:: deepxube.base.domain.StartGoalWalkable
          :summary:
   * - :py:obj:`GoalStartRevWalkable <deepxube.base.domain.GoalStartRevWalkable>`
     -
   * - :py:obj:`GoalStartRevWalkableActsRev <deepxube.base.domain.GoalStartRevWalkableActsRev>`
     -
   * - :py:obj:`NextStateNP <deepxube.base.domain.NextStateNP>`
     -
   * - :py:obj:`NextStateNPActsFixed <deepxube.base.domain.NextStateNPActsFixed>`
     -
   * - :py:obj:`NextStateNPActsEnumFixed <deepxube.base.domain.NextStateNPActsEnumFixed>`
     -
   * - :py:obj:`SupportsPDDL <deepxube.base.domain.SupportsPDDL>`
     -
   * - :py:obj:`GoalGrndAtoms <deepxube.base.domain.GoalGrndAtoms>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`S <deepxube.base.domain.S>`
     - .. autodoc2-docstring:: deepxube.base.domain.S
          :summary:
   * - :py:obj:`A <deepxube.base.domain.A>`
     - .. autodoc2-docstring:: deepxube.base.domain.A
          :summary:
   * - :py:obj:`G <deepxube.base.domain.G>`
     - .. autodoc2-docstring:: deepxube.base.domain.G
          :summary:

API
~~~

.. py:class:: State
   :canonical: deepxube.base.domain.State

   Bases: :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.domain.State

   .. py:method:: __hash__() -> int
      :canonical: deepxube.base.domain.State.__hash__
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.State.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.base.domain.State.__eq__
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.State.__eq__

.. py:class:: Action
   :canonical: deepxube.base.domain.Action

   Bases: :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.domain.Action

   .. py:method:: __hash__() -> int
      :canonical: deepxube.base.domain.Action.__hash__
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.Action.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.base.domain.Action.__eq__
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.Action.__eq__

.. py:class:: Goal
   :canonical: deepxube.base.domain.Goal

   Bases: :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.domain.Goal

.. py:data:: S
   :canonical: deepxube.base.domain.S
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.domain.S

.. py:data:: A
   :canonical: deepxube.base.domain.A
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.domain.A

.. py:data:: G
   :canonical: deepxube.base.domain.G
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.domain.G

.. py:class:: Domain(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.Domain

   Bases: :py:obj:`abc.ABC`, :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. autodoc2-docstring:: deepxube.base.domain.Domain

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.Domain.__init__

   .. py:method:: sample_problem_instances(num_steps_l: typing.List[int], times: typing.Optional[deepxube.utils.timing_utils.Times] = None) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G]]
      :canonical: deepxube.base.domain.Domain.sample_problem_instances
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.Domain.sample_problem_instances

   .. py:method:: sample_state_action(states: typing.List[deepxube.base.domain.S]) -> typing.List[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.Domain.sample_state_action
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.Domain.sample_state_action

   .. py:method:: next_state(states: typing.List[deepxube.base.domain.S], actions: typing.List[deepxube.base.domain.A]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[float]]
      :canonical: deepxube.base.domain.Domain.next_state
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.Domain.next_state

   .. py:method:: is_solved(states: typing.List[deepxube.base.domain.S], goals: typing.List[deepxube.base.domain.G]) -> typing.List[bool]
      :canonical: deepxube.base.domain.Domain.is_solved
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.Domain.is_solved

   .. py:method:: sample_next_state(states: typing.List[deepxube.base.domain.S]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.A], typing.List[float]]
      :canonical: deepxube.base.domain.Domain.sample_next_state

      .. autodoc2-docstring:: deepxube.base.domain.Domain.sample_next_state

   .. py:method:: random_walk(states: typing.List[deepxube.base.domain.S], num_steps_l: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[typing.List[deepxube.base.domain.A]], typing.List[float]]
      :canonical: deepxube.base.domain.Domain.random_walk

      .. autodoc2-docstring:: deepxube.base.domain.Domain.random_walk

   .. py:method:: get_nnet_par_dict() -> typing.Dict[str, typing.Tuple[str, deepxube.nnet.nnet_utils.NNetPar]]
      :canonical: deepxube.base.domain.Domain.get_nnet_par_dict

      .. autodoc2-docstring:: deepxube.base.domain.Domain.get_nnet_par_dict

   .. py:method:: set_nnet_fns(nnet_fn_dict: typing.Dict[str, deepxube.nnet.nnet_utils.NNetCallable]) -> None
      :canonical: deepxube.base.domain.Domain.set_nnet_fns

      .. autodoc2-docstring:: deepxube.base.domain.Domain.set_nnet_fns

   .. py:method:: get_nnet_fn(nnet_fn_name: str) -> deepxube.nnet.nnet_utils.NNetCallable
      :canonical: deepxube.base.domain.Domain.get_nnet_fn

      .. autodoc2-docstring:: deepxube.base.domain.Domain.get_nnet_fn

   .. py:method:: _add_nnet_par(nnet_name: str, nnet_file: str, nnet_par: deepxube.nnet.nnet_utils.NNetPar) -> None
      :canonical: deepxube.base.domain.Domain._add_nnet_par

      .. autodoc2-docstring:: deepxube.base.domain.Domain._add_nnet_par

   .. py:method:: __getstate__() -> typing.Dict
      :canonical: deepxube.base.domain.Domain.__getstate__

      .. autodoc2-docstring:: deepxube.base.domain.Domain.__getstate__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.domain.Domain.__repr__

.. py:class:: StateGoalVizable(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.StateGoalVizable

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. autodoc2-docstring:: deepxube.base.domain.StateGoalVizable

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.StateGoalVizable.__init__

   .. py:method:: visualize_state_goal(state: deepxube.base.domain.S, goal: deepxube.base.domain.G, fig: matplotlib.figure.Figure) -> None
      :canonical: deepxube.base.domain.StateGoalVizable.visualize_state_goal
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.StateGoalVizable.visualize_state_goal

.. py:class:: StringToAct(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.StringToAct

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. autodoc2-docstring:: deepxube.base.domain.StringToAct

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.StringToAct.__init__

   .. py:method:: string_to_action(act_str: str) -> typing.Optional[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.StringToAct.string_to_action
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.StringToAct.string_to_action

   .. py:method:: string_to_action_help() -> str
      :canonical: deepxube.base.domain.StringToAct.string_to_action_help
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.StringToAct.string_to_action_help

.. py:class:: ActsFixed(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.ActsFixed

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: sample_action(num: int) -> typing.List[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.ActsFixed.sample_action
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.ActsFixed.sample_action

   .. py:method:: sample_state_action(states: typing.List[deepxube.base.domain.S]) -> typing.List[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.ActsFixed.sample_state_action

.. py:class:: ActsRev(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.ActsRev

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.domain.ActsRev

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.ActsRev.__init__

   .. py:method:: sample_rev_state(states: typing.List[deepxube.base.domain.S]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.A], typing.List[float]]
      :canonical: deepxube.base.domain.ActsRev.sample_rev_state
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.ActsRev.sample_rev_state

.. py:class:: ActsEnum(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.ActsEnum

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: get_state_actions(states: typing.List[deepxube.base.domain.S]) -> typing.List[typing.List[deepxube.base.domain.A]]
      :canonical: deepxube.base.domain.ActsEnum.get_state_actions
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.ActsEnum.get_state_actions

   .. py:method:: sample_state_action(states: typing.List[deepxube.base.domain.S]) -> typing.List[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.ActsEnum.sample_state_action

   .. py:method:: expand(states: typing.List[deepxube.base.domain.S]) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.S]], typing.List[typing.List[deepxube.base.domain.A]], typing.List[typing.List[float]]]
      :canonical: deepxube.base.domain.ActsEnum.expand

      .. autodoc2-docstring:: deepxube.base.domain.ActsEnum.expand

.. py:class:: ActsEnumFixed(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.ActsEnumFixed

   Bases: :py:obj:`deepxube.base.domain.ActsEnum`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.ActsFixed`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: sample_action(num: int) -> typing.List[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.ActsEnumFixed.sample_action

   .. py:method:: get_state_actions(states: typing.List[deepxube.base.domain.S]) -> typing.List[typing.List[deepxube.base.domain.A]]
      :canonical: deepxube.base.domain.ActsEnumFixed.get_state_actions

   .. py:method:: get_actions_fixed() -> typing.List[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.ActsEnumFixed.get_actions_fixed
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.ActsEnumFixed.get_actions_fixed

   .. py:method:: get_num_acts() -> int
      :canonical: deepxube.base.domain.ActsEnumFixed.get_num_acts

      .. autodoc2-docstring:: deepxube.base.domain.ActsEnumFixed.get_num_acts

.. py:class:: NodesSupervisable(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.NodesSupervisable

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: samp_nodes_and_labels(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[float]]
      :canonical: deepxube.base.domain.NodesSupervisable.samp_nodes_and_labels
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.NodesSupervisable.samp_nodes_and_labels

.. py:class:: EdgesSupervisable(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.EdgesSupervisable

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: samp_edges_and_labels(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[deepxube.base.domain.A], typing.List[float]]
      :canonical: deepxube.base.domain.EdgesSupervisable.samp_edges_and_labels
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.EdgesSupervisable.samp_edges_and_labels

.. py:class:: EdgesSampleable(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.EdgesSampleable

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: samp_edges(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[deepxube.base.domain.A]]
      :canonical: deepxube.base.domain.EdgesSampleable.samp_edges
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.EdgesSampleable.samp_edges

.. py:class:: GoalSampleable(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalSampleable

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. autodoc2-docstring:: deepxube.base.domain.GoalSampleable

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.GoalSampleable.__init__

   .. py:method:: sample_goals(num: int) -> typing.List[deepxube.base.domain.G]
      :canonical: deepxube.base.domain.GoalSampleable.sample_goals
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalSampleable.sample_goals

.. py:class:: GoalStateSampleable(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalStateSampleable

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. autodoc2-docstring:: deepxube.base.domain.GoalStateSampleable

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.GoalStateSampleable.__init__

   .. py:method:: sample_goal_states(num: int) -> typing.List[deepxube.base.domain.S]
      :canonical: deepxube.base.domain.GoalStateSampleable.sample_goal_states
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalStateSampleable.sample_goal_states

.. py:class:: GoalSampleableFromState(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalSampleableFromState

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. autodoc2-docstring:: deepxube.base.domain.GoalSampleableFromState

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.GoalSampleableFromState.__init__

   .. py:method:: sample_goal_from_state(states_start: typing.Optional[typing.List[deepxube.base.domain.S]], states_goal: typing.List[deepxube.base.domain.S]) -> typing.List[deepxube.base.domain.G]
      :canonical: deepxube.base.domain.GoalSampleableFromState.sample_goal_from_state
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalSampleableFromState.sample_goal_from_state

.. py:class:: StateSampleableFromGoal(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.StateSampleableFromGoal

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. autodoc2-docstring:: deepxube.base.domain.StateSampleableFromGoal

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.StateSampleableFromGoal.__init__

   .. py:method:: sample_state_from_goal(goals: typing.List[deepxube.base.domain.G]) -> typing.List[deepxube.base.domain.S]
      :canonical: deepxube.base.domain.StateSampleableFromGoal.sample_state_from_goal
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.StateSampleableFromGoal.sample_state_from_goal

.. py:class:: GoalFixed(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalFixed

   Bases: :py:obj:`deepxube.base.domain.GoalSampleable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. autodoc2-docstring:: deepxube.base.domain.GoalFixed

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.GoalFixed.__init__

   .. py:method:: get_goal() -> deepxube.base.domain.G
      :canonical: deepxube.base.domain.GoalFixed.get_goal
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalFixed.get_goal

   .. py:method:: sample_goals(num: int) -> typing.List[deepxube.base.domain.G]
      :canonical: deepxube.base.domain.GoalFixed.sample_goals

.. py:class:: GoalStateGoalPairSampleable(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalStateGoalPairSampleable

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: sample_goalstate_goal_pairs(num: int) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G]]
      :canonical: deepxube.base.domain.GoalStateGoalPairSampleable.sample_goalstate_goal_pairs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalStateGoalPairSampleable.sample_goalstate_goal_pairs

.. py:class:: GoalStateSampGoalSamp(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalStateSampGoalSamp

   Bases: :py:obj:`deepxube.base.domain.GoalStateGoalPairSampleable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.GoalStateSampleable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.domain.GoalStateSampGoalSamp

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.GoalStateSampGoalSamp.__init__

   .. py:method:: sample_goalstate_goal_pairs(num: int) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G]]
      :canonical: deepxube.base.domain.GoalStateSampGoalSamp.sample_goalstate_goal_pairs

.. py:class:: GoalSampGoalStateSamp(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalSampGoalStateSamp

   Bases: :py:obj:`deepxube.base.domain.GoalStateGoalPairSampleable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.GoalSampleable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.StateSampleableFromGoal`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`abc.ABC`

   .. autodoc2-docstring:: deepxube.base.domain.GoalSampGoalStateSamp

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.GoalSampGoalStateSamp.__init__

   .. py:method:: sample_goalstate_goal_pairs(num: int) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G]]
      :canonical: deepxube.base.domain.GoalSampGoalStateSamp.sample_goalstate_goal_pairs

.. py:class:: StartGoalWalkable(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.StartGoalWalkable

   Bases: :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.NodesSupervisable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.EdgesSupervisable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.EdgesSampleable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. autodoc2-docstring:: deepxube.base.domain.StartGoalWalkable

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.domain.StartGoalWalkable.__init__

   .. py:method:: sample_start_states(num_states: int) -> typing.List[deepxube.base.domain.S]
      :canonical: deepxube.base.domain.StartGoalWalkable.sample_start_states
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.StartGoalWalkable.sample_start_states

   .. py:method:: sample_problem_instances(num_steps_l: typing.List[int], times: typing.Optional[deepxube.utils.timing_utils.Times] = None) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G]]
      :canonical: deepxube.base.domain.StartGoalWalkable.sample_problem_instances

   .. py:method:: samp_nodes_and_labels(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[float]]
      :canonical: deepxube.base.domain.StartGoalWalkable.samp_nodes_and_labels

   .. py:method:: samp_edges_and_labels(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[deepxube.base.domain.A], typing.List[float]]
      :canonical: deepxube.base.domain.StartGoalWalkable.samp_edges_and_labels

   .. py:method:: samp_edges(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[deepxube.base.domain.A]]
      :canonical: deepxube.base.domain.StartGoalWalkable.samp_edges

   .. py:method:: _get_edges_and_labels(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[deepxube.base.domain.A], typing.List[float]]
      :canonical: deepxube.base.domain.StartGoalWalkable._get_edges_and_labels

      .. autodoc2-docstring:: deepxube.base.domain.StartGoalWalkable._get_edges_and_labels

.. py:class:: GoalStartRevWalkable(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalStartRevWalkable

   Bases: :py:obj:`deepxube.base.domain.GoalStateGoalPairSampleable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: sample_problem_instances(num_steps_l: typing.List[int], times: typing.Optional[deepxube.utils.timing_utils.Times] = None) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G]]
      :canonical: deepxube.base.domain.GoalStartRevWalkable.sample_problem_instances

   .. py:method:: random_walk_rev_no_path_cost(states: typing.List[deepxube.base.domain.S], num_steps_l: typing.List[int]) -> typing.List[deepxube.base.domain.S]
      :canonical: deepxube.base.domain.GoalStartRevWalkable.random_walk_rev_no_path_cost
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalStartRevWalkable.random_walk_rev_no_path_cost

.. py:class:: GoalStartRevWalkableActsRev(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalStartRevWalkableActsRev

   Bases: :py:obj:`deepxube.base.domain.GoalStartRevWalkable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.ActsRev`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.NodesSupervisable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.EdgesSupervisable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.EdgesSampleable`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`abc.ABC`

   .. py:method:: random_walk_rev_no_path_cost(states: typing.List[deepxube.base.domain.S], num_steps_l: typing.List[int]) -> typing.List[deepxube.base.domain.S]
      :canonical: deepxube.base.domain.GoalStartRevWalkableActsRev.random_walk_rev_no_path_cost

   .. py:method:: random_walk_rev(states: typing.List[deepxube.base.domain.S], num_steps_l: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[typing.List[deepxube.base.domain.A]], typing.List[float]]
      :canonical: deepxube.base.domain.GoalStartRevWalkableActsRev.random_walk_rev

      .. autodoc2-docstring:: deepxube.base.domain.GoalStartRevWalkableActsRev.random_walk_rev

   .. py:method:: samp_nodes_and_labels(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[float]]
      :canonical: deepxube.base.domain.GoalStartRevWalkableActsRev.samp_nodes_and_labels

   .. py:method:: samp_edges_and_labels(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[deepxube.base.domain.A], typing.List[float]]
      :canonical: deepxube.base.domain.GoalStartRevWalkableActsRev.samp_edges_and_labels

   .. py:method:: samp_edges(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[deepxube.base.domain.A]]
      :canonical: deepxube.base.domain.GoalStartRevWalkableActsRev.samp_edges

   .. py:method:: _get_edges_and_labels(steps_gen: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[deepxube.base.domain.G], typing.List[deepxube.base.domain.A], typing.List[float]]
      :canonical: deepxube.base.domain.GoalStartRevWalkableActsRev._get_edges_and_labels

      .. autodoc2-docstring:: deepxube.base.domain.GoalStartRevWalkableActsRev._get_edges_and_labels

.. py:class:: NextStateNP(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.NextStateNP

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: next_state(states: typing.List[deepxube.base.domain.S], actions: typing.List[deepxube.base.domain.A]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[float]]
      :canonical: deepxube.base.domain.NextStateNP.next_state

   .. py:method:: random_walk(states: typing.List[deepxube.base.domain.S], num_steps_l: typing.List[int]) -> typing.Tuple[typing.List[deepxube.base.domain.S], typing.List[typing.List[deepxube.base.domain.A]], typing.List[float]]
      :canonical: deepxube.base.domain.NextStateNP.random_walk

   .. py:method:: _states_to_np(states: typing.List[deepxube.base.domain.S]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.domain.NextStateNP._states_to_np
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.NextStateNP._states_to_np

   .. py:method:: _np_to_states(states_np_l: typing.List[numpy.typing.NDArray]) -> typing.List[deepxube.base.domain.S]
      :canonical: deepxube.base.domain.NextStateNP._np_to_states
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.NextStateNP._np_to_states

   .. py:method:: _sample_state_np_action(states_np: typing.List[numpy.typing.NDArray]) -> typing.List[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.NextStateNP._sample_state_np_action
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.NextStateNP._sample_state_np_action

   .. py:method:: _next_state_np(states_np: typing.List[numpy.typing.NDArray], actions: typing.List[deepxube.base.domain.A]) -> typing.Tuple[typing.List[numpy.typing.NDArray], typing.List[float]]
      :canonical: deepxube.base.domain.NextStateNP._next_state_np
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.NextStateNP._next_state_np

.. py:class:: NextStateNPActsFixed(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.NextStateNPActsFixed

   Bases: :py:obj:`deepxube.base.domain.NextStateNP`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.ActsFixed`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`abc.ABC`

   .. py:method:: _sample_state_np_action(states_np: typing.List[numpy.typing.NDArray]) -> typing.List[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.NextStateNPActsFixed._sample_state_np_action

      .. autodoc2-docstring:: deepxube.base.domain.NextStateNPActsFixed._sample_state_np_action

.. py:class:: NextStateNPActsEnumFixed(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.NextStateNPActsEnumFixed

   Bases: :py:obj:`deepxube.base.domain.NextStateNP`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`deepxube.base.domain.ActsEnumFixed`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`abc.ABC`

   .. py:method:: _get_state_np_actions(states_np: typing.List[numpy.typing.NDArray]) -> typing.List[typing.List[deepxube.base.domain.A]]
      :canonical: deepxube.base.domain.NextStateNPActsEnumFixed._get_state_np_actions

      .. autodoc2-docstring:: deepxube.base.domain.NextStateNPActsEnumFixed._get_state_np_actions

   .. py:method:: _sample_state_np_action(states_np: typing.List[numpy.typing.NDArray]) -> typing.List[deepxube.base.domain.A]
      :canonical: deepxube.base.domain.NextStateNPActsEnumFixed._sample_state_np_action

      .. autodoc2-docstring:: deepxube.base.domain.NextStateNPActsEnumFixed._sample_state_np_action

.. py:class:: SupportsPDDL(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.SupportsPDDL

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_pddl_domain() -> typing.List[str]
      :canonical: deepxube.base.domain.SupportsPDDL.get_pddl_domain
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.SupportsPDDL.get_pddl_domain

   .. py:method:: prob_inst_to_pddl_inst(state: deepxube.base.domain.S, goal: deepxube.base.domain.G) -> typing.List[str]
      :canonical: deepxube.base.domain.SupportsPDDL.prob_inst_to_pddl_inst
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.SupportsPDDL.prob_inst_to_pddl_inst

   .. py:method:: pddl_action_to_action(pddl_action: str) -> deepxube.base.domain.A
      :canonical: deepxube.base.domain.SupportsPDDL.pddl_action_to_action
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.SupportsPDDL.pddl_action_to_action

.. py:class:: GoalGrndAtoms(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.domain.GoalGrndAtoms

   Bases: :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ [\ :py:obj:`deepxube.base.domain.S`\ , :py:obj:`deepxube.base.domain.A`\ , :py:obj:`deepxube.base.domain.G`\ ]

   .. py:method:: state_to_model(states: typing.List[deepxube.base.domain.S]) -> typing.List[deepxube.logic.logic_objects.Model]
      :canonical: deepxube.base.domain.GoalGrndAtoms.state_to_model
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalGrndAtoms.state_to_model

   .. py:method:: model_to_state(models: typing.List[deepxube.logic.logic_objects.Model]) -> typing.List[deepxube.base.domain.S]
      :canonical: deepxube.base.domain.GoalGrndAtoms.model_to_state
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalGrndAtoms.model_to_state

   .. py:method:: goal_to_model(goals: typing.List[deepxube.base.domain.G]) -> typing.List[deepxube.logic.logic_objects.Model]
      :canonical: deepxube.base.domain.GoalGrndAtoms.goal_to_model
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalGrndAtoms.goal_to_model

   .. py:method:: model_to_goal(models: typing.List[deepxube.logic.logic_objects.Model]) -> typing.List[deepxube.base.domain.G]
      :canonical: deepxube.base.domain.GoalGrndAtoms.model_to_goal
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalGrndAtoms.model_to_goal

   .. py:method:: is_solved(states: typing.List[deepxube.base.domain.S], goals: typing.List[deepxube.base.domain.G]) -> typing.List[bool]
      :canonical: deepxube.base.domain.GoalGrndAtoms.is_solved

      .. autodoc2-docstring:: deepxube.base.domain.GoalGrndAtoms.is_solved

   .. py:method:: sample_goal_from_state(states_start: typing.Optional[typing.List[deepxube.base.domain.S]], states_goal: typing.List[deepxube.base.domain.S]) -> typing.List[deepxube.base.domain.G]
      :canonical: deepxube.base.domain.GoalGrndAtoms.sample_goal_from_state

   .. py:method:: get_bk() -> typing.List[str]
      :canonical: deepxube.base.domain.GoalGrndAtoms.get_bk
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalGrndAtoms.get_bk

   .. py:method:: get_ground_atoms() -> typing.List[deepxube.logic.logic_objects.Atom]
      :canonical: deepxube.base.domain.GoalGrndAtoms.get_ground_atoms
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalGrndAtoms.get_ground_atoms

   .. py:method:: on_model(m: clingo.solving.Model) -> deepxube.logic.logic_objects.Model
      :canonical: deepxube.base.domain.GoalGrndAtoms.on_model
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalGrndAtoms.on_model

   .. py:method:: start_state_fixed(states: typing.List[deepxube.base.domain.S]) -> typing.List[deepxube.logic.logic_objects.Model]
      :canonical: deepxube.base.domain.GoalGrndAtoms.start_state_fixed
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.domain.GoalGrndAtoms.start_state_fixed
