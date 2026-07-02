:py:mod:`deepxube.base.nnet_input`
==================================

.. py:module:: deepxube.base.nnet_input

.. autodoc2-docstring:: deepxube.base.nnet_input
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NNetInput <deepxube.base.nnet_input.NNetInput>`
     - .. autodoc2-docstring:: deepxube.base.nnet_input.NNetInput
          :summary:
   * - :py:obj:`FlatIn <deepxube.base.nnet_input.FlatIn>`
     -
   * - :py:obj:`TwoDIn <deepxube.base.nnet_input.TwoDIn>`
     -
   * - :py:obj:`StateGoalIn <deepxube.base.nnet_input.StateGoalIn>`
     -
   * - :py:obj:`StateGoalActFixIn <deepxube.base.nnet_input.StateGoalActFixIn>`
     -
   * - :py:obj:`StateGoalActIn <deepxube.base.nnet_input.StateGoalActIn>`
     -
   * - :py:obj:`PolicyNNetIn <deepxube.base.nnet_input.PolicyNNetIn>`
     -
   * - :py:obj:`FlatInPolicy <deepxube.base.nnet_input.FlatInPolicy>`
     -
   * - :py:obj:`DynamicNNetInput <deepxube.base.nnet_input.DynamicNNetInput>`
     -
   * - :py:obj:`HasFlatSGIn <deepxube.base.nnet_input.HasFlatSGIn>`
     - .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGIn
          :summary:
   * - :py:obj:`HasActsEnumFixedIn <deepxube.base.nnet_input.HasActsEnumFixedIn>`
     -
   * - :py:obj:`HasFlatSGActsEnumFixedIn <deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn>`
     -
   * - :py:obj:`HasFlatSGAIn <deepxube.base.nnet_input.HasFlatSGAIn>`
     -
   * - :py:obj:`HasTwoDSGIn <deepxube.base.nnet_input.HasTwoDSGIn>`
     - .. autodoc2-docstring:: deepxube.base.nnet_input.HasTwoDSGIn
          :summary:
   * - :py:obj:`HasTwoDSGActsEnumFixedIn <deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`D <deepxube.base.nnet_input.D>`
     - .. autodoc2-docstring:: deepxube.base.nnet_input.D
          :summary:
   * - :py:obj:`S <deepxube.base.nnet_input.S>`
     - .. autodoc2-docstring:: deepxube.base.nnet_input.S
          :summary:
   * - :py:obj:`A <deepxube.base.nnet_input.A>`
     - .. autodoc2-docstring:: deepxube.base.nnet_input.A
          :summary:
   * - :py:obj:`G <deepxube.base.nnet_input.G>`
     - .. autodoc2-docstring:: deepxube.base.nnet_input.G
          :summary:

API
~~~

.. py:data:: D
   :canonical: deepxube.base.nnet_input.D
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.nnet_input.D

.. py:data:: S
   :canonical: deepxube.base.nnet_input.S
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.nnet_input.S

.. py:data:: A
   :canonical: deepxube.base.nnet_input.A
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.nnet_input.A

.. py:data:: G
   :canonical: deepxube.base.nnet_input.G
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.nnet_input.G

.. py:class:: NNetInput(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.base.nnet_input.NNetInput

   Bases: :py:obj:`abc.ABC`, :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ ]

   .. autodoc2-docstring:: deepxube.base.nnet_input.NNetInput

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.nnet_input.NNetInput.__init__

   .. py:method:: get_input_info() -> typing.Any
      :canonical: deepxube.base.nnet_input.NNetInput.get_input_info
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.NNetInput.get_input_info

   .. py:method:: to_np(*args: typing.Any) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.nnet_input.NNetInput.to_np
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.NNetInput.to_np

.. py:class:: FlatIn(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.base.nnet_input.FlatIn

   Bases: :py:obj:`deepxube.base.nnet_input.NNetInput`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ ]

   .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.base.nnet_input.FlatIn.get_input_info
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.FlatIn.get_input_info

.. py:class:: TwoDIn(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.base.nnet_input.TwoDIn

   Bases: :py:obj:`deepxube.base.nnet_input.NNetInput`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ ]

   .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.Tuple[int, int], typing.List[int], typing.Optional[int]]
      :canonical: deepxube.base.nnet_input.TwoDIn.get_input_info
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.TwoDIn.get_input_info

.. py:class:: StateGoalIn(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.base.nnet_input.StateGoalIn

   Bases: :py:obj:`deepxube.base.nnet_input.NNetInput`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ ], :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ , :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.G`\ ]

   .. py:method:: to_np(states: typing.List[deepxube.base.nnet_input.S], goals: typing.List[deepxube.base.nnet_input.G]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.nnet_input.StateGoalIn.to_np
      :abstractmethod:

.. py:class:: StateGoalActFixIn(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.base.nnet_input.StateGoalActFixIn

   Bases: :py:obj:`deepxube.base.nnet_input.NNetInput`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ ], :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ , :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.G`\ , :py:obj:`deepxube.base.nnet_input.A`\ ]

   .. py:method:: to_np(states: typing.List[deepxube.base.nnet_input.S], goals: typing.List[deepxube.base.nnet_input.G], actions_l: typing.List[typing.List[deepxube.base.nnet_input.A]]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.nnet_input.StateGoalActFixIn.to_np
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.StateGoalActFixIn.to_np

.. py:class:: StateGoalActIn(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.base.nnet_input.StateGoalActIn

   Bases: :py:obj:`deepxube.base.nnet_input.NNetInput`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ ], :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ , :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.G`\ , :py:obj:`deepxube.base.nnet_input.A`\ ]

   .. py:method:: to_np(states: typing.List[deepxube.base.nnet_input.S], goals: typing.List[deepxube.base.nnet_input.G], actions: typing.List[deepxube.base.nnet_input.A]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.nnet_input.StateGoalActIn.to_np
      :abstractmethod:

.. py:class:: PolicyNNetIn(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.base.nnet_input.PolicyNNetIn

   Bases: :py:obj:`deepxube.base.nnet_input.NNetInput`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ ], :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ , :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.G`\ , :py:obj:`deepxube.base.nnet_input.A`\ ]

   .. py:method:: to_np(states: typing.List[deepxube.base.nnet_input.S], goals: typing.List[deepxube.base.nnet_input.G], actions: typing.List[deepxube.base.nnet_input.A]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.nnet_input.PolicyNNetIn.to_np
      :abstractmethod:

   .. py:method:: to_np_fn(states: typing.List[deepxube.base.nnet_input.S], goals: typing.List[deepxube.base.nnet_input.G]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.nnet_input.PolicyNNetIn.to_np_fn
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.PolicyNNetIn.to_np_fn

   .. py:method:: nnet_out_to_actions(nnet_out: typing.List[numpy.typing.NDArray[numpy.float64]]) -> typing.List[deepxube.base.nnet_input.A]
      :canonical: deepxube.base.nnet_input.PolicyNNetIn.nnet_out_to_actions
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.PolicyNNetIn.nnet_out_to_actions

   .. py:method:: states_goals_actions_split_idx() -> int
      :canonical: deepxube.base.nnet_input.PolicyNNetIn.states_goals_actions_split_idx
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.PolicyNNetIn.states_goals_actions_split_idx

.. py:class:: FlatInPolicy(domain: deepxube.base.nnet_input.D)
   :canonical: deepxube.base.nnet_input.FlatInPolicy

   Bases: :py:obj:`deepxube.base.nnet_input.FlatIn`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ ], :py:obj:`deepxube.base.nnet_input.PolicyNNetIn`\ [\ :py:obj:`deepxube.base.nnet_input.D`\ , :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.G`\ , :py:obj:`deepxube.base.nnet_input.A`\ ], :py:obj:`abc.ABC`

.. py:class:: DynamicNNetInput(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet_input.DynamicNNetInput

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.A`\ , :py:obj:`deepxube.base.nnet_input.G`\ ], :py:obj:`abc.ABC`

   .. py:attribute:: _nnet_input_register
      :canonical: deepxube.base.nnet_input.DynamicNNetInput._nnet_input_register
      :type: typing.ClassVar[typing.Dict[str, typing.Type[deepxube.base.nnet_input.NNetInput]]]
      :value: 'dict(...)'

      .. autodoc2-docstring:: deepxube.base.nnet_input.DynamicNNetInput._nnet_input_register

   .. py:method:: __init_subclass__(**kwargs: typing.Any)
      :canonical: deepxube.base.nnet_input.DynamicNNetInput.__init_subclass__
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.DynamicNNetInput.__init_subclass__

   .. py:method:: register_nnet_input(nnet_input_t: typing.Type[deepxube.base.nnet_input.NNetInput], nnet_input_name: str) -> None
      :canonical: deepxube.base.nnet_input.DynamicNNetInput.register_nnet_input
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.DynamicNNetInput.register_nnet_input

   .. py:method:: get_dynamic_nnet_inputs() -> typing.Dict[str, typing.Type[deepxube.base.nnet_input.NNetInput]]
      :canonical: deepxube.base.nnet_input.DynamicNNetInput.get_dynamic_nnet_inputs
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.DynamicNNetInput.get_dynamic_nnet_inputs

.. py:class:: HasFlatSGIn(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet_input.HasFlatSGIn

   Bases: :py:obj:`deepxube.base.nnet_input.DynamicNNetInput`\ [\ :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.A`\ , :py:obj:`deepxube.base.nnet_input.G`\ ]

   .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGIn

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGIn.__init__

   .. py:class:: FlatSGConcrete(domain: deepxube.base.nnet_input.HasFlatSGIn)
      :canonical: deepxube.base.nnet_input.HasFlatSGIn.FlatSGConcrete

      Bases: :py:obj:`deepxube.base.nnet_input.FlatIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasFlatSGIn`\ ], :py:obj:`deepxube.base.nnet_input.StateGoalIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasFlatSGIn`\ , :py:obj:`deepxube.base.domain.State`\ , :py:obj:`deepxube.base.domain.Goal`\ ]

      .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.List[int]]
         :canonical: deepxube.base.nnet_input.HasFlatSGIn.FlatSGConcrete.get_input_info

      .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[numpy.typing.NDArray]
         :canonical: deepxube.base.nnet_input.HasFlatSGIn.FlatSGConcrete.to_np

   .. py:method:: __init_subclass__(**kwargs: typing.Any) -> None
      :canonical: deepxube.base.nnet_input.HasFlatSGIn.__init_subclass__
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGIn.__init_subclass__

   .. py:method:: get_input_info_flat_sg() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.base.nnet_input.HasFlatSGIn.get_input_info_flat_sg
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGIn.get_input_info_flat_sg

   .. py:method:: to_np_flat_sg(states: typing.List[deepxube.base.nnet_input.S], goals: typing.List[deepxube.base.nnet_input.G]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.nnet_input.HasFlatSGIn.to_np_flat_sg
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGIn.to_np_flat_sg

.. py:class:: HasActsEnumFixedIn(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet_input.HasActsEnumFixedIn

   Bases: :py:obj:`deepxube.base.domain.Domain`\ [\ :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.A`\ , :py:obj:`deepxube.base.nnet_input.G`\ ]

   .. py:method:: actions_to_indices(actions: typing.List[deepxube.base.nnet_input.A]) -> typing.List[int]
      :canonical: deepxube.base.nnet_input.HasActsEnumFixedIn.actions_to_indices
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasActsEnumFixedIn.actions_to_indices

.. py:class:: HasFlatSGActsEnumFixedIn(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn

   Bases: :py:obj:`deepxube.base.nnet_input.HasFlatSGIn`\ [\ :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.A`\ , :py:obj:`deepxube.base.nnet_input.G`\ ], :py:obj:`deepxube.base.nnet_input.HasActsEnumFixedIn`\ [\ :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.A`\ , :py:obj:`deepxube.base.nnet_input.G`\ ], :py:obj:`abc.ABC`

   .. py:class:: FlatSGActFixConcrete(domain: deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn)
      :canonical: deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn.FlatSGActFixConcrete

      Bases: :py:obj:`deepxube.base.nnet_input.FlatIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn`\ ], :py:obj:`deepxube.base.nnet_input.StateGoalActFixIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn`\ , :py:obj:`deepxube.base.domain.State`\ , :py:obj:`deepxube.base.domain.Goal`\ , :py:obj:`deepxube.base.domain.Action`\ ]

      .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.List[int]]
         :canonical: deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn.FlatSGActFixConcrete.get_input_info

      .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[numpy.typing.NDArray]
         :canonical: deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn.FlatSGActFixConcrete.to_np

   .. py:method:: __init_subclass__(**kwargs: typing.Any) -> None
      :canonical: deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn.__init_subclass__
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGActsEnumFixedIn.__init_subclass__

.. py:class:: HasFlatSGAIn(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet_input.HasFlatSGAIn

   Bases: :py:obj:`deepxube.base.nnet_input.DynamicNNetInput`\ [\ :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.A`\ , :py:obj:`deepxube.base.nnet_input.G`\ ]

   .. py:class:: FlatSGAConcrete(domain: deepxube.base.nnet_input.HasFlatSGAIn)
      :canonical: deepxube.base.nnet_input.HasFlatSGAIn.FlatSGAConcrete

      Bases: :py:obj:`deepxube.base.nnet_input.FlatIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasFlatSGAIn`\ ], :py:obj:`deepxube.base.nnet_input.StateGoalActIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasFlatSGAIn`\ , :py:obj:`deepxube.base.domain.State`\ , :py:obj:`deepxube.base.domain.Goal`\ , :py:obj:`deepxube.base.domain.Action`\ ]

      .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.List[int]]
         :canonical: deepxube.base.nnet_input.HasFlatSGAIn.FlatSGAConcrete.get_input_info

      .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions: typing.List[deepxube.base.domain.Action]) -> typing.List[numpy.typing.NDArray]
         :canonical: deepxube.base.nnet_input.HasFlatSGAIn.FlatSGAConcrete.to_np

   .. py:method:: __init_subclass__(**kwargs: typing.Any) -> None
      :canonical: deepxube.base.nnet_input.HasFlatSGAIn.__init_subclass__
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGAIn.__init_subclass__

   .. py:method:: get_input_info_flat_sga() -> typing.Tuple[typing.List[int], typing.List[int]]
      :canonical: deepxube.base.nnet_input.HasFlatSGAIn.get_input_info_flat_sga
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGAIn.get_input_info_flat_sga

   .. py:method:: to_np_flat_sga(states: typing.List[deepxube.base.nnet_input.S], goals: typing.List[deepxube.base.nnet_input.G], actions: typing.List[deepxube.base.nnet_input.A]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.nnet_input.HasFlatSGAIn.to_np_flat_sga
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasFlatSGAIn.to_np_flat_sga

.. py:class:: HasTwoDSGIn(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet_input.HasTwoDSGIn

   Bases: :py:obj:`deepxube.base.nnet_input.DynamicNNetInput`\ [\ :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.A`\ , :py:obj:`deepxube.base.nnet_input.G`\ ]

   .. autodoc2-docstring:: deepxube.base.nnet_input.HasTwoDSGIn

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.nnet_input.HasTwoDSGIn.__init__

   .. py:class:: TwoDSGConcrete(domain: deepxube.base.nnet_input.HasTwoDSGIn)
      :canonical: deepxube.base.nnet_input.HasTwoDSGIn.TwoDSGConcrete

      Bases: :py:obj:`deepxube.base.nnet_input.TwoDIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasTwoDSGIn`\ ], :py:obj:`deepxube.base.nnet_input.StateGoalIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasTwoDSGIn`\ , :py:obj:`deepxube.base.domain.State`\ , :py:obj:`deepxube.base.domain.Goal`\ ]

      .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.Tuple[int, int], typing.List[int], typing.Optional[int]]
         :canonical: deepxube.base.nnet_input.HasTwoDSGIn.TwoDSGConcrete.get_input_info

      .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[numpy.typing.NDArray]
         :canonical: deepxube.base.nnet_input.HasTwoDSGIn.TwoDSGConcrete.to_np

   .. py:method:: __init_subclass__(**kwargs: typing.Any) -> None
      :canonical: deepxube.base.nnet_input.HasTwoDSGIn.__init_subclass__
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasTwoDSGIn.__init_subclass__

   .. py:method:: get_input_info_2d_sg() -> typing.Tuple[typing.List[int], typing.Tuple[int, int], typing.List[int], typing.Optional[int]]
      :canonical: deepxube.base.nnet_input.HasTwoDSGIn.get_input_info_2d_sg
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasTwoDSGIn.get_input_info_2d_sg

   .. py:method:: to_np_2d_sg(states: typing.List[deepxube.base.nnet_input.S], goals: typing.List[deepxube.base.nnet_input.G]) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.nnet_input.HasTwoDSGIn.to_np_2d_sg
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasTwoDSGIn.to_np_2d_sg

.. py:class:: HasTwoDSGActsEnumFixedIn(*args: typing.Any, **kwargs: typing.Any)
   :canonical: deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn

   Bases: :py:obj:`deepxube.base.nnet_input.HasTwoDSGIn`\ [\ :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.A`\ , :py:obj:`deepxube.base.nnet_input.G`\ ], :py:obj:`deepxube.base.nnet_input.HasActsEnumFixedIn`\ [\ :py:obj:`deepxube.base.nnet_input.S`\ , :py:obj:`deepxube.base.nnet_input.A`\ , :py:obj:`deepxube.base.nnet_input.G`\ ], :py:obj:`abc.ABC`

   .. py:class:: TwoDSGActFixConcrete(domain: deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn)
      :canonical: deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn.TwoDSGActFixConcrete

      Bases: :py:obj:`deepxube.base.nnet_input.TwoDIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn`\ ], :py:obj:`deepxube.base.nnet_input.StateGoalActFixIn`\ [\ :py:obj:`deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn`\ , :py:obj:`deepxube.base.domain.State`\ , :py:obj:`deepxube.base.domain.Goal`\ , :py:obj:`deepxube.base.domain.Action`\ ]

      .. py:method:: get_input_info() -> typing.Tuple[typing.List[int], typing.Tuple[int, int], typing.List[int], typing.Optional[int]]
         :canonical: deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn.TwoDSGActFixConcrete.get_input_info

      .. py:method:: to_np(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> typing.List[numpy.typing.NDArray]
         :canonical: deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn.TwoDSGActFixConcrete.to_np

   .. py:method:: __init_subclass__(**kwargs: typing.Any) -> None
      :canonical: deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn.__init_subclass__
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.nnet_input.HasTwoDSGActsEnumFixedIn.__init_subclass__
