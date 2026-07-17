:py:mod:`deepxube.base.pathfinding`
===================================

.. py:module:: deepxube.base.pathfinding

.. autodoc2-docstring:: deepxube.base.pathfinding
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Node <deepxube.base.pathfinding.Node>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.Node
          :summary:
   * - :py:obj:`EdgeQ <deepxube.base.pathfinding.EdgeQ>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.EdgeQ
          :summary:
   * - :py:obj:`Instance <deepxube.base.pathfinding.Instance>`
     -
   * - :py:obj:`FNsHeurV <deepxube.base.pathfinding.FNsHeurV>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurV
          :summary:
   * - :py:obj:`FNsHeurQ <deepxube.base.pathfinding.FNsHeurQ>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurQ
          :summary:
   * - :py:obj:`FNsPolicy <deepxube.base.pathfinding.FNsPolicy>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNsPolicy
          :summary:
   * - :py:obj:`FNsHeurVPolicy <deepxube.base.pathfinding.FNsHeurVPolicy>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurVPolicy
          :summary:
   * - :py:obj:`FNsHeurQPolicy <deepxube.base.pathfinding.FNsHeurQPolicy>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurQPolicy
          :summary:
   * - :py:obj:`PathFind <deepxube.base.pathfinding.PathFind>`
     -
   * - :py:obj:`InstanceNode <deepxube.base.pathfinding.InstanceNode>`
     -
   * - :py:obj:`InstanceEdge <deepxube.base.pathfinding.InstanceEdge>`
     -
   * - :py:obj:`PathFindNode <deepxube.base.pathfinding.PathFindNode>`
     -
   * - :py:obj:`PathFindEdge <deepxube.base.pathfinding.PathFindEdge>`
     -
   * - :py:obj:`PathFindSetPolicy <deepxube.base.pathfinding.PathFindSetPolicy>`
     -
   * - :py:obj:`PathFindSetHeurV <deepxube.base.pathfinding.PathFindSetHeurV>`
     -
   * - :py:obj:`PathFindSetHeurQ <deepxube.base.pathfinding.PathFindSetHeurQ>`
     -
   * - :py:obj:`PathFindActsEnum <deepxube.base.pathfinding.PathFindActsEnum>`
     -
   * - :py:obj:`PathFindActsPolicy <deepxube.base.pathfinding.PathFindActsPolicy>`
     -
   * - :py:obj:`PathFindSup <deepxube.base.pathfinding.PathFindSup>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSup
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_path <deepxube.base.pathfinding.get_path>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.get_path
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`FNsHeur <deepxube.base.pathfinding.FNsHeur>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeur
          :summary:
   * - :py:obj:`I <deepxube.base.pathfinding.I>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.I
          :summary:
   * - :py:obj:`D <deepxube.base.pathfinding.D>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.D
          :summary:
   * - :py:obj:`FNs <deepxube.base.pathfinding.FNs>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNs
          :summary:
   * - :py:obj:`INode <deepxube.base.pathfinding.INode>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.INode
          :summary:
   * - :py:obj:`IEdge <deepxube.base.pathfinding.IEdge>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.IEdge
          :summary:
   * - :py:obj:`FNsP <deepxube.base.pathfinding.FNsP>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNsP
          :summary:
   * - :py:obj:`FNsHV <deepxube.base.pathfinding.FNsHV>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHV
          :summary:
   * - :py:obj:`FNsHQ <deepxube.base.pathfinding.FNsHQ>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHQ
          :summary:
   * - :py:obj:`DActsEnum <deepxube.base.pathfinding.DActsEnum>`
     - .. autodoc2-docstring:: deepxube.base.pathfinding.DActsEnum
          :summary:

API
~~~

.. py:class:: Node(state: deepxube.base.domain.State, goal: deepxube.base.domain.Goal, path_cost: float, heuristic: float, q_values: typing.Optional[typing.Tuple[typing.List[deepxube.base.domain.Action], typing.List[float]]], is_solved: typing.Optional[bool], parent_action: typing.Optional[deepxube.base.domain.Action], parent_t_cost: typing.Optional[float], parent: typing.Optional[deepxube.base.pathfinding.Node])
   :canonical: deepxube.base.pathfinding.Node

   .. autodoc2-docstring:: deepxube.base.pathfinding.Node

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.pathfinding.Node.__init__

   .. py:attribute:: __slots__
      :canonical: deepxube.base.pathfinding.Node.__slots__
      :value: ['state', 'goal', 'path_cost', 'heuristic', 'q_values', 'act_probs', 'is_solved', 'parent_action', '...

      .. autodoc2-docstring:: deepxube.base.pathfinding.Node.__slots__

   .. py:method:: add_edge(action: deepxube.base.domain.Action, t_cost: float, node_next: deepxube.base.pathfinding.Node) -> None
      :canonical: deepxube.base.pathfinding.Node.add_edge

      .. autodoc2-docstring:: deepxube.base.pathfinding.Node.add_edge

   .. py:method:: bellman_backup() -> float
      :canonical: deepxube.base.pathfinding.Node.bellman_backup

      .. autodoc2-docstring:: deepxube.base.pathfinding.Node.bellman_backup

   .. py:method:: upper_bound_parent_path(ctg_ub: float) -> None
      :canonical: deepxube.base.pathfinding.Node.upper_bound_parent_path

      .. autodoc2-docstring:: deepxube.base.pathfinding.Node.upper_bound_parent_path

   .. py:method:: tree_backup() -> float
      :canonical: deepxube.base.pathfinding.Node.tree_backup

      .. autodoc2-docstring:: deepxube.base.pathfinding.Node.tree_backup

   .. py:method:: backup_act(action: deepxube.base.domain.Action) -> float
      :canonical: deepxube.base.pathfinding.Node.backup_act

      .. autodoc2-docstring:: deepxube.base.pathfinding.Node.backup_act

   .. py:method:: get_all_descendants() -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.base.pathfinding.Node.get_all_descendants

      .. autodoc2-docstring:: deepxube.base.pathfinding.Node.get_all_descendants

.. py:function:: get_path(node: deepxube.base.pathfinding.Node) -> typing.Tuple[typing.List[deepxube.base.domain.State], typing.List[deepxube.base.domain.Action], typing.List[float], float]
   :canonical: deepxube.base.pathfinding.get_path

   .. autodoc2-docstring:: deepxube.base.pathfinding.get_path

.. py:class:: EdgeQ(node: deepxube.base.pathfinding.Node, action: deepxube.base.domain.Action, q_val: float)
   :canonical: deepxube.base.pathfinding.EdgeQ

   .. autodoc2-docstring:: deepxube.base.pathfinding.EdgeQ

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.pathfinding.EdgeQ.__init__

   .. py:attribute:: __slots__
      :canonical: deepxube.base.pathfinding.EdgeQ.__slots__
      :value: ['node', 'action', 'q_val']

      .. autodoc2-docstring:: deepxube.base.pathfinding.EdgeQ.__slots__

.. py:class:: Instance(root_node: deepxube.base.pathfinding.Node, inst_info: typing.Any)
   :canonical: deepxube.base.pathfinding.Instance

   Bases: :py:obj:`abc.ABC`

   .. py:method:: frontier_size() -> int
      :canonical: deepxube.base.pathfinding.Instance.frontier_size
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.frontier_size

   .. py:method:: get_nodes() -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.base.pathfinding.Instance.get_nodes

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.get_nodes

   .. py:method:: set_next_nodes(nodes_next: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.base.pathfinding.Instance.set_next_nodes

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.set_next_nodes

   .. py:method:: record_goal(nodes: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.base.pathfinding.Instance.record_goal
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.record_goal

   .. py:method:: add_nodes_popped(nodes_popped: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.base.pathfinding.Instance.add_nodes_popped

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.add_nodes_popped

   .. py:method:: get_nodes_popped() -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.base.pathfinding.Instance.get_nodes_popped

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.get_nodes_popped

   .. py:method:: add_edges_popped(edges_popped: typing.List[deepxube.base.pathfinding.EdgeQ]) -> None
      :canonical: deepxube.base.pathfinding.Instance.add_edges_popped

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.add_edges_popped

   .. py:method:: get_edges_popped() -> typing.List[deepxube.base.pathfinding.EdgeQ]
      :canonical: deepxube.base.pathfinding.Instance.get_edges_popped

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.get_edges_popped

   .. py:method:: has_soln() -> bool
      :canonical: deepxube.base.pathfinding.Instance.has_soln

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.has_soln

   .. py:method:: path_cost() -> float
      :canonical: deepxube.base.pathfinding.Instance.path_cost

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.path_cost

   .. py:method:: finished() -> bool
      :canonical: deepxube.base.pathfinding.Instance.finished
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.Instance.finished

.. py:class:: FNsHeurV
   :canonical: deepxube.base.pathfinding.FNsHeurV

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurV

   .. py:attribute:: heur_fn_v
      :canonical: deepxube.base.pathfinding.FNsHeurV.heur_fn_v
      :type: deepxube.base.heuristic.HeurFnV
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurV.heur_fn_v

.. py:class:: FNsHeurQ
   :canonical: deepxube.base.pathfinding.FNsHeurQ

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurQ

   .. py:attribute:: heur_fn_q
      :canonical: deepxube.base.pathfinding.FNsHeurQ.heur_fn_q
      :type: deepxube.base.heuristic.HeurFnQ
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurQ.heur_fn_q

.. py:data:: FNsHeur
   :canonical: deepxube.base.pathfinding.FNsHeur
   :value: None

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeur

.. py:class:: FNsPolicy
   :canonical: deepxube.base.pathfinding.FNsPolicy

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNsPolicy

   .. py:attribute:: policy_fn
      :canonical: deepxube.base.pathfinding.FNsPolicy.policy_fn
      :type: deepxube.base.heuristic.PolicyFn
      :value: None

      .. autodoc2-docstring:: deepxube.base.pathfinding.FNsPolicy.policy_fn

.. py:class:: FNsHeurVPolicy
   :canonical: deepxube.base.pathfinding.FNsHeurVPolicy

   Bases: :py:obj:`deepxube.base.pathfinding.FNsPolicy`, :py:obj:`deepxube.base.pathfinding.FNsHeurV`

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurVPolicy

.. py:class:: FNsHeurQPolicy
   :canonical: deepxube.base.pathfinding.FNsHeurQPolicy

   Bases: :py:obj:`deepxube.base.pathfinding.FNsPolicy`, :py:obj:`deepxube.base.pathfinding.FNsHeurQ`

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHeurQPolicy

.. py:data:: I
   :canonical: deepxube.base.pathfinding.I
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfinding.I

.. py:data:: D
   :canonical: deepxube.base.pathfinding.D
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfinding.D

.. py:data:: FNs
   :canonical: deepxube.base.pathfinding.FNs
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNs

.. py:class:: PathFind(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.base.pathfinding.PathFind

   Bases: :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.pathfinding.D`\ , :py:obj:`deepxube.base.pathfinding.FNs`\ , :py:obj:`deepxube.base.pathfinding.I`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.pathfinding.D]
      :canonical: deepxube.base.pathfinding.PathFind.domain_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.domain_type

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNs]
      :canonical: deepxube.base.pathfinding.PathFind.functions_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.functions_type

   .. py:method:: make_instances(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], inst_infos: typing.Optional[typing.List[typing.Any]] = None, compute_root_vals: bool = True) -> typing.List[deepxube.base.pathfinding.I]
      :canonical: deepxube.base.pathfinding.PathFind.make_instances
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.make_instances

   .. py:method:: add_instances(instances: typing.List[deepxube.base.pathfinding.I]) -> None
      :canonical: deepxube.base.pathfinding.PathFind.add_instances

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.add_instances

   .. py:method:: expand_states(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.State]], typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
      :canonical: deepxube.base.pathfinding.PathFind.expand_states
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.expand_states

   .. py:method:: get_state_actions(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[typing.List[deepxube.base.domain.Action]]
      :canonical: deepxube.base.pathfinding.PathFind.get_state_actions
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.get_state_actions

   .. py:method:: step(verbose: bool = False) -> typing.Tuple[typing.List[deepxube.base.pathfinding.Node], typing.List[deepxube.base.pathfinding.EdgeQ]]
      :canonical: deepxube.base.pathfinding.PathFind.step
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.step

   .. py:method:: remove_finished_instances(itr_max: int) -> typing.List[deepxube.base.pathfinding.I]
      :canonical: deepxube.base.pathfinding.PathFind.remove_finished_instances

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.remove_finished_instances

   .. py:method:: remove_instances(test_rem: typing.Callable[[deepxube.base.pathfinding.I], bool]) -> typing.List[deepxube.base.pathfinding.I]
      :canonical: deepxube.base.pathfinding.PathFind.remove_instances

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.remove_instances

   .. py:method:: set_is_solved(nodes: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.base.pathfinding.PathFind.set_is_solved

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind.set_is_solved

   .. py:method:: _set_node_vals(nodes: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.base.pathfinding.PathFind._set_node_vals
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind._set_node_vals

   .. py:method:: _create_root_nodes(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], compute_root_vals: bool) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.base.pathfinding.PathFind._create_root_nodes

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind._create_root_nodes

   .. py:method:: _verbose(instances: typing.List[deepxube.base.pathfinding.I], nodes_by_inst: typing.List[typing.List[deepxube.base.pathfinding.Node]]) -> None
      :canonical: deepxube.base.pathfinding.PathFind._verbose

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFind._verbose

.. py:class:: InstanceNode(root_node: deepxube.base.pathfinding.Node, inst_info: typing.Any)
   :canonical: deepxube.base.pathfinding.InstanceNode

   Bases: :py:obj:`deepxube.base.pathfinding.Instance`, :py:obj:`abc.ABC`

   .. py:method:: filter_expanded_nodes(nodes: typing.List[deepxube.base.pathfinding.Node]) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.base.pathfinding.InstanceNode.filter_expanded_nodes
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.InstanceNode.filter_expanded_nodes

   .. py:method:: push_pop_nodes(nodes: typing.List[deepxube.base.pathfinding.Node], costs: typing.List[float]) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.base.pathfinding.InstanceNode.push_pop_nodes
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.InstanceNode.push_pop_nodes

.. py:data:: INode
   :canonical: deepxube.base.pathfinding.INode
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfinding.INode

.. py:class:: InstanceEdge(root_node: deepxube.base.pathfinding.Node, inst_info: typing.Any)
   :canonical: deepxube.base.pathfinding.InstanceEdge

   Bases: :py:obj:`deepxube.base.pathfinding.Instance`, :py:obj:`abc.ABC`

   .. py:method:: filter_popped_nodes(nodes: typing.List[deepxube.base.pathfinding.Node]) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.base.pathfinding.InstanceEdge.filter_popped_nodes
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.InstanceEdge.filter_popped_nodes

   .. py:method:: push_pop_edges(edges: typing.List[deepxube.base.pathfinding.EdgeQ], costs: typing.List[float]) -> typing.List[deepxube.base.pathfinding.EdgeQ]
      :canonical: deepxube.base.pathfinding.InstanceEdge.push_pop_edges
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.InstanceEdge.push_pop_edges

.. py:data:: IEdge
   :canonical: deepxube.base.pathfinding.IEdge
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfinding.IEdge

.. py:class:: PathFindNode(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.base.pathfinding.PathFindNode

   Bases: :py:obj:`deepxube.base.pathfinding.PathFind`\ [\ :py:obj:`deepxube.base.pathfinding.D`\ , :py:obj:`deepxube.base.pathfinding.FNs`\ , :py:obj:`deepxube.base.pathfinding.INode`\ ]

   .. py:method:: step(verbose: bool = False) -> typing.Tuple[typing.List[deepxube.base.pathfinding.Node], typing.List[deepxube.base.pathfinding.EdgeQ]]
      :canonical: deepxube.base.pathfinding.PathFindNode.step

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindNode.step

   .. py:method:: _expand(instances: typing.List[deepxube.base.pathfinding.INode], nodes_by_inst: typing.List[typing.List[deepxube.base.pathfinding.Node]]) -> typing.List[typing.List[deepxube.base.pathfinding.Node]]
      :canonical: deepxube.base.pathfinding.PathFindNode._expand

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindNode._expand

   .. py:method:: _compute_costs(instances: typing.List[deepxube.base.pathfinding.INode], nodes_by_inst: typing.List[typing.List[deepxube.base.pathfinding.Node]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.base.pathfinding.PathFindNode._compute_costs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindNode._compute_costs

.. py:class:: PathFindEdge(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.base.pathfinding.PathFindEdge

   Bases: :py:obj:`deepxube.base.pathfinding.PathFind`\ [\ :py:obj:`deepxube.base.pathfinding.D`\ , :py:obj:`deepxube.base.pathfinding.FNs`\ , :py:obj:`deepxube.base.pathfinding.IEdge`\ ]

   .. py:method:: step(verbose: bool = False) -> typing.Tuple[typing.List[deepxube.base.pathfinding.Node], typing.List[deepxube.base.pathfinding.EdgeQ]]
      :canonical: deepxube.base.pathfinding.PathFindEdge.step

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindEdge.step

   .. py:method:: get_next_nodes(instances: typing.List[deepxube.base.pathfinding.IEdge], edges_by_inst: typing.List[typing.List[deepxube.base.pathfinding.EdgeQ]]) -> typing.List[typing.List[deepxube.base.pathfinding.Node]]
      :canonical: deepxube.base.pathfinding.PathFindEdge.get_next_nodes

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindEdge.get_next_nodes

   .. py:method:: _get_edges(nodes_by_inst: typing.List[typing.List[deepxube.base.pathfinding.Node]]) -> typing.List[typing.List[deepxube.base.pathfinding.EdgeQ]]
      :canonical: deepxube.base.pathfinding.PathFindEdge._get_edges

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindEdge._get_edges

   .. py:method:: _compute_costs(instances: typing.List[deepxube.base.pathfinding.IEdge], edges_by_inst: typing.List[typing.List[deepxube.base.pathfinding.EdgeQ]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.base.pathfinding.PathFindEdge._compute_costs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindEdge._compute_costs

.. py:data:: FNsP
   :canonical: deepxube.base.pathfinding.FNsP
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNsP

.. py:data:: FNsHV
   :canonical: deepxube.base.pathfinding.FNsHV
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHV

.. py:data:: FNsHQ
   :canonical: deepxube.base.pathfinding.FNsHQ
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfinding.FNsHQ

.. py:class:: PathFindSetPolicy(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.base.pathfinding.PathFindSetPolicy

   Bases: :py:obj:`deepxube.base.pathfinding.PathFind`\ [\ :py:obj:`deepxube.base.pathfinding.D`\ , :py:obj:`deepxube.base.pathfinding.FNsP`\ , :py:obj:`deepxube.base.pathfinding.I`\ ], :py:obj:`abc.ABC`

   .. py:method:: _set_node_vals(nodes: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.base.pathfinding.PathFindSetPolicy._set_node_vals

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSetPolicy._set_node_vals

.. py:class:: PathFindSetHeurV(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.base.pathfinding.PathFindSetHeurV

   Bases: :py:obj:`deepxube.base.pathfinding.PathFind`\ [\ :py:obj:`deepxube.base.pathfinding.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHV`\ , :py:obj:`deepxube.base.pathfinding.I`\ ], :py:obj:`abc.ABC`

   .. py:method:: _set_node_vals(nodes: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.base.pathfinding.PathFindSetHeurV._set_node_vals

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSetHeurV._set_node_vals

.. py:class:: PathFindSetHeurQ(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.base.pathfinding.PathFindSetHeurQ

   Bases: :py:obj:`deepxube.base.pathfinding.PathFind`\ [\ :py:obj:`deepxube.base.pathfinding.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHQ`\ , :py:obj:`deepxube.base.pathfinding.I`\ ], :py:obj:`abc.ABC`

   .. py:method:: _set_node_vals(nodes: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.base.pathfinding.PathFindSetHeurQ._set_node_vals

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSetHeurQ._set_node_vals

.. py:data:: DActsEnum
   :canonical: deepxube.base.pathfinding.DActsEnum
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.pathfinding.DActsEnum

.. py:class:: PathFindActsEnum(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.base.pathfinding.PathFindActsEnum

   Bases: :py:obj:`deepxube.base.pathfinding.PathFind`\ [\ :py:obj:`deepxube.base.pathfinding.DActsEnum`\ , :py:obj:`deepxube.base.pathfinding.FNs`\ , :py:obj:`deepxube.base.pathfinding.I`\ ], :py:obj:`abc.ABC`

   .. py:method:: expand_states(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.State]], typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
      :canonical: deepxube.base.pathfinding.PathFindActsEnum.expand_states

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindActsEnum.expand_states

   .. py:method:: get_state_actions(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[typing.List[deepxube.base.domain.Action]]
      :canonical: deepxube.base.pathfinding.PathFindActsEnum.get_state_actions

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindActsEnum.get_state_actions

.. py:class:: PathFindActsPolicy(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.base.pathfinding.PathFindActsPolicy

   Bases: :py:obj:`deepxube.base.pathfinding.PathFind`\ [\ :py:obj:`deepxube.base.pathfinding.D`\ , :py:obj:`deepxube.base.pathfinding.FNsP`\ , :py:obj:`deepxube.base.pathfinding.I`\ ], :py:obj:`abc.ABC`

   .. py:property:: num_rand_edges
      :canonical: deepxube.base.pathfinding.PathFindActsPolicy.num_rand_edges
      :abstractmethod:
      :type: int

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindActsPolicy.num_rand_edges

   .. py:method:: expand_states(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.State]], typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
      :canonical: deepxube.base.pathfinding.PathFindActsPolicy.expand_states

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindActsPolicy.expand_states

   .. py:method:: get_state_actions(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[typing.List[deepxube.base.domain.Action]]
      :canonical: deepxube.base.pathfinding.PathFindActsPolicy.get_state_actions

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindActsPolicy.get_state_actions

   .. py:method:: _get_actions(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[typing.List[deepxube.base.domain.Action]]
      :canonical: deepxube.base.pathfinding.PathFindActsPolicy._get_actions

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindActsPolicy._get_actions

.. py:class:: PathFindSup(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.base.pathfinding.PathFindSup

   Bases: :py:obj:`deepxube.base.pathfinding.PathFind`\ [\ :py:obj:`deepxube.base.pathfinding.D`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.base.pathfinding.I`\ ]

   .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSup

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSup.__init__

   .. py:method:: functions_type() -> typing.Type[typing.Any]
      :canonical: deepxube.base.pathfinding.PathFindSup.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSup.functions_type

   .. py:method:: make_instances(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], inst_infos: typing.Optional[typing.List[typing.Any]] = None, compute_root_vals: bool = True) -> typing.List[deepxube.base.pathfinding.I]
      :canonical: deepxube.base.pathfinding.PathFindSup.make_instances
      :abstractmethod:

   .. py:method:: expand_states(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.State]], typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
      :canonical: deepxube.base.pathfinding.PathFindSup.expand_states
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSup.expand_states

   .. py:method:: get_state_actions(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal]) -> typing.List[typing.List[deepxube.base.domain.Action]]
      :canonical: deepxube.base.pathfinding.PathFindSup.get_state_actions
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSup.get_state_actions

   .. py:method:: make_instances_sup(steps_gen: typing.List[int], inst_infos: typing.Optional[typing.List[typing.Any]]) -> typing.List[deepxube.base.pathfinding.I]
      :canonical: deepxube.base.pathfinding.PathFindSup.make_instances_sup
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSup.make_instances_sup

   .. py:method:: _set_node_vals(nodes: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.base.pathfinding.PathFindSup._set_node_vals
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.pathfinding.PathFindSup._set_node_vals

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.pathfinding.PathFindSup.__repr__
