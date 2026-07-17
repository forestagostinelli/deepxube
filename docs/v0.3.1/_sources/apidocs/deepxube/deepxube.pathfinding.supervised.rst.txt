:py:mod:`deepxube.pathfinding.supervised`
=========================================

.. py:module:: deepxube.pathfinding.supervised

.. autodoc2-docstring:: deepxube.pathfinding.supervised
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`InstanceSup <deepxube.pathfinding.supervised.InstanceSup>`
     -
   * - :py:obj:`InstanceNodeSup <deepxube.pathfinding.supervised.InstanceNodeSup>`
     -
   * - :py:obj:`InstanceEdgeSup <deepxube.pathfinding.supervised.InstanceEdgeSup>`
     -
   * - :py:obj:`PathFindNodeSup <deepxube.pathfinding.supervised.PathFindNodeSup>`
     -
   * - :py:obj:`PathFindEdgeSup <deepxube.pathfinding.supervised.PathFindEdgeSup>`
     -
   * - :py:obj:`PathFindEdgeSamp <deepxube.pathfinding.supervised.PathFindEdgeSamp>`
     -

API
~~~

.. py:class:: InstanceSup(root_node: deepxube.base.pathfinding.Node, inst_info: typing.Any)
   :canonical: deepxube.pathfinding.supervised.InstanceSup

   Bases: :py:obj:`deepxube.base.pathfinding.Instance`

   .. py:method:: frontier_size() -> int
      :canonical: deepxube.pathfinding.supervised.InstanceSup.frontier_size
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.InstanceSup.frontier_size

   .. py:method:: record_goal(nodes: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.pathfinding.supervised.InstanceSup.record_goal
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.InstanceSup.record_goal

   .. py:method:: finished() -> bool
      :canonical: deepxube.pathfinding.supervised.InstanceSup.finished

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.InstanceSup.finished

.. py:class:: InstanceNodeSup(root_node: deepxube.base.pathfinding.Node, path_cost_sup: float, inst_info: typing.Any)
   :canonical: deepxube.pathfinding.supervised.InstanceNodeSup

   Bases: :py:obj:`deepxube.base.pathfinding.InstanceNode`, :py:obj:`deepxube.pathfinding.supervised.InstanceSup`

   .. py:method:: filter_expanded_nodes(nodes: typing.List[deepxube.base.pathfinding.Node]) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.pathfinding.supervised.InstanceNodeSup.filter_expanded_nodes
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.InstanceNodeSup.filter_expanded_nodes

   .. py:method:: push_pop_nodes(nodes: typing.List[deepxube.base.pathfinding.Node], costs: typing.List[float]) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.pathfinding.supervised.InstanceNodeSup.push_pop_nodes
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.InstanceNodeSup.push_pop_nodes

.. py:class:: InstanceEdgeSup(root_node: deepxube.base.pathfinding.Node, action: deepxube.base.domain.Action, path_cost_sup: float, inst_info: typing.Any)
   :canonical: deepxube.pathfinding.supervised.InstanceEdgeSup

   Bases: :py:obj:`deepxube.base.pathfinding.InstanceEdge`, :py:obj:`deepxube.pathfinding.supervised.InstanceSup`

   .. py:method:: filter_popped_nodes(nodes: typing.List[deepxube.base.pathfinding.Node]) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.pathfinding.supervised.InstanceEdgeSup.filter_popped_nodes
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.InstanceEdgeSup.filter_popped_nodes

   .. py:method:: push_pop_edges(edges: typing.List[deepxube.base.pathfinding.EdgeQ], costs: typing.List[float]) -> typing.List[deepxube.base.pathfinding.EdgeQ]
      :canonical: deepxube.pathfinding.supervised.InstanceEdgeSup.push_pop_edges
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.InstanceEdgeSup.push_pop_edges

.. py:class:: PathFindNodeSup(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.pathfinding.supervised.PathFindNodeSup

   Bases: :py:obj:`deepxube.base.pathfinding.PathFindNode`\ [\ :py:obj:`deepxube.base.domain.NodesSupervisable`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.pathfinding.supervised.InstanceNodeSup`\ ], :py:obj:`deepxube.base.pathfinding.PathFindSup`\ [\ :py:obj:`deepxube.base.domain.NodesSupervisable`\ , :py:obj:`deepxube.pathfinding.supervised.InstanceNodeSup`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.NodesSupervisable]
      :canonical: deepxube.pathfinding.supervised.PathFindNodeSup.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.PathFindNodeSup.domain_type

   .. py:method:: step(verbose: bool = False) -> typing.Tuple[typing.List[deepxube.base.pathfinding.Node], typing.List[deepxube.base.pathfinding.EdgeQ]]
      :canonical: deepxube.pathfinding.supervised.PathFindNodeSup.step

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.PathFindNodeSup.step

   .. py:method:: _compute_costs(instances: typing.List[deepxube.pathfinding.supervised.InstanceNodeSup], nodes_by_inst: typing.List[typing.List[deepxube.base.pathfinding.Node]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.pathfinding.supervised.PathFindNodeSup._compute_costs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.PathFindNodeSup._compute_costs

   .. py:method:: make_instances_sup(steps_gen: typing.List[int], inst_infos: typing.Optional[typing.List[typing.Any]]) -> typing.List[deepxube.pathfinding.supervised.InstanceNodeSup]
      :canonical: deepxube.pathfinding.supervised.PathFindNodeSup.make_instances_sup

.. py:class:: PathFindEdgeSup(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.pathfinding.supervised.PathFindEdgeSup

   Bases: :py:obj:`deepxube.base.pathfinding.PathFindEdge`\ [\ :py:obj:`deepxube.base.domain.EdgesSupervisable`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.pathfinding.supervised.InstanceEdgeSup`\ ], :py:obj:`deepxube.base.pathfinding.PathFindSup`\ [\ :py:obj:`deepxube.base.domain.EdgesSupervisable`\ , :py:obj:`deepxube.pathfinding.supervised.InstanceEdgeSup`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.EdgesSupervisable]
      :canonical: deepxube.pathfinding.supervised.PathFindEdgeSup.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.PathFindEdgeSup.domain_type

   .. py:method:: step(verbose: bool = False) -> typing.Tuple[typing.List[deepxube.base.pathfinding.Node], typing.List[deepxube.base.pathfinding.EdgeQ]]
      :canonical: deepxube.pathfinding.supervised.PathFindEdgeSup.step

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.PathFindEdgeSup.step

   .. py:method:: _compute_costs(instances: typing.List[deepxube.pathfinding.supervised.InstanceEdgeSup], edges_by_inst: typing.List[typing.List[deepxube.base.pathfinding.EdgeQ]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.pathfinding.supervised.PathFindEdgeSup._compute_costs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.PathFindEdgeSup._compute_costs

   .. py:method:: make_instances_sup(steps_gen: typing.List[int], inst_infos: typing.Optional[typing.List[typing.Any]]) -> typing.List[deepxube.pathfinding.supervised.InstanceEdgeSup]
      :canonical: deepxube.pathfinding.supervised.PathFindEdgeSup.make_instances_sup

.. py:class:: PathFindEdgeSamp(domain: deepxube.base.pathfinding.D, functions: deepxube.base.pathfinding.FNs)
   :canonical: deepxube.pathfinding.supervised.PathFindEdgeSamp

   Bases: :py:obj:`deepxube.base.pathfinding.PathFindEdge`\ [\ :py:obj:`deepxube.base.domain.EdgesSampleable`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.pathfinding.supervised.InstanceEdgeSup`\ ], :py:obj:`deepxube.base.pathfinding.PathFindSup`\ [\ :py:obj:`deepxube.base.domain.EdgesSampleable`\ , :py:obj:`deepxube.pathfinding.supervised.InstanceEdgeSup`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.EdgesSampleable]
      :canonical: deepxube.pathfinding.supervised.PathFindEdgeSamp.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.PathFindEdgeSamp.domain_type

   .. py:method:: step(verbose: bool = False) -> typing.Tuple[typing.List[deepxube.base.pathfinding.Node], typing.List[deepxube.base.pathfinding.EdgeQ]]
      :canonical: deepxube.pathfinding.supervised.PathFindEdgeSamp.step

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.PathFindEdgeSamp.step

   .. py:method:: _compute_costs(instances: typing.List[deepxube.pathfinding.supervised.InstanceEdgeSup], edges_by_inst: typing.List[typing.List[deepxube.base.pathfinding.EdgeQ]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.pathfinding.supervised.PathFindEdgeSamp._compute_costs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.supervised.PathFindEdgeSamp._compute_costs

   .. py:method:: make_instances_sup(steps_gen: typing.List[int], inst_infos: typing.Optional[typing.List[typing.Any]]) -> typing.List[deepxube.pathfinding.supervised.InstanceEdgeSup]
      :canonical: deepxube.pathfinding.supervised.PathFindEdgeSamp.make_instances_sup
