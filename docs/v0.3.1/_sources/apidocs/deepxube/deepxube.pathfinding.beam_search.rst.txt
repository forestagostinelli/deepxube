:py:mod:`deepxube.pathfinding.beam_search`
==========================================

.. py:module:: deepxube.pathfinding.beam_search

.. autodoc2-docstring:: deepxube.pathfinding.beam_search
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`InstanceBeam <deepxube.pathfinding.beam_search.InstanceBeam>`
     -
   * - :py:obj:`BeamSearch <deepxube.pathfinding.beam_search.BeamSearch>`
     -
   * - :py:obj:`InstanceNodeBeam <deepxube.pathfinding.beam_search.InstanceNodeBeam>`
     -
   * - :py:obj:`InstanceEdgeBeam <deepxube.pathfinding.beam_search.InstanceEdgeBeam>`
     -
   * - :py:obj:`BeamSearchPolicy <deepxube.pathfinding.beam_search.BeamSearchPolicy>`
     -
   * - :py:obj:`BeamSearchHeurNode <deepxube.pathfinding.beam_search.BeamSearchHeurNode>`
     -
   * - :py:obj:`BeamSearchHeurEdge <deepxube.pathfinding.beam_search.BeamSearchHeurEdge>`
     -
   * - :py:obj:`BeamSearchHeurNodeActsEnum <deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsEnum>`
     -
   * - :py:obj:`BeamSearchHeurEdgeActsEnum <deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsEnum>`
     -
   * - :py:obj:`BeamSearchHeurNodeActsPolicy <deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsPolicy>`
     -
   * - :py:obj:`BeamSearchHeurEdgeActsPolicy <deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsPolicy>`
     -
   * - :py:obj:`BeamSearchParser <deepxube.pathfinding.beam_search.BeamSearchParser>`
     -
   * - :py:obj:`BeamSearchPolicyParser <deepxube.pathfinding.beam_search.BeamSearchPolicyParser>`
     -
   * - :py:obj:`BeamSearchNodeParser <deepxube.pathfinding.beam_search.BeamSearchNodeParser>`
     -
   * - :py:obj:`BeamSearchEdgeParser <deepxube.pathfinding.beam_search.BeamSearchEdgeParser>`
     -
   * - :py:obj:`BeamSearchHasPolicyParser <deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser>`
     -
   * - :py:obj:`BeamSearchNodeHasPolicyParser <deepxube.pathfinding.beam_search.BeamSearchNodeHasPolicyParser>`
     -
   * - :py:obj:`BeamSearchEdgeHasPolicyParser <deepxube.pathfinding.beam_search.BeamSearchEdgeHasPolicyParser>`
     -
   * - :py:obj:`RolloutPolicy <deepxube.pathfinding.beam_search.RolloutPolicy>`
     -
   * - :py:obj:`RolloutParser <deepxube.pathfinding.beam_search.RolloutParser>`
     -
   * - :py:obj:`RolloutPolicyParser <deepxube.pathfinding.beam_search.RolloutPolicyParser>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`D <deepxube.pathfinding.beam_search.D>`
     - .. autodoc2-docstring:: deepxube.pathfinding.beam_search.D
          :summary:
   * - :py:obj:`IBeam <deepxube.pathfinding.beam_search.IBeam>`
     - .. autodoc2-docstring:: deepxube.pathfinding.beam_search.IBeam
          :summary:

API
~~~

.. py:class:: InstanceBeam(root_node: deepxube.base.pathfinding.Node, inst_info: typing.Any)
   :canonical: deepxube.pathfinding.beam_search.InstanceBeam

   Bases: :py:obj:`deepxube.base.pathfinding.Instance`, :py:obj:`abc.ABC`

   .. py:method:: set_beam_size(beam_size: int) -> None
      :canonical: deepxube.pathfinding.beam_search.InstanceBeam.set_beam_size

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceBeam.set_beam_size

   .. py:method:: set_temp(temp: float) -> None
      :canonical: deepxube.pathfinding.beam_search.InstanceBeam.set_temp

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceBeam.set_temp

   .. py:method:: set_eps(eps: float) -> None
      :canonical: deepxube.pathfinding.beam_search.InstanceBeam.set_eps

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceBeam.set_eps

   .. py:method:: set_rollout(rollout: bool) -> None
      :canonical: deepxube.pathfinding.beam_search.InstanceBeam.set_rollout

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceBeam.set_rollout

   .. py:method:: frontier_size() -> int
      :canonical: deepxube.pathfinding.beam_search.InstanceBeam.frontier_size

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceBeam.frontier_size

   .. py:method:: record_goal(nodes: typing.List[deepxube.base.pathfinding.Node]) -> None
      :canonical: deepxube.pathfinding.beam_search.InstanceBeam.record_goal

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceBeam.record_goal

   .. py:method:: select_idxs_from_logits(logits: typing.List[float]) -> typing.List[int]
      :canonical: deepxube.pathfinding.beam_search.InstanceBeam.select_idxs_from_logits

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceBeam.select_idxs_from_logits

   .. py:method:: finished() -> bool
      :canonical: deepxube.pathfinding.beam_search.InstanceBeam.finished

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceBeam.finished

.. py:data:: D
   :canonical: deepxube.pathfinding.beam_search.D
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.pathfinding.beam_search.D

.. py:data:: IBeam
   :canonical: deepxube.pathfinding.beam_search.IBeam
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.pathfinding.beam_search.IBeam

.. py:class:: BeamSearch(domain: deepxube.pathfinding.beam_search.D, functions: deepxube.base.pathfinding.FNs, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False)
   :canonical: deepxube.pathfinding.beam_search.BeamSearch

   Bases: :py:obj:`deepxube.base.pathfinding.PathFind`\ [\ :py:obj:`deepxube.pathfinding.beam_search.D`\ , :py:obj:`deepxube.base.pathfinding.FNs`\ , :py:obj:`deepxube.pathfinding.beam_search.IBeam`\ ], :py:obj:`abc.ABC`

   .. py:method:: _construct_instances(inst_cls: type[deepxube.pathfinding.beam_search.IBeam], nodes_root: typing.List[deepxube.base.pathfinding.Node], inst_infos: typing.Optional[typing.List[typing.Any]], beam_size: typing.Optional[int], temp: typing.Optional[float], eps: typing.Optional[float]) -> typing.List[deepxube.pathfinding.beam_search.IBeam]
      :canonical: deepxube.pathfinding.beam_search.BeamSearch._construct_instances

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearch._construct_instances

   .. py:method:: __repr__() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearch.__repr__

.. py:class:: InstanceNodeBeam(root_node: deepxube.base.pathfinding.Node, inst_info: typing.Any)
   :canonical: deepxube.pathfinding.beam_search.InstanceNodeBeam

   Bases: :py:obj:`deepxube.base.pathfinding.InstanceNode`, :py:obj:`deepxube.pathfinding.beam_search.InstanceBeam`

   .. py:method:: filter_expanded_nodes(nodes: typing.List[deepxube.base.pathfinding.Node]) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.pathfinding.beam_search.InstanceNodeBeam.filter_expanded_nodes

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceNodeBeam.filter_expanded_nodes

   .. py:method:: push_pop_nodes(nodes: typing.List[deepxube.base.pathfinding.Node], costs: typing.List[float]) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.pathfinding.beam_search.InstanceNodeBeam.push_pop_nodes

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceNodeBeam.push_pop_nodes

.. py:class:: InstanceEdgeBeam(root_node: deepxube.base.pathfinding.Node, inst_info: typing.Any)
   :canonical: deepxube.pathfinding.beam_search.InstanceEdgeBeam

   Bases: :py:obj:`deepxube.base.pathfinding.InstanceEdge`, :py:obj:`deepxube.pathfinding.beam_search.InstanceBeam`

   .. py:method:: filter_popped_nodes(nodes: typing.List[deepxube.base.pathfinding.Node]) -> typing.List[deepxube.base.pathfinding.Node]
      :canonical: deepxube.pathfinding.beam_search.InstanceEdgeBeam.filter_popped_nodes

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceEdgeBeam.filter_popped_nodes

   .. py:method:: push_pop_edges(edges: typing.List[deepxube.base.pathfinding.EdgeQ], costs: typing.List[float]) -> typing.List[deepxube.base.pathfinding.EdgeQ]
      :canonical: deepxube.pathfinding.beam_search.InstanceEdgeBeam.push_pop_edges

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.InstanceEdgeBeam.push_pop_edges

.. py:class:: BeamSearchPolicy(domain: deepxube.pathfinding.beam_search.D, functions: deepxube.base.pathfinding.FNs, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False)
   :canonical: deepxube.pathfinding.beam_search.BeamSearchPolicy

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearch`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsPolicy`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceEdgeBeam`\ ], :py:obj:`deepxube.base.pathfinding.PathFindEdge`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsPolicy`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceEdgeBeam`\ ], :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsPolicy`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceEdgeBeam`\ ], :py:obj:`deepxube.base.pathfinding.PathFindSetPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsPolicy`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceEdgeBeam`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchPolicy.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchPolicy.domain_type

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsPolicy]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchPolicy.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchPolicy.functions_type

   .. py:property:: num_rand_edges
      :canonical: deepxube.pathfinding.beam_search.BeamSearchPolicy.num_rand_edges
      :type: int

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchPolicy.num_rand_edges

   .. py:method:: make_instances(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], inst_infos: typing.Optional[typing.List[typing.Any]] = None, compute_root_vals: bool = True, beam_size: typing.Optional[int] = None, temp: typing.Optional[float] = None, eps: typing.Optional[float] = None) -> typing.List[deepxube.pathfinding.beam_search.InstanceEdgeBeam]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchPolicy.make_instances

   .. py:method:: _compute_costs(instances: typing.List[deepxube.pathfinding.beam_search.InstanceEdgeBeam], edges_by_inst: typing.List[typing.List[deepxube.base.pathfinding.EdgeQ]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchPolicy._compute_costs

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchPolicy._compute_costs

.. py:class:: BeamSearchHeurNode(domain: deepxube.pathfinding.beam_search.D, functions: deepxube.base.pathfinding.FNs, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False)
   :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNode

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearch`\ [\ :py:obj:`deepxube.pathfinding.beam_search.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHV`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceNodeBeam`\ ], :py:obj:`deepxube.base.pathfinding.PathFindNode`\ [\ :py:obj:`deepxube.pathfinding.beam_search.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHV`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceNodeBeam`\ ], :py:obj:`deepxube.base.pathfinding.PathFindSetHeurV`\ [\ :py:obj:`deepxube.pathfinding.beam_search.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHV`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceNodeBeam`\ ], :py:obj:`abc.ABC`

   .. py:method:: make_instances(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], inst_infos: typing.Optional[typing.List[typing.Any]] = None, compute_root_vals: bool = True, beam_size: typing.Optional[int] = None, temp: typing.Optional[float] = None, eps: typing.Optional[float] = None) -> typing.List[deepxube.pathfinding.beam_search.InstanceNodeBeam]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNode.make_instances

   .. py:method:: _compute_costs(instances: typing.List[deepxube.pathfinding.beam_search.InstanceNodeBeam], nodes_by_inst: typing.List[typing.List[deepxube.base.pathfinding.Node]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNode._compute_costs

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurNode._compute_costs

.. py:class:: BeamSearchHeurEdge(domain: deepxube.pathfinding.beam_search.D, functions: deepxube.base.pathfinding.FNs, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False)
   :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdge

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearch`\ [\ :py:obj:`deepxube.pathfinding.beam_search.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHQ`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceEdgeBeam`\ ], :py:obj:`deepxube.base.pathfinding.PathFindEdge`\ [\ :py:obj:`deepxube.pathfinding.beam_search.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHQ`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceEdgeBeam`\ ], :py:obj:`deepxube.base.pathfinding.PathFindSetHeurQ`\ [\ :py:obj:`deepxube.pathfinding.beam_search.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHQ`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceEdgeBeam`\ ], :py:obj:`abc.ABC`

   .. py:method:: make_instances(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], inst_infos: typing.Optional[typing.List[typing.Any]] = None, compute_root_vals: bool = True, beam_size: typing.Optional[int] = None, temp: typing.Optional[float] = None, eps: typing.Optional[float] = None) -> typing.List[deepxube.pathfinding.beam_search.InstanceEdgeBeam]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdge.make_instances

   .. py:method:: _compute_costs(instances: typing.List[deepxube.pathfinding.beam_search.InstanceEdgeBeam], edges_by_inst: typing.List[typing.List[deepxube.base.pathfinding.EdgeQ]]) -> typing.List[typing.List[float]]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdge._compute_costs

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurEdge._compute_costs

.. py:class:: BeamSearchHeurNodeActsEnum(domain: deepxube.pathfinding.beam_search.D, functions: deepxube.base.pathfinding.FNs, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False)
   :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsEnum

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchHeurNode`\ [\ :py:obj:`deepxube.base.domain.ActsEnum`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurV`\ ], :py:obj:`deepxube.base.pathfinding.PathFindActsEnum`\ [\ :py:obj:`deepxube.base.domain.ActsEnum`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurV`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceNodeBeam`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.ActsEnum]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsEnum.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsEnum.domain_type

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurV]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsEnum.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsEnum.functions_type

.. py:class:: BeamSearchHeurEdgeActsEnum(domain: deepxube.pathfinding.beam_search.D, functions: deepxube.base.pathfinding.FNs, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False)
   :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsEnum

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchHeurEdge`\ [\ :py:obj:`deepxube.base.domain.ActsEnum`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurQ`\ ], :py:obj:`deepxube.base.pathfinding.PathFindActsEnum`\ [\ :py:obj:`deepxube.base.domain.ActsEnum`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurQ`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceEdgeBeam`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.ActsEnum]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsEnum.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsEnum.domain_type

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurQ]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsEnum.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsEnum.functions_type

.. py:class:: BeamSearchHeurNodeActsPolicy(domain: deepxube.base.domain.Domain, functions: deepxube.base.pathfinding.FNsHeurVPolicy, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False, num_rand_edges: int = 0)
   :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsPolicy

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchHeurNode`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ ], :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurVPolicy`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceNodeBeam`\ ]

   .. py:property:: num_rand_edges
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsPolicy.num_rand_edges
      :type: int

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsPolicy.num_rand_edges

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsPolicy.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsPolicy.domain_type

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurVPolicy]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsPolicy.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsPolicy.functions_type

   .. py:method:: __repr__() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurNodeActsPolicy.__repr__

.. py:class:: BeamSearchHeurEdgeActsPolicy(domain: deepxube.base.domain.Domain, functions: deepxube.base.pathfinding.FNsHeurQPolicy, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False, num_rand_edges: int = 0)
   :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsPolicy

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchHeurEdge`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ ], :py:obj:`deepxube.base.pathfinding.PathFindActsPolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.pathfinding.FNsHeurQPolicy`\ , :py:obj:`deepxube.pathfinding.beam_search.InstanceEdgeBeam`\ ]

   .. py:property:: num_rand_edges
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsPolicy.num_rand_edges
      :type: int

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsPolicy.num_rand_edges

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsPolicy.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsPolicy.domain_type

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNsHeurQPolicy]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsPolicy.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsPolicy.functions_type

   .. py:method:: __repr__() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHeurEdgeActsPolicy.__repr__

.. py:class:: BeamSearchParser
   :canonical: deepxube.pathfinding.beam_search.BeamSearchParser

   Bases: :py:obj:`deepxube.base.factory.Parser`, :py:obj:`abc.ABC`

   .. py:method:: parse(args_str: str) -> typing.Dict[str, typing.Any]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchParser.parse

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchParser.parse

   .. py:method:: help() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchParser.help

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchParser.help

   .. py:method:: _alg_name() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchParser._alg_name
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchParser._alg_name

.. py:class:: BeamSearchPolicyParser
   :canonical: deepxube.pathfinding.beam_search.BeamSearchPolicyParser

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchParser`

   .. py:method:: _alg_name() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchPolicyParser._alg_name

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchPolicyParser._alg_name

.. py:class:: BeamSearchNodeParser
   :canonical: deepxube.pathfinding.beam_search.BeamSearchNodeParser

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchParser`

   .. py:method:: _alg_name() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchNodeParser._alg_name

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchNodeParser._alg_name

.. py:class:: BeamSearchEdgeParser
   :canonical: deepxube.pathfinding.beam_search.BeamSearchEdgeParser

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchParser`

   .. py:method:: _alg_name() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchEdgeParser._alg_name

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchEdgeParser._alg_name

.. py:class:: BeamSearchHasPolicyParser
   :canonical: deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser

   Bases: :py:obj:`deepxube.base.factory.Parser`, :py:obj:`abc.ABC`

   .. py:method:: parse(args_str: str) -> typing.Dict[str, typing.Any]
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser.parse

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser.parse

   .. py:method:: help() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser.help

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser.help

   .. py:method:: _alg_name() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser._alg_name
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser._alg_name

.. py:class:: BeamSearchNodeHasPolicyParser
   :canonical: deepxube.pathfinding.beam_search.BeamSearchNodeHasPolicyParser

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser`

   .. py:method:: _alg_name() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchNodeHasPolicyParser._alg_name

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchNodeHasPolicyParser._alg_name

.. py:class:: BeamSearchEdgeHasPolicyParser
   :canonical: deepxube.pathfinding.beam_search.BeamSearchEdgeHasPolicyParser

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchHasPolicyParser`

   .. py:method:: _alg_name() -> str
      :canonical: deepxube.pathfinding.beam_search.BeamSearchEdgeHasPolicyParser._alg_name

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.BeamSearchEdgeHasPolicyParser._alg_name

.. py:class:: RolloutPolicy(domain: deepxube.pathfinding.beam_search.D, functions: deepxube.base.pathfinding.FNs, beam_size: int = 1, temp: float = 0.0, eps: float = 0.0, rollout: bool = False)
   :canonical: deepxube.pathfinding.beam_search.RolloutPolicy

   Bases: :py:obj:`deepxube.pathfinding.beam_search.BeamSearchPolicy`

.. py:class:: RolloutParser
   :canonical: deepxube.pathfinding.beam_search.RolloutParser

   Bases: :py:obj:`deepxube.base.factory.Parser`, :py:obj:`abc.ABC`

   .. py:method:: parse(args_str: str) -> typing.Dict[str, typing.Any]
      :canonical: deepxube.pathfinding.beam_search.RolloutParser.parse

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.RolloutParser.parse

   .. py:method:: help() -> str
      :canonical: deepxube.pathfinding.beam_search.RolloutParser.help

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.RolloutParser.help

   .. py:method:: _alg_name() -> str
      :canonical: deepxube.pathfinding.beam_search.RolloutParser._alg_name
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.RolloutParser._alg_name

.. py:class:: RolloutPolicyParser
   :canonical: deepxube.pathfinding.beam_search.RolloutPolicyParser

   Bases: :py:obj:`deepxube.pathfinding.beam_search.RolloutParser`

   .. py:method:: _alg_name() -> str
      :canonical: deepxube.pathfinding.beam_search.RolloutPolicyParser._alg_name

      .. autodoc2-docstring:: deepxube.pathfinding.beam_search.RolloutPolicyParser._alg_name
