:py:mod:`deepxube.logic.logic_objects`
======================================

.. py:module:: deepxube.logic.logic_objects

.. autodoc2-docstring:: deepxube.logic.logic_objects
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Literal <deepxube.logic.logic_objects.Literal>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.Literal
          :summary:
   * - :py:obj:`VarNode <deepxube.logic.logic_objects.VarNode>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.VarNode
          :summary:
   * - :py:obj:`LitNode <deepxube.logic.logic_objects.LitNode>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.LitNode
          :summary:
   * - :py:obj:`Clause <deepxube.logic.logic_objects.Clause>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.Clause
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`prop_across <deepxube.logic.logic_objects.prop_across>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.prop_across
          :summary:
   * - :py:obj:`prune_lit <deepxube.logic.logic_objects.prune_lit>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.prune_lit
          :summary:
   * - :py:obj:`theta_sub_lits <deepxube.logic.logic_objects.theta_sub_lits>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.theta_sub_lits
          :summary:
   * - :py:obj:`theta_sub_args <deepxube.logic.logic_objects.theta_sub_args>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.theta_sub_args
          :summary:
   * - :py:obj:`make_subs_lit <deepxube.logic.logic_objects.make_subs_lit>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.make_subs_lit
          :summary:
   * - :py:obj:`make_subs <deepxube.logic.logic_objects.make_subs>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.make_subs
          :summary:
   * - :py:obj:`theta_sub_replace <deepxube.logic.logic_objects.theta_sub_replace>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.theta_sub_replace
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Atom <deepxube.logic.logic_objects.Atom>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.Atom
          :summary:
   * - :py:obj:`Model <deepxube.logic.logic_objects.Model>`
     - .. autodoc2-docstring:: deepxube.logic.logic_objects.Model
          :summary:

API
~~~

.. py:class:: Literal(predicate: str, arguments: typing.Tuple[str, ...], directions: typing.Tuple[str, ...], positive: bool = True)
   :canonical: deepxube.logic.logic_objects.Literal

   .. autodoc2-docstring:: deepxube.logic.logic_objects.Literal

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.logic.logic_objects.Literal.__init__

   .. py:method:: to_code() -> str
      :canonical: deepxube.logic.logic_objects.Literal.to_code

      .. autodoc2-docstring:: deepxube.logic.logic_objects.Literal.to_code

   .. py:method:: get_pred_arity_pos_id() -> typing.Tuple[str, int, bool]
      :canonical: deepxube.logic.logic_objects.Literal.get_pred_arity_pos_id

      .. autodoc2-docstring:: deepxube.logic.logic_objects.Literal.get_pred_arity_pos_id

   .. py:method:: __str__() -> str
      :canonical: deepxube.logic.logic_objects.Literal.__str__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.logic.logic_objects.Literal.__repr__

   .. py:method:: __hash__() -> int
      :canonical: deepxube.logic.logic_objects.Literal.__hash__

.. py:class:: VarNode()
   :canonical: deepxube.logic.logic_objects.VarNode

   .. autodoc2-docstring:: deepxube.logic.logic_objects.VarNode

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.logic.logic_objects.VarNode.__init__

   .. py:method:: add_neighbor(neighbor: deepxube.logic.logic_objects.VarNode) -> None
      :canonical: deepxube.logic.logic_objects.VarNode.add_neighbor

      .. autodoc2-docstring:: deepxube.logic.logic_objects.VarNode.add_neighbor

.. py:class:: LitNode(predicate: str, in_body: bool, arguments: typing.Tuple[str, ...])
   :canonical: deepxube.logic.logic_objects.LitNode

   .. autodoc2-docstring:: deepxube.logic.logic_objects.LitNode

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.logic.logic_objects.LitNode.__init__

   .. py:method:: prop_up() -> None
      :canonical: deepxube.logic.logic_objects.LitNode.prop_up

      .. autodoc2-docstring:: deepxube.logic.logic_objects.LitNode.prop_up

   .. py:method:: prop_down() -> None
      :canonical: deepxube.logic.logic_objects.LitNode.prop_down

      .. autodoc2-docstring:: deepxube.logic.logic_objects.LitNode.prop_down

.. py:function:: prop_across(var_nodes: typing.List[deepxube.logic.logic_objects.VarNode]) -> None
   :canonical: deepxube.logic.logic_objects.prop_across

   .. autodoc2-docstring:: deepxube.logic.logic_objects.prop_across

.. py:class:: Clause(head: deepxube.logic.logic_objects.Literal, body: typing.Tuple[deepxube.logic.logic_objects.Literal, ...])
   :canonical: deepxube.logic.logic_objects.Clause

   .. autodoc2-docstring:: deepxube.logic.logic_objects.Clause

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.logic.logic_objects.Clause.__init__

   .. py:method:: is_in_out_consistent_body() -> bool
      :canonical: deepxube.logic.logic_objects.Clause.is_in_out_consistent_body

      .. autodoc2-docstring:: deepxube.logic.logic_objects.Clause.is_in_out_consistent_body

   .. py:method:: can_ground() -> bool
      :canonical: deepxube.logic.logic_objects.Clause.can_ground

      .. autodoc2-docstring:: deepxube.logic.logic_objects.Clause.can_ground

   .. py:method:: to_code() -> str
      :canonical: deepxube.logic.logic_objects.Clause.to_code

      .. autodoc2-docstring:: deepxube.logic.logic_objects.Clause.to_code

   .. py:method:: get_lit_id_count_dict() -> typing.Dict[typing.Tuple[str, int, bool], int]
      :canonical: deepxube.logic.logic_objects.Clause.get_lit_id_count_dict

      .. autodoc2-docstring:: deepxube.logic.logic_objects.Clause.get_lit_id_count_dict

   .. py:method:: theta_sub(other: deepxube.logic.logic_objects.Clause, subs_prev: typing.Optional[typing.Dict[str, str]] = None, negate_l: typing.Optional[typing.List[bool]] = None, subs_forbid: typing.Optional[typing.Dict[str, typing.List[str]]] = None, ignore_head: bool = False) -> typing.Optional[typing.Dict[str, str]]
      :canonical: deepxube.logic.logic_objects.Clause.theta_sub

      .. autodoc2-docstring:: deepxube.logic.logic_objects.Clause.theta_sub

   .. py:method:: __str__() -> str
      :canonical: deepxube.logic.logic_objects.Clause.__str__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.logic.logic_objects.Clause.__repr__

   .. py:method:: __hash__() -> int
      :canonical: deepxube.logic.logic_objects.Clause.__hash__

   .. py:method:: __eq__(other: object) -> bool
      :canonical: deepxube.logic.logic_objects.Clause.__eq__

.. py:function:: prune_lit(lit: deepxube.logic.logic_objects.Literal, lit_to_prune: deepxube.logic.logic_objects.Literal, idxs_vars_req: typing.List[typing.Tuple[int, str]]) -> bool
   :canonical: deepxube.logic.logic_objects.prune_lit

   .. autodoc2-docstring:: deepxube.logic.logic_objects.prune_lit

.. py:function:: theta_sub_lits(lits1: typing.List[deepxube.logic.logic_objects.Literal], lits2_dict: typing.Dict[str, typing.List[deepxube.logic.logic_objects.Literal]], negate_l: typing.List[bool], subs_prev: typing.Dict[str, str], subs_forbid: typing.Dict[str, typing.List[str]]) -> typing.Optional[typing.Dict[str, str]]
   :canonical: deepxube.logic.logic_objects.theta_sub_lits

   .. autodoc2-docstring:: deepxube.logic.logic_objects.theta_sub_lits

.. py:function:: theta_sub_args(args1: typing.Tuple[str, ...], args2: typing.Tuple[str, ...], idxs_not_subbed: typing.List[int], subs_prev: typing.Dict[str, str], subs_forbid: typing.Dict[str, typing.List[str]]) -> typing.Optional[typing.Dict[str, str]]
   :canonical: deepxube.logic.logic_objects.theta_sub_args

   .. autodoc2-docstring:: deepxube.logic.logic_objects.theta_sub_args

.. py:function:: make_subs_lit(lit: deepxube.logic.logic_objects.Literal, subs: typing.Dict[str, str]) -> deepxube.logic.logic_objects.Literal
   :canonical: deepxube.logic.logic_objects.make_subs_lit

   .. autodoc2-docstring:: deepxube.logic.logic_objects.make_subs_lit

.. py:function:: make_subs(clause: deepxube.logic.logic_objects.Clause, subs: typing.Dict[str, str]) -> deepxube.logic.logic_objects.Clause
   :canonical: deepxube.logic.logic_objects.make_subs

   .. autodoc2-docstring:: deepxube.logic.logic_objects.make_subs

.. py:function:: theta_sub_replace(clause1: deepxube.logic.logic_objects.Clause, clause2: deepxube.logic.logic_objects.Clause, ignore_head: bool = False) -> deepxube.logic.logic_objects.Clause
   :canonical: deepxube.logic.logic_objects.theta_sub_replace

   .. autodoc2-docstring:: deepxube.logic.logic_objects.theta_sub_replace

.. py:data:: Atom
   :canonical: deepxube.logic.logic_objects.Atom
   :value: None

   .. autodoc2-docstring:: deepxube.logic.logic_objects.Atom

.. py:data:: Model
   :canonical: deepxube.logic.logic_objects.Model
   :value: None

   .. autodoc2-docstring:: deepxube.logic.logic_objects.Model
