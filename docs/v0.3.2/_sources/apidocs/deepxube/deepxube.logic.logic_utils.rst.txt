:py:mod:`deepxube.logic.logic_utils`
====================================

.. py:module:: deepxube.logic.logic_utils

.. autodoc2-docstring:: deepxube.logic.logic_utils
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`parse_literal <deepxube.logic.logic_utils.parse_literal>`
     - .. autodoc2-docstring:: deepxube.logic.logic_utils.parse_literal
          :summary:
   * - :py:obj:`replace_anon_vars <deepxube.logic.logic_utils.replace_anon_vars>`
     - .. autodoc2-docstring:: deepxube.logic.logic_utils.replace_anon_vars
          :summary:
   * - :py:obj:`parse_clause <deepxube.logic.logic_utils.parse_clause>`
     - .. autodoc2-docstring:: deepxube.logic.logic_utils.parse_clause
          :summary:
   * - :py:obj:`copy_clause_with_new_head <deepxube.logic.logic_utils.copy_clause_with_new_head>`
     - .. autodoc2-docstring:: deepxube.logic.logic_utils.copy_clause_with_new_head
          :summary:
   * - :py:obj:`atom_to_str <deepxube.logic.logic_utils.atom_to_str>`
     - .. autodoc2-docstring:: deepxube.logic.logic_utils.atom_to_str
          :summary:

API
~~~

.. py:function:: parse_literal(lit_str: str) -> deepxube.logic.logic_objects.Literal
   :canonical: deepxube.logic.logic_utils.parse_literal

   .. autodoc2-docstring:: deepxube.logic.logic_utils.parse_literal

.. py:function:: replace_anon_vars(lit: deepxube.logic.logic_objects.Literal, all_lits: typing.List[deepxube.logic.logic_objects.Literal]) -> deepxube.logic.logic_objects.Literal
   :canonical: deepxube.logic.logic_utils.replace_anon_vars

   .. autodoc2-docstring:: deepxube.logic.logic_utils.replace_anon_vars

.. py:function:: parse_clause(constr_str: str) -> typing.Tuple[deepxube.logic.logic_objects.Clause, typing.Dict[str, typing.List[str]]]
   :canonical: deepxube.logic.logic_utils.parse_clause

   .. autodoc2-docstring:: deepxube.logic.logic_utils.parse_clause

.. py:function:: copy_clause_with_new_head(clause: deepxube.logic.logic_objects.Clause, head_name_new: str) -> deepxube.logic.logic_objects.Clause
   :canonical: deepxube.logic.logic_utils.copy_clause_with_new_head

   .. autodoc2-docstring:: deepxube.logic.logic_utils.copy_clause_with_new_head

.. py:function:: atom_to_str(atom: deepxube.logic.logic_objects.Atom) -> str
   :canonical: deepxube.logic.logic_utils.atom_to_str

   .. autodoc2-docstring:: deepxube.logic.logic_utils.atom_to_str
