:py:mod:`deepxube.logic.asp`
============================

.. py:module:: deepxube.logic.asp

.. autodoc2-docstring:: deepxube.logic.asp
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Spec <deepxube.logic.asp.Spec>`
     - .. autodoc2-docstring:: deepxube.logic.asp.Spec
          :summary:
   * - :py:obj:`Solver <deepxube.logic.asp.Solver>`
     - .. autodoc2-docstring:: deepxube.logic.asp.Solver
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`model_to_body <deepxube.logic.asp.model_to_body>`
     - .. autodoc2-docstring:: deepxube.logic.asp.model_to_body
          :summary:
   * - :py:obj:`on_model_var_vals <deepxube.logic.asp.on_model_var_vals>`
     - .. autodoc2-docstring:: deepxube.logic.asp.on_model_var_vals
          :summary:
   * - :py:obj:`parse_clingo_line <deepxube.logic.asp.parse_clingo_line>`
     - .. autodoc2-docstring:: deepxube.logic.asp.parse_clingo_line
          :summary:
   * - :py:obj:`ctl_add_check <deepxube.logic.asp.ctl_add_check>`
     - .. autodoc2-docstring:: deepxube.logic.asp.ctl_add_check
          :summary:

API
~~~

.. py:function:: model_to_body(model: deepxube.logic.logic_objects.Model) -> str
   :canonical: deepxube.logic.asp.model_to_body

   .. autodoc2-docstring:: deepxube.logic.asp.model_to_body

.. py:function:: on_model_var_vals(m: clingo.solving.Model) -> frozenset[str]
   :canonical: deepxube.logic.asp.on_model_var_vals

   .. autodoc2-docstring:: deepxube.logic.asp.on_model_var_vals

.. py:function:: parse_clingo_line(line: str) -> str
   :canonical: deepxube.logic.asp.parse_clingo_line

   .. autodoc2-docstring:: deepxube.logic.asp.parse_clingo_line

.. py:class:: Spec(goal_true: typing.Optional[typing.List[deepxube.logic.logic_objects.Clause]] = None, goal_false: typing.Optional[typing.List[deepxube.logic.logic_objects.Clause]] = None, atoms_true: typing.Optional[typing.List[deepxube.logic.logic_objects.Atom]] = None, atoms_false: typing.Optional[typing.List[deepxube.logic.logic_objects.Atom]] = None, models_banned: typing.Optional[typing.List[deepxube.logic.logic_objects.Model]] = None, num_atoms_gt: typing.Optional[int] = None)
   :canonical: deepxube.logic.asp.Spec

   .. autodoc2-docstring:: deepxube.logic.asp.Spec

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.logic.asp.Spec.__init__

   .. py:method:: add(spec_add: deepxube.logic.asp.Spec) -> deepxube.logic.asp.Spec
      :canonical: deepxube.logic.asp.Spec.add

      .. autodoc2-docstring:: deepxube.logic.asp.Spec.add

.. py:function:: ctl_add_check(ctl: clingo.Control, block: str, add_line: str) -> None
   :canonical: deepxube.logic.asp.ctl_add_check

   .. autodoc2-docstring:: deepxube.logic.asp.ctl_add_check

.. py:class:: Solver(ground_atoms: typing.List[deepxube.logic.logic_objects.Atom], bk: typing.List[str])
   :canonical: deepxube.logic.asp.Solver

   .. autodoc2-docstring:: deepxube.logic.asp.Solver

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.logic.asp.Solver.__init__

   .. py:method:: get_num_ground_rules() -> int
      :canonical: deepxube.logic.asp.Solver.get_num_ground_rules

      .. autodoc2-docstring:: deepxube.logic.asp.Solver.get_num_ground_rules

   .. py:method:: get_models(spec: deepxube.logic.asp.Spec, on_model: typing.Callable[[typing.Any], deepxube.logic.logic_objects.Model], num_models: int, minimal: bool) -> typing.List[deepxube.logic.logic_objects.Model]
      :canonical: deepxube.logic.asp.Solver.get_models

      .. autodoc2-docstring:: deepxube.logic.asp.Solver.get_models

   .. py:method:: check_model(spec: deepxube.logic.asp.Spec, model: deepxube.logic.logic_objects.Model, timeout: typing.Optional[float] = None) -> bool
      :canonical: deepxube.logic.asp.Solver.check_model

      .. autodoc2-docstring:: deepxube.logic.asp.Solver.check_model

   .. py:method:: sample_minimal_model(spec: deepxube.logic.asp.Spec, model: deepxube.logic.logic_objects.Model, on_model: typing.Callable[[clingo.solving.Model], deepxube.logic.logic_objects.Model]) -> deepxube.logic.logic_objects.Model
      :canonical: deepxube.logic.asp.Solver.sample_minimal_model

      .. autodoc2-docstring:: deepxube.logic.asp.Solver.sample_minimal_model

   .. py:method:: sample_minimal_model_old(spec: deepxube.logic.asp.Spec, model: deepxube.logic.logic_objects.Model) -> deepxube.logic.logic_objects.Model
      :canonical: deepxube.logic.asp.Solver.sample_minimal_model_old

      .. autodoc2-docstring:: deepxube.logic.asp.Solver.sample_minimal_model_old

   .. py:method:: _add_goal(goal: typing.List[deepxube.logic.logic_objects.Clause]) -> str
      :canonical: deepxube.logic.asp.Solver._add_goal

      .. autodoc2-docstring:: deepxube.logic.asp.Solver._add_goal

   .. py:method:: _make_assumptions(spec: deepxube.logic.asp.Spec) -> typing.List[typing.Tuple[clingo.Symbol, bool]]
      :canonical: deepxube.logic.asp.Solver._make_assumptions

      .. autodoc2-docstring:: deepxube.logic.asp.Solver._make_assumptions
