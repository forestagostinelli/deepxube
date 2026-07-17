:py:mod:`deepxube.pathfinding.utils.performance`
================================================

.. py:module:: deepxube.pathfinding.utils.performance

.. autodoc2-docstring:: deepxube.pathfinding.utils.performance
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PathFindPerf <deepxube.pathfinding.utils.performance.PathFindPerf>`
     - .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.PathFindPerf
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_eq_weighted_perf <deepxube.pathfinding.utils.performance.get_eq_weighted_perf>`
     - .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.get_eq_weighted_perf
          :summary:
   * - :py:obj:`print_pathfindperf <deepxube.pathfinding.utils.performance.print_pathfindperf>`
     - .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.print_pathfindperf
          :summary:
   * - :py:obj:`is_valid_soln <deepxube.pathfinding.utils.performance.is_valid_soln>`
     - .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.is_valid_soln
          :summary:

API
~~~

.. py:class:: PathFindPerf()
   :canonical: deepxube.pathfinding.utils.performance.PathFindPerf

   .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.PathFindPerf

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.PathFindPerf.__init__

   .. py:method:: update_perf(instance: deepxube.base.pathfinding.Instance) -> None
      :canonical: deepxube.pathfinding.utils.performance.PathFindPerf.update_perf

      .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.PathFindPerf.update_perf

   .. py:method:: comb_perf(search_perf2: deepxube.pathfinding.utils.performance.PathFindPerf) -> deepxube.pathfinding.utils.performance.PathFindPerf
      :canonical: deepxube.pathfinding.utils.performance.PathFindPerf.comb_perf

      .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.PathFindPerf.comb_perf

   .. py:method:: per_solved() -> float
      :canonical: deepxube.pathfinding.utils.performance.PathFindPerf.per_solved

      .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.PathFindPerf.per_solved

   .. py:method:: stats() -> typing.Tuple[float, float, float]
      :canonical: deepxube.pathfinding.utils.performance.PathFindPerf.stats

      .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.PathFindPerf.stats

   .. py:method:: to_string() -> str
      :canonical: deepxube.pathfinding.utils.performance.PathFindPerf.to_string

      .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.PathFindPerf.to_string

.. py:function:: get_eq_weighted_perf(step_to_search_perf: typing.Dict[int, deepxube.pathfinding.utils.performance.PathFindPerf]) -> typing.Tuple[float, float, float]
   :canonical: deepxube.pathfinding.utils.performance.get_eq_weighted_perf

   .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.get_eq_weighted_perf

.. py:function:: print_pathfindperf(step_to_pathfindperf: typing.Dict[int, deepxube.pathfinding.utils.performance.PathFindPerf]) -> None
   :canonical: deepxube.pathfinding.utils.performance.print_pathfindperf

   .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.print_pathfindperf

.. py:function:: is_valid_soln(state: deepxube.base.domain.State, goal: deepxube.base.domain.Goal, soln: typing.List[deepxube.base.domain.Action], domain: deepxube.base.domain.Domain) -> bool
   :canonical: deepxube.pathfinding.utils.performance.is_valid_soln

   .. autodoc2-docstring:: deepxube.pathfinding.utils.performance.is_valid_soln
