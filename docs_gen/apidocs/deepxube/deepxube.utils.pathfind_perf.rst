:py:mod:`deepxube.utils.pathfind_perf`
======================================

.. py:module:: deepxube.utils.pathfind_perf

.. autodoc2-docstring:: deepxube.utils.pathfind_perf
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PathFindPerf <deepxube.utils.pathfind_perf.PathFindPerf>`
     - .. autodoc2-docstring:: deepxube.utils.pathfind_perf.PathFindPerf
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_eq_weighted_perf <deepxube.utils.pathfind_perf.get_eq_weighted_perf>`
     - .. autodoc2-docstring:: deepxube.utils.pathfind_perf.get_eq_weighted_perf
          :summary:
   * - :py:obj:`print_pathfindperf <deepxube.utils.pathfind_perf.print_pathfindperf>`
     - .. autodoc2-docstring:: deepxube.utils.pathfind_perf.print_pathfindperf
          :summary:
   * - :py:obj:`is_valid_soln <deepxube.utils.pathfind_perf.is_valid_soln>`
     - .. autodoc2-docstring:: deepxube.utils.pathfind_perf.is_valid_soln
          :summary:

API
~~~

.. py:class:: PathFindPerf()
   :canonical: deepxube.utils.pathfind_perf.PathFindPerf

   .. autodoc2-docstring:: deepxube.utils.pathfind_perf.PathFindPerf

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.utils.pathfind_perf.PathFindPerf.__init__

   .. py:method:: update_perf(instance: deepxube.base.pathfinding.Instance) -> None
      :canonical: deepxube.utils.pathfind_perf.PathFindPerf.update_perf

      .. autodoc2-docstring:: deepxube.utils.pathfind_perf.PathFindPerf.update_perf

   .. py:method:: comb_perf(search_perf2: deepxube.utils.pathfind_perf.PathFindPerf) -> deepxube.utils.pathfind_perf.PathFindPerf
      :canonical: deepxube.utils.pathfind_perf.PathFindPerf.comb_perf

      .. autodoc2-docstring:: deepxube.utils.pathfind_perf.PathFindPerf.comb_perf

   .. py:method:: per_solved() -> float
      :canonical: deepxube.utils.pathfind_perf.PathFindPerf.per_solved

      .. autodoc2-docstring:: deepxube.utils.pathfind_perf.PathFindPerf.per_solved

   .. py:method:: stats() -> typing.Tuple[float, float, float]
      :canonical: deepxube.utils.pathfind_perf.PathFindPerf.stats

      .. autodoc2-docstring:: deepxube.utils.pathfind_perf.PathFindPerf.stats

   .. py:method:: to_string() -> str
      :canonical: deepxube.utils.pathfind_perf.PathFindPerf.to_string

      .. autodoc2-docstring:: deepxube.utils.pathfind_perf.PathFindPerf.to_string

.. py:function:: get_eq_weighted_perf(step_to_search_perf: typing.Dict[int, deepxube.utils.pathfind_perf.PathFindPerf]) -> typing.Tuple[float, float, float]
   :canonical: deepxube.utils.pathfind_perf.get_eq_weighted_perf

   .. autodoc2-docstring:: deepxube.utils.pathfind_perf.get_eq_weighted_perf

.. py:function:: print_pathfindperf(step_to_pathfindperf: typing.Dict[int, deepxube.utils.pathfind_perf.PathFindPerf]) -> None
   :canonical: deepxube.utils.pathfind_perf.print_pathfindperf

   .. autodoc2-docstring:: deepxube.utils.pathfind_perf.print_pathfindperf

.. py:function:: is_valid_soln(state: deepxube.base.domain.State, goal: deepxube.base.domain.Goal, soln: typing.List[deepxube.base.domain.Action], domain: deepxube.base.domain.Domain) -> bool
   :canonical: deepxube.utils.pathfind_perf.is_valid_soln

   .. autodoc2-docstring:: deepxube.utils.pathfind_perf.is_valid_soln
