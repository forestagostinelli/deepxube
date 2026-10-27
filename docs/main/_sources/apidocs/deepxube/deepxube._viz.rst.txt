:py:mod:`deepxube._viz`
=======================

.. py:module:: deepxube._viz

.. autodoc2-docstring:: deepxube._viz
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`parse_viz <deepxube._viz.parse_viz>`
     - .. autodoc2-docstring:: deepxube._viz.parse_viz
          :summary:
   * - :py:obj:`fig_to_rgba <deepxube._viz.fig_to_rgba>`
     - .. autodoc2-docstring:: deepxube._viz.fig_to_rgba
          :summary:
   * - :py:obj:`_viz_state_goal_update <deepxube._viz._viz_state_goal_update>`
     - .. autodoc2-docstring:: deepxube._viz._viz_state_goal_update
          :summary:
   * - :py:obj:`viz_step <deepxube._viz.viz_step>`
     - .. autodoc2-docstring:: deepxube._viz.viz_step
          :summary:
   * - :py:obj:`viz_instance <deepxube._viz.viz_instance>`
     - .. autodoc2-docstring:: deepxube._viz.viz_instance
          :summary:
   * - :py:obj:`viz <deepxube._viz.viz>`
     - .. autodoc2-docstring:: deepxube._viz.viz
          :summary:

API
~~~

.. py:function:: parse_viz(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._viz.parse_viz

   .. autodoc2-docstring:: deepxube._viz.parse_viz

.. py:function:: fig_to_rgba(fig: matplotlib.figure.Figure) -> numpy.typing.NDArray
   :canonical: deepxube._viz.fig_to_rgba

   .. autodoc2-docstring:: deepxube._viz.fig_to_rgba

.. py:function:: _viz_state_goal_update(domain: deepxube.base.domain.StateGoalVizable, state: deepxube.base.domain.State, goal: deepxube.base.domain.Goal, fig: matplotlib.figure.Figure) -> None
   :canonical: deepxube._viz._viz_state_goal_update

   .. autodoc2-docstring:: deepxube._viz._viz_state_goal_update

.. py:function:: viz_step(domain: deepxube.base.domain.StateGoalVizable, data: typing.Dict, idx: int, state_idx: int, state_idx_max: int, states_on_path: typing.List[deepxube.base.domain.State], state: deepxube.base.domain.State, goal: deepxube.base.domain.Goal, no_act: bool, fig: matplotlib.figure.Figure) -> typing.Tuple[deepxube.base.domain.State, int]
   :canonical: deepxube._viz.viz_step

   .. autodoc2-docstring:: deepxube._viz.viz_step

.. py:function:: viz_instance(domain: deepxube.base.domain.StateGoalVizable, fig: matplotlib.figure.Figure, file: str, idx: int, v_time: float) -> None
   :canonical: deepxube._viz.viz_instance

   .. autodoc2-docstring:: deepxube._viz.viz_instance

.. py:function:: viz(args: argparse.Namespace) -> None
   :canonical: deepxube._viz.viz

   .. autodoc2-docstring:: deepxube._viz.viz
