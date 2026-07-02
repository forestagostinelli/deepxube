:py:mod:`deepxube._cli`
=======================

.. py:module:: deepxube._cli

.. autodoc2-docstring:: deepxube._cli
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`plot_scatter <deepxube._cli.plot_scatter>`
     - .. autodoc2-docstring:: deepxube._cli.plot_scatter
          :summary:
   * - :py:obj:`get_immediate_mixins <deepxube._cli.get_immediate_mixins>`
     - .. autodoc2-docstring:: deepxube._cli.get_immediate_mixins
          :summary:
   * - :py:obj:`domain_info <deepxube._cli.domain_info>`
     - .. autodoc2-docstring:: deepxube._cli.domain_info
          :summary:
   * - :py:obj:`heur_info <deepxube._cli.heur_info>`
     - .. autodoc2-docstring:: deepxube._cli.heur_info
          :summary:
   * - :py:obj:`pathfinding_info <deepxube._cli.pathfinding_info>`
     - .. autodoc2-docstring:: deepxube._cli.pathfinding_info
          :summary:
   * - :py:obj:`fig_to_rgba <deepxube._cli.fig_to_rgba>`
     - .. autodoc2-docstring:: deepxube._cli.fig_to_rgba
          :summary:
   * - :py:obj:`viz_step <deepxube._cli.viz_step>`
     - .. autodoc2-docstring:: deepxube._cli.viz_step
          :summary:
   * - :py:obj:`viz <deepxube._cli.viz>`
     - .. autodoc2-docstring:: deepxube._cli.viz
          :summary:
   * - :py:obj:`_viz_state_goal_update <deepxube._cli._viz_state_goal_update>`
     - .. autodoc2-docstring:: deepxube._cli._viz_state_goal_update
          :summary:
   * - :py:obj:`time_test_args <deepxube._cli.time_test_args>`
     - .. autodoc2-docstring:: deepxube._cli.time_test_args
          :summary:
   * - :py:obj:`plot_itr_data <deepxube._cli.plot_itr_data>`
     - .. autodoc2-docstring:: deepxube._cli.plot_itr_data
          :summary:
   * - :py:obj:`train_summary <deepxube._cli.train_summary>`
     - .. autodoc2-docstring:: deepxube._cli.train_summary
          :summary:
   * - :py:obj:`problem_inst_gen <deepxube._cli.problem_inst_gen>`
     - .. autodoc2-docstring:: deepxube._cli.problem_inst_gen
          :summary:
   * - :py:obj:`main <deepxube._cli.main>`
     - .. autodoc2-docstring:: deepxube._cli.main
          :summary:
   * - :py:obj:`_parser_domain_info <deepxube._cli._parser_domain_info>`
     - .. autodoc2-docstring:: deepxube._cli._parser_domain_info
          :summary:
   * - :py:obj:`_parser_heur_info <deepxube._cli._parser_heur_info>`
     - .. autodoc2-docstring:: deepxube._cli._parser_heur_info
          :summary:
   * - :py:obj:`_parser_pathfind_info <deepxube._cli._parser_pathfind_info>`
     - .. autodoc2-docstring:: deepxube._cli._parser_pathfind_info
          :summary:
   * - :py:obj:`_parse_viz_info <deepxube._cli._parse_viz_info>`
     - .. autodoc2-docstring:: deepxube._cli._parse_viz_info
          :summary:
   * - :py:obj:`_parse_time <deepxube._cli._parse_time>`
     - .. autodoc2-docstring:: deepxube._cli._parse_time
          :summary:
   * - :py:obj:`_parse_problem_instance <deepxube._cli._parse_problem_instance>`
     - .. autodoc2-docstring:: deepxube._cli._parse_problem_instance
          :summary:
   * - :py:obj:`_parse_train_summary <deepxube._cli._parse_train_summary>`
     - .. autodoc2-docstring:: deepxube._cli._parse_train_summary
          :summary:

API
~~~

.. py:function:: plot_scatter(ax: matplotlib.axes.Axes, x: typing.Any, y: typing.Any, x_label: str, y_label: str, xy_line: bool, alpha: float = 1.0, title: str = '') -> None
   :canonical: deepxube._cli.plot_scatter

   .. autodoc2-docstring:: deepxube._cli.plot_scatter

.. py:function:: get_immediate_mixins(cls: typing.Type[object], mixin_base: typing.Type) -> typing.List[typing.Type]
   :canonical: deepxube._cli.get_immediate_mixins

   .. autodoc2-docstring:: deepxube._cli.get_immediate_mixins

.. py:function:: domain_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.domain_info

   .. autodoc2-docstring:: deepxube._cli.domain_info

.. py:function:: heur_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.heur_info

   .. autodoc2-docstring:: deepxube._cli.heur_info

.. py:function:: pathfinding_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.pathfinding_info

   .. autodoc2-docstring:: deepxube._cli.pathfinding_info

.. py:function:: fig_to_rgba(fig: matplotlib.figure.Figure) -> numpy.typing.NDArray
   :canonical: deepxube._cli.fig_to_rgba

   .. autodoc2-docstring:: deepxube._cli.fig_to_rgba

.. py:function:: viz_step(domain: deepxube.base.domain.StateGoalVizable, data: typing.Dict, idx: int, state_idx: int, state_idx_max: int, states_on_path: typing.List[deepxube.base.domain.State], state: deepxube.base.domain.State, goal: deepxube.base.domain.Goal, no_act: bool, fig: matplotlib.figure.Figure) -> typing.Tuple[deepxube.base.domain.State, int]
   :canonical: deepxube._cli.viz_step

   .. autodoc2-docstring:: deepxube._cli.viz_step

.. py:function:: viz(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.viz

   .. autodoc2-docstring:: deepxube._cli.viz

.. py:function:: _viz_state_goal_update(domain: deepxube.base.domain.StateGoalVizable, state: deepxube.base.domain.State, goal: deepxube.base.domain.Goal, fig: matplotlib.figure.Figure) -> None
   :canonical: deepxube._cli._viz_state_goal_update

   .. autodoc2-docstring:: deepxube._cli._viz_state_goal_update

.. py:function:: time_test_args(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.time_test_args

   .. autodoc2-docstring:: deepxube._cli.time_test_args

.. py:function:: plot_itr_data(axs: typing.List[matplotlib.axes.Axes], step_slider: matplotlib.widgets.Slider, itr: int, itr_to_in_out: typing.Dict[int, typing.Tuple[numpy.typing.NDArray, numpy.typing.NDArray]], itr_to_steps_to_pathfindstats: typing.Dict[int, typing.Dict[int, typing.Dict]]) -> None
   :canonical: deepxube._cli.plot_itr_data

   .. autodoc2-docstring:: deepxube._cli.plot_itr_data

.. py:function:: train_summary(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.train_summary

   .. autodoc2-docstring:: deepxube._cli.train_summary

.. py:function:: problem_inst_gen(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.problem_inst_gen

   .. autodoc2-docstring:: deepxube._cli.problem_inst_gen

.. py:function:: main() -> None
   :canonical: deepxube._cli.main

   .. autodoc2-docstring:: deepxube._cli.main

.. py:function:: _parser_domain_info(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parser_domain_info

   .. autodoc2-docstring:: deepxube._cli._parser_domain_info

.. py:function:: _parser_heur_info(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parser_heur_info

   .. autodoc2-docstring:: deepxube._cli._parser_heur_info

.. py:function:: _parser_pathfind_info(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parser_pathfind_info

   .. autodoc2-docstring:: deepxube._cli._parser_pathfind_info

.. py:function:: _parse_viz_info(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parse_viz_info

   .. autodoc2-docstring:: deepxube._cli._parse_viz_info

.. py:function:: _parse_time(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parse_time

   .. autodoc2-docstring:: deepxube._cli._parse_time

.. py:function:: _parse_problem_instance(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parse_problem_instance

   .. autodoc2-docstring:: deepxube._cli._parse_problem_instance

.. py:function:: _parse_train_summary(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parse_train_summary

   .. autodoc2-docstring:: deepxube._cli._parse_train_summary
