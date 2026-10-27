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
   * - :py:obj:`get_names_match_type <deepxube._cli.get_names_match_type>`
     - .. autodoc2-docstring:: deepxube._cli.get_names_match_type
          :summary:
   * - :py:obj:`domain_info <deepxube._cli.domain_info>`
     - .. autodoc2-docstring:: deepxube._cli.domain_info
          :summary:
   * - :py:obj:`nnet_info <deepxube._cli.nnet_info>`
     - .. autodoc2-docstring:: deepxube._cli.nnet_info
          :summary:
   * - :py:obj:`fn_info <deepxube._cli.fn_info>`
     - .. autodoc2-docstring:: deepxube._cli.fn_info
          :summary:
   * - :py:obj:`pathfind_fns_info <deepxube._cli.pathfind_fns_info>`
     - .. autodoc2-docstring:: deepxube._cli.pathfind_fns_info
          :summary:
   * - :py:obj:`updater_fns_info <deepxube._cli.updater_fns_info>`
     - .. autodoc2-docstring:: deepxube._cli.updater_fns_info
          :summary:
   * - :py:obj:`pathfind_info <deepxube._cli.pathfind_info>`
     - .. autodoc2-docstring:: deepxube._cli.pathfind_info
          :summary:
   * - :py:obj:`updater_info <deepxube._cli.updater_info>`
     - .. autodoc2-docstring:: deepxube._cli.updater_info
          :summary:
   * - :py:obj:`trainer_info <deepxube._cli.trainer_info>`
     - .. autodoc2-docstring:: deepxube._cli.trainer_info
          :summary:
   * - :py:obj:`fig_to_rgba <deepxube._cli.fig_to_rgba>`
     - .. autodoc2-docstring:: deepxube._cli.fig_to_rgba
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

.. py:function:: get_names_match_type(req_type: typing.Type, factory: deepxube.base.factory.Factory) -> typing.List[str]
   :canonical: deepxube._cli.get_names_match_type

   .. autodoc2-docstring:: deepxube._cli.get_names_match_type

.. py:function:: domain_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.domain_info

   .. autodoc2-docstring:: deepxube._cli.domain_info

.. py:function:: nnet_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.nnet_info

   .. autodoc2-docstring:: deepxube._cli.nnet_info

.. py:function:: fn_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.fn_info

   .. autodoc2-docstring:: deepxube._cli.fn_info

.. py:function:: pathfind_fns_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.pathfind_fns_info

   .. autodoc2-docstring:: deepxube._cli.pathfind_fns_info

.. py:function:: updater_fns_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.updater_fns_info

   .. autodoc2-docstring:: deepxube._cli.updater_fns_info

.. py:function:: pathfind_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.pathfind_info

   .. autodoc2-docstring:: deepxube._cli.pathfind_info

.. py:function:: updater_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.updater_info

   .. autodoc2-docstring:: deepxube._cli.updater_info

.. py:function:: trainer_info(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.trainer_info

   .. autodoc2-docstring:: deepxube._cli.trainer_info

.. py:function:: fig_to_rgba(fig: matplotlib.figure.Figure) -> numpy.typing.NDArray
   :canonical: deepxube._cli.fig_to_rgba

   .. autodoc2-docstring:: deepxube._cli.fig_to_rgba

.. py:function:: time_test_args(args: argparse.Namespace) -> None
   :canonical: deepxube._cli.time_test_args

   .. autodoc2-docstring:: deepxube._cli.time_test_args

.. py:function:: plot_itr_data(axs: typing.List[matplotlib.axes.Axes], step_slider: matplotlib.widgets.Slider, itr: int, train_summ: deepxube.base.trainer.TrainSummary) -> None
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

.. py:function:: _parse_time(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parse_time

   .. autodoc2-docstring:: deepxube._cli._parse_time

.. py:function:: _parse_problem_instance(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parse_problem_instance

   .. autodoc2-docstring:: deepxube._cli._parse_problem_instance

.. py:function:: _parse_train_summary(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._cli._parse_train_summary

   .. autodoc2-docstring:: deepxube._cli._parse_train_summary
