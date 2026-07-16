:py:mod:`deepxube._solve`
=========================

.. py:module:: deepxube._solve

.. autodoc2-docstring:: deepxube._solve
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`policy_fn_rand <deepxube._solve.policy_fn_rand>`
     - .. autodoc2-docstring:: deepxube._solve.policy_fn_rand
          :summary:
   * - :py:obj:`parse_solve <deepxube._solve.parse_solve>`
     - .. autodoc2-docstring:: deepxube._solve.parse_solve
          :summary:
   * - :py:obj:`solve_cli <deepxube._solve.solve_cli>`
     - .. autodoc2-docstring:: deepxube._solve.solve_cli
          :summary:
   * - :py:obj:`_get_mean <deepxube._solve._get_mean>`
     - .. autodoc2-docstring:: deepxube._solve._get_mean
          :summary:

API
~~~

.. py:function:: policy_fn_rand(domain: deepxube.base.domain.Domain, states: typing.List[deepxube.base.domain.State], num_rand: int) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
   :canonical: deepxube._solve.policy_fn_rand

   .. autodoc2-docstring:: deepxube._solve.policy_fn_rand

.. py:function:: parse_solve(parser: argparse.ArgumentParser) -> None
   :canonical: deepxube._solve.parse_solve

   .. autodoc2-docstring:: deepxube._solve.parse_solve

.. py:function:: solve_cli(args: argparse.Namespace) -> None
   :canonical: deepxube._solve.solve_cli

   .. autodoc2-docstring:: deepxube._solve.solve_cli

.. py:function:: _get_mean(results: typing.Dict[str, typing.Any], key: str) -> float
   :canonical: deepxube._solve._get_mean

   .. autodoc2-docstring:: deepxube._solve._get_mean
