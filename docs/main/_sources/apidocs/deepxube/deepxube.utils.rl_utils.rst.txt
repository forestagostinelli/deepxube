:py:mod:`deepxube.utils.rl_utils`
=================================

.. py:module:: deepxube.utils.rl_utils

.. autodoc2-docstring:: deepxube.utils.rl_utils
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`vi_backup <deepxube.utils.rl_utils.vi_backup>`
     - .. autodoc2-docstring:: deepxube.utils.rl_utils.vi_backup
          :summary:

API
~~~

.. py:function:: vi_backup(is_solved: typing.List[bool], goals: typing.List[deepxube.base.domain.Goal], contexts: typing.List[typing.Any], states_exp: typing.List[typing.List[deepxube.base.domain.State]], tcs_l: typing.List[typing.List[float]], heur_fn: deepxube.base.pathfind_fns.HeurVFn) -> typing.List[float]
   :canonical: deepxube.utils.rl_utils.vi_backup

   .. autodoc2-docstring:: deepxube.utils.rl_utils.vi_backup
