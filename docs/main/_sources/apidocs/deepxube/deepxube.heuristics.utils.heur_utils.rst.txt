:py:mod:`deepxube.heuristics.utils.heur_utils`
==============================================

.. py:module:: deepxube.heuristics.utils.heur_utils

.. autodoc2-docstring:: deepxube.heuristics.utils.heur_utils
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_zero_heur <deepxube.heuristics.utils.heur_utils.get_zero_heur>`
     - .. autodoc2-docstring:: deepxube.heuristics.utils.heur_utils.get_zero_heur
          :summary:
   * - :py:obj:`policy_fn_rand <deepxube.heuristics.utils.heur_utils.policy_fn_rand>`
     - .. autodoc2-docstring:: deepxube.heuristics.utils.heur_utils.policy_fn_rand
          :summary:
   * - :py:obj:`get_rand_policy <deepxube.heuristics.utils.heur_utils.get_rand_policy>`
     - .. autodoc2-docstring:: deepxube.heuristics.utils.heur_utils.get_rand_policy
          :summary:

API
~~~

.. py:function:: get_zero_heur(heur_type: str) -> deepxube.base.heuristic.HeurFn
   :canonical: deepxube.heuristics.utils.heur_utils.get_zero_heur

   .. autodoc2-docstring:: deepxube.heuristics.utils.heur_utils.get_zero_heur

.. py:function:: policy_fn_rand(domain: deepxube.base.domain.Domain, states: typing.List[deepxube.base.domain.State], num_rand: int) -> typing.Tuple[typing.List[typing.List[deepxube.base.domain.Action]], typing.List[typing.List[float]]]
   :canonical: deepxube.heuristics.utils.heur_utils.policy_fn_rand

   .. autodoc2-docstring:: deepxube.heuristics.utils.heur_utils.policy_fn_rand

.. py:function:: get_rand_policy(domain: deepxube.base.domain.Domain, policy_samp: int) -> deepxube.base.heuristic.PolicyFn
   :canonical: deepxube.heuristics.utils.heur_utils.get_rand_policy

   .. autodoc2-docstring:: deepxube.heuristics.utils.heur_utils.get_rand_policy
