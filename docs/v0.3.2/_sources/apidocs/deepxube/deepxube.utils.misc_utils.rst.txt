:py:mod:`deepxube.utils.misc_utils`
===================================

.. py:module:: deepxube.utils.misc_utils

.. autodoc2-docstring:: deepxube.utils.misc_utils
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`flatten <deepxube.utils.misc_utils.flatten>`
     - .. autodoc2-docstring:: deepxube.utils.misc_utils.flatten
          :summary:
   * - :py:obj:`unflatten <deepxube.utils.misc_utils.unflatten>`
     - .. autodoc2-docstring:: deepxube.utils.misc_utils.unflatten
          :summary:
   * - :py:obj:`split_evenly <deepxube.utils.misc_utils.split_evenly>`
     - .. autodoc2-docstring:: deepxube.utils.misc_utils.split_evenly
          :summary:
   * - :py:obj:`split_evenly_w_max <deepxube.utils.misc_utils.split_evenly_w_max>`
     - .. autodoc2-docstring:: deepxube.utils.misc_utils.split_evenly_w_max
          :summary:
   * - :py:obj:`remove_all_whitespace <deepxube.utils.misc_utils.remove_all_whitespace>`
     - .. autodoc2-docstring:: deepxube.utils.misc_utils.remove_all_whitespace
          :summary:
   * - :py:obj:`random_subset <deepxube.utils.misc_utils.random_subset>`
     - .. autodoc2-docstring:: deepxube.utils.misc_utils.random_subset
          :summary:
   * - :py:obj:`boltzmann <deepxube.utils.misc_utils.boltzmann>`
     - .. autodoc2-docstring:: deepxube.utils.misc_utils.boltzmann
          :summary:
   * - :py:obj:`scalar_stats <deepxube.utils.misc_utils.scalar_stats>`
     - .. autodoc2-docstring:: deepxube.utils.misc_utils.scalar_stats
          :summary:

API
~~~

.. py:function:: flatten(data: typing.List[typing.List[typing.Any]]) -> typing.Tuple[typing.List[typing.Any], typing.List[int]]
   :canonical: deepxube.utils.misc_utils.flatten

   .. autodoc2-docstring:: deepxube.utils.misc_utils.flatten

.. py:function:: unflatten(data: typing.Union[typing.List[typing.Any], numpy.typing.NDArray[typing.Any]], split_idxs: typing.List[int]) -> typing.List[typing.List[typing.Any]]
   :canonical: deepxube.utils.misc_utils.unflatten

   .. autodoc2-docstring:: deepxube.utils.misc_utils.unflatten

.. py:function:: split_evenly(num_total: int, num_splits: int) -> typing.List[int]
   :canonical: deepxube.utils.misc_utils.split_evenly

   .. autodoc2-docstring:: deepxube.utils.misc_utils.split_evenly

.. py:function:: split_evenly_w_max(num_total: int, num_splits: int, max_per: int) -> typing.List[int]
   :canonical: deepxube.utils.misc_utils.split_evenly_w_max

   .. autodoc2-docstring:: deepxube.utils.misc_utils.split_evenly_w_max

.. py:function:: remove_all_whitespace(val: str) -> str
   :canonical: deepxube.utils.misc_utils.remove_all_whitespace

   .. autodoc2-docstring:: deepxube.utils.misc_utils.remove_all_whitespace

.. py:function:: random_subset(set_orig: typing.Union[typing.Set[typing.Any], frozenset[typing.Any]], keep_prob: bool) -> typing.Set[typing.Any]
   :canonical: deepxube.utils.misc_utils.random_subset

   .. autodoc2-docstring:: deepxube.utils.misc_utils.random_subset

.. py:function:: boltzmann(vals: typing.List[float], temp: float) -> typing.List[float]
   :canonical: deepxube.utils.misc_utils.boltzmann

   .. autodoc2-docstring:: deepxube.utils.misc_utils.boltzmann

.. py:function:: scalar_stats(data: numpy.typing.NDArray) -> str
   :canonical: deepxube.utils.misc_utils.scalar_stats

   .. autodoc2-docstring:: deepxube.utils.misc_utils.scalar_stats
