:py:mod:`deepxube.utils.timing_utils`
=====================================

.. py:module:: deepxube.utils.timing_utils

.. autodoc2-docstring:: deepxube.utils.timing_utils
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Times <deepxube.utils.timing_utils.Times>`
     - .. autodoc2-docstring:: deepxube.utils.timing_utils.Times
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`add_times <deepxube.utils.timing_utils.add_times>`
     - .. autodoc2-docstring:: deepxube.utils.timing_utils.add_times
          :summary:
   * - :py:obj:`add_counts <deepxube.utils.timing_utils.add_counts>`
     - .. autodoc2-docstring:: deepxube.utils.timing_utils.add_counts
          :summary:
   * - :py:obj:`init_times <deepxube.utils.timing_utils.init_times>`
     - .. autodoc2-docstring:: deepxube.utils.timing_utils.init_times
          :summary:
   * - :py:obj:`init_counts <deepxube.utils.timing_utils.init_counts>`
     - .. autodoc2-docstring:: deepxube.utils.timing_utils.init_counts
          :summary:

API
~~~

.. py:function:: add_times(times: collections.OrderedDict[str, float], times_to_add: collections.OrderedDict[str, float]) -> None
   :canonical: deepxube.utils.timing_utils.add_times

   .. autodoc2-docstring:: deepxube.utils.timing_utils.add_times

.. py:function:: add_counts(counts: collections.OrderedDict[str, int], counts_to_add: collections.OrderedDict[str, int]) -> None
   :canonical: deepxube.utils.timing_utils.add_counts

   .. autodoc2-docstring:: deepxube.utils.timing_utils.add_counts

.. py:function:: init_times(time_names: typing.List[str]) -> collections.OrderedDict[str, float]
   :canonical: deepxube.utils.timing_utils.init_times

   .. autodoc2-docstring:: deepxube.utils.timing_utils.init_times

.. py:function:: init_counts(time_names: typing.List[str]) -> collections.OrderedDict[str, int]
   :canonical: deepxube.utils.timing_utils.init_counts

   .. autodoc2-docstring:: deepxube.utils.timing_utils.init_counts

.. py:class:: Times(time_names: typing.Optional[typing.List[str]] = None)
   :canonical: deepxube.utils.timing_utils.Times

   .. autodoc2-docstring:: deepxube.utils.timing_utils.Times

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.utils.timing_utils.Times.__init__

   .. py:method:: record_time(time_name: str, time_elapsed: float, path: typing.Optional[typing.List[str]] = None) -> None
      :canonical: deepxube.utils.timing_utils.Times.record_time

      .. autodoc2-docstring:: deepxube.utils.timing_utils.Times.record_time

   .. py:method:: add_times(time: deepxube.utils.timing_utils.Times, path: typing.Optional[typing.List[str]] = None) -> None
      :canonical: deepxube.utils.timing_utils.Times.add_times

      .. autodoc2-docstring:: deepxube.utils.timing_utils.Times.add_times

   .. py:method:: reset_times() -> None
      :canonical: deepxube.utils.timing_utils.Times.reset_times

      .. autodoc2-docstring:: deepxube.utils.timing_utils.Times.reset_times

   .. py:method:: get_total_time() -> float
      :canonical: deepxube.utils.timing_utils.Times.get_total_time

      .. autodoc2-docstring:: deepxube.utils.timing_utils.Times.get_total_time

   .. py:method:: get_time_str(prefix: str = '', decplace: int = 2) -> str
      :canonical: deepxube.utils.timing_utils.Times.get_time_str

      .. autodoc2-docstring:: deepxube.utils.timing_utils.Times.get_time_str

   .. py:method:: __str__() -> str
      :canonical: deepxube.utils.timing_utils.Times.__str__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.utils.timing_utils.Times.__repr__
