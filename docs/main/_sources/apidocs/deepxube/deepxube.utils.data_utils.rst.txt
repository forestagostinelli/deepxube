:py:mod:`deepxube.utils.data_utils`
===================================

.. py:module:: deepxube.utils.data_utils

.. autodoc2-docstring:: deepxube.utils.data_utils
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Logger <deepxube.utils.data_utils.Logger>`
     - .. autodoc2-docstring:: deepxube.utils.data_utils.Logger
          :summary:
   * - :py:obj:`SharedNDArray <deepxube.utils.data_utils.SharedNDArray>`
     - .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_nowait_noerr <deepxube.utils.data_utils.get_nowait_noerr>`
     - .. autodoc2-docstring:: deepxube.utils.data_utils.get_nowait_noerr
          :summary:
   * - :py:obj:`get_while_not_empty <deepxube.utils.data_utils.get_while_not_empty>`
     - .. autodoc2-docstring:: deepxube.utils.data_utils.get_while_not_empty
          :summary:
   * - :py:obj:`get_in_order <deepxube.utils.data_utils.get_in_order>`
     - .. autodoc2-docstring:: deepxube.utils.data_utils.get_in_order
          :summary:
   * - :py:obj:`copy_dir_files <deepxube.utils.data_utils.copy_dir_files>`
     - .. autodoc2-docstring:: deepxube.utils.data_utils.copy_dir_files
          :summary:
   * - :py:obj:`sel_l <deepxube.utils.data_utils.sel_l>`
     - .. autodoc2-docstring:: deepxube.utils.data_utils.sel_l
          :summary:
   * - :py:obj:`combine_l_l <deepxube.utils.data_utils.combine_l_l>`
     - .. autodoc2-docstring:: deepxube.utils.data_utils.combine_l_l
          :summary:
   * - :py:obj:`np_to_shnd <deepxube.utils.data_utils.np_to_shnd>`
     - .. autodoc2-docstring:: deepxube.utils.data_utils.np_to_shnd
          :summary:

API
~~~

.. py:class:: Logger(filename: str, mode: str = 'a', echo: bool = True)
   :canonical: deepxube.utils.data_utils.Logger

   Bases: :py:obj:`object`

   .. autodoc2-docstring:: deepxube.utils.data_utils.Logger

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.utils.data_utils.Logger.__init__

   .. py:method:: write(message: str) -> None
      :canonical: deepxube.utils.data_utils.Logger.write

      .. autodoc2-docstring:: deepxube.utils.data_utils.Logger.write

   .. py:method:: flush() -> None
      :canonical: deepxube.utils.data_utils.Logger.flush

      .. autodoc2-docstring:: deepxube.utils.data_utils.Logger.flush

.. py:function:: get_nowait_noerr(q: multiprocessing.Queue) -> typing.Any
   :canonical: deepxube.utils.data_utils.get_nowait_noerr

   .. autodoc2-docstring:: deepxube.utils.data_utils.get_nowait_noerr

.. py:function:: get_while_not_empty(q: multiprocessing.Queue) -> typing.List[typing.Any]
   :canonical: deepxube.utils.data_utils.get_while_not_empty

   .. autodoc2-docstring:: deepxube.utils.data_utils.get_while_not_empty

.. py:function:: get_in_order(q: multiprocessing.Queue, num: int) -> typing.List[typing.Any]
   :canonical: deepxube.utils.data_utils.get_in_order

   .. autodoc2-docstring:: deepxube.utils.data_utils.get_in_order

.. py:function:: copy_dir_files(src_dir: str, dest_dir: str) -> None
   :canonical: deepxube.utils.data_utils.copy_dir_files

   .. autodoc2-docstring:: deepxube.utils.data_utils.copy_dir_files

.. py:function:: sel_l(data_l: typing.List[numpy.typing.NDArray], idxs: numpy.typing.NDArray) -> typing.List[numpy.typing.NDArray]
   :canonical: deepxube.utils.data_utils.sel_l

   .. autodoc2-docstring:: deepxube.utils.data_utils.sel_l

.. py:function:: combine_l_l(l_l: typing.List[typing.List[numpy.typing.NDArray]], comb: str) -> typing.List[numpy.typing.NDArray]
   :canonical: deepxube.utils.data_utils.combine_l_l

   .. autodoc2-docstring:: deepxube.utils.data_utils.combine_l_l

.. py:class:: SharedNDArray(shape: typing.Tuple[int, ...], dtype: numpy.dtype, name: typing.Optional[str], create: bool)
   :canonical: deepxube.utils.data_utils.SharedNDArray

   .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray.__init__

   .. py:property:: name
      :canonical: deepxube.utils.data_utils.SharedNDArray.name
      :type: str

      .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray.name

   .. py:method:: close() -> None
      :canonical: deepxube.utils.data_utils.SharedNDArray.close

      .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray.close

   .. py:method:: unlink() -> None
      :canonical: deepxube.utils.data_utils.SharedNDArray.unlink

      .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray.unlink

   .. py:method:: __reduce__() -> typing.Tuple[typing.Type, typing.Tuple[typing.Tuple[int, ...], numpy.dtype, str, bool]]
      :canonical: deepxube.utils.data_utils.SharedNDArray.__reduce__

      .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray.__reduce__

   .. py:method:: __getitem__(key: typing.Any) -> numpy.typing.NDArray
      :canonical: deepxube.utils.data_utils.SharedNDArray.__getitem__

      .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray.__getitem__

   .. py:method:: __setitem__(key: typing.Any, value: numpy.typing.ArrayLike) -> None
      :canonical: deepxube.utils.data_utils.SharedNDArray.__setitem__

      .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray.__setitem__

   .. py:method:: __array__() -> numpy.typing.NDArray
      :canonical: deepxube.utils.data_utils.SharedNDArray.__array__

      .. autodoc2-docstring:: deepxube.utils.data_utils.SharedNDArray.__array__

   .. py:method:: __repr__() -> str
      :canonical: deepxube.utils.data_utils.SharedNDArray.__repr__

.. py:function:: np_to_shnd(arr: numpy.typing.NDArray) -> deepxube.utils.data_utils.SharedNDArray
   :canonical: deepxube.utils.data_utils.np_to_shnd

   .. autodoc2-docstring:: deepxube.utils.data_utils.np_to_shnd
