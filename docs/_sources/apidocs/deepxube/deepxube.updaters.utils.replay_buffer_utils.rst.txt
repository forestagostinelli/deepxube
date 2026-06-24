:py:mod:`deepxube.updaters.utils.replay_buffer_utils`
=====================================================

.. py:module:: deepxube.updaters.utils.replay_buffer_utils

.. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ReplayBuffer <deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer>`
     -
   * - :py:obj:`ReplayBufferV <deepxube.updaters.utils.replay_buffer_utils.ReplayBufferV>`
     -
   * - :py:obj:`ReplayBufferQ <deepxube.updaters.utils.replay_buffer_utils.ReplayBufferQ>`
     -
   * - :py:obj:`ReplayBufferP <deepxube.updaters.utils.replay_buffer_utils.ReplayBufferP>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ReplayVElem <deepxube.updaters.utils.replay_buffer_utils.ReplayVElem>`
     - .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayVElem
          :summary:
   * - :py:obj:`ReplayQElem <deepxube.updaters.utils.replay_buffer_utils.ReplayQElem>`
     - .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayQElem
          :summary:
   * - :py:obj:`ReplayPElem <deepxube.updaters.utils.replay_buffer_utils.ReplayPElem>`
     - .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayPElem
          :summary:
   * - :py:obj:`ReplayVRet <deepxube.updaters.utils.replay_buffer_utils.ReplayVRet>`
     - .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayVRet
          :summary:
   * - :py:obj:`ReplayQRet <deepxube.updaters.utils.replay_buffer_utils.ReplayQRet>`
     - .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayQRet
          :summary:
   * - :py:obj:`ReplayPRet <deepxube.updaters.utils.replay_buffer_utils.ReplayPRet>`
     - .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayPRet
          :summary:
   * - :py:obj:`Elem <deepxube.updaters.utils.replay_buffer_utils.Elem>`
     - .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.Elem
          :summary:
   * - :py:obj:`SampRet <deepxube.updaters.utils.replay_buffer_utils.SampRet>`
     - .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.SampRet
          :summary:

API
~~~

.. py:data:: ReplayVElem
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayVElem
   :value: None

   .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayVElem

.. py:data:: ReplayQElem
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayQElem
   :value: None

   .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayQElem

.. py:data:: ReplayPElem
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayPElem
   :value: None

   .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayPElem

.. py:data:: ReplayVRet
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayVRet
   :value: None

   .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayVRet

.. py:data:: ReplayQRet
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayQRet
   :value: None

   .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayQRet

.. py:data:: ReplayPRet
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayPRet
   :value: None

   .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayPRet

.. py:data:: Elem
   :canonical: deepxube.updaters.utils.replay_buffer_utils.Elem
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.Elem

.. py:data:: SampRet
   :canonical: deepxube.updaters.utils.replay_buffer_utils.SampRet
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.SampRet

.. py:class:: ReplayBuffer(max_size: int)
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer

   Bases: :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.updaters.utils.replay_buffer_utils.Elem`\ , :py:obj:`deepxube.updaters.utils.replay_buffer_utils.SampRet`\ ], :py:obj:`abc.ABC`

   .. py:method:: add(data: typing.List[deepxube.updaters.utils.replay_buffer_utils.Elem]) -> None
      :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer.add

      .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer.add

   .. py:method:: sample(num: int) -> deepxube.updaters.utils.replay_buffer_utils.SampRet
      :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer.sample

      .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer.sample

   .. py:method:: size() -> int
      :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer.size

      .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer.size

   .. py:method:: max_size() -> int
      :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer.max_size

      .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer.max_size

   .. py:method:: _elems_to_ret(elems: typing.List[deepxube.updaters.utils.replay_buffer_utils.Elem]) -> deepxube.updaters.utils.replay_buffer_utils.SampRet
      :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer._elems_to_ret
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer._elems_to_ret

.. py:class:: ReplayBufferV(max_size: int)
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBufferV

   Bases: :py:obj:`deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer`\ [\ :py:obj:`deepxube.updaters.utils.replay_buffer_utils.ReplayVElem`\ , :py:obj:`deepxube.updaters.utils.replay_buffer_utils.ReplayVRet`\ ]

   .. py:method:: _elems_to_ret(elems: typing.List[deepxube.updaters.utils.replay_buffer_utils.ReplayVElem]) -> deepxube.updaters.utils.replay_buffer_utils.ReplayVRet
      :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBufferV._elems_to_ret

      .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayBufferV._elems_to_ret

.. py:class:: ReplayBufferQ(max_size: int)
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBufferQ

   Bases: :py:obj:`deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer`\ [\ :py:obj:`deepxube.updaters.utils.replay_buffer_utils.ReplayQElem`\ , :py:obj:`deepxube.updaters.utils.replay_buffer_utils.ReplayQRet`\ ]

   .. py:method:: _elems_to_ret(elems: typing.List[deepxube.updaters.utils.replay_buffer_utils.ReplayQElem]) -> deepxube.updaters.utils.replay_buffer_utils.ReplayQRet
      :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBufferQ._elems_to_ret

      .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayBufferQ._elems_to_ret

.. py:class:: ReplayBufferP(max_size: int)
   :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBufferP

   Bases: :py:obj:`deepxube.updaters.utils.replay_buffer_utils.ReplayBuffer`\ [\ :py:obj:`deepxube.updaters.utils.replay_buffer_utils.ReplayPElem`\ , :py:obj:`deepxube.updaters.utils.replay_buffer_utils.ReplayPRet`\ ]

   .. py:method:: _elems_to_ret(elems: typing.List[deepxube.updaters.utils.replay_buffer_utils.ReplayPElem]) -> deepxube.updaters.utils.replay_buffer_utils.ReplayPRet
      :canonical: deepxube.updaters.utils.replay_buffer_utils.ReplayBufferP._elems_to_ret

      .. autodoc2-docstring:: deepxube.updaters.utils.replay_buffer_utils.ReplayBufferP._elems_to_ret
