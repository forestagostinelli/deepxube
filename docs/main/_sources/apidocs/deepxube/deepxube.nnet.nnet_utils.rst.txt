:py:mod:`deepxube.nnet.nnet_utils`
==================================

.. py:module:: deepxube.nnet.nnet_utils

.. autodoc2-docstring:: deepxube.nnet.nnet_utils
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NNetParInfo <deepxube.nnet.nnet_utils.NNetParInfo>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetParInfo
          :summary:
   * - :py:obj:`NNetPar <deepxube.nnet.nnet_utils.NNetPar>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetPar
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`to_pytorch_input <deepxube.nnet.nnet_utils.to_pytorch_input>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.to_pytorch_input
          :summary:
   * - :py:obj:`get_device <deepxube.nnet.nnet_utils.get_device>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.get_device
          :summary:
   * - :py:obj:`load_nnet <deepxube.nnet.nnet_utils.load_nnet>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.load_nnet
          :summary:
   * - :py:obj:`get_available_gpu_nums <deepxube.nnet.nnet_utils.get_available_gpu_nums>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.get_available_gpu_nums
          :summary:
   * - :py:obj:`nnet_batched <deepxube.nnet.nnet_utils.nnet_batched>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.nnet_batched
          :summary:
   * - :py:obj:`nnet_in_out_shared_q <deepxube.nnet.nnet_utils.nnet_in_out_shared_q>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.nnet_in_out_shared_q
          :summary:
   * - :py:obj:`nnet_fn_runner <deepxube.nnet.nnet_utils.nnet_fn_runner>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.nnet_fn_runner
          :summary:
   * - :py:obj:`get_nnet_par_infos <deepxube.nnet.nnet_utils.get_nnet_par_infos>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.get_nnet_par_infos
          :summary:
   * - :py:obj:`start_nnet_fn_runners <deepxube.nnet.nnet_utils.start_nnet_fn_runners>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.start_nnet_fn_runners
          :summary:
   * - :py:obj:`stop_nnet_runners <deepxube.nnet.nnet_utils.stop_nnet_runners>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.stop_nnet_runners
          :summary:
   * - :py:obj:`get_nnet_par_out <deepxube.nnet.nnet_utils.get_nnet_par_out>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.get_nnet_par_out
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NNetCallable <deepxube.nnet.nnet_utils.NNetCallable>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetCallable
          :summary:
   * - :py:obj:`NNetFn <deepxube.nnet.nnet_utils.NNetFn>`
     - .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetFn
          :summary:

API
~~~

.. py:function:: to_pytorch_input(states_nnet: typing.List[numpy.typing.NDArray[typing.Any]], device: torch.device) -> typing.List[torch.Tensor]
   :canonical: deepxube.nnet.nnet_utils.to_pytorch_input

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.to_pytorch_input

.. py:function:: get_device() -> typing.Tuple[torch.device, typing.List[int], bool]
   :canonical: deepxube.nnet.nnet_utils.get_device

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.get_device

.. py:function:: load_nnet(model_file: str, nnet: torch.nn.Module, device: typing.Optional[torch.device] = None) -> torch.nn.Module
   :canonical: deepxube.nnet.nnet_utils.load_nnet

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.load_nnet

.. py:function:: get_available_gpu_nums() -> typing.List[int]
   :canonical: deepxube.nnet.nnet_utils.get_available_gpu_nums

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.get_available_gpu_nums

.. py:function:: nnet_batched(nnet: torch.nn.Module, inputs: typing.List[numpy.typing.NDArray[typing.Any]], batch_size: typing.Optional[int], device: torch.device) -> typing.List[numpy.typing.NDArray[numpy.float64]]
   :canonical: deepxube.nnet.nnet_utils.nnet_batched

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.nnet_batched

.. py:class:: NNetParInfo
   :canonical: deepxube.nnet.nnet_utils.NNetParInfo

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetParInfo

   .. py:attribute:: nnet_i_q
      :canonical: deepxube.nnet.nnet_utils.NNetParInfo.nnet_i_q
      :type: torch.multiprocessing.Queue
      :value: None

      .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetParInfo.nnet_i_q

   .. py:attribute:: nnet_o_q
      :canonical: deepxube.nnet.nnet_utils.NNetParInfo.nnet_o_q
      :type: torch.multiprocessing.Queue
      :value: None

      .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetParInfo.nnet_o_q

   .. py:attribute:: proc_id
      :canonical: deepxube.nnet.nnet_utils.NNetParInfo.proc_id
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetParInfo.proc_id

.. py:function:: nnet_in_out_shared_q(nnet: torch.nn.Module, inputs_nnet_shm: typing.List[deepxube.utils.data_utils.SharedNDArray], batch_size: typing.Optional[int], device: torch.device, nnet_o_q: torch.multiprocessing.Queue) -> None
   :canonical: deepxube.nnet.nnet_utils.nnet_in_out_shared_q

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.nnet_in_out_shared_q

.. py:function:: nnet_fn_runner(nnet_i_q: torch.multiprocessing.Queue, nnet_o_qs: typing.List[torch.multiprocessing.Queue], model_file: str, device: torch.device, on_gpu: bool, gpu_num: int, get_nnet: typing.Callable[[], torch.nn.Module], batch_size: typing.Optional[int]) -> None
   :canonical: deepxube.nnet.nnet_utils.nnet_fn_runner

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.nnet_fn_runner

.. py:function:: get_nnet_par_infos(num_procs: int) -> typing.List[deepxube.nnet.nnet_utils.NNetParInfo]
   :canonical: deepxube.nnet.nnet_utils.get_nnet_par_infos

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.get_nnet_par_infos

.. py:function:: start_nnet_fn_runners(get_nnet: typing.Callable[[], torch.nn.Module], nnet_par_infos: typing.List[deepxube.nnet.nnet_utils.NNetParInfo], model_file: str, device: torch.device, on_gpu: bool, batch_size: typing.Optional[int] = None) -> typing.List[multiprocessing.process.BaseProcess]
   :canonical: deepxube.nnet.nnet_utils.start_nnet_fn_runners

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.start_nnet_fn_runners

.. py:function:: stop_nnet_runners(nnet_fn_procs: typing.List[multiprocessing.process.BaseProcess], nnet_par_infos: typing.List[deepxube.nnet.nnet_utils.NNetParInfo]) -> None
   :canonical: deepxube.nnet.nnet_utils.stop_nnet_runners

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.stop_nnet_runners

.. py:data:: NNetCallable
   :canonical: deepxube.nnet.nnet_utils.NNetCallable
   :value: None

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetCallable

.. py:data:: NNetFn
   :canonical: deepxube.nnet.nnet_utils.NNetFn
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetFn

.. py:class:: NNetPar
   :canonical: deepxube.nnet.nnet_utils.NNetPar

   Bases: :py:obj:`abc.ABC`, :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.nnet.nnet_utils.NNetFn`\ ]

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetPar

   .. py:method:: get_nnet_fn(nnet: torch.nn.Module, batch_size: typing.Optional[int], device: torch.device, update_num: typing.Optional[int]) -> deepxube.nnet.nnet_utils.NNetFn
      :canonical: deepxube.nnet.nnet_utils.NNetPar.get_nnet_fn
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetPar.get_nnet_fn

   .. py:method:: get_nnet_par_fn(nnet_par_info: deepxube.nnet.nnet_utils.NNetParInfo, update_num: typing.Optional[int]) -> deepxube.nnet.nnet_utils.NNetFn
      :canonical: deepxube.nnet.nnet_utils.NNetPar.get_nnet_par_fn
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetPar.get_nnet_par_fn

   .. py:method:: get_nnet() -> torch.nn.Module
      :canonical: deepxube.nnet.nnet_utils.NNetPar.get_nnet
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.nnet.nnet_utils.NNetPar.get_nnet

   .. py:method:: __repr__() -> str
      :canonical: deepxube.nnet.nnet_utils.NNetPar.__repr__

.. py:function:: get_nnet_par_out(inputs_nnet: typing.List[numpy.typing.NDArray], nnet_par_info: deepxube.nnet.nnet_utils.NNetParInfo) -> typing.List[numpy.typing.NDArray]
   :canonical: deepxube.nnet.nnet_utils.get_nnet_par_out

   .. autodoc2-docstring:: deepxube.nnet.nnet_utils.get_nnet_par_out
