:py:mod:`deepxube.pytorch.nnet_utils`
=====================================

.. py:module:: deepxube.pytorch.nnet_utils

.. autodoc2-docstring:: deepxube.pytorch.nnet_utils
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NNetParInfo <deepxube.pytorch.nnet_utils.NNetParInfo>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetParInfo
          :summary:
   * - :py:obj:`ProcessedInput <deepxube.pytorch.nnet_utils.ProcessedInput>`
     -
   * - :py:obj:`NNetPar <deepxube.pytorch.nnet_utils.NNetPar>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`to_pytorch_input <deepxube.pytorch.nnet_utils.to_pytorch_input>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.to_pytorch_input
          :summary:
   * - :py:obj:`get_device <deepxube.pytorch.nnet_utils.get_device>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.get_device
          :summary:
   * - :py:obj:`load_nnet <deepxube.pytorch.nnet_utils.load_nnet>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.load_nnet
          :summary:
   * - :py:obj:`get_available_gpu_nums <deepxube.pytorch.nnet_utils.get_available_gpu_nums>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.get_available_gpu_nums
          :summary:
   * - :py:obj:`nnet_batched <deepxube.pytorch.nnet_utils.nnet_batched>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.nnet_batched
          :summary:
   * - :py:obj:`nnet_in_out_shared_q <deepxube.pytorch.nnet_utils.nnet_in_out_shared_q>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.nnet_in_out_shared_q
          :summary:
   * - :py:obj:`nnet_fn_runner <deepxube.pytorch.nnet_utils.nnet_fn_runner>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.nnet_fn_runner
          :summary:
   * - :py:obj:`get_nnet_par_infos <deepxube.pytorch.nnet_utils.get_nnet_par_infos>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.get_nnet_par_infos
          :summary:
   * - :py:obj:`start_nnet_fn_runners <deepxube.pytorch.nnet_utils.start_nnet_fn_runners>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.start_nnet_fn_runners
          :summary:
   * - :py:obj:`stop_nnet_fn_runners <deepxube.pytorch.nnet_utils.stop_nnet_fn_runners>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.stop_nnet_fn_runners
          :summary:
   * - :py:obj:`get_nnet_par_out <deepxube.pytorch.nnet_utils.get_nnet_par_out>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.get_nnet_par_out
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NNetCallable <deepxube.pytorch.nnet_utils.NNetCallable>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetCallable
          :summary:
   * - :py:obj:`NNF_T <deepxube.pytorch.nnet_utils.NNF_T>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNF_T
          :summary:
   * - :py:obj:`CTX_T <deepxube.pytorch.nnet_utils.CTX_T>`
     - .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.CTX_T
          :summary:

API
~~~

.. py:function:: to_pytorch_input(states_nnet: typing.List[numpy.typing.NDArray[typing.Any]], device: torch.device) -> typing.List[torch.Tensor]
   :canonical: deepxube.pytorch.nnet_utils.to_pytorch_input

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.to_pytorch_input

.. py:function:: get_device() -> typing.Tuple[torch.device, typing.List[int], bool]
   :canonical: deepxube.pytorch.nnet_utils.get_device

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.get_device

.. py:function:: load_nnet(model_file: str, nnet: torch.nn.Module, device: typing.Optional[torch.device] = None) -> torch.nn.Module
   :canonical: deepxube.pytorch.nnet_utils.load_nnet

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.load_nnet

.. py:function:: get_available_gpu_nums() -> typing.List[int]
   :canonical: deepxube.pytorch.nnet_utils.get_available_gpu_nums

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.get_available_gpu_nums

.. py:function:: nnet_batched(nnet: torch.nn.Module, inputs: typing.List[numpy.typing.NDArray[typing.Any]], batch_size: typing.Optional[int], device: torch.device) -> typing.List[numpy.typing.NDArray[numpy.float64]]
   :canonical: deepxube.pytorch.nnet_utils.nnet_batched

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.nnet_batched

.. py:class:: NNetParInfo
   :canonical: deepxube.pytorch.nnet_utils.NNetParInfo

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetParInfo

   .. py:attribute:: nnet_i_q
      :canonical: deepxube.pytorch.nnet_utils.NNetParInfo.nnet_i_q
      :type: torch.multiprocessing.Queue
      :value: None

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetParInfo.nnet_i_q

   .. py:attribute:: nnet_o_q
      :canonical: deepxube.pytorch.nnet_utils.NNetParInfo.nnet_o_q
      :type: torch.multiprocessing.Queue
      :value: None

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetParInfo.nnet_o_q

   .. py:attribute:: proc_id
      :canonical: deepxube.pytorch.nnet_utils.NNetParInfo.proc_id
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetParInfo.proc_id

.. py:function:: nnet_in_out_shared_q(nnet: torch.nn.Module, inputs_nnet_shm: typing.List[deepxube.utils.data_utils.SharedNDArray], batch_size: typing.Optional[int], device: torch.device, nnet_o_q: torch.multiprocessing.Queue) -> None
   :canonical: deepxube.pytorch.nnet_utils.nnet_in_out_shared_q

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.nnet_in_out_shared_q

.. py:function:: nnet_fn_runner(nnet_i_q: torch.multiprocessing.Queue, nnet_o_qs: typing.List[torch.multiprocessing.Queue], model_file: str, device: torch.device, on_gpu: bool, gpu_num: int, get_nnet: typing.Callable[[], torch.nn.Module], batch_size: typing.Optional[int]) -> None
   :canonical: deepxube.pytorch.nnet_utils.nnet_fn_runner

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.nnet_fn_runner

.. py:function:: get_nnet_par_infos(num_procs: int) -> typing.List[deepxube.pytorch.nnet_utils.NNetParInfo]
   :canonical: deepxube.pytorch.nnet_utils.get_nnet_par_infos

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.get_nnet_par_infos

.. py:function:: start_nnet_fn_runners(get_nnet: typing.Callable[[], torch.nn.Module], nnet_par_infos: typing.List[deepxube.pytorch.nnet_utils.NNetParInfo], model_file: str, device: torch.device, on_gpu: bool, batch_size: typing.Optional[int] = None) -> typing.List[multiprocessing.process.BaseProcess]
   :canonical: deepxube.pytorch.nnet_utils.start_nnet_fn_runners

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.start_nnet_fn_runners

.. py:function:: stop_nnet_fn_runners(nnet_fn_procs: typing.List[multiprocessing.process.BaseProcess], nnet_par_infos: typing.List[deepxube.pytorch.nnet_utils.NNetParInfo]) -> None
   :canonical: deepxube.pytorch.nnet_utils.stop_nnet_fn_runners

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.stop_nnet_fn_runners

.. py:function:: get_nnet_par_out(inputs_nnet: typing.List[numpy.typing.NDArray], nnet_par_info: deepxube.pytorch.nnet_utils.NNetParInfo) -> typing.List[numpy.typing.NDArray]
   :canonical: deepxube.pytorch.nnet_utils.get_nnet_par_out

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.get_nnet_par_out

.. py:data:: NNetCallable
   :canonical: deepxube.pytorch.nnet_utils.NNetCallable
   :value: None

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetCallable

.. py:data:: NNF_T
   :canonical: deepxube.pytorch.nnet_utils.NNF_T
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNF_T

.. py:data:: CTX_T
   :canonical: deepxube.pytorch.nnet_utils.CTX_T
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.CTX_T

.. py:class:: ProcessedInput
   :canonical: deepxube.pytorch.nnet_utils.ProcessedInput

   Bases: :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.pytorch.nnet_utils.CTX_T`\ ]

   .. py:attribute:: inputs_nnet
      :canonical: deepxube.pytorch.nnet_utils.ProcessedInput.inputs_nnet
      :type: typing.List[numpy.typing.NDArray]
      :value: None

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.ProcessedInput.inputs_nnet

   .. py:attribute:: ctx
      :canonical: deepxube.pytorch.nnet_utils.ProcessedInput.ctx
      :type: deepxube.pytorch.nnet_utils.CTX_T
      :value: None

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.ProcessedInput.ctx

.. py:class:: NNetPar(**kwargs: typing.Any)
   :canonical: deepxube.pytorch.nnet_utils.NNetPar

   Bases: :py:obj:`abc.ABC`, :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.pytorch.nnet_utils.NNF_T`\ , :py:obj:`deepxube.pytorch.nnet_utils.CTX_T`\ ]

   .. py:method:: get_nnet_fn(nnet: torch.nn.Module, batch_size: typing.Optional[int], device: torch.device) -> deepxube.pytorch.nnet_utils.NNF_T
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.get_nnet_fn

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.get_nnet_fn

   .. py:method:: get_nnet_par_fn_w_info(nnet_par_info: deepxube.pytorch.nnet_utils.NNetParInfo) -> deepxube.pytorch.nnet_utils.NNF_T
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.get_nnet_par_fn_w_info

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.get_nnet_par_fn_w_info

   .. py:method:: get_nnet() -> torch.nn.Module
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.get_nnet
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.get_nnet

   .. py:method:: process_inputs(*args: typing.Any) -> deepxube.pytorch.nnet_utils.ProcessedInput[deepxube.pytorch.nnet_utils.CTX_T]
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.process_inputs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.process_inputs

   .. py:method:: process_outputs(outs: typing.List[numpy.typing.NDArray], ctx: deepxube.pytorch.nnet_utils.CTX_T) -> typing.Any
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.process_outputs
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.process_outputs

   .. py:method:: get_default_fn() -> deepxube.pytorch.nnet_utils.NNF_T
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.get_default_fn
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.get_default_fn

   .. py:method:: set_nnet_file(nnet_file: str) -> None
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.set_nnet_file

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.set_nnet_file

   .. py:method:: get_nnet_par_infos(proc_idx: int) -> deepxube.pytorch.nnet_utils.NNetParInfo
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.get_nnet_par_infos

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.get_nnet_par_infos

   .. py:method:: get_nnet_par_fn() -> deepxube.pytorch.nnet_utils.NNF_T
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.get_nnet_par_fn

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.get_nnet_par_fn

   .. py:method:: set_nnet_par_info_l(num_procs: int) -> None
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.set_nnet_par_info_l

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.set_nnet_par_info_l

   .. py:method:: set_nnet_par_info(nnet_par_info: deepxube.pytorch.nnet_utils.NNetParInfo) -> None
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.set_nnet_par_info

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.set_nnet_par_info

   .. py:method:: start_nnet_runners(device: torch.device, on_gpu: bool, nnet_batch_size: typing.Optional[int]) -> None
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.start_nnet_runners

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.start_nnet_runners

   .. py:method:: stop_nnet_runners() -> None
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.stop_nnet_runners

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.stop_nnet_runners

   .. py:method:: init_nnet_par_fn() -> None
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.init_nnet_par_fn

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.init_nnet_par_fn

   .. py:method:: clear_nnet_fn() -> None
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.clear_nnet_fn

      .. autodoc2-docstring:: deepxube.pytorch.nnet_utils.NNetPar.clear_nnet_fn

   .. py:method:: __repr__() -> str
      :canonical: deepxube.pytorch.nnet_utils.NNetPar.__repr__
