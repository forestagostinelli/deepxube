:py:mod:`deepxube.base.trainer`
===============================

.. py:module:: deepxube.base.trainer

.. autodoc2-docstring:: deepxube.base.trainer
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`TrainArgs <deepxube.base.trainer.TrainArgs>`
     - .. autodoc2-docstring:: deepxube.base.trainer.TrainArgs
          :summary:
   * - :py:obj:`DataBuffer <deepxube.base.trainer.DataBuffer>`
     - .. autodoc2-docstring:: deepxube.base.trainer.DataBuffer
          :summary:
   * - :py:obj:`Status <deepxube.base.trainer.Status>`
     - .. autodoc2-docstring:: deepxube.base.trainer.Status
          :summary:
   * - :py:obj:`TrainSummary <deepxube.base.trainer.TrainSummary>`
     - .. autodoc2-docstring:: deepxube.base.trainer.TrainSummary
          :summary:
   * - :py:obj:`Train <deepxube.base.trainer.Train>`
     -
   * - :py:obj:`TrainParser <deepxube.base.trainer.TrainParser>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`update_optimizer <deepxube.base.trainer.update_optimizer>`
     - .. autodoc2-docstring:: deepxube.base.trainer.update_optimizer
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NNet <deepxube.base.trainer.NNet>`
     - .. autodoc2-docstring:: deepxube.base.trainer.NNet
          :summary:
   * - :py:obj:`Up <deepxube.base.trainer.Up>`
     - .. autodoc2-docstring:: deepxube.base.trainer.Up
          :summary:

API
~~~

.. py:class:: TrainArgs
   :canonical: deepxube.base.trainer.TrainArgs

   .. autodoc2-docstring:: deepxube.base.trainer.TrainArgs

   .. py:attribute:: batch_size
      :canonical: deepxube.base.trainer.TrainArgs.batch_size
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.trainer.TrainArgs.batch_size

   .. py:attribute:: max_itrs
      :canonical: deepxube.base.trainer.TrainArgs.max_itrs
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.trainer.TrainArgs.max_itrs

   .. py:attribute:: balance_steps
      :canonical: deepxube.base.trainer.TrainArgs.balance_steps
      :type: bool
      :value: None

      .. autodoc2-docstring:: deepxube.base.trainer.TrainArgs.balance_steps

   .. py:attribute:: loss_thresh
      :canonical: deepxube.base.trainer.TrainArgs.loss_thresh
      :type: float
      :value: None

      .. autodoc2-docstring:: deepxube.base.trainer.TrainArgs.loss_thresh

   .. py:attribute:: checkpoint
      :canonical: deepxube.base.trainer.TrainArgs.checkpoint
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.trainer.TrainArgs.checkpoint

   .. py:attribute:: grad_accum
      :canonical: deepxube.base.trainer.TrainArgs.grad_accum
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.trainer.TrainArgs.grad_accum

   .. py:attribute:: display
      :canonical: deepxube.base.trainer.TrainArgs.display
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.trainer.TrainArgs.display

.. py:class:: DataBuffer(max_size: int, shapes: typing.List[typing.Tuple[int, ...]], dtypes: typing.List[numpy.dtype])
   :canonical: deepxube.base.trainer.DataBuffer

   .. autodoc2-docstring:: deepxube.base.trainer.DataBuffer

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.trainer.DataBuffer.__init__

   .. py:method:: add(arrays_add: typing.List[numpy.typing.NDArray]) -> None
      :canonical: deepxube.base.trainer.DataBuffer.add

      .. autodoc2-docstring:: deepxube.base.trainer.DataBuffer.add

   .. py:method:: sample(sel_idxs: numpy.typing.NDArray) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.trainer.DataBuffer.sample

      .. autodoc2-docstring:: deepxube.base.trainer.DataBuffer.sample

   .. py:method:: size() -> int
      :canonical: deepxube.base.trainer.DataBuffer.size

      .. autodoc2-docstring:: deepxube.base.trainer.DataBuffer.size

   .. py:method:: clear() -> None
      :canonical: deepxube.base.trainer.DataBuffer.clear

      .. autodoc2-docstring:: deepxube.base.trainer.DataBuffer.clear

   .. py:method:: _add_circular(arrays_add: typing.List[numpy.typing.NDArray]) -> None
      :canonical: deepxube.base.trainer.DataBuffer._add_circular

      .. autodoc2-docstring:: deepxube.base.trainer.DataBuffer._add_circular

.. py:class:: Status(step_max: int, balance_steps: bool)
   :canonical: deepxube.base.trainer.Status

   .. autodoc2-docstring:: deepxube.base.trainer.Status

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.trainer.Status.__init__

   .. py:method:: update_step_probs(step_to_search_perf: typing.Dict[int, deepxube.pathfinding.utils.performance.PathFindPerf]) -> None
      :canonical: deepxube.base.trainer.Status.update_step_probs

      .. autodoc2-docstring:: deepxube.base.trainer.Status.update_step_probs

.. py:class:: TrainSummary()
   :canonical: deepxube.base.trainer.TrainSummary

   .. autodoc2-docstring:: deepxube.base.trainer.TrainSummary

   .. rubric:: Initialization

   .. autodoc2-docstring:: deepxube.base.trainer.TrainSummary.__init__

   .. py:method:: update_pathfindstats(step_to_pathfindperf: typing.Dict[int, deepxube.pathfinding.utils.performance.PathFindPerf], itr: int) -> None
      :canonical: deepxube.base.trainer.TrainSummary.update_pathfindstats

      .. autodoc2-docstring:: deepxube.base.trainer.TrainSummary.update_pathfindstats

.. py:data:: NNet
   :canonical: deepxube.base.trainer.NNet
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.trainer.NNet

.. py:data:: Up
   :canonical: deepxube.base.trainer.Up
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.trainer.Up

.. py:function:: update_optimizer(optimizer: torch.optim.Optimizer, nnet: typing.Union[torch.nn.DataParallel, deepxube.base.nnet.DeepXubeNNet], train_itr: int) -> None
   :canonical: deepxube.base.trainer.update_optimizer

   .. autodoc2-docstring:: deepxube.base.trainer.update_optimizer

.. py:class:: Train(nnet_dir: str, updater: deepxube.base.trainer.Up, device: torch.device, on_gpu: bool, batch_size: int = 100, max_itrs: int = 100000, balance_steps: bool = False, loss_thresh: float = np.inf, checkpoint: int = 0, grad_accum: int = 1, display: int = 100)
   :canonical: deepxube.base.trainer.Train

   Bases: :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.trainer.NNet`\ , :py:obj:`deepxube.base.trainer.Up`\ ], :py:obj:`abc.ABC`

   .. py:method:: data_parallel() -> bool
      :canonical: deepxube.base.trainer.Train.data_parallel
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.trainer.Train.data_parallel

   .. py:method:: nnet_type() -> typing.Type[deepxube.base.trainer.NNet]
      :canonical: deepxube.base.trainer.Train.nnet_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.trainer.Train.nnet_type

   .. py:method:: updater_type() -> typing.Type[deepxube.base.trainer.Up]
      :canonical: deepxube.base.trainer.Train.updater_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.trainer.Train.updater_type

   .. py:method:: get_incompat_reason(updater: deepxube.base.updater.Update) -> typing.Optional[str]
      :canonical: deepxube.base.trainer.Train.get_incompat_reason
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.trainer.Train.get_incompat_reason

   .. py:method:: get_nnet_name() -> str
      :canonical: deepxube.base.trainer.Train.get_nnet_name
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.trainer.Train.get_nnet_name

   .. py:method:: train_loop() -> None
      :canonical: deepxube.base.trainer.Train.train_loop

      .. autodoc2-docstring:: deepxube.base.trainer.Train.train_loop

   .. py:method:: _update_step(to_main_q: multiprocessing.Queue, from_main_qs: typing.List[multiprocessing.Queue]) -> None
      :canonical: deepxube.base.trainer.Train._update_step

      .. autodoc2-docstring:: deepxube.base.trainer.Train._update_step

   .. py:method:: _get_update_data(num_gen: int, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.base.trainer.Train._get_update_data

      .. autodoc2-docstring:: deepxube.base.trainer.Train._get_update_data

   .. py:method:: _train(times: deepxube.utils.timing_utils.Times) -> float
      :canonical: deepxube.base.trainer.Train._train

      .. autodoc2-docstring:: deepxube.base.trainer.Train._train

   .. py:method:: _train_sync_main(num_gen: int, times: deepxube.utils.timing_utils.Times, to_main_q: multiprocessing.Queue, from_main_qs: typing.List[multiprocessing.Queue]) -> float
      :canonical: deepxube.base.trainer.Train._train_sync_main

      .. autodoc2-docstring:: deepxube.base.trainer.Train._train_sync_main

   .. py:method:: _train_itr(batch: typing.List[numpy.typing.NDArray], first_itr_in_update: bool, times: deepxube.utils.timing_utils.Times) -> float
      :canonical: deepxube.base.trainer.Train._train_itr
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.trainer.Train._train_itr

   .. py:method:: _end_update(itr_init: int, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.base.trainer.Train._end_update

      .. autodoc2-docstring:: deepxube.base.trainer.Train._end_update

   .. py:method:: _save_checkpoint() -> None
      :canonical: deepxube.base.trainer.Train._save_checkpoint

      .. autodoc2-docstring:: deepxube.base.trainer.Train._save_checkpoint

   .. py:method:: _add_post_up_info() -> typing.List[str]
      :canonical: deepxube.base.trainer.Train._add_post_up_info
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.trainer.Train._add_post_up_info

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.trainer.Train.__repr__

.. py:class:: TrainParser()
   :canonical: deepxube.base.trainer.TrainParser

   Bases: :py:obj:`deepxube.base.factory.DelimParser`

   .. py:property:: delim
      :canonical: deepxube.base.trainer.TrainParser.delim
      :type: str

      .. autodoc2-docstring:: deepxube.base.trainer.TrainParser.delim
