:py:mod:`deepxube.trainers.train_policy`
========================================

.. py:module:: deepxube.trainers.train_policy

.. autodoc2-docstring:: deepxube.trainers.train_policy
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`TrainPolicy <deepxube.trainers.train_policy.TrainPolicy>`
     -

API
~~~

.. py:class:: TrainPolicy(nnet: deepxube.base.trainer.NNet, updater: deepxube.base.trainer.Up, to_main_q: multiprocessing.Queue, from_main_qs: typing.List[multiprocessing.Queue], nnet_file: str, nnet_targ_file: str, status_file: str, train_summary_file: str, device: torch.device, on_gpu: bool, writer: torch.utils.tensorboard.SummaryWriter, train_args: deepxube.base.trainer.TrainArgs)
   :canonical: deepxube.trainers.train_policy.TrainPolicy

   Bases: :py:obj:`deepxube.base.trainer.Train`\ [\ :py:obj:`deepxube.base.heuristic.PolicyNNet`\ , :py:obj:`deepxube.base.updater.UpdatePolicy`\ ]

   .. py:method:: data_parallel() -> bool
      :canonical: deepxube.trainers.train_policy.TrainPolicy.data_parallel
      :staticmethod:

      .. autodoc2-docstring:: deepxube.trainers.train_policy.TrainPolicy.data_parallel

   .. py:method:: _train_itr(batch: typing.List[numpy.typing.NDArray], first_itr_in_update: bool, times: deepxube.utils.timing_utils.Times) -> float
      :canonical: deepxube.trainers.train_policy.TrainPolicy._train_itr

      .. autodoc2-docstring:: deepxube.trainers.train_policy.TrainPolicy._train_itr

   .. py:method:: _add_post_up_info() -> typing.List[str]
      :canonical: deepxube.trainers.train_policy.TrainPolicy._add_post_up_info

      .. autodoc2-docstring:: deepxube.trainers.train_policy.TrainPolicy._add_post_up_info

   .. py:method:: _get_shapes_dtypes() -> typing.List[typing.Tuple[typing.Tuple[int, ...], numpy.dtype]]
      :canonical: deepxube.trainers.train_policy.TrainPolicy._get_shapes_dtypes

      .. autodoc2-docstring:: deepxube.trainers.train_policy.TrainPolicy._get_shapes_dtypes
