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

.. py:class:: TrainPolicy(nnet_dir: str, updater: deepxube.base.trainer.Up, device: torch.device, on_gpu: bool, batch_size: int = 100, max_itrs: int = 100000, balance_steps: bool = False, loss_thresh: float = np.inf, checkpoint: int = 0, grad_accum: int = 1, display: int = 100)
   :canonical: deepxube.trainers.train_policy.TrainPolicy

   Bases: :py:obj:`deepxube.base.trainer.Train`\ [\ :py:obj:`deepxube.base.nnet.PolicyNNet`\ , :py:obj:`deepxube.base.updater.UpdatePolicy`\ ]

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
