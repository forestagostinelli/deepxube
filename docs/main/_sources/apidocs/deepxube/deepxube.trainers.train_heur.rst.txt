:py:mod:`deepxube.trainers.train_heur`
======================================

.. py:module:: deepxube.trainers.train_heur

.. autodoc2-docstring:: deepxube.trainers.train_heur
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`TrainHeur <deepxube.trainers.train_heur.TrainHeur>`
     -
   * - :py:obj:`TrainHeurParser <deepxube.trainers.train_heur.TrainHeurParser>`
     -

API
~~~

.. py:class:: TrainHeur(nnet_dir: str, updater: deepxube.base.trainer.Up, device: torch.device, on_gpu: bool, batch_size: int = 100, max_itrs: int = 100000, balance_steps: bool = False, loss_thresh: float = np.inf, checkpoint: int = 0, grad_accum: int = 1, display: int = 100)
   :canonical: deepxube.trainers.train_heur.TrainHeur

   Bases: :py:obj:`deepxube.base.trainer.Train`\ [\ :py:obj:`deepxube.base.nnet.HeurNNet`\ , :py:obj:`deepxube.base.updater.UpdateHeur`\ ], :py:obj:`abc.ABC`

   .. py:method:: data_parallel() -> bool
      :canonical: deepxube.trainers.train_heur.TrainHeur.data_parallel
      :staticmethod:

      .. autodoc2-docstring:: deepxube.trainers.train_heur.TrainHeur.data_parallel

   .. py:method:: nnet_type() -> typing.Type[deepxube.base.nnet.HeurNNet]
      :canonical: deepxube.trainers.train_heur.TrainHeur.nnet_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.trainers.train_heur.TrainHeur.nnet_type

   .. py:method:: updater_type() -> typing.Type[deepxube.base.updater.UpdateHeur]
      :canonical: deepxube.trainers.train_heur.TrainHeur.updater_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.trainers.train_heur.TrainHeur.updater_type

   .. py:method:: get_nnet_name() -> str
      :canonical: deepxube.trainers.train_heur.TrainHeur.get_nnet_name
      :staticmethod:

      .. autodoc2-docstring:: deepxube.trainers.train_heur.TrainHeur.get_nnet_name

   .. py:method:: _train_itr(batch: typing.List[numpy.typing.NDArray], first_itr_in_update: bool, times: deepxube.utils.timing_utils.Times) -> float
      :canonical: deepxube.trainers.train_heur.TrainHeur._train_itr

      .. autodoc2-docstring:: deepxube.trainers.train_heur.TrainHeur._train_itr

   .. py:method:: _add_post_up_info() -> typing.List[str]
      :canonical: deepxube.trainers.train_heur.TrainHeur._add_post_up_info

      .. autodoc2-docstring:: deepxube.trainers.train_heur.TrainHeur._add_post_up_info

.. py:class:: TrainHeurParser()
   :canonical: deepxube.trainers.train_heur.TrainHeurParser

   Bases: :py:obj:`deepxube.base.trainer.TrainParser`
