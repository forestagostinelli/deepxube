:py:mod:`deepxube.trainers.utils.train_utils`
=============================================

.. py:module:: deepxube.trainers.utils.train_utils

.. autodoc2-docstring:: deepxube.trainers.utils.train_utils
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ctgs_summary <deepxube.trainers.utils.train_utils.ctgs_summary>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_utils.ctgs_summary
          :summary:
   * - :py:obj:`get_deepxube_nnet <deepxube.trainers.utils.train_utils.get_deepxube_nnet>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_utils.get_deepxube_nnet
          :summary:
   * - :py:obj:`train_nnet_step <deepxube.trainers.utils.train_utils.train_nnet_step>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_utils.train_nnet_step
          :summary:

API
~~~

.. py:function:: ctgs_summary(ctgs_l: typing.List[numpy.typing.NDArray]) -> typing.Tuple[float, float, float]
   :canonical: deepxube.trainers.utils.train_utils.ctgs_summary

   .. autodoc2-docstring:: deepxube.trainers.utils.train_utils.ctgs_summary

.. py:function:: get_deepxube_nnet(nnet: typing.Union[deepxube.base.heuristic.DeepXubeNNet, torch.nn.DataParallel]) -> deepxube.base.heuristic.DeepXubeNNet
   :canonical: deepxube.trainers.utils.train_utils.get_deepxube_nnet

   .. autodoc2-docstring:: deepxube.trainers.utils.train_utils.get_deepxube_nnet

.. py:function:: train_nnet_step(nnet: typing.Union[deepxube.base.heuristic.DeepXubeNNet, torch.nn.DataParallel], data_np: typing.List[numpy.typing.NDArray], optimizer: torch.optim.optimizer.Optimizer, device: torch.device, train_itr: int, train_args: deepxube.base.trainer.TrainArgs, start_time: float) -> typing.Tuple[typing.List[numpy.typing.NDArray], float]
   :canonical: deepxube.trainers.utils.train_utils.train_nnet_step

   .. autodoc2-docstring:: deepxube.trainers.utils.train_utils.train_nnet_step
